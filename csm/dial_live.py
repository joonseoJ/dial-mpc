"""Interactive DIAL-MPC viewer: live weights, live temperature, live pushes.

Runs real DIAL-MPC in a background thread and streams it as MJPEG, with every
knob that matters exposed to a browser on the other end of an ``ssh -L``
tunnel.  Nothing here is a trained policy -- it is the planner itself, so what
you see is what the objective actually buys.

Three things are adjustable without recompiling anything:

  * the reward weights, because they live in ``state.info["reward_weights"]``
    and are read by the rollouts through the state;
  * the sampling temperature, because it is passed into the jitted update as a
    traced scalar rather than closed over;
  * disturbances, applied on demand as a velocity impulse on the base.

The episode has no step limit.  It ends when the robot goes down, and the page
reports how long the *current* episode has survived -- previous episodes are
kept only as a history line.

The temperature panel exists because DIAL's softmax is easy to misjudge.
``reverse_once`` divides sample returns by their own standard deviation before
the softmax, so the useful temperature range depends on nothing but the shape
of the return distribution, and at the stock 0.05 the update is frequently a
hard argmax over a couple of thousand samples.  The panel reports effective
sample size and where the weight mass sits, so a temperature can be chosen by
looking rather than by guessing.

Note that the weight vector's *scale* is inert for the same reason -- the
standard-deviation normalisation makes ``omega`` and ``2 * omega`` produce
identical updates.  Only the direction matters, and the page shows it.
"""

from __future__ import annotations

import os

# Set before anything pulls in mujoco: the renderer needs a headless GL backend
# and picking one after the library is imported has no effect.
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import dataclasses
import io
import threading
import time
from pathlib import Path

import numpy as np
import yaml

import jax
import jax.numpy as jnp

from brax import envs as brax_envs
from flask import Flask, Response, jsonify, request

import dial_mpc.envs as dial_envs
from dial_mpc.core.dial_config import DialConfig
from dial_mpc.core.dial_core import make_controller
from dial_mpc.utils.io_utils import get_example_path, load_dataclass_from_dict

from csm.dial_lean import make_rollout, make_sampler
from csm.dial_score_serve import FrameRenderer, _label

from PIL import Image


# Cumulative weight mass is reported at these ranks; the shape of this list is
# what tells "one sample owns the update" apart from "the cloud is voting".
MASS_RANKS = (1, 4, 16, 64, 256, 1024)


def _load(example: str | None, config_path: Path | None):
    if example is not None:
        config_dict = yaml.safe_load(open(get_example_path(example + ".yaml")))
    else:
        config_dict = yaml.safe_load(open(config_path))
    dial_config = load_dataclass_from_dict(DialConfig, config_dict)
    env_config = load_dataclass_from_dict(
        dial_envs.get_config(dial_config.env_name),
        config_dict,
        convert_list_to_array=True,
    )
    return dial_config, env_config


def make_interactive_step(env, mbdpi, dial_config):
    """One DIAL control step with the temperature left traced.

    Returns the softmax weights of the final annealing level as well, since
    that is the distribution the applied action actually came from.
    """

    sample = make_sampler(mbdpi, dial_config)
    rollout_vmap = jax.vmap(make_rollout(env), in_axes=(None, 0))
    sigma = mbdpi.sigma_control
    factors = dial_config.traj_diffuse_factor ** jnp.arange(dial_config.Ndiffuse)

    def anneal_once(state, rng, plan, noise_scale, temp):
        rng, key = jax.random.split(rng)
        nodes = sample(key, plan, noise_scale)
        us = mbdpi.node2u_vvmap(nodes)
        returns = rollout_vmap(state, us).mean(axis=-1)
        # Exactly DIAL's normalisation: score against the proposal centre and
        # divide by the spread of the cloud itself.
        scale = jnp.maximum(returns.std(), 1e-6)
        weights = jax.nn.softmax((returns - returns[-1]) / scale / temp)
        return rng, jnp.einsum("n,nij->ij", weights, nodes), weights, returns

    @jax.jit
    def control_step(state, rng, plan, temp, shift):
        plan = jnp.where(shift, mbdpi.shift(plan), plan)

        def body(carry, factor):
            key, current = carry
            key, current, weights, returns = anneal_once(
                state, key, current, sigma * factor, temp
            )
            return (key, current), (weights, returns)

        (rng, plan), (weights, returns) = jax.lax.scan(body, (rng, plan), factors)
        state = env.step(state, plan[0])
        return state, rng, plan, weights[-1], returns[-1]

    return control_step


class LiveDial:
    def __init__(self, args) -> None:
        self.args = args
        dial_config, env_config = _load(args.example, args.config)
        overrides = {"time_correlated": False}
        if args.samples is not None:
            overrides["Nsample"] = int(args.samples)
        if args.ndiffuse is not None:
            overrides["Ndiffuse"] = int(args.ndiffuse)
        self.dial_config = dataclasses.replace(dial_config, **overrides)
        self.env = brax_envs.get_environment(
            self.dial_config.env_name, config=env_config
        )
        self.mbdpi = make_controller(self.dial_config, self.env)
        self.n_sample = self.dial_config.Nsample + 1
        self.n_rows = int(np.asarray(env_config.reward_weights).shape[0])
        self.row_names = list(args.row_names or [])
        while len(self.row_names) < self.n_rows:
            self.row_names.append(f"row{len(self.row_names)}")

        self._reset_env = jax.jit(self.env.reset)
        self._control = make_interactive_step(self.env, self.mbdpi, self.dial_config)

        @jax.jit
        def push(state, impulse):
            ps = state.pipeline_state
            pipeline_state = self.env.pipeline_init(
                ps.qpos, ps.qvel.at[:3].add(impulse)
            )
            obs = self.env._get_obs(pipeline_state, state.info)
            return state.replace(pipeline_state=pipeline_state, obs=obs)

        self._push = push

        self.control_dt = float(self.env.dt)
        self.camera = args.camera or "track"
        self.render_every = max(
            1, int(round((1.0 / self.control_dt) / max(args.fps, 1e-6)))
        )

        self._lock = threading.Lock()
        self._condition = threading.Condition()
        self._frame: bytes | None = None
        self._commands: list[tuple[str, object]] = []
        self.controls = {
            "omega": [float(v) for v in np.asarray(env_config.reward_weights)],
            "temp": float(self.dial_config.temp_sample),
        }
        self.stats: dict[str, object] = {
            "status": "starting",
            "env": self.dial_config.env_name,
            "n_sample": self.n_sample,
            "n_diffuse": int(self.dial_config.Ndiffuse),
            "row_names": self.row_names,
            "mass_ranks": list(MASS_RANKS),
        }

    # ---------------------------------------------------------------- frames
    def publish(self, pixels, caption: str) -> None:
        image = _label(Image.fromarray(np.asarray(pixels)), caption)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=self.args.quality)
        with self._condition:
            self._frame = buffer.getvalue()
            self._condition.notify_all()

    def frames(self):
        previous = None
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._frame is not None and self._frame is not previous
                )
                previous = self._frame
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + previous + b"\r\n"

    # -------------------------------------------------------------- commands
    def submit(self, name: str, payload: object = None) -> None:
        with self._lock:
            self._commands.append((name, payload))

    def set_controls(self, omega=None, temp=None) -> None:
        with self._lock:
            if omega is not None:
                self.controls["omega"] = [float(v) for v in omega]
            if temp is not None:
                self.controls["temp"] = max(float(temp), 1e-4)

    def _drain(self):
        with self._lock:
            pending, self._commands = self._commands, []
            omega = jnp.asarray(self.controls["omega"], dtype=jnp.float32)
            temp = jnp.asarray(self.controls["temp"], dtype=jnp.float32)
        return pending, omega, temp

    # ------------------------------------------------------------ diagnostics
    def weight_report(self, weights: np.ndarray) -> dict:
        """How concentrated is the softmax that produced this action?"""

        ess = float(1.0 / np.sum(weights**2))
        order = np.sort(weights)[::-1]
        cumulative = np.cumsum(order)
        mass = {
            str(k): float(cumulative[min(k, len(order)) - 1]) for k in MASS_RANKS
        }
        share = ess / self.n_sample
        if ess < 3:
            verdict, hint = "argmax", "온도를 올리세요 — 사실상 최고 샘플 하나만 씁니다"
        elif ess < 20:
            verdict, hint = "sharp", "소수 샘플이 지배합니다 — 업데이트가 공격적입니다"
        elif share < 0.25:
            verdict, hint = "healthy", "적당한 가중 평균입니다"
        elif share < 0.7:
            verdict, hint = "soft", "분포가 넓습니다 — 온도를 낮춰도 됩니다"
        else:
            verdict, hint = "flat", "온도가 너무 높습니다 — 거의 균등 평균입니다"
        return {
            "ess": round(ess, 1),
            "ess_share": round(share, 4),
            "top1": round(float(order[0]), 4),
            "mass": mass,
            "verdict": verdict,
            "hint": hint,
        }

    # ------------------------------------------------------------------ loop
    def run(self) -> None:
        args = self.args
        seed = int(args.seed)
        rng = jax.random.PRNGKey(seed + 7919)
        state = self._reset_env(jax.random.PRNGKey(seed))
        plan = jnp.zeros(
            (self.dial_config.Hnode + 1, int(self.env.action_size)), dtype=jnp.float32
        )
        renderer = FrameRenderer(self.env.sys, args.width, args.height, self.camera)

        self.stats["status"] = "compiling"
        episode_step = 0
        episode_reward = 0.0
        step_index = 0
        falls = 0
        history: list[float] = []
        rate_window = time.perf_counter()
        rate_steps = 0
        control_hz = 0.0
        fresh = True

        while True:
            pending, omega, temp = self._drain()
            reset_now = False
            for name, payload in pending:
                if name == "reset":
                    reset_now = True
                elif name == "push":
                    state = self._push(state, jnp.asarray(payload, dtype=jnp.float32))

            if reset_now:
                seed += 1
                state = self._reset_env(jax.random.PRNGKey(seed))
                plan = jnp.zeros_like(plan)
                episode_step, episode_reward, fresh = 0, 0.0, True

            # The weights are read out of the state by the rollouts, so writing
            # them here is all it takes for a slider to reach the planner.
            state.info["reward_weights"] = omega

            state, rng, plan, weights, returns = self._control(
                state, rng, plan, temp, not fresh
            )
            plan.block_until_ready()
            fresh = False

            reward = float(state.reward)
            terms = np.asarray(state.info["reward_terms"])
            episode_reward += reward
            episode_step += 1
            step_index += 1
            rate_steps += 1

            if step_index % self.render_every == 0:
                self.publish(
                    renderer.render(state.pipeline_state),
                    f"t={episode_step * self.control_dt:6.2f}s  cost={-reward:8.3f}",
                )

            now = time.perf_counter()
            if now - rate_window >= 1.0:
                control_hz = rate_steps / (now - rate_window)
                rate_window, rate_steps = now, 0

            omega_np = np.asarray(omega, dtype=float)
            total = float(np.abs(omega_np).sum()) or 1.0
            report = self.weight_report(np.asarray(weights, dtype=np.float64))
            self.stats.update(
                status="running",
                episode_seconds=round(episode_step * self.control_dt, 2),
                episode_step=episode_step,
                episode_mean_cost=round(-episode_reward / max(episode_step, 1), 4),
                cost=round(-reward, 4),
                terms=[round(float(v), 4) for v in terms],
                weighted_terms=[
                    round(float(w * v), 4) for w, v in zip(omega_np, terms)
                ],
                omega=[round(float(v), 3) for v in omega_np],
                omega_direction=[round(float(v / total), 3) for v in omega_np],
                temp=round(float(temp), 5),
                falls=falls,
                last_episodes=[round(v, 2) for v in history[-8:]],
                best_episode=round(max(history), 2) if history else None,
                control_hz=round(control_hz, 1),
                realtime=round(control_hz * self.control_dt, 3),
                seed=seed,
                sample=report,
                return_spread=round(float(np.asarray(returns).std()), 4),
            )

            if float(state.done) > 0.5:
                falls += 1
                history.append(episode_step * self.control_dt)
                seed += 1
                state = self._reset_env(jax.random.PRNGKey(seed))
                plan = jnp.zeros_like(plan)
                episode_step, episode_reward, fresh = 0, 0.0, True


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>DIAL-MPC live</title>
<style>
 body{margin:0;background:#12151b;color:#e8ecf2;
      font:13px ui-monospace,SFMono-Regular,Menlo,monospace}
 header{padding:11px 18px;border-bottom:1px solid #2b323e;display:flex;
        gap:16px;align-items:baseline}
 h1{font-size:13px;margin:0;letter-spacing:.14em;text-transform:uppercase;color:#8d99a9}
 .env{color:#5f6b7a;font-size:12px}
 main{display:grid;grid-template-columns:minmax(360px,1fr) 380px;gap:18px;
      padding:18px;align-items:start}
 @media(max-width:900px){main{grid-template-columns:1fr}}
 img{background:#000;width:100%;border:1px solid #2b323e;display:block}
 .card{background:#181c24;border:1px solid #2b323e;padding:14px 16px;margin-bottom:14px}
 .card h2{font-size:11px;letter-spacing:.13em;text-transform:uppercase;
          color:#8d99a9;margin:0 0 12px}
 table{border-collapse:collapse;width:100%}
 td{padding:3px 0}
 td:first-child{color:#6d7887;font-size:11px;letter-spacing:.06em;
                text-transform:uppercase;width:52%}
 td:last-child{text-align:right;font-variant-numeric:tabular-nums}
 .big{font-size:26px;font-variant-numeric:tabular-nums;letter-spacing:-.01em}
 .row{display:flex;align-items:center;gap:9px;margin:7px 0}
 .row label{width:74px;color:#8d99a9;font-size:11px;text-transform:uppercase;
            letter-spacing:.06em}
 input[type=range]{flex:1;accent-color:#5b9dd9}
 .num{width:56px;font-variant-numeric:tabular-nums;text-align:right;color:#c7d0dc}
 button{font:inherit;background:#20262f;color:#e8ecf2;
        border:1px solid #3a4350;padding:6px 11px;cursor:pointer;border-radius:2px}
 button:hover{border-color:#6d7887}
 .btnrow{display:flex;gap:7px;flex-wrap:wrap;margin-top:10px}
 .bars{display:flex;align-items:flex-end;gap:3px;height:56px;margin:10px 0 4px}
 .bars div{flex:1;background:#5b9dd9;min-height:1px}
 .barlab{display:flex;gap:3px;color:#5f6b7a;font-size:10px}
 .barlab span{flex:1;text-align:center}
 .verdict{font-size:15px;letter-spacing:.04em}
 .hint{color:#8d99a9;font-size:11px;margin-top:5px;line-height:1.5}
 .v-argmax,.v-flat{color:#e0925f}
 .v-healthy{color:#7fc088}
 .v-sharp,.v-soft{color:#d9c86b}
</style></head><body>
<header><h1>DIAL-MPC live</h1><span class="env" id="env"></span></header>
<main>
 <div>
  <img id="view" src="/stream.mjpg" alt="live rollout">
  <div class="card" style="margin-top:14px">
   <h2>현재 에피소드</h2>
   <div class="big"><span id="secs">–</span> s</div>
   <table><tbody id="ep"></tbody></table>
   <div class="btnrow">
    <button onclick="post('/reset')">리셋</button>
    <button onclick="push(0)">밀기 →</button>
    <button onclick="push(180)">밀기 ←</button>
    <button onclick="push(90)">밀기 ↑</button>
    <button onclick="push(-1)">랜덤 방향</button>
   </div>
   <div class="row"><label>세기</label>
    <input type="range" id="pmag" min="0.1" max="2.5" step="0.05" value="0.8"
           oninput="pmagv.textContent=this.value">
    <span class="num" id="pmagv">0.80</span></div>
  </div>
 </div>
 <div>
  <div class="card">
   <h2>보상 가중치 ω</h2>
   <div id="omega"></div>
   <div class="hint">스케일은 무시됩니다 — DIAL이 리턴을 표준편차로 나누므로
    ω와 2ω의 업데이트가 동일합니다. 방향만 의미가 있습니다.</div>
   <div class="btnrow" id="presets"></div>
  </div>
  <div class="card">
   <h2>온도</h2>
   <div class="row"><label>temp</label>
    <input type="range" id="temp" min="-3" max="0.3" step="0.02"
           oninput="applyTemp(this.value)">
    <span class="num" id="tempv">–</span></div>
   <div class="verdict" id="verdict">–</div>
   <div class="bars" id="bars"></div>
   <div class="barlab" id="barlab"></div>
   <div class="hint" id="hint"></div>
   <table><tbody id="samp"></tbody></table>
  </div>
  <div class="card"><h2>비용</h2><table><tbody id="cost"></tbody></table></div>
  <div class="card"><h2>실행</h2><table><tbody id="run"></tbody></table></div>
 </div>
</main>
<script>
let S={}, touching=false;
const post=(u,b)=>fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},
                          body:JSON.stringify(b||{})});
const push=deg=>post('/push',{magnitude:+pmag.value,
   heading:deg<0?null:deg*Math.PI/180});
function rows(el,pairs){el.innerHTML=pairs.map(([k,v])=>
  `<tr><td>${k}</td><td>${v}</td></tr>`).join('')}

function buildOmega(names,values){
  omega.innerHTML=names.map((n,i)=>`
   <div class="row"><label>${n}</label>
    <input type="range" id="w${i}" min="0" max="5" step="0.05" value="${values[i]}"
     oninput="wv${i}.textContent=(+this.value).toFixed(2);sendOmega()">
    <span class="num" id="wv${i}">${(+values[i]).toFixed(2)}</span></div>`).join('');
  const sets={'균등':names.map(()=>1)};
  names.forEach((n,i)=>sets['only '+n]=names.map((_,j)=>j===i?1:0));
  names.forEach((n,i)=>sets['boost '+n]=names.map((_,j)=>j===i?3:1));
  presets.innerHTML=Object.keys(sets).map(k=>
    `<button onclick='setOmega(${JSON.stringify(sets[k])})'>${k}</button>`).join('');
}
function setOmega(v){v.forEach((x,i)=>{const s=document.getElementById('w'+i);
  s.value=x;document.getElementById('wv'+i).textContent=(+x).toFixed(2)});sendOmega()}
function sendOmega(){const v=S.row_names.map((_,i)=>+document.getElementById('w'+i).value);
  post('/controls',{omega:v})}
function applyTemp(log){const t=Math.pow(10,+log);tempv.textContent=t.toFixed(4);
  post('/controls',{temp:t})}

async function tick(){
  const s=await (await fetch('/stats')).json(); S=s;
  env.textContent=`${s.env} · ${s.n_sample} samples · ${s.n_diffuse} anneal levels`;
  if(s.status!=='running')return;
  if(!document.getElementById('w0')) buildOmega(s.row_names,s.omega);
  if(tempv.textContent==='–'){temp.value=Math.log10(s.temp);
    tempv.textContent=s.temp.toFixed(4)}
  secs.textContent=s.episode_seconds.toFixed(2);
  rows(ep,[['스텝',s.episode_step],['평균 비용',s.episode_mean_cost],
    ['낙상 횟수',s.falls],['최장 에피소드',s.best_episode??'–'],
    ['최근 (s)',(s.last_episodes||[]).join(' ')||'–']]);
  const v=s.sample;
  verdict.textContent=v.verdict.toUpperCase();
  verdict.className='verdict v-'+v.verdict;
  hint.textContent=v.hint;
  const m=s.mass_ranks.map(k=>v.mass[String(k)]);
  bars.innerHTML=m.map(x=>`<div style="height:${Math.max(x*100,1)}%"></div>`).join('');
  barlab.innerHTML=s.mass_ranks.map(k=>`<span>${k}</span>`).join('');
  rows(samp,[['유효 샘플수 (ESS)',`${v.ess} / ${s.n_sample}`],
    ['ESS 비율',(v.ess_share*100).toFixed(2)+' %'],
    ['최대 가중치',v.top1],['리턴 표준편차',s.return_spread]]);
  rows(cost,[['현재 비용',s.cost],
    ...s.row_names.map((n,i)=>[n,`${s.terms[i]}  ×${s.omega[i]} = ${s.weighted_terms[i]}`]),
    ['ω 방향',s.omega_direction.join('  ')]]);
  rows(run,[['제어 Hz',s.control_hz],['실시간 배속',s.realtime],['시드',s.seed]]);
}
tick();setInterval(tick,400);
</script></body></html>"""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--example", default=None)
    source.add_argument("--config", type=Path, default=None)
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address; keep the default and use ssh -L")
    parser.add_argument("--port", type=int, default=8083)
    parser.add_argument("--width", type=int, default=560)
    parser.add_argument("--height", type=int, default=420)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--quality", type=int, default=80)
    parser.add_argument("--camera", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=None,
                        help="override Nsample; lower it for a snappier viewer")
    parser.add_argument("--ndiffuse", type=int, default=None)
    parser.add_argument("--row-names", nargs="+", default=None,
                        help="labels for the reward rows, in order")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.row_names is None and args.example == "unitree_go2_push_recover":
        args.row_names = ["tilt", "base", "feet", "shape"]
    live = LiveDial(args)

    app = Flask("dial_live")

    @app.get("/")
    def index():
        return PAGE

    @app.get("/stream.mjpg")
    def stream():
        return Response(
            live.frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
        )

    @app.get("/stats")
    def stats():
        return jsonify(live.stats)

    @app.post("/controls")
    def controls():
        body = request.get_json(silent=True) or {}
        live.set_controls(omega=body.get("omega"), temp=body.get("temp"))
        return jsonify(ok=True, **live.controls)

    @app.post("/reset")
    def reset():
        live.submit("reset")
        return jsonify(ok=True)

    @app.post("/push")
    def push():
        body = request.get_json(silent=True) or {}
        magnitude = float(body.get("magnitude", 0.8))
        heading = body.get("heading")
        if heading is None:
            heading = float(np.random.uniform(-np.pi, np.pi))
        impulse = [
            magnitude * float(np.cos(heading)),
            magnitude * float(np.sin(heading)),
            0.0,
        ]
        live.submit("push", impulse)
        return jsonify(ok=True, impulse=impulse)

    threading.Thread(target=live.run, daemon=True).start()
    print(f"serving http://{args.host}:{args.port}  (stream at /stream.mjpg)")
    print("the first frame appears once JAX has finished compiling")
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
