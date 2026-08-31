"""Interactive viewer for a trained composed score policy.

The DIAL viewer exposes the planner's knobs -- sampling temperature, effective
sample size -- because those are what a sampler is.  A composed student has a
different set: the weight vector picks a direction, the target temperature
picks a sharpness, and together they fix four coefficients that mix four fixed
networks.  Those coefficients are the thing worth watching, so they are what
this page shows.

Everything is live and nothing recompiles.  The weights and the target
temperature enter the jitted step as traced arrays and the composition solve
happens inside it, which is a 4x4 product against a stored pseudoinverse.
Retempering is available for the same reason: a target temperature different
from the one the fields were fitted at simply scales the coefficients, so it
costs nothing to expose.

``--with-dial`` runs the planner beside the student from the same state and
applies the same shove to both.  Expect the pair to advance at DIAL's pace,
which on this plant is roughly twenty times slower than the student.
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

import jax
import jax.numpy as jnp

from brax import envs as brax_envs
from flask import Flask, Response, jsonify, request

from dial_mpc.core.dial_core import make_controller

from csm.basis_screen import _load_config, build_omegas
from csm.dial_lean import make_dial_step
from csm.dial_score import ComposedDialScorePolicy, factor_to_t
from csm.dial_score_serve import FrameRenderer, _label
from csm.omega import mixture_from_pinv, normalize_omega, normalize_omega_np

from PIL import Image


def make_student_step(env, policy, dial_config, init_passes: int):
    """One control step of the composed policy, with its mixing exposed.

    Two jitted entry points rather than a traced branch: the first control step
    of an episode tiles the annealing schedule `init_passes` times from a zero
    plan, and folding that into one program would make every later step pay for
    a schedule it does not use.
    """

    fields = policy.policies
    factors = jnp.asarray(fields[0].factors)
    factor_min, factor_max = float(jnp.min(factors)), float(jnp.max(factors))
    shift = jnp.asarray(fields[0].shift_matrix)
    pinv_nu = getattr(policy, "pinv_nu_weights", None)
    pinv_mode = policy.pinv_mode_weights

    def coefficients(omega, temperature):
        return mixture_from_pinv(omega, temperature, pinv_nu, pinv_mode)

    def refine(plan, obs, mixture, passes):
        def level(carry, factor):
            current = carry
            t = factor_to_t(factor, factor_min, factor_max).reshape(1)
            parts = jnp.stack([f.delta(current, obs, t) for f in fields])
            update = jnp.einsum("k,kij->ij", mixture, parts)
            return jnp.clip(current + update, -1.0, 1.0), (parts, update)

        plan, (parts, updates) = jax.lax.scan(
            level, plan, jnp.tile(factors, passes)
        )
        # Report the finest level, the one that decides the applied action.
        return plan, parts[-1], updates[-1]

    def build(passes):
        @jax.jit
        def run(state, plan, omega, temperature):
            mixture = coefficients(omega, temperature)
            plan, parts, update = refine(plan, state.obs, mixture, passes)
            state = env.step(state, plan[0])
            plan_next = jnp.einsum("ij,ja->ia", shift, plan)
            diagnostics = {
                "mixture": mixture,
                "field_norm": jnp.linalg.norm(parts.reshape(parts.shape[0], -1),
                                              axis=-1),
                "contribution": jnp.abs(mixture) * jnp.linalg.norm(
                    parts.reshape(parts.shape[0], -1), axis=-1
                ),
                "update_norm": jnp.linalg.norm(update),
            }
            return state, plan_next, diagnostics

        return run

    return build(init_passes), build(1)


class LiveComposed:
    def __init__(self, args) -> None:
        self.args = args
        dial_config, env_config = _load_config(args.example, args.config)
        dial_config = dataclasses.replace(
            dial_config, temp_sample=args.temperature, time_correlated=False
        )
        if args.samples is not None:
            dial_config = dataclasses.replace(dial_config, Nsample=int(args.samples))
        self.dial_config = dial_config
        self.env = brax_envs.get_environment(dial_config.env_name, config=env_config)
        self.policy = ComposedDialScorePolicy.load(args.policy)
        self.n_rows = int(np.asarray(env_config.reward_weights).shape[0])
        self.row_names = list(args.row_names or [])
        while len(self.row_names) < self.n_rows:
            self.row_names.append(f"row{len(self.row_names)}")
        self.basis_names = list(args.basis)

        self._first, self._later = make_student_step(
            self.env, self.policy, dial_config, args.init_passes
        )
        self._reset = jax.jit(self.env.reset)

        @jax.jit
        def push(state, impulse):
            ps = state.pipeline_state
            pipeline_state = self.env.pipeline_init(
                ps.qpos, ps.qvel.at[:3].add(impulse)
            )
            return state.replace(
                pipeline_state=pipeline_state,
                obs=self.env._get_obs(pipeline_state, state.info),
            )

        self._push = push

        self.with_dial = bool(args.with_dial)
        if self.with_dial:
            self.mbdpi = make_controller(dial_config, self.env)
            self._dial = jax.jit(make_dial_step(
                self.env, self.mbdpi, dial_config, std_normalize=False,
                level_scales=tuple(args.level_scales),
            ))

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
            "omega": [1.0] * self.n_rows,
            "temperature": float(self.policy.temperature or args.temperature),
        }
        self.stats: dict[str, object] = {
            "status": "starting",
            "env": dial_config.env_name,
            "policy": str(args.policy),
            "row_names": self.row_names,
            "basis_names": self.basis_names,
            "fit_temperature": float(self.policy.temperature or args.temperature),
            "with_dial": self.with_dial,
        }

    # ---------------------------------------------------------------- frames
    def publish(self, panels) -> None:
        images = [_label(Image.fromarray(np.asarray(p)), name) for name, p in panels]
        if len(images) == 1:
            canvas = images[0]
        else:
            width = sum(i.width for i in images) + 2 * (len(images) - 1)
            canvas = Image.new("RGB", (width, images[0].height), (18, 21, 27))
            offset = 0
            for image in images:
                canvas.paste(image, (offset, 0))
                offset += image.width + 2
        buffer = io.BytesIO()
        canvas.save(buffer, format="JPEG", quality=self.args.quality)
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

    def set_controls(self, omega=None, temperature=None) -> None:
        with self._lock:
            if omega is not None:
                self.controls["omega"] = [float(v) for v in omega]
            if temperature is not None:
                self.controls["temperature"] = max(float(temperature), 1e-4)

    def _drain(self):
        with self._lock:
            pending, self._commands = self._commands, []
            omega = jnp.asarray(self.controls["omega"], dtype=jnp.float32)
            temperature = jnp.asarray(
                self.controls["temperature"], dtype=jnp.float32
            )
        return pending, omega, temperature

    # ------------------------------------------------------------------ loop
    def run(self) -> None:
        args = self.args
        seed = int(args.seed)
        state = self._reset(jax.random.PRNGKey(seed))
        dial_state = state if self.with_dial else None
        rng = jax.random.PRNGKey(seed + 7919)
        plan = jnp.zeros(
            (self.dial_config.Hnode + 1, int(self.env.action_size)),
            dtype=jnp.float32,
        )
        dial_plan = plan
        renderer = FrameRenderer(self.env.sys, args.width, args.height, self.camera)
        self.stats["status"] = "compiling"

        episode_step = 0
        episode_cost = 0.0
        dial_cost = 0.0
        step_index = 0
        falls = 0
        history: list[float] = []
        rate_window, rate_steps, control_hz = time.perf_counter(), 0, 0.0
        fresh = True
        period = 1.0 / (1.0 / self.control_dt)
        deadline = time.perf_counter()

        while True:
            pending, omega, temperature = self._drain()
            reset_now = False
            for name, payload in pending:
                if name == "reset":
                    reset_now = True
                elif name == "push":
                    impulse = jnp.asarray(payload, dtype=jnp.float32)
                    state = self._push(state, impulse)
                    if self.with_dial:
                        dial_state = self._push(dial_state, impulse)

            if reset_now:
                seed += 1
                state = self._reset(jax.random.PRNGKey(seed))
                dial_state = state if self.with_dial else None
                plan = jnp.zeros_like(plan)
                dial_plan = plan
                episode_step, episode_cost, dial_cost, fresh = 0, 0.0, 0.0, True

            state.info["reward_weights"] = omega
            step_fn = self._first if fresh else self._later
            state, plan, diagnostics = step_fn(state, plan, omega, temperature)
            plan.block_until_ready()
            fresh = False

            if self.with_dial:
                dial_state.info["reward_weights"] = omega
                dial_state, rng, dial_plan = self._dial(dial_state, rng, dial_plan)
                dial_cost += -float(dial_state.reward)

            cost = -float(state.reward)
            # Not every environment publishes per-row rewards; the viewer
            # should still run on the ones that do not.
            terms = np.asarray(
                state.info.get("reward_terms", np.zeros(len(self.row_names)))
            )
            episode_cost += cost
            episode_step += 1
            step_index += 1
            rate_steps += 1

            if step_index % self.render_every == 0:
                panels = [(
                    f"student  t={episode_step * self.control_dt:6.2f}s "
                    f"cost={cost:7.3f}",
                    renderer.render(state.pipeline_state),
                )]
                if self.with_dial:
                    panels.append(
                        ("DIAL-MPC", renderer.render(dial_state.pipeline_state))
                    )
                self.publish(panels)

            now = time.perf_counter()
            if now - rate_window >= 1.0:
                control_hz = rate_steps / (now - rate_window)
                rate_window, rate_steps = now, 0

            omega_np = np.asarray(omega, dtype=float)
            unit = normalize_omega_np(omega_np)
            mixture = np.asarray(diagnostics["mixture"], dtype=float)
            contribution = np.asarray(diagnostics["contribution"], dtype=float)
            share = contribution / max(contribution.sum(), 1e-12)
            self.stats.update(
                status="running",
                episode_seconds=round(episode_step * self.control_dt, 2),
                episode_step=episode_step,
                cost=round(cost, 4),
                episode_mean_cost=round(episode_cost / max(episode_step, 1), 4),
                terms=[round(float(v), 4) for v in terms],
                weighted_terms=[round(float(w * v), 4) for w, v in zip(unit, terms)],
                omega=[round(float(v), 3) for v in omega_np],
                omega_unit=[round(float(v), 3) for v in unit],
                temperature=round(float(temperature), 4),
                mixture=[round(float(v), 4) for v in mixture],
                field_share=[round(float(v), 4) for v in share],
                update_norm=round(float(diagnostics["update_norm"]), 4),
                falls=falls,
                last_episodes=[round(v, 2) for v in history[-8:]],
                best_episode=round(max(history), 2) if history else None,
                control_hz=round(control_hz, 1),
                realtime=round(control_hz * self.control_dt, 3),
                seed=seed,
            )
            if self.with_dial:
                self.stats["dial_mean_cost"] = round(
                    dial_cost / max(episode_step, 1), 4
                )
                self.stats["cost_ratio"] = round(
                    (episode_cost / max(episode_step, 1))
                    / max(dial_cost / max(episode_step, 1), 1e-9), 3
                )

            if float(state.done) > 0.5 or (
                self.with_dial and float(dial_state.done) > 0.5
            ):
                falls += 1
                history.append(episode_step * self.control_dt)
                seed += 1
                state = self._reset(jax.random.PRNGKey(seed))
                dial_state = state if self.with_dial else None
                plan = jnp.zeros_like(plan)
                dial_plan = plan
                episode_step, episode_cost, dial_cost, fresh = 0, 0.0, 0.0, True

            if args.realtime and not self.with_dial:
                deadline += period
                sleep = deadline - time.perf_counter()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    deadline = time.perf_counter()


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>composed score policy — live</title>
<style>
 body{margin:0;background:#12151b;color:#e8ecf2;
      font:13px ui-monospace,SFMono-Regular,Menlo,monospace}
 header{padding:11px 18px;border-bottom:1px solid #2b323e;display:flex;
        gap:16px;align-items:baseline;flex-wrap:wrap}
 h1{font-size:13px;margin:0;letter-spacing:.14em;text-transform:uppercase;color:#8d99a9}
 .path{color:#5f6b7a;font-size:11px;word-break:break-all}
 main{display:grid;grid-template-columns:minmax(360px,1fr) 400px;gap:18px;
      padding:18px;align-items:start}
 @media(max-width:920px){main{grid-template-columns:1fr}}
 img{background:#000;width:100%;border:1px solid #2b323e;display:block}
 .card{background:#181c24;border:1px solid #2b323e;padding:14px 16px;margin-bottom:14px}
 .card h2{font-size:11px;letter-spacing:.13em;text-transform:uppercase;
          color:#8d99a9;margin:0 0 12px}
 table{border-collapse:collapse;width:100%}
 td{padding:3px 0}
 td:first-child{color:#6d7887;font-size:11px;letter-spacing:.06em;
                text-transform:uppercase;width:50%}
 td:last-child{text-align:right;font-variant-numeric:tabular-nums}
 .big{font-size:26px;font-variant-numeric:tabular-nums}
 .row{display:flex;align-items:center;gap:9px;margin:7px 0}
 .row label{width:74px;color:#8d99a9;font-size:11px;text-transform:uppercase;
            letter-spacing:.06em}
 input[type=range]{flex:1;accent-color:#5b9dd9}
 .num{width:58px;font-variant-numeric:tabular-nums;text-align:right;color:#c7d0dc}
 button{font:inherit;background:#20262f;color:#e8ecf2;
        border:1px solid #3a4350;padding:6px 11px;cursor:pointer;border-radius:2px}
 button:hover{border-color:#6d7887}
 .btnrow{display:flex;gap:7px;flex-wrap:wrap;margin-top:10px}
 .mix{display:flex;flex-direction:column;gap:6px;margin:4px 0 8px}
 .mixrow{display:flex;align-items:center;gap:8px}
 .mixrow span:first-child{width:64px;color:#8d99a9;font-size:11px}
 .bar{flex:1;height:12px;background:#20262f;position:relative}
 .bar i{position:absolute;top:0;bottom:0;left:50%;background:#5b9dd9}
 .bar i.neg{background:#e0925f}
 .mixrow span:last-child{width:58px;text-align:right;
                         font-variant-numeric:tabular-nums;color:#c7d0dc}
 .hint{color:#8d99a9;font-size:11px;margin-top:6px;line-height:1.5}
</style></head><body>
<header><h1>composed score policy — live</h1>
 <span class="path" id="policy"></span></header>
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
           oninput="pmagv.textContent=(+this.value).toFixed(2)">
    <span class="num" id="pmagv">0.80</span></div>
  </div>
 </div>
 <div>
  <div class="card">
   <h2>보상 가중치 ω</h2>
   <div id="omega"></div>
   <div class="btnrow" id="presets"></div>
   <div class="hint">스케일은 무시됩니다 — 경계에서 단위 벡터로 정규화되고,
    길이는 온도가 담당합니다.</div>
  </div>
  <div class="card">
   <h2>합성 계수</h2>
   <div class="mix" id="mix"></div>
   <table><tbody id="mixstats"></tbody></table>
   <div class="hint">계수는 ν 공간에서 풀립니다:
    a = (T_fit / T*) · ω̂ᵀ · pinv(B). 기저 방향을 정확히 요청하면
    해당 필드 하나만 1이 됩니다.</div>
  </div>
  <div class="card">
   <h2>목표 온도 T*</h2>
   <div class="row"><label>T*</label>
    <input type="range" id="temp" min="-1.4" max="0.0" step="0.01"
           oninput="applyTemp(this.value)">
    <span class="num" id="tempv">–</span></div>
   <div class="hint">필드는 <span id="tfit">–</span>에서 학습됐습니다.
    T*를 낮추면 계수가 그만큼 커집니다 — 같은 방향을 더 날카롭게 요청하는 것이고,
    스코어 공간에서는 정확하지만 유계 업데이트에서는 근사입니다.</div>
  </div>
  <div class="card"><h2>비용</h2><table><tbody id="cost"></tbody></table></div>
  <div class="card"><h2>실행</h2><table><tbody id="run"></tbody></table></div>
 </div>
</main>
<script>
let S={};
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
  names.forEach((n,i)=>sets['boost '+n]=names.map((_,j)=>j===i?3:1));
  sets['(2,1,1,1)']=[2,1,1,1]; sets['(1,2,2,1)']=[1,2,2,1];
  presets.innerHTML=Object.keys(sets).map(k=>
    `<button onclick='setOmega(${JSON.stringify(sets[k])})'>${k}</button>`).join('');
}
function setOmega(v){v.forEach((x,i)=>{const s=document.getElementById('w'+i);
  if(!s)return; s.value=x;document.getElementById('wv'+i).textContent=(+x).toFixed(2)});
  sendOmega()}
function sendOmega(){const v=S.row_names.map((_,i)=>+document.getElementById('w'+i).value);
  post('/controls',{omega:v})}
function applyTemp(log){const t=Math.pow(10,+log);tempv.textContent=t.toFixed(4);
  post('/controls',{temperature:t})}

async function tick(){
  const s=await (await fetch('/stats')).json(); S=s;
  policy.textContent=s.policy;
  if(s.status!=='running')return;
  if(!document.getElementById('w0')) buildOmega(s.row_names,s.omega);
  if(tempv.textContent==='–'){temp.value=Math.log10(s.temperature);
    tempv.textContent=s.temperature.toFixed(4);
    tfit.textContent='T = '+s.fit_temperature}
  secs.textContent=s.episode_seconds.toFixed(2);
  rows(ep,[['스텝',s.episode_step],['평균 비용',s.episode_mean_cost],
    ['낙상 횟수',s.falls],['최장 에피소드',s.best_episode??'–'],
    ['최근 (s)',(s.last_episodes||[]).join(' ')||'–']]);
  const m=s.mixture, lim=Math.max(1e-6,...m.map(Math.abs));
  mix.innerHTML=s.basis_names.map((n,i)=>{
    const w=Math.abs(m[i])/lim*50, left=m[i]<0?50-w:50;
    return `<div class="mixrow"><span>${n}</span><div class="bar">
      <i class="${m[i]<0?'neg':''}" style="left:${left}%;width:${w}%"></i></div>
      <span>${m[i].toFixed(3)}</span></div>`}).join('');
  rows(mixstats,[['계수 합',m.reduce((a,b)=>a+b,0).toFixed(3)],
    ['업데이트 크기',s.update_norm],
    ...s.basis_names.map((n,i)=>[n+' 기여도',(s.field_share[i]*100).toFixed(1)+' %'])]);
  rows(cost,[['현재 비용',s.cost],
    ...s.row_names.map((n,i)=>[n,`${s.terms[i]}  ×${s.omega_unit[i]} = ${s.weighted_terms[i]}`]),
    ['ω 단위벡터',s.omega_unit.join('  ')]]);
  const r=[['제어 Hz',s.control_hz],['실시간 배속',s.realtime],['시드',s.seed]];
  if(s.with_dial)r.push(['DIAL 평균 비용',s.dial_mean_cost],['비용 비율',s.cost_ratio]);
  rows(run,r);
}
tick();setInterval(tick,400);
</script></body></html>"""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--policy", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--example", default=None)
    source.add_argument("--config", type=Path, default=None)
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address; keep the default and use ssh -L")
    parser.add_argument("--port", type=int, default=8084)
    parser.add_argument("--width", type=int, default=560)
    parser.add_argument("--height", type=int, default=420)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--quality", type=int, default=80)
    parser.add_argument("--camera", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-passes", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--level-scales", type=float, nargs="+",
                        default=[3.175, 1.0])
    parser.add_argument("--samples", type=int, default=None,
                        help="Nsample for the optional DIAL panel")
    parser.add_argument("--basis", nargs="+",
                        default=["boost0", "boost1", "boost2", "boost3"])
    parser.add_argument("--row-names", nargs="+", default=None)
    parser.add_argument("--with-dial", action="store_true",
                        help="run DIAL beside the student; the pair advances at "
                             "DIAL's pace")
    parser.add_argument("--fast", dest="realtime", action="store_false",
                        help="run as fast as possible instead of real time")
    parser.set_defaults(realtime=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    names = {
        "unitree_go2_push_recover": ["tilt", "base", "feet", "shape"],
        "unitree_go2_trot": ["tracking", "stability", "gait"],
    }
    if args.row_names is None:
        key = args.example or (str(args.config) if args.config else "")
        for tag, labels in names.items():
            if tag in str(key):
                args.row_names = labels
                break
    live = LiveComposed(args)

    app = Flask("csm_live")

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
        live.set_controls(omega=body.get("omega"),
                          temperature=body.get("temperature"))
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
        impulse = [magnitude * float(np.cos(heading)),
                   magnitude * float(np.sin(heading)), 0.0]
        live.submit("push", impulse)
        return jsonify(ok=True, impulse=impulse)

    threading.Thread(target=live.run, daemon=True).start()
    print(f"policy={args.policy}")
    print(f"serving http://{args.host}:{args.port}  (stream at /stream.mjpg)")
    print("the first frame appears once JAX has finished compiling")
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
