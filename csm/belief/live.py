"""Watch robust MPPI carry an unknown payload, and change what it believes.

The arm is driven by MPPI over a *belief*: a set of hypotheses about
`theta = (m, c_x, c_y, c_z, mu)` with weights that sum to one.  The plant runs
under one true `theta` that the planner never sees.  Everything the panel
changes is either what is true or what is believed, so the two failure modes
this task is about are visible side by side:

    wrong belief   the planner is confident and mistaken
    wide belief    the planner is uncertain and has to hedge

Three readouts carry the argument.  `progress` is whether the task is being
done at all -- a controller that hedges by standing still is not robust, it is
broken, and this number is the one that catches it.  `ESS` is whether the
softmax is averaging or picking one lucky sample.  `belief error` is how far
the belief's mean is from the truth, which is what the mismatch matrix measures
offline and what a particle filter would shrink over time.

Rendering is the real MuJoCo renderer here -- the plant is MuJoCo, so unlike
the other viewers in this package there is no need to draw the scene by hand.
The planner runs on MJX and the renderer on an `MjData` synced from it each
frame.
"""
from __future__ import annotations

import os

# Before anything imports mujoco: the renderer needs a headless GL backend and
# choosing one afterwards has no effect.
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import io
import threading
import time

import numpy as np
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from flask import Flask, Response, jsonify, request
from PIL import Image

from csm.belief import plant as P
from csm.belief.oracle import make_planner, make_world

BELIEF_MODES = ("truth", "narrow", "prior")


class LiveBelief:
    def __init__(self, args):
        self.args = args
        self.model, self.bare, self.ids, self.mj = P.load()
        self.world = make_world(self.model, self.bare, self.ids)
        self.renderer = None                      # built in the render thread
        self.controls = {
            "mass": 1.0, "cy": 0.0, "mu": 0.5,
            "belief": "prior", "M": 8,
            "lam_scale": 1.0, "sigma_scale": 1.0, "paused": False,
        }
        self._pending = []
        self._lock = threading.Lock()
        self._planner_for = None
        self.stats = {"status": "starting"}
        self.frame = None
        self.reset()

    # -- planner, rebuilt only when M changes (the jit is keyed on it) -------
    def planner(self):
        M = int(self.controls["M"])
        if self._planner_for != M:
            self._planner_for = M
            # One jitted planner per annealing level: the temperature is a
            # closure constant, and the two levels need different ones -- the
            # cost spread across the cloud differs fivefold between them.
            self._upd = make_planner(self.model, self.bare, self.ids,
                                     self.args.plans)
        return self._upd

    def theta_true(self):
        c = self.controls
        return jnp.array([c["mass"], 0.0, c["cy"], 0.0, c["mu"]])

    def belief(self):
        """Particles and weights for the chosen mode.

        `truth` is the oracle -- one particle, exactly right.  `prior` is the
        hardest case, hypotheses drawn from the whole box.  `narrow` sits
        between: a particle filter part way through converging.
        """
        c, M = self.controls, int(self.controls["M"])
        th_t = self.theta_true()
        if c["belief"] == "truth":
            return th_t[None, :], jnp.ones((1,))
        key = jax.random.PRNGKey(self._bkey)
        if c["belief"] == "narrow":
            spread = jnp.array([0.25, 0.0, 0.015, 0.0, 0.08])
            th = th_t[None, :] + spread * jax.random.normal(key, (M, 5))
            th = jnp.clip(th, P.THETA_LO, P.THETA_HI)
        else:
            th = P.sample_theta(key, M)
        return th, jnp.full((M,), 1.0 / th.shape[0])

    # -- state ---------------------------------------------------------------
    def reset(self):
        self._bkey = int(time.time()) & 0xFFFF
        d = mjx.make_data(self.model).replace(qpos=P.Q_CARRY)
        self.data = mjx.forward(P.apply_theta(self.model, self.ids,
                                              self.theta_true()), d)
        self.U = jnp.zeros(P.DIM_U)
        self.key = jax.random.PRNGKey(self._bkey)
        self.t = 0.0
        self.prof = np.zeros(7)
        self.ess = 0.0
        self.d0 = float(jnp.linalg.norm(
            self.data.xpos[self.ids["payload"]] - P.GOAL))

    def submit(self, name):
        with self._lock:
            self._pending.append(name)

    def set_controls(self, **kw):
        with self._lock:
            for k, v in kw.items():
                if v is not None and k in self.controls:
                    self.controls[k] = v
            if any(k in kw and kw[k] is not None
                   for k in ("mass", "cy", "mu")):
                self._pending.append("reset")

    # -- the loop -------------------------------------------------------------
    def run(self):
        while True:
            t0 = time.time()
            with self._lock:
                pend, self._pending = self._pending, []
            if "reset" in pend:
                self.reset()
            if not self.controls["paused"]:
                self._step()
            self.stats = self._stats()
            dt = time.time() - t0
            if dt < 1.0 / self.args.hz:
                time.sleep(1.0 / self.args.hz - dt)

    def _step(self):
        t_wall = time.time()
        c = self.controls
        upd = self.planner()
        th_b, bel = self.belief()
        esss = []
        for sg, lm in zip(P.SIGMA_SCHEDULE, P.LAM_SCHEDULE):
            self.key, k = jax.random.split(self.key)
            self.U, ess = upd(self.data, self.U, th_b, bel,
                              sg * c["sigma_scale"], lm, P.GOAL, k)
            esss.append(float(ess))
        self.ess = float(np.mean(esss))
        self.data, rows = self.world(self.data, self.U[:P.NU],
                                     self.theta_true(), P.GOAL)
        self.prof = self.prof + np.asarray(rows)
        self.U = P.shift_nodes(self.U)
        self.t += P.DT
        dt_wall = time.time() - t_wall
        self._rt = 0.9 * getattr(self, "_rt", P.DT / max(dt_wall, 1e-6)) \
            + 0.1 * (P.DT / max(dt_wall, 1e-6))
        self._sec_per_step = dt_wall
        self._belief_err = float(jnp.linalg.norm(
            (th_b * bel[:, None]).sum(0)[jnp.array([0, 2, 4])]
            - self.theta_true()[jnp.array([0, 2, 4])]))
        if self.t > self.args.episode:
            self.reset()

    # -- rendering -------------------------------------------------------------
    def render_loop(self):
        # A daemon thread that raises dies quietly, and the page then shows a
        # panel with no video and no clue why -- which is exactly what happened
        # when the requested width exceeded the model's offscreen framebuffer.
        try:
            self._render_loop()
        except Exception:
            import traceback
            traceback.print_exc()

    def _camera(self):
        """A free camera, not the model's fixed one.

        Framing is the one visual parameter worth being able to change without
        a restart, and a restart here costs a recompile of the double-vmapped
        rollout -- 152 s measured, because editing the XML changes the MJX
        model's array shapes and misses the compilation cache.  An `MjvCamera`
        lives entirely on the render thread's CPU model, so it costs nothing.

        Azimuth 90 puts the camera out along `-y`, looking across the transport
        rather than down it: the payload travels 0.33 m in mostly `+x`, so from
        the model's own camera position the whole motion was foreshortened into
        a few pixels of apparent depth.
        """
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        mid = 0.5 * (np.asarray(P.P_START) + np.asarray(P.GOAL))
        cam.lookat[:] = [mid[0], mid[1], 0.55]   # below the payload, to keep
        cam.distance = self.args.cam_dist        # the base in frame
        cam.azimuth = self.args.cam_azim
        cam.elevation = self.args.cam_elev
        return cam

    def _render_loop(self):
        self.renderer = mujoco.Renderer(self.mj, height=self.args.height,
                                        width=self.args.width)
        cam = self._camera()
        md = mujoco.MjData(self.mj)
        while True:
            t0 = time.time()
            md.qpos[:] = np.asarray(self.data.qpos)
            md.qvel[:] = np.asarray(self.data.qvel)
            self.mj.body_mass[self.ids["payload"]] = float(self.controls["mass"])
            mujoco.mj_forward(self.mj, md)
            self.renderer.update_scene(md, camera=cam)
            img = Image.fromarray(self.renderer.render())
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=self.args.quality)
            self.frame = buf.getvalue()
            self._nframe = getattr(self, "_nframe", 0) + 1
            dt = time.time() - t0
            if dt < 1.0 / self.args.fps:
                time.sleep(1.0 / self.args.fps - dt)

    def frames(self):
        while True:
            if self.frame is not None:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + self.frame + b"\r\n")
            time.sleep(1.0 / self.args.fps)

    def _stats(self):
        gap = float(jnp.linalg.norm(
            self.data.xpos[self.ids["payload"]] - P.GOAL))
        return {
            "status": "running",
            "t": round(self.t, 2),
            "gap": round(gap, 3),
            "progress": round(100.0 * (1.0 - gap / max(self.d0, 1e-6)), 1),
            "slip": round(float(self.prof[2]), 4),
            "tilt": round(float(self.prof[3]), 4),
            "tau": round(float(self.prof[1]), 4),
            "ess": round(100.0 * self.ess, 1),
            "belief_err": round(getattr(self, "_belief_err", 0.0), 3),
            "controls": dict(self.controls),
            "plans": self.args.plans,
            "frame_bytes": len(self.frame or b""),
            "frames": getattr(self, "_nframe", 0),
            # how far from real time: 1.0 would be live, 0.02 is fifty times
            # slower.  This is what robust MPPI over M hypotheses costs, and
            # measuring it is the point of the spec's E2.
            "realtime": round(float(getattr(self, "_rt", 0.0)), 4),
            "slowdown": round(1.0 / max(float(getattr(self, "_rt", 1e-9)), 1e-9), 1),
            "sec_per_step": round(float(getattr(self, "_sec_per_step", 0.0)), 3),
        }


PAGE = """<!doctype html><meta charset=utf-8><title>belief-conditioned MPPI</title>
<style>
 body{margin:0;background:#12151b;color:#e8ecf2;
      font:13px ui-monospace,SFMono-Regular,Menlo,monospace}
 header{padding:11px 18px;border-bottom:1px solid #2b323e}
 h1{font-size:13px;margin:0;letter-spacing:.14em;text-transform:uppercase;color:#8d99a9}
 main{display:grid;grid-template-columns:minmax(360px,1fr) 340px;gap:18px;padding:18px;
      align-items:start}
 @media(max-width:900px){main{grid-template-columns:1fr}}
 img{background:#000;width:100%;border:1px solid #2b323e;display:block}
 .card{background:#181c24;border:1px solid #2b323e;padding:14px 16px;margin-bottom:14px}
 .card h2{font-size:11px;letter-spacing:.13em;text-transform:uppercase;
          color:#8d99a9;margin:0 0 10px}
 .row{display:flex;align-items:center;gap:9px;margin:7px 0}
 label{width:66px;color:#6d7887;font-size:11px;text-transform:uppercase}
 input[type=range]{flex:1}
 .num{width:54px;text-align:right;font-variant-numeric:tabular-nums}
 button{background:#232a35;color:#e8ecf2;border:1px solid #39424f;padding:5px 10px;
        margin:2px;cursor:pointer;font:inherit}
 button.on{background:#2f4a63;border-color:#4d7ea8}
 table{border-collapse:collapse;width:100%}
 td{padding:3px 0} td:first-child{color:#6d7887;font-size:11px;text-transform:uppercase}
 td:last-child{text-align:right;font-variant-numeric:tabular-nums}
 .hint{color:#5f6b7a;font-size:11px;margin-top:8px;line-height:1.5}
</style>
<header><h1>belief-conditioned MPPI &middot; unknown payload</h1></header>
<main>
 <div><img id=v src="/stream.mjpg"></div>
 <div>
  <div class="card"><h2>what is true (the plant)</h2>
   <div class="row"><label>mass</label><input type=range id=mass min=0.2 max=2.5
     step=0.05 oninput="set('mass',+this.value)"><span class=num id=massv>-</span></div>
   <div class="row"><label>CoM y</label><input type=range id=cy min=-0.06 max=0.06
     step=0.005 oninput="set('cy',+this.value)"><span class=num id=cyv>-</span></div>
   <div class="row"><label>friction</label><input type=range id=mu min=0.2 max=0.8
     step=0.02 oninput="set('mu',+this.value)"><span class=num id=muv>-</span></div>
   <div class="hint">Changing any of these restarts the run &mdash; the payload
    is different, so the episode is a different one.  <b>CoM y</b> is the axis
    that flips the sign of the wrist torque, so two hypotheses either side of
    zero ask for opposite corrections rather than the same one harder.</div>
  </div>
  <div class="card"><h2>what is believed (the planner)</h2>
   <div id=bel></div>
   <div class="row"><label>M</label><input type=range id=M min=1 max=48 step=1
     oninput="set('M',+this.value)"><span class=num id=Mv>-</span></div>
   <div class="hint"><b>truth</b> is the oracle: one particle, exactly right.
    <b>prior</b> is the hardest case, hypotheses from the whole box.
    <b>narrow</b> is a filter part way through converging.  The planner never
    sees the true value in the last two.</div>
  </div>
  <div class="card"><h2>planner</h2>
   <div class="row"><label>sigma</label><input type=range id=sigma_scale min=0.3
     max=2 step=0.05 oninput="set('sigma_scale',+this.value)">
     <span class=num id=sigma_scalev>-</span></div>
   <div><button id=pz onclick="set('paused',!S.controls.paused)">pause</button>
    <button onclick="post('/reset')">reset</button></div>
   <div class="hint">Temperatures are the measured per-level pair (4.0, 0.8);
    a single value for both leaves the fine level at 99% effective sample size,
    which is noise.</div>
  </div>
  <div class="card"><h2>state</h2><table><tbody id=st></tbody></table>
   <div class="hint"><b>progress</b> is the one that catches a controller that
    "hedges" by standing still.  <b>belief error</b> is the distance from the
    belief's mean to the truth in (mass, CoM y, friction).</div>
  </div>
 </div>
</main>
<script>
let S={controls:{}};
const post=(u,b)=>fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},
                          body:JSON.stringify(b||{})});
function set(k,v){S.controls[k]=v;post('/controls',{[k]:v});sync()}
function sync(){
  const c=S.controls; if(!c||c.mass===undefined)return;
  for(const k of ['mass','cy','mu','M','sigma_scale']){
    const el=document.getElementById(k);
    if(el&&document.activeElement!==el)el.value=c[k];
    const s=document.getElementById(k+'v');
    if(s)s.textContent=(k==='M')?c[k]:(+c[k]).toFixed(3);
  }
  pz.className=c.paused?'on':'';
  bel.innerHTML=['truth','narrow','prior'].map(k=>
    `<button class="${c.belief===k?'on':''}" onclick="set('belief','${k}')">${k}</button>`
  ).join('');
}
async function tick(){
  try{
    const s=await (await fetch('/stats')).json(); S=s;
    if(s.status!=='running')return; sync();
    st.innerHTML=[['t',s.t+' s'],['progress',s.progress+' %'],
      ['goal gap',s.gap+' m'],['belief error',s.belief_err],
      ['ESS',s.ess+' %'],['slip',s.slip],['tilt',s.tilt],['torque',s.tau],
      ['plans',s.plans]].map(([k,v])=>`<tr><td>${k}</td><td>${v}</td></tr>`).join('');
  }catch(e){}
}
setInterval(tick,400); tick();
</script>
"""


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8087)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--hz", type=float, default=6.0,
                    help="control steps per second of wall clock")
    ap.add_argument("--quality", type=int, default=82)
    # Free-camera framing.  Azimuth 90 looks along +y, i.e. across the +x
    # transport; see `_camera`.
    ap.add_argument("--cam-dist", type=float, default=1.55)
    ap.add_argument("--cam-azim", type=float, default=96.0)
    ap.add_argument("--cam-elev", type=float, default=-16.0)
    ap.add_argument("--cache", default="/tmp/jax_cache_belief",
                    help="persistent XLA compilation cache")
    ap.add_argument("--substeps", type=int, default=P.SUBSTEPS,
                    help="physics substeps per control step for the **world** "
                         "-- the plant every reported number is measured on. "
                         "Leave it at 25.")
    ap.add_argument("--plan-substeps", type=int, default=P.PLAN_SUBSTEPS,
                    help="substeps inside the planner's own rollouts, which may "
                         "be coarser than the world's; see plant.PLAN_SUBSTEPS. "
                         "Measured closed-loop: 10 against 25 moves progress by "
                         "0.1 points and leaves every constraint row at zero, "
                         "for 2.3x.")
    ap.add_argument("--plans", type=int, default=1024,
                    help="MPPI samples per update.  Cheap but no longer free: "
                         "128, 256 and 512 plans all cost the same 0.133 s "
                         "because the rollout's sequential depth (H x substeps "
                         "kernel launches) sets a floor, and above ~512 the "
                         "dispatch becomes GPU-bound -- 1024 costs 15% over the "
                         "floor, 4096 costs 2.1x.")
    ap.add_argument("--episode", type=float, default=2.0,
                    help="seconds before the run restarts")
    args = ap.parse_args(argv)
    # The first call to the planner compiles a double-vmapped MJX rollout, which
    # costs minutes; a persistent cache makes that a once-ever cost rather than
    # a once-per-restart one, which is what made the viewer look broken while it
    # was only starting up.  The cache is keyed on the model, so editing the XML
    # -- adding a site, say -- misses it and pays the compile again.
    jax.config.update("jax_compilation_cache_dir", args.cache)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 2.0)
    # Before the plant is loaded: both are read when the model is built.
    P.SUBSTEPS = int(args.substeps)
    P.PLAN_SUBSTEPS = int(args.plan_substeps)

    live = LiveBelief(args)
    app = Flask("belief_live")

    @app.get("/")
    def index():
        return PAGE

    @app.get("/stream.mjpg")
    def stream():
        return Response(live.frames(),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/stats")
    def stats():
        return jsonify(live.stats)

    @app.post("/controls")
    def controls():
        b = request.get_json(silent=True) or {}
        live.set_controls(**{k: b.get(k) for k in live.controls})
        return jsonify(ok=True)

    @app.post("/reset")
    def reset():
        live.submit("reset")
        return jsonify(ok=True)

    threading.Thread(target=live.run, daemon=True).start()
    threading.Thread(target=live.render_loop, daemon=True).start()
    print(f"serving http://{args.host}:{args.port}  (stream at /stream.mjpg)")
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
