"""Did the contrast decomposition hurt pace's *representation*, or only its closed loop?

On the same held-out v2 rows, relabel under pace (e2) and walk (e0) and compare
what the whole-label field and the contrast field predict against the label.
If the contrast pace field agrees with the label as well as the whole-label
one does, the decomposition kept the information and pace's failure is a
closed-loop effect; if it agrees worse, the decomposition itself lost pace --
which decides whether mixing the two fits per gait is principled or lucky.
"""
import sys, dataclasses, numpy as np, jax, jax.numpy as jnp
import brax.envs as brax_envs, dial_mpc.envs
from dial_mpc.core.dial_core import make_controller
from csm.basis_screen import _load_config
from csm.cloud_data import load_clouds, make_relabeler, query_temperatures
from csm.dial_score import DialScorePolicy, factor_to_t
T = 0.15
V2, C = sys.argv[1], sys.argv[2]
dc0, ec = _load_config("unitree_go2_gait", None)
dc = dataclasses.replace(dc0, temp_sample=T)
env = brax_envs.get_environment(dc.env_name, config=ec)
mbdpi = make_controller(dc, env)
relabel, _ = make_relabeler(mbdpi, dc)
# last shard, last rows: the fit's validation split is random, so this is
# not strictly held out -- but both fields saw the same rows, and it is the
# comparison between them that matters
cl = jax.tree.map(lambda x: x[-160:], load_clouds("csm_runs/gait_v2/clouds_0007.npz"))
keep = np.asarray(cl.qpos)[:, 2] >= 0.19
cl = jax.tree.map(lambda x: x[keep], cl)
temps = query_temperatures(cl, T, (1.0, 1.0))
fields = {"v2": {i: DialScorePolicy.load(f"{V2}/field_e{i}.pkl") for i in (0, 2)},
          "contrast": {i: DialScorePolicy.load(f"{C}/field_e{i}.pkl") for i in (0, 2)}}
lo, hi = float(jnp.min(fields["v2"][0].factors)), float(jnp.max(fields["v2"][0].factors))
tt = jax.vmap(lambda f: factor_to_t(f, lo, hi).reshape(1))(cl.factor)
print(f"{len(keep.nonzero()[0])} rows, T={T}")
print(f"{'gait':<6}{'fit':<10}{'cosine':>8}{'rel rms':>9}{'|pred|/|lbl|':>13}")
for i, name in ((2, "pace"), (0, "walk")):
    om = np.zeros(4, np.float32); om[i] = 1
    lbl = np.asarray(relabel(cl, jnp.asarray(om), temps, False, None)[0]).reshape(len(cl.u), -1)
    for fit in ("v2", "contrast"):
        f = fields[fit][i]
        pred = np.asarray(jax.vmap(lambda u, o, t: f.delta(u, o, t))(cl.u, cl.obs, tt)).reshape(len(cl.u), -1)
        cos = ((pred * lbl).sum(1) / np.maximum(np.linalg.norm(pred, axis=1) * np.linalg.norm(lbl, axis=1), 1e-9)).mean()
        rel = np.sqrt(((pred - lbl) ** 2).mean() / (lbl ** 2).mean())
        ratio = (np.linalg.norm(pred, axis=1) / np.maximum(np.linalg.norm(lbl, axis=1), 1e-9)).mean()
        print(f"{name:<6}{fit:<10}{cos:8.3f}{rel:9.3f}{ratio:13.2f}")
print("REPRDONE")
