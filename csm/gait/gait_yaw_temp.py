"""Temperature and label quality on the objective that now prices heading.

Heading went into the shared floor, so the common part of every return grew
while the gait-differentiating part did not: at a fixed temperature the softmax
is sharper (lower ESS) and the gait's share of the label is smaller.  Both are
the quantities that decided this task before, so both are re-measured here
rather than assumed, against the old objective's own clouds at the same
repeats and by the same method.

SNR is the gait-specific part of the label over its own noise, the inequality
that separated the fits that worked from the ones that did not.
"""
import sys, dataclasses, numpy as np, jax, jax.numpy as jnp
import brax.envs as brax_envs, dial_mpc.envs
from dial_mpc.core.dial_core import make_controller
from csm.basis_screen import _load_config
from csm.cloud_data import load_clouds, make_relabeler, query_temperatures
E = np.eye(4, dtype=np.float32); Q = 160
dc0, ec = _load_config("unitree_go2_gait", None)
env = brax_envs.get_environment(dc0.env_name, config=ec)

def scan(path, tag, temps):
    cl = jax.tree.map(lambda x: x[:Q], load_clouds(path))
    R = cl.terms.shape[1]
    halves = [jax.tree.map(lambda x: x[:, s] if (hasattr(x,'ndim') and x.ndim>1 and x.shape[1]==R) else x, cl)
              for s in (slice(0, R//2), slice(R//2, R))]
    print(f"\n--- {tag}  ({path}, repeats {R}) ---")
    print(f"{'T':>6}{'ESS%':>7}{'eff.n':>7}{'resid/lbl':>11}{'noise/lbl':>11}{'SNR':>7}")
    for T in temps:
        dc = dataclasses.replace(dc0, temp_sample=T)
        mbdpi = make_controller(dc, env)
        relabel, ess_fn = make_relabeler(mbdpi, dc)
        temps_q = query_temperatures(cl, T, (1.0, 1.0))
        th = query_temperatures(halves[0], T, (1.0, 1.0))
        ess = np.mean([float(np.asarray(ess_fn(cl, jnp.asarray(E[i]), temps_q, False)).mean()) for i in range(4)])
        full, h0, h1 = [], [], []
        for i in range(4):
            om = jnp.asarray(E[i])
            full.append(np.asarray(relabel(cl, om, temps_q, False, None)[0]).reshape(Q, -1))
            h0.append(np.asarray(relabel(halves[0], om, th, False, None)[0]).reshape(Q, -1))
            h1.append(np.asarray(relabel(halves[1], om, th, False, None)[0]).reshape(Q, -1))
        full, h0, h1 = np.stack(full), np.stack(h0), np.stack(h1)
        lbl = np.linalg.norm(full, axis=-1).mean()
        resid = np.linalg.norm(full - full.mean(0, keepdims=True), axis=-1).mean()
        nR = (np.linalg.norm(h0 - h1, axis=-1).mean() / np.sqrt(2)) / np.sqrt(2)  # noise of the R-repeat label
        print(f"{T:6.2f}{ess/(dc.Nsample+1)*100:7.1f}{ess:7.0f}{resid/lbl:11.3f}{nR/lbl:11.3f}{resid/nR:7.2f}")

scan("csm_runs/gait_v2/clouds_0000.npz", "OLD objective (no heading)", [0.10, 0.15, 0.20])
scan("csm_runs/gait_probe_yaw/clouds_0000.npz", "NEW objective (yaw_weight 0.6)", [0.10, 0.15, 0.20, 0.25, 0.30])
print("TEMPSCANDONE")
