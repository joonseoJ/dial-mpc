"""Are the four gait labels actually different?  Measure it on stored clouds."""
import sys, numpy as np, jax, jax.numpy as jnp, dataclasses
import brax.envs as brax_envs, dial_mpc.envs
from dial_mpc.core.dial_core import make_controller
from csm.basis_screen import _load_config
from csm.cloud_data import load_clouds, make_relabeler, query_temperatures
T=0.10
dc, ec = _load_config("unitree_go2_gait", None); dc = dataclasses.replace(dc, temp_sample=T)
env = brax_envs.get_environment(dc.env_name, config=ec); mbdpi = make_controller(dc, env)
relabel, ess_fn = make_relabeler(mbdpi, dc)
for path in sys.argv[1:]:
    cl = load_clouds(path); Q=192
    cl = jax.tree.map(lambda x: x[:Q], cl)
    terms = np.asarray(cl.terms)                      # (Q,R,N,k)
    print(f"\n### {path}  terms {terms.shape}  T={T}")
    m = terms.mean(-1, keepdims=True); d = terms - m   # common part / gait-differentiating part
    # spread across the N samples of one query, averaged over queries, in temperature units
    sd_common = terms[...,0].std(-1).mean()/T if False else m[...,0].std(-1).mean()/T
    sd_diff = np.sqrt((d.std(-2)**2).mean(-1)).mean()/T
    sd_row = terms.std(-2).mean()/T
    print(f"sample spread / T :  full row {sd_row:.3f}   common(floor+mean gait) {sd_common:.3f}   gait-differentiating {sd_diff:.3f}")
    rng = terms.max(-2)-terms.min(-2)
    print(f"gait row diff between two gaits, same sample (|c_i-c_j| mean): {np.abs(d).mean():.4f}  vs row sample-std {terms.std(-2).mean():.4f}")
    temps = query_temperatures(cl, T, (1.0,1.0))
    labs=[]
    for i in range(4):
        om = np.zeros(4,np.float32); om[i]=1
        lab,_ = relabel(cl, jnp.asarray(om), temps, False, None)
        labs.append(np.asarray(lab).reshape(Q,-1))
    L = np.stack(labs)                                # (4,Q,D)
    norm = np.linalg.norm(L,axis=-1)                  # (4,Q)
    mean = L.mean(0, keepdims=True); res = L-mean
    print(f"label norm per row      : {norm.mean(1).round(4)}")
    print(f"residual(label - row-mean) norm: {np.linalg.norm(res,axis=-1).mean(1).round(4)}   ratio to label norm {(np.linalg.norm(res,axis=-1)/norm).mean():.3f}")
    names=["walk","trot","pace","bound"]
    print("pairwise cosine between row labels (same query):")
    for i in range(4):
        row=[]
        for j in range(4):
            c=(L[i]*L[j]).sum(-1)/np.maximum(norm[i]*norm[j],1e-9); row.append(f"{c.mean():.3f}")
        print(f"  {names[i]:<6}", " ".join(row))
    # what a field must resolve, vs the label noise it is fit against
    print(f"ESS share per row: {[round(float(np.asarray(ess_fn(cl, jnp.asarray(np.eye(4,dtype=np.float32)[i]), temps, False)).mean())/(dc.Nsample+1),3) for i in range(4)]}")
