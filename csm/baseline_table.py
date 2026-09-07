"""CSM's 64 cells against PPO's 64, on one objective and one cached teacher."""
import json, sys
from pathlib import Path
import numpy as np

CMDS = ["box_fast", "box_slow", "box_turn", "box_strafe"]
NAMES = ["uniform", "boost0", "boost1", "boost2", "2,1,1", "1,2,1", "1,1,2", "3,1,2"]
SLUGS = ["uniform", "boost0", "boost1", "boost2", "211", "121", "112", "312"]
H = sys.argv[1] if len(sys.argv) > 1 else "1500"
suffix = "" if H == "1500" else "_150"

csm = json.load(open(f"csm_runs/walk_screen/compose_v5_cached{suffix}.json"))
ppo = {}
for name, slug in zip(NAMES, SLUGS):
    p = Path(f"csm_runs/rl-sweep/{slug}_eval{suffix}.json")
    if not p.exists():
        continue
    d = json.load(open(p))
    # The evaluation labels the row with the policy's own omega name; a policy
    # saved before that field existed falls back to "rl".
    key = {k.split("/")[1] for k in d}
    assert len(key) == 1, key
    tag = key.pop()
    ppo[name] = {c: d[f"{c}/{tag}"] for c in CMDS if f"{c}/{tag}" in d}

print(f"cost ratio against DIAL, {H} steps  (1.00 = matches its teacher)\n")
head = f"{'weight':<9}" + "".join(f"{c.replace('box_',''):>17}" for c in CMDS)
print(head); print(f"{'':<9}" + "".join(f"{'CSM':>8}{'PPO':>9}" for _ in CMDS))
print("-" * len(head))
cr, pr, cf, pf = [], [], 0, 0
for name in NAMES:
    line = f"{name:<9}"
    for c in CMDS:
        a = csm.get(f"{c}/{name}")
        line += f"{a['ratio']:8.3f}" if a else f"{'-':>8}"
        if a: cr.append(a["ratio"]); cf += a["student_falls"]
        b = ppo.get(name, {}).get(c)
        if b:
            mark = "*" if a and b["ratio"] < a["ratio"] else " "
            line += f"{b['ratio']:8.3f}{mark}"
            pr.append(b["ratio"]); pf += b["student_falls"]
        else:
            line += f"{'-':>9}"
    print(line)
print("-" * len(head))
print("  * = PPO beats CSM in that cell\n")
if pr:
    print(f"{'':<12}{'cells':>7}{'median':>9}{'mean':>8}{'worst':>8}{'falls':>8}")
    for lbl, v, f in (("CSM", cr, cf), ("PPO", pr, pf)):
        print(f"{lbl:<12}{len(v):7d}{np.median(v):9.3f}{np.mean(v):8.3f}"
              f"{max(v):8.3f}{f:8d}")
    wins = sum(1 for n in NAMES for c in CMDS
               if n in ppo and c in ppo[n] and f"{c}/{n}" in csm
               and ppo[n][c]["ratio"] < csm[f"{c}/{n}"]["ratio"])
    print(f"\nPPO wins {wins} of {len(pr)} cells")
