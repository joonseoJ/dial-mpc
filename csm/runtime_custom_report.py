"""Tables and figures for the runtime-customisation experiments.

Reads the result files `csm.policy_completed_mppi` writes and prints the
markdown tables used in `docs/runtime_customization_ko.md`, plus two figures:
the height slider (torso height against task cost) and the chunk-length
ablation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RUNS = Path("csm_runs")


def load(*names):
    out = {}
    for n in names:
        p = RUNS / n
        if p.exists():
            out.update(json.loads(p.read_text()))
    return out


def cell(res, key, metric=True):
    if key not in res:
        return "—"
    v = res[key]
    c = sum(v["collapsed"])
    s = f"{np.mean(v['task_cost']):.4f}"
    if metric:
        s += f" / {np.mean(v['metric']):.3g}"
    return s + (f" ({c}/{len(v['collapsed'])} 붕괴)" if c else "")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--figures", type=Path, default=Path("docs/assets"))
    a = ap.parse_args(argv)
    zero = load("runtime_custom.json", "runtime_custom_r2.json")
    retrain = {}
    for slug, spec in (("height_0.22_4", "height:0.22:4"), ("step_0.14_0.05", "step:0.14:0.05"),
                       ("energy_0.02", "energy:0.02")):
        r = load(f"runtime_retrain_{slug}.json")
        if f"policy-retrained/{spec}" in r:
            retrain[spec] = r[f"policy-retrained/{spec}"]
    bppo = load("constraint_eval_bppo.json")
    for name in ("none", "knee", "front", "crouch", "lock"):
        v = bppo.get(f"bppo/{name}")
        if v is not None:
            spec = "none" if name == "none" else f"band:{name}"
            retrain.setdefault(spec, {"task_cost": v["costs"], "metric": None,
                                      "collapsed": v["collapsed"], "family": True})
    abl = load("runtime_ablation.json")
    front = load("runtime_frontier.json")
    res_ = load("runtime_residual.json", "runtime_residual_slider.json")
    res2 = load("runtime_residual02.json")

    specs = ["none", "band:knee", "band:front", "band:crouch", "band:lock",
             "height:0.22:4", "step:0.14:0.05", "energy:0.02"]
    print("| 실행 중 목적 | 정책만 | **+ 계획층** (chunk + 잔차 σ=0.02) | 그 목적으로 재학습한 PPO |")
    print("|---|---|---|---|")
    for s in specs:
        rt = retrain.get(s)
        if rt is None:
            rt_s = "—"
        elif rt.get("family"):
            rt_s = f"{np.mean(rt['task_cost']):.4f} (제약 족으로 학습한 PPO, 200M)"
        else:
            rt_s = f"{np.mean(rt['task_cost']):.4f} / {np.mean(rt['metric']):.3g}"
        print(f"| `{s}` | {cell(zero, 'policy/' + s)} | {cell(res2, 'pc-res02/' + s)} | {rt_s} |")

    print("\n| 잔차 완성 σ_b | " + " | ".join(f"`{s}`" for s in specs) + " |")
    print("|---|" + "---|" * len(specs))
    for lbl, src, name in (("0 (chunk만)", zero, "pc"), ("0.02", res2, "pc-res02"),
                           ("0.05", res_, "pc-res")):
        print(f"| {lbl} | " + " | ".join(cell(src, f"{name}/{s}") for s in specs) + " |")

    print("\n| chunk K (잔차 없음) | " + " | ".join(f"`{s}`" for s in
                                       ("none", "band:lock", "height:0.22:4", "step:0.14:0.05")) + " |")
    print("|---|---|---|---|---|")
    for k, name in (("16 (개루프 잔차)", "pc-k16"), ("8", "pc-k8"), ("4 (기본)", "pc")):
        src = zero if name == "pc" else abl
        print(f"| {k} | " + " | ".join(cell(src, f"{name}/{s}") for s in
                                      ("none", "band:lock", "height:0.22:4", "step:0.14:0.05")) + " |")

    print("\n| 조합 | 정책만 | + chunk 계획 | + 계획층 (잔차 0.02) |")
    print("|---|---|---|---|")
    for s in ("band:lock+height:0.22:4", "band:knee+step:0.14:0.05"):
        print(f"| `{s}` | {cell(front, 'policy/' + s)} | {cell(front, 'pc/' + s)} | "
              f"{cell(res2, 'pc-res02/' + s)} |")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return 0
    targets = [0.26, 0.24, 0.22, 0.20, 0.18]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for arm, color, label in (("policy", "#8a8f98", "PPO alone"),
                              ("pc", "#7fa7cf", "+ chunk planner (no residual)"),
                              ("pc-res02", "#1f4e8c", "+ chunk planner + planned residual")):
        pts = []
        for z in targets:
            key = f"{arm}/height:{z:.2f}:4"
            src = next((d for d in (zero, front, res2) if key in d), {})
            if key in src:
                pts.append((z, np.mean(src[key]["metric"]), np.mean(src[key]["task_cost"])))
        if pts:
            pts = np.array(pts)
            ax.plot(pts[:, 0], pts[:, 1], "o-", color=color, label=label)
            for z, m, c in pts:
                ax.annotate(f"{c:.4f}", (z, m), textcoords="offset points", xytext=(4, 4),
                            fontsize=7, color=color)
    if "height:0.22:4" in retrain:
        r = retrain["height:0.22:4"]
        ax.plot([0.22], [np.mean(r["metric"])], "s", color="#c0392b",
                label="PPO retrained for 0.22 (5.6 min)")
    ax.plot(targets, targets, ":", color="#444", lw=1, label="target")
    ax.set_xlabel("runtime torso-height target (m)")
    ax.set_ylabel("achieved mean torso height (m)")
    ax.set_title("A runtime slider the policy was never trained on\n(labels: task cost)",
                 fontsize=9)
    ax.invert_xaxis()
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    a.figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.figures / "runtime_height_slider.png", dpi=160)
    print(f"\nwrote {a.figures / 'runtime_height_slider.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
