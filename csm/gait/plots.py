"""Figures for the gait report, from the head-to-head json.

Four panels, each answering one question the tables answer less legibly:

  `mixing`     signal on the axis the weight asks for, against how many rows
               the weight mixes -- the shape of the whole comparison
  `midpoints`  the six 50:50 pairs, per arm, where the aggregate hides that
               one pair collapses the RL blend and another favours it
  `robust`     collapse rate inside and outside the command box
  `runtime`    cost per control step against the 20 ms period, log scale

Written with matplotlib's Agg backend; no display needed.
"""
from __future__ import annotations

import argparse, json, collections
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = [("csm", "CSM (score mix)", "#5b9dd9"),
        ("rl_mix", "RL blend (action mix)", "#e0925f"),
        ("rl_cond", "RL conditioned", "#8fb85f")]
BG, FG, GRID = "#12151b", "#e8ecf2", "#2b323e"


def style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors="#8d99a9", labelsize=8)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    if title: ax.set_title(title, color=FG, fontsize=10, pad=8)
    if xlabel: ax.set_xlabel(xlabel, color="#8d99a9", fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color="#8d99a9", fontsize=9)


def load(paths):
    rows = []
    for p, box in paths:
        for r in json.load(open(p)):
            r["box"] = box
            rows.append(r)
    return rows


def n_mixed(r):
    return int(sum(1 for v in r["omega"] if v > 1e-6))


def defining(r):
    """Strongest gait signature present, whichever axis carries it.

    This measures whether the motion *is* a gait, not whether it is the right
    one -- and for a weight that asks equally for mutually exclusive rows there
    is no right one.  Read it with `fig_deep` beside it: at the four-way weight
    CSM scores 1.00 here by producing a clean trot and ignoring the other three
    requests, which is a different thing from interpolating between them.
    """
    nz = [i for i, v in enumerate(r["omega"]) if v > 1e-6]
    ax = {1: r["diag"], 2: r["lat"], 3: r["fh"]}
    if nz == [0]:
        return -np.nanmean([r["diag"], r["lat"], r["fh"]])
    vals = [ax[i] for i in nz if i in ax]
    return max(vals) if vals else np.nan


def fig_mixing(rows, out):
    fig, ax = plt.subplots(figsize=(6.4, 3.6), facecolor=BG)
    xs = [1, 2, 3, 4]
    for key, label, colour in ARMS:
        ys, es = [], []
        for n in xs:
            v = [defining(r) for r in rows
                 if r["arm"] == key and r["box"] == "in" and n_mixed(r) == n]
            v = [x for x in v if not np.isnan(x)]
            ys.append(np.mean(v) if v else np.nan)
            es.append(np.std(v) / max(np.sqrt(len(v)), 1) if v else 0)
        ax.errorbar(xs, ys, yerr=es, marker="o", color=colour, label=label,
                    lw=2, capsize=3, ms=5)
    style(ax, "Is it still a gait?  Strongest signature against rows mixed",
          "rows mixed (1 = a pure gait, 4 = all of them equally)",
          "strongest gait signature")
    ax.set_xticks(xs); ax.set_xticklabels(["pure", "pair", "triple", "all four"])
    ax.axhline(0, color=GRID, lw=1)
    ax.annotate("CSM reverts to a\nclean trot here",
                xy=(4, 0.93), xytext=(3.2, 0.62), color="#c96a6a", fontsize=8,
                ha="center", arrowprops=dict(arrowstyle="->", color="#c96a6a", lw=1))
    leg = ax.legend(facecolor="#181c24", edgecolor=GRID, fontsize=8, labelcolor=FG)
    fig.tight_layout(); fig.savefig(out, dpi=150, facecolor=BG); plt.close(fig)
    print("wrote", out)


def fig_midpoints(rows, out):
    mids = collections.defaultdict(lambda: collections.defaultdict(list))
    downs = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r["box"] != "in" or n_mixed(r) != 2:
            continue
        vals = sorted([v for v in r["omega"] if v > 1e-6], reverse=True)
        if abs(vals[0] - vals[1]) > 1e-6:
            continue
        name = r["weight"].replace("0.50", "").replace("+", " + ")
        mids[name][r["arm"]].append(defining(r))
        downs[name][r["arm"]].append(r["down"] / r["seeds"])
    names = sorted(mids)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.2, 5.2), facecolor=BG,
                                  gridspec_kw=dict(height_ratios=[2, 1]))
    w, xs = 0.26, np.arange(len(names))
    for k, (key, label, colour) in enumerate(ARMS):
        ax.bar(xs + (k - 1) * w, [np.nanmean(mids[n][key]) for n in names],
               w, color=colour, label=label)
        ax2.bar(xs + (k - 1) * w, [100 * np.mean(downs[n][key]) for n in names],
                w, color=colour)
    style(ax, "Every pair at 50:50, the weight furthest from anything fitted",
          None, "signal on the requested axis")
    ax.set_xticks(xs); ax.set_xticklabels([])
    ax.axhline(0, color=GRID, lw=1)
    ax.legend(facecolor="#181c24", edgecolor=GRID, fontsize=8, labelcolor=FG)
    style(ax2, None, None, "collapsed %")
    ax2.set_xticks(xs); ax2.set_xticklabels(names, fontsize=8, color="#8d99a9")
    fig.tight_layout(); fig.savefig(out, dpi=150, facecolor=BG); plt.close(fig)
    print("wrote", out)


def fig_robust(rows, out):
    fig, ax = plt.subplots(figsize=(6.4, 3.4), facecolor=BG)
    groups = [("in", 1, "pure"), ("in", 2, "pair"), ("in", 3, "triple"), ("in", 4, "all four"),
              ("out", 1, "pure !"), ("out", 2, "pair !"), ("out", 3, "triple !"), ("out", 4, "all four !")]
    xs = np.arange(len(groups)); w = 0.26
    for k, (key, label, colour) in enumerate(ARMS):
        ys = []
        for box, n, _ in groups:
            g = [r for r in rows if r["arm"] == key and r["box"] == box and n_mixed(r) == n]
            tot = sum(r["seeds"] for r in g) or 1
            ys.append(100 * sum(r["down"] for r in g) / tot)
        ax.bar(xs + (k - 1) * w, ys, w, color=colour, label=label)
    style(ax, "Collapse rate; ! is a command outside the training box",
          None, "collapsed % of rollouts")
    ax.set_xticks(xs); ax.set_xticklabels([g[2] for g in groups], fontsize=8)
    ax.axvline(3.5, color="#6d7887", lw=1, ls="--")
    ax.legend(facecolor="#181c24", edgecolor=GRID, fontsize=8, labelcolor=FG)
    fig.tight_layout(); fig.savefig(out, dpi=150, facecolor=BG); plt.close(fig)
    print("wrote", out)


def fig_runtime(out, dt_ms=20.0):
    names = ["DIAL-MPC\n(teacher)", "CSM composed\n(4 fields x 6)", "PPO\n(1 pass)"]
    ms = [125.83, 0.76, 0.05]
    fig, ax = plt.subplots(figsize=(5.4, 3.4), facecolor=BG)
    colours = ["#c96a6a", "#5b9dd9", "#8fb85f"]
    ax.bar(names, ms, color=colours, width=0.55)
    ax.axhline(dt_ms, color="#e8ecf2", ls="--", lw=1.2)
    ax.text(2.42, dt_ms * 1.15, "20 ms control period", color=FG, fontsize=8, ha="right")
    ax.set_yscale("log")
    for i, v in enumerate(ms):
        ax.text(i, v * 1.25, f"{v:.2f} ms\n{v / dt_ms:.2f}x dt", ha="center",
                color=FG, fontsize=8)
    style(ax, "Cost per control step", None, "ms per step (log)")
    ax.set_ylim(0.02, 400)
    fig.tight_layout(); fig.savefig(out, dpi=150, facecolor=BG); plt.close(fig)
    print("wrote", out)


def fig_deep(rows, out):
    """What the deep mixtures actually produce, axis by axis."""
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), facecolor=BG)
    for ax, n, title in ((axes[0], 3, "three rows mixed"),
                         (axes[1], 4, "all four mixed")):
        xs, w = np.arange(3), 0.26
        for k, (key, label, colour) in enumerate(ARMS):
            g = [r for r in rows if r["arm"] == key and r["box"] == "in" and n_mixed(r) == n]
            ys = [np.nanmean([r[a] for r in g]) for a in ("diag", "lat", "fh")]
            ax.bar(xs + (k - 1) * w, ys, w, color=colour, label=label if n == 3 else None)
        style(ax, title, None, "contact correlation" if n == 3 else None)
        ax.set_xticks(xs)
        ax.set_xticklabels(["diagonal\n(trot)", "lateral\n(pace)", "front/hind\n(bound)"],
                           fontsize=8)
        ax.axhline(0, color=GRID, lw=1); ax.set_ylim(-0.75, 1.05)
    axes[0].legend(facecolor="#181c24", edgecolor=GRID, fontsize=8, labelcolor=FG)
    fig.suptitle("A mixture of mutually exclusive rows has no right answer; "
                 "these are the three on offer", color=FG, fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=150, facecolor=BG); plt.close(fig)
    print("wrote", out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage1", type=Path, default=Path("csm_runs/h2h_stage1.json"))
    p.add_argument("--stage2", type=Path, default=Path("csm_runs/h2h_stage2.json"))
    p.add_argument("--out-dir", type=Path, default=Path("docs/assets"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = load([(args.stage1, "in"), (args.stage2, "out")])
    print(f"{len(rows)} rows")
    fig_mixing(rows, args.out_dir / "gait_mixing.png")
    fig_midpoints(rows, args.out_dir / "gait_midpoints.png")
    fig_deep(rows, args.out_dir / "gait_deep.png")
    fig_robust(rows, args.out_dir / "gait_robustness.png")
    fig_runtime(args.out_dir / "gait_runtime.png")


if __name__ == "__main__":
    main()
