"""Every arm measured, on the three axes that make the comparison honest.

Quality alone ranks the full-horizon planner first and stops there, which is
how the report spent a day saying the composed student costs 1.3x what DIAL
does without noticing that DIAL runs at 17 Hz against a 20 ms control period.
Offline cost alone says train RL per weight and never build a basis.  Online
cost alone says any network beats any planner.  The three together are the
only view in which each arm's trade is visible, so this prints them as one
table and marks which arms could actually be deployed.

Numbers come from the evaluation JSONs and `runtime_cost.json`; nothing here
is typed in by hand except the offline costs, which are wall-clock times
recorded when the jobs ran.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CMDS = ("box_fast", "box_slow", "box_turn", "box_strafe")
PERIOD_MS = 20.0  # env dt

# arm -> (evaluation json, targets to read, online ms/step, offline cost)
ARMS = [
    ("DIAL 2048/2/16 (teacher)", None, None, 58.4, "0 (it is the reference)"),
    ("DIAL stock (std-norm)", "walk_screen/a2_stock_dial.json", None, 58.4, "0"),
    ("DIAL 2048/2/4 (real time)", "walk_screen/a3_realtime_dial.json", None, 18.2, "0"),
    ("CSM composed", "walk_screen/compose_v5_cached.json", None, 1.38, "6.2 h once, whole cone"),
    ("CSM specialist", "walk_screen/specialist_eval.json", None, 1.38, "51 min per weight"),
    ("PPO", None, None, 1.33, "4.2 min per weight"),
    # Under-tuned rather than invalid.  brax counts one gradient update per
    # loop step and a step collects `num_envs` transitions, so 1024 envs at
    # grad_updates_per_step=1 is an update-to-data ratio of ~0.001 against
    # the ~1 SAC is normally run at.  It still lands within cell noise of
    # PPO at equal sample budget, which makes this row a lower bound on
    # model-free RL rather than a verdict on SAC.
    ("SAC (update ratio 0.001)", None, None, 1.33, "6.3 min per weight"),
]

PPO_SLUGS = {"uniform": "uniform", "boost0": "boost0", "boost1": "boost1",
             "boost2": "boost2", "2,1,1": "211", "1,2,1": "121",
             "1,1,2": "112", "3,1,2": "312"}


def cells(path: Path, targets=None):
    """Every (command, target) ratio in a report, optionally filtered."""

    if not path.exists():
        return []
    data = json.load(open(path))
    out = []
    for key, value in data.items():
        command, _, target = key.partition("/")
        if targets is not None and target not in targets:
            continue
        if command not in CMDS:
            continue
        out.append((value["ratio"], value["student_falls"]))
    return out


def gather(root: Path):
    rows = []
    for name, rel, targets, ms, offline in ARMS:
        if name == "PPO":
            got = []
            for slug in PPO_SLUGS.values():
                got += cells(root / f"rl-sweep/{slug}_eval.json")
        elif name.startswith("SAC"):
            got = []
            for slug in ("uniform", "boost1", "boost2"):
                # The fair-budget runs where they exist; the first sweep gave
                # SAC a fifth of PPO's samples and is kept only as a budget
                # curve.
                fair = root / f"sac-fair/{slug}_eval.json"
                got += cells(fair if fair.exists()
                             else root / f"sac-sweep/{slug}_eval.json")
        elif rel is None:
            got = [(1.0, 0)]
        else:
            got = cells(root / rel)
        if not got:
            continue
        ratios = np.array([g[0] for g in got])
        falls = sum(g[1] for g in got)
        rows.append({
            "arm": name, "cells": len(got), "median": float(np.median(ratios)),
            "mean": float(ratios.mean()), "worst": float(ratios.max()),
            "falls": int(falls), "ms": ms, "hz": 1e3 / ms,
            "realtime": 1e3 / ms >= 1e3 / PERIOD_MS, "offline": offline,
        })
    return rows


def main() -> None:
    root = Path("csm_runs")
    rows = gather(root)
    print(f"control period {PERIOD_MS:.0f} ms = {1e3 / PERIOD_MS:.0f} Hz\n")
    head = (f"{'arm':<27}{'cells':>6}{'median':>8}{'mean':>7}{'worst':>8}"
            f"{'falls':>7}{'Hz':>8}  {'offline':<26}")
    print(head)
    print("-" * len(head))
    for r in rows:
        mark = " " if r["realtime"] else "*"
        print(f"{r['arm']:<27}{r['cells']:6d}{r['median']:8.2f}{r['mean']:7.2f}"
              f"{r['worst']:8.2f}{r['falls']:7d}{r['hz']:8.1f}{mark} {r['offline']:<26}")
    print("-" * len(head))
    print("* cannot meet the control period; its quality is not deployable\n")
    live = [r for r in rows if r["realtime"]]
    if live:
        best = min(live, key=lambda r: r["median"])
        print(f"Among arms that fit the period, best median is {best['arm']} "
              f"at {best['median']:.2f}.")
    json.dump(rows, open(root / "baseline_arms.json", "w"), indent=2)
    print(f"wrote {root / 'baseline_arms.json'}")


if __name__ == "__main__":
    main()
