# Gait basis: the screens, and what each one was built to catch

The gait task (`unitree_go2_gait`: omega picks walk / trot / pace / bound) took
five failed rounds before it worked, and every failure was a measurement that
was not being taken rather than a method that was wrong. These scripts are
those measurements. They live here rather than in `csm_runs/` because the
lessons are not reproducible from the data alone — a cloud does not record
which question was never asked of it.

Run them from the repository root. Data, logs and traces stay in `csm_runs/`.

## Judging a policy

| script | what it does | why it exists |
| --- | --- | --- |
| `gait_verify2.py` | 170-step screen: pattern, speed, heading drift, turn error, per gait | the short screen, for gating a chain |
| `gait_verify_long2.py` | 1500-step verify, five windows, collapse-aware | **both traps that preceded it.** The student is deterministic and `reset` does not randomise, so three "seeds" were three identical rollouts — the real diversity is the command. And `done` is recomputed every step (torso under 0.18 m) and does *not* stick, so scoring the first `done` as a death reported "0/3 survived" for policies that dipped once in the ramp and then walked for thirty seconds. A collapse is `done` held 100 steps, or a torso ending under 0.15 m |

Both carry a `|drift|` column and run a turning command. The objective now
prices heading; before it did, nothing measured heading, and the two absences
hid each other for five rounds while a policy drove in circles.

## Choosing the objective

| script | what it sweeps |
| --- | --- |
| `gaitscale2.py` | gait-term strength under raw Gibbs — found the collected objective so weak that DIAL never lifted its feet (max foot lift ~0 against a 0.08 m target) |
| `gait_pick.py` | `gait_scale` × `track_floor`, 3 seeds × 3 commands — raising the gait term alone buys the pattern and loses the command and then stability; raising the tracking floor with it recovers both |
| `gait_yaw_pick.py` | `yaw_weight`, including 0 as the control, with both turn directions — at 0 the planner ignores a yaw command outright (0.45 rad/s of error) |

Measure under the convention the *labels* use (raw Gibbs, `std_normalize=False`).
`gait_ab.py` is why: the gaits had been verified under `std_normalize=True`
while collection used raw Gibbs, so every field was distilled from a teacher
that shuffled rather than walked.

## Label quality

| script | what it answers |
| --- | --- |
| `label_diag.py` | are the four gait labels actually different? (they were 0.88–0.94 cosine — the four fields were one field) |
| `gait_yaw_temp.py` | ESS, gait share of the label, and its noise, across temperature, old objective against new |

The inequality that separated the fits that worked from the ones that did not
is **gait-specific signal > its own label noise**. ESS is *not* the thing to
tune: it is high exactly when the softmax averages everything and every weight
returns the same update. Repeats is the only knob that raises the ratio.

## Composition

| script | what it does |
| --- | --- |
| `gait_blend2.py` | all six gait pairs × five ratios, CSM against DIAL at the same mixed weight |
| `gait_mid_long.py` | the six 50:50 midpoints for 1500 steps |
| `gait_assemble_fields.py` | build one composed policy from the best field per gait across several fits |

"Matches DIAL" is not the bar at a mixture: the planner compromises at a
midpoint while the composition commits to a gait, and the composition is
usually the sharper of the two.

## Forensics, kept because the traps recur

| script | the question it settled |
| --- | --- |
| `gait_fallcheck.py` | is a `done` step a fall, or a crouch grazing the height line? |
| `gait_collapse.py` | what happens in the ten steps before the torso goes through the floor? (only feet collide with the ground, so lifting all four sinks the body — and a "pattern" scored after that is a collapsed robot's legs) |
| `gait_speed_probe.py` | does a gait fail because the field is wrong, or because it cannot hold the commanded speed? |
| `gait_repr_probe.py` | two fits with indistinguishable one-step accuracy (cosine 0.832 vs 0.843) had opposite closed-loop outcomes — validation does not predict control, even when it is *identical* |

## The pipeline

`pipeline.sh` is the recipe that produced the working policy: collect at the
chosen objective, fit with `--min-height 0.19`, verify, one DAgger round driven
by that student, refit, verify long.

`--min-height` filters on base height and the plant stands at 0.28 m, so the
old default of 0.25 discarded the 0.19–0.25 m crouch band — 15% of the rows,
and exactly where the walk and bound students live. It is also where a DAgger
round's whole value sits, so a 0.25 cut throws away the data it just collected.
