set -u; cd /workspace/dial-mpc; source .venv/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# The RL side of the composition question.  Four specialists, one per pure gait,
# and one omega-conditioned policy trained across the weight sphere -- the
# latter is CSM's own claim made by RL instead ("train once, serve the cone"),
# and the fair opponent for the composition experiment.
# Episode 1500 matches the horizon the CSM policies are verified at.
#
# Termination is off, and that is not a detail.  Every row of this objective is
# a cost -- there is no alive bonus anywhere in reward_components -- so letting
# the learner end the episode makes falling over the optimal policy.  Measured:
# with it on, all five runs converged to an average episode length of 7.5-7.8
# steps out of 1500, and the "improving" reward curve (-56.8 -> -4.9) was PPO
# getting better at dying quickly.  FixedHorizonWrapper suppresses the done
# flag while the physics keeps running, so a fallen robot keeps paying --
# exactly as it does in the evaluation, and exactly as the CSM student loop
# does, since that loop never resets either.
N=${N:-100000000}
for g in 0 1 2 3; do
  echo "=== specialist e$g ==="
  python -m csm.rl_baseline --example unitree_go2_gait --omega e$g --algo ppo \
    --num-timesteps $N --episode-length 1500 --num-evals 11 \
    --output csm_runs/rl-gait-e$g > csm_runs/rl_gait_e$g.log 2>&1 \
    && grep -a "trained in" csm_runs/rl_gait_e$g.log \
    || { echo "FAILED e$g"; grep -avE "Warning|warp|pkg_res|autotun" csm_runs/rl_gait_e$g.log | tail -8; exit 1; }
  grep -a "^\[ppo\]" csm_runs/rl_gait_e$g.log | tail -3
done
echo "=== omega-conditioned ==="
python -m csm.rl_baseline --example unitree_go2_gait --omega uniform --algo ppo \
  --condition-omega --num-timesteps $N --episode-length 1500 --num-evals 11 \
  --output csm_runs/rl-gait-cond > csm_runs/rl_gait_cond.log 2>&1 \
  && grep -a "trained in" csm_runs/rl_gait_cond.log \
  || { echo "FAILED cond"; grep -avE "Warning|warp|pkg_res|autotun" csm_runs/rl_gait_cond.log | tail -8; exit 1; }
grep -a "^\[ppo\]" csm_runs/rl_gait_cond.log | tail -3
echo "RL TRAIN DONE"
