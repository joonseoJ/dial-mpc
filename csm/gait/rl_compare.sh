set -u; cd /workspace/dial-mpc; source .venv/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false
while ps -eo args --no-headers | grep -q "[r]l_baseline"; do sleep 30; done
S="csm_runs/rl-gait-e0/policy.pkl csm_runs/rl-gait-e1/policy.pkl csm_runs/rl-gait-e2/policy.pkl csm_runs/rl-gait-e3/policy.pkl"
echo "=== 1/3 RL specialists at their own gait ==="
python csm/gait/rl_eval.py --mode specialists --policies $S > csm_runs/rl_eval_spec.log 2>&1 \
  && grep -aE "^(walk|trot|pace|bound) |^gait" csm_runs/rl_eval_spec.log \
  || { echo "SPEC FAILED"; grep -avE "Warning|warp|pkg_res|autotun" csm_runs/rl_eval_spec.log | tail -10; }
echo "=== 2/3 RL blend (action averaging) ==="
python csm/gait/rl_eval.py --mode blend --policies $S > csm_runs/rl_eval_blend.log 2>&1 \
  && grep -aE "^===|^ *[01]\.[0-9]{2}:[01]\.[0-9]{2}" csm_runs/rl_eval_blend.log \
  || { echo "BLEND FAILED"; grep -avE "Warning|warp|pkg_res|autotun" csm_runs/rl_eval_blend.log | tail -10; }
echo "=== 3/3 omega-conditioned PPO ==="
python csm/gait/rl_eval.py --mode conditioned --policies csm_runs/rl-gait-cond/policy.pkl \
  > csm_runs/rl_eval_cond.log 2>&1 \
  && grep -aE "^(walk|trot|pace|bound)|^target" csm_runs/rl_eval_cond.log \
  || { echo "COND FAILED"; grep -avE "Warning|warp|pkg_res|autotun" csm_runs/rl_eval_cond.log | tail -10; }
echo "RL COMPARE DONE"
