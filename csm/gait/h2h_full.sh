set -u; cd /workspace/dial-mpc; source .venv/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Wait for the smoke run to release the GPU.
while ps -eo args --no-headers | grep -q "[h]ead2head.py"; do sleep 30; done
P="--csm-policy csm_runs/gait-fit-d2/clouds-fit-20260916-090712/policy.pkl \
   --rl-specialists csm_runs/rl-gait-e0/policy.pkl csm_runs/rl-gait-e1/policy.pkl \
                    csm_runs/rl-gait-e2/policy.pkl csm_runs/rl-gait-e3/policy.pkl \
   --rl-conditioned csm_runs/rl-gait-cond/policy.pkl"
# Step 1: the comparison that was resting on one sample per command.
echo "### STAGE 1: pure + pairwise weights, in-box commands ###"
python csm/gait/head2head.py $P --weights all --commands inbox \
  --seeds 6 --steps 900 --out csm_runs/h2h_stage1.json
# Step 2: where CSM should separate -- deeper mixtures and commands outside the box.
echo "### STAGE 2: out-of-box commands ###"
python csm/gait/head2head.py $P --weights all --commands outbox \
  --seeds 6 --steps 900 --out csm_runs/h2h_stage2.json
echo "H2H FULL DONE"
