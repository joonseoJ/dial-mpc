set -u; cd /workspace/dial-mpc; source .venv/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Objective with heading (yaw_weight 0.6), T=0.15, repeats 8, height filter 0.19.
# Every stage of this pipeline is the one that was measured, not a default:
# yaw_weight from the DIAL sweep, T from the label scan, 0.19 because 0.25 cut
# the crouch band the walk and bound students live in.
GB="--example unitree_go2_gait --basis e0 e1 e2 e3 --temperature 0.15 --level-scales 1.0 1.0"
while ps -eo args --no-headers | grep -q "[c]sm.collect_cli.*probe_yaw"; do sleep 60; done

echo "=== 1/5 collect v3 (teacher-driven) ==="
python -m csm.collect_cli $GB --steps 2000 --num-envs 8 --repeats 8 \
  --mix-basis-rows 0.7 --out csm_runs/gait_v3 > csm_runs/gait_v3_collect.log 2>&1 \
  && echo "collect v3 done" || { echo "COLLECT V3 FAILED"; tail -6 csm_runs/gait_v3_collect.log; exit 1; }

echo "=== 2/5 fit v3 ==="
python -m csm.fit_from_clouds --clouds csm_runs/gait_v3 $GB --relabel-chunk 1024 \
  --min-height 0.19 --train-iters 300000 --output csm_runs/gait-fit-v3 \
  > csm_runs/gait_v3_fit.log 2>&1 && echo "fit v3 done" || { echo "FIT V3 FAILED"; grep -avE "Warning|warp|pkg_res|autotun" csm_runs/gait_v3_fit.log | tail -10; exit 1; }
grep -aE "height filter|^  e[0-3]: rms|training rows" csm_runs/gait_v3_fit.log
V3=$(ls -td csm_runs/gait-fit-v3/clouds-fit-* | head -1)
python csm/gait/gait_verify2.py $V3/policy.pkl 0.15 > csm_runs/gait_v3_verify.log 2>&1
grep -aE "VERDICT|^(walk|trot|pace|bound) " csm_runs/gait_v3_verify.log

echo "=== 3/5 DAgger d2 (driven by the v3 student) ==="
python -m csm.collect_cli $GB --steps 1500 --num-envs 8 --repeats 8 \
  --mix-basis-rows 0.5 --student-policy $V3/policy.pkl --out csm_runs/gait_d2 \
  > csm_runs/gait_d2_collect.log 2>&1 && echo "collect d2 done" || { echo "COLLECT D2 FAILED"; tail -6 csm_runs/gait_d2_collect.log; exit 1; }

echo "=== 4/5 refit v3 + d2 ==="
python -m csm.fit_from_clouds --clouds csm_runs/gait_v3 csm_runs/gait_d2 $GB --relabel-chunk 1024 \
  --min-height 0.19 --train-iters 300000 --output csm_runs/gait-fit-d2 \
  > csm_runs/gait_d2_fit.log 2>&1 && echo "fit d2 done" || { echo "FIT D2 FAILED"; grep -avE "Warning|warp|pkg_res|autotun" csm_runs/gait_d2_fit.log | tail -10; exit 1; }
grep -aE "height filter|^  e[0-3]: rms|training rows" csm_runs/gait_d2_fit.log
D2=$(ls -td csm_runs/gait-fit-d2/clouds-fit-* | head -1)

echo "=== 5/5 verify ==="
python csm/gait/gait_verify2.py $D2/policy.pkl 0.15 > csm_runs/gait_d2_verify.log 2>&1
grep -aE "VERDICT|^(walk|trot|pace|bound) " csm_runs/gait_d2_verify.log
python csm/gait/gait_verify_long2.py $D2/policy.pkl d2 0.15 > csm_runs/gait_long2_d2.log 2>&1
grep -aE "^(walk|trot|pace|bound) |alive" csm_runs/gait_long2_d2.log
echo "V3 PIPELINE DONE"
