set -u; cd /workspace/dial-mpc; source .venv/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Every 5th step at 64 colours: the walking report's clips sit at 1-7 MB and a
# 14 MB gif is not a figure, it is a download.
C="csm_runs/gait-fit-d2/clouds-fit-20260916-090712/policy.pkl"
S="csm_runs/rl-gait-e0/policy.pkl csm_runs/rl-gait-e1/policy.pkl csm_runs/rl-gait-e2/policy.pkl csm_runs/rl-gait-e3/policy.pkl"
COMMON="--steps 400 --every 5 --colors 64"
while ps -eo args --no-headers | grep -q "[r]ender_gaits.py"; do sleep 20; done
echo "=== 1/4 four pure gaits ==="
python csm/gait/render_gaits.py --mode pure --csm-policy $C $COMMON \
  --width 250 --height 195 --out docs/assets/gait_pure.gif
echo "=== 2/4 trot -> pace sweep ==="
python csm/gait/render_gaits.py --mode sweep --csm-policy $C --pair trot pace $COMMON \
  --width 230 --height 180 --out docs/assets/gait_sweep_trot_pace.gif
echo "=== 3/4 CSM vs RL at trot+bound ==="
python csm/gait/render_gaits.py --mode versus --csm-policy $C --rl-specialists $S \
  --pair trot bound $COMMON --width 320 --height 240 \
  --out docs/assets/gait_versus_trot_bound.gif
echo "=== 4/4 CSM vs RL at trot+pace ==="
python csm/gait/render_gaits.py --mode versus --csm-policy $C --rl-specialists $S \
  --pair trot pace $COMMON --width 320 --height 240 \
  --out docs/assets/gait_versus_trot_pace.gif
echo "RENDER ALL DONE"
