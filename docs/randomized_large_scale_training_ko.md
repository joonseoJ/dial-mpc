# Randomized large-scale compositional-energy 학습

이 설정은 기존 최고 성능 `objective_compositional_energy` 방법론을 바꾸지
않는다. objective heads, exact DIAL update teacher, 8회 teacher 평균, value /
direction / calibration / Sobolev loss, fixed-beta query DAgger, trust-region
student inference는 그대로다. 달라지는 것은 학습 query를 만드는 초기 상태와
target command의 분포, 그리고 데이터·gradient-step 규모뿐이다.

## Randomization 범위

환경 설정은 `dial_mpc/examples/unitree_go2_trot_randomized.yaml`에 있다.

| 항목 | 분포 |
|---|---|
| target forward velocity | $v_x^*\sim U(0.4,1.1)$ m/s |
| target lateral velocity | $v_y^*\sim U(-0.2,0.2)$ m/s |
| target yaw velocity | $\omega_z^*\sim U(-0.5,0.5)$ rad/s |
| base $x,y$ 위치 변화 | 각 축 $U(-0.05,0.05)$ m |
| base height 변화 | $U(-0.015,0.015)$ m |
| base roll/pitch/yaw 변화 | 각 축 $U(-0.05,0.05)$ rad |
| joint position 변화 | 각 joint $U(-0.08,0.08)$ rad, joint limit 안으로 clip |
| base linear velocity | 각 축 $U(-0.15,0.15)$ m/s |
| base angular velocity | 각 축 $U(-0.15,0.15)$ rad/s |
| joint velocity | 각 joint $U(-0.30,0.30)$ rad/s |

target command는 원래 환경의 ramp-up을 거쳐 observation의 `vel_tar`와
`ang_vel_tar`에 들어간다. 학습 collection은 100 control step마다 episode를
의도적으로 reset하므로, expert가 넘어지지 않아도 새로운 초기 상태와 target을
계속 방문한다. 환경 자체의 command resampling 주기는 250 step이지만 이 학습
명령에서는 periodic reset이 더 빠르므로 대체로 episode마다 새 command를
사용한다.

기존 `unitree_go2_trot.yaml`은 모든 randomization의 기본값이 꺼져 있으므로
기존 학습 및 저장 정책의 동작은 바뀌지 않는다.

## 권장 대규모 명령

```bash
cd /workspace/dial-mpc
source .venv/bin/activate

dial-csm --example unitree_go2_trot_randomized \
  --output csm_runs/compositional-energy-randomized-large \
  --mode-weights csm/go2_modes.json \
  --samples 2048 \
  --collect-steps 2000 --collection-episode-steps 100 \
  --energy-candidates 64 --teacher-repeats 8 \
  --train-iters 100000 \
  --dagger-rounds 5 --dagger-steps 400 \
  --dagger-train-iters 30000 --dagger-beta 0.5 \
  --batch-size 256 --learning-rate 1e-4 \
  --guidance-weight 0.2 --calibration-weight 0.1 --sobolev-weight 0.1 \
  --energy-steps 8 --energy-step-size 1.0 --trust-radius 0.05 \
  --minimum-mean-vx 0.3 --selection-track-command \
  --checkpoint-every 5000 \
  --selection-steps 300 --selection-seeds 5 --eval-steps 500
```

규모는 대략 다음과 같다.

- base: mode당 2,000 state, 약 12,024개의 DIAL query
- DAgger: 5 round × mode당 400 state, 약 12,000개의 DIAL query
- 각 query: 2,049개 MPPI proposal을 사용하는 exact DIAL update를 8회 평균
- value labels: query당 64개, 전체 약 154만 trajectory
- exact objective-gradient anchors: 전체 약 2.4만 query
- optimization: base 100,000 + DAgger 150,000 = 총 250,000 gradient step

첫 state에서는 initial diffusion 10개, 이후에는 regular diffusion 2개를 쓰기
때문에 query 수가 단순히 state 수와 같지는 않다. exact MJX rollout Jacobian과
8회 MPPI 평균 때문에 collection이 학습보다 훨씬 오래 걸릴 수 있다.

## Randomized checkpoint selection

`--selection-track-command`는 기존 고정 0.8 m/s 대신 각 randomized rollout의
실제 $(v_x^*,v_y^*,\omega_z^*)$를 사용한다.

$$
e_t^{\mathrm{track}}
=\|v_{t,xy}^{b}-v_{t,xy}^*\|_2^2
+(\omega_{t,z}^{b}-\omega_{t,z}^*)^2,
\qquad
\mathrm{RMSE}_{\mathrm{track}}
=\sqrt{\frac{1}{T}\sum_t e_t^{\mathrm{track}}}.
$$

selection score의 survival, tilt, jerk 구조는 기존과 같다. forward command의
최솟값이 0.4 m/s이므로 생존 인정 threshold만 0.3 m/s로 낮춘다. 이 옵션을
사용하지 않으면 기존 checkpoint selector는 기존 방식대로 0.8 m/s를 기준으로
계산한다.

## 먼저 실행할 검증

대규모 작업 전에 같은 randomization 경로를 최소 설정으로 확인한다.

```bash
dial-csm --example unitree_go2_trot_randomized \
  --output csm_runs/randomized-smoke \
  --collection-episode-steps 1 \
  --selection-track-command --smoke
```

완료된 run의 `run_config.json`에는 실제 environment randomization 범위가
`environment_randomization`으로 저장된다.
