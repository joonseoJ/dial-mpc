# Randomized large-scale compositional-energy 학습

이 설정은 기존 `objective_compositional_energy` 구조를 유지한다. objective
heads, exact DIAL update teacher, 8회 teacher 평균, value / direction /
calibration / Sobolev loss, trust-region student inference는 그대로다. 이전
대규모 run의 분석에 따라 episode horizon, gait 관측, gradient-loss 비중,
후반 DAgger 실행 분포만 수정한다.

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
`ang_vel_tar`에 들어간다. Collection episode 길이는 `80,80,400` pool에서
균등하게 선택한다. 중복된 80은 short recovery episode를 자주 시작하기
위함이다. 실제 수집 step의 기대 비율은 short 약 28.6%, long 약 71.4%다.
400-step episode는 100-step 이후의 정상 보행과 250-step command resampling을
모두 포함한다.

Randomized 설정의 observation에는 기존 55차원에 다음 8차원이 추가된다.

$$
o_{\mathrm{new}}=[o_{55},z_{\mathrm{feet}}^{\mathrm{target}},
z_{\mathrm{feet}}^{\mathrm{actual}}]\in\mathbb R^{63}.
$$

각 foot-height vector는 FL, FR, RL, RR 네 발에 대응한다. Gait head가 숨겨진
phase를 추측하지 않고 현재 gait target과 실제 발 높이를 직접 조건으로 사용할
수 있게 한다.

기존 `unitree_go2_trot.yaml`은 모든 randomization의 기본값이 꺼져 있으므로
기존 학습 및 저장 정책의 동작은 바뀌지 않는다.

## 권장 대규모 명령

```bash
cd /workspace/dial-mpc
source .venv/bin/activate

dial-csm --example unitree_go2_trot_randomized \
  --output csm_runs/compositional-energy-randomized-mixed \
  --mode-weights csm/go2_modes.json \
  --samples 2048 \
  --collect-steps 2000 --collection-episode-lengths 80,80,400 \
  --energy-candidates 64 --teacher-repeats 8 \
  --train-iters 100000 \
  --dagger-rounds 5 --dagger-steps 400 \
  --dagger-train-iters 30000 --dagger-beta 0.5 \
  --student-only-dagger-rounds 2 \
  --student-only-learning-rate 2e-5 --student-only-train-iters 10000 \
  --student-only-eval-every 2500 --student-only-early-stop-patience 2 \
  --batch-size 256 --learning-rate 1e-4 \
  --guidance-weight 0.3 --calibration-weight 0.1 --sobolev-weight 0.2 \
  --deployment-weight 0.2 --deployment-direction-weight 0.3 \
  --conditional-magnitude-weight 0.1 \
  --conditional-magnitude-cosine 0.7 \
  --conditional-magnitude-temperature 0.1 \
  --deployment-batch-size 8 \
  --sobolev-influence-cap 2.0 \
  --closed-loop-weight 0.05 \
  --closed-loop-horizon-curriculum 4,4,6,8,12 \
  --closed-loop-batch-size 8 --closed-loop-every 4 \
  --energy-steps 8 --energy-step-size 1.0 --trust-radius 0.05 \
  --minimum-mean-vx 0.3 --selection-track-command \
  --checkpoint-every 5000 \
  --selection-steps 300 --selection-seeds 5 \
  --final-selection-steps 500 --final-selection-seeds 10 \
  --selection-finalists 5 \
  --hard-recovery-steps 500 --hard-recovery-window 24 \
  --hard-recovery-queries-per-mode 64 \
  --hard-recovery-teacher-repeats 16 \
  --hard-recovery-train-iters 5000 --hard-recovery-learning-rate 1e-5 \
  --hard-recovery-eval-every 1000 \
  --hard-recovery-early-stop-patience 2 \
  --eval-steps 500
```

규모는 대략 다음과 같다.

- base: mode당 2,000 state, 약 12,024개의 DIAL query
- DAgger: 5 round × mode당 400 state, 약 12,000개의 DIAL query
- 각 query: 2,049개 MPPI proposal을 사용하는 exact DIAL update를 8회 평균
- value labels: query당 64개, 전체 약 154만 trajectory
- exact objective-gradient anchors: 전체 약 2.4만 query
- optimization 상한: base 100,000 + mixed DAgger 90,000 + student-only
  DAgger 20,000 + hard recovery 5,000 = 총 215,000 gradient step. Student-only
  및 hard-recovery 단계는 rollout early stopping에 따라 더 일찍 끝날 수 있다.
- DAgger beta schedule: `[0.5, 0.5, 0.5, 0.0, 0.0]`

## Hard-query 및 deployment supervision

Student-only state의 exact gradient가 정상 보행 query보다 한두 자릿수 커져도
target 자체는 clipping하지 않는다. 대신 composed exact-gradient RMS가
`--sobolev-influence-cap`을 넘으면 그 query가 batch parameter update에 미치는
가중치만 역비례로 낮춘다.

Raw objective energy와 Sobolev Jacobian 학습은 그대로 유지한다. 전역 scalar
`log_update_scale` 대신 observation, 현재 plan, normalized preference를 입력으로
받는 독립 magnitude head가 다음 값을 예측한다.

$$
0\le m_\theta(o,U,\omega)
=r\,\operatorname{sigmoid}(f_\theta(o,U,\omega))\le r.
$$

이 값은 inner step 하나의 크기가 아니라 전체 inference 호출에서 사용할 수 있는
최종 RMS displacement budget이다. Inner optimization step 수를 $T$라 할 때 step
$t=0,\ldots,T-1$의 허용 반경은

$$
b_t=\frac{t+1}{T}m_\theta(o,U_0,\omega)
$$

로 증가한다. RMS-normalized energy descent와 momentum으로 만든 proposal을 매
step $U_0$ 중심의 반경 $b_t$에 projection한다.

$$
\widetilde U_{t+1}
=U_t-\eta v_t,
\qquad
U_{t+1}
=U_0+\operatorname{Proj}_{\operatorname{RMS}\le b_t}
(\widetilde U_{t+1}-U_0).
$$

따라서 8번의 inner update가 magnitude를 반복해서 누적하지 않으며 항상

$$
\operatorname{RMS}(U_T-U_0)\le m_\theta\le r
$$

가 성립한다. Magnitude head는 energy encoder와 parameter를 공유하지 않으므로
controller 크기 학습이 objective energy geometry를 직접 바꾸지 않는다.

Calibration loss도 더 이상 head scalar $m_\theta$를 teacher RMS와 직접 비교하지
않는다. 실제 deployment와 동일한 energy gradient, momentum, action clipping,
progressive projection을 $T$ step 미분 가능하게 전개하고 최종 student update를

$$
\Delta U_\theta=U_T^\theta-U_0
$$

로 정의한다. Exact DIAL target은

$$
\Delta U^*=\operatorname{Proj}_{\operatorname{RMS}\le r}
(U_{\mathrm{DIAL}}-U_0)
$$

이다. Calibration 항은 두 최종 update의 RMS를 비교한다.

$$
\mathcal L_{\mathrm{cal}}
=\frac1B\sum_b h\!\left(
\frac{\operatorname{RMS}(\Delta U_{\theta,b})
-\operatorname{RMS}(\Delta U_b^*)}{r}
\right).
$$

최종 update 방향 항은 momentum과 projection까지 지난 실제 controller update의
cosine을 비교한다.

$$
\mathcal L_{\mathrm{final-dir}}
=\frac1B\sum_b\left(1-
\frac{\langle\Delta U_{\theta,b},\Delta U_b^*\rangle}
{\|\Delta U_{\theta,b}\|_2\|\Delta U_b^*\|_2+\epsilon}
\right).
$$

`--deployment-direction-weight 0.3`이 이 항의 weight다. 기존 raw-gradient
direction loss와 달리 실제 inference의 모든 inner step을 지난 방향을 감독한다.

Deployment vector 항은 최종 update 전체를 비교한다.

$$
\mathcal L_{\mathrm{deploy}}
=\frac1B\sum_b h\!\left(
\frac{\Delta U_{\theta,b}-\Delta U_b^*}{r}
\right).
$$

이 unroll은 second-order autodiff를 포함해 비싸므로 scalar energy, raw direction,
Sobolev loss는 기존 전체 guidance batch를 사용하되 최종-update 세 항만 별도
sub-batch를 사용한다. `--deployment-batch-size 8`이 그 크기다.

Sub-batch는 teacher update RMS에 따라 무작위로만 뽑지 않고 다음처럼 구성한다.

- $\operatorname{RMS}(\Delta U^{\mathrm{DIAL}})<0.8r$: 25%
- $0.8r\le\operatorname{RMS}(\Delta U^{\mathrm{DIAL}})<r$: 25%
- $\operatorname{RMS}(\Delta U^{\mathrm{DIAL}})\ge r$: 50%

따라서 batch size 8에서는 각각 2/2/4 query가 들어간다. Raw exact-gradient가 큰
query의 영향력을 줄이는 `--sobolev-influence-cap`은 Sobolev 항에는 계속
적용하지만, 이미 trust radius로 bounded된 final magnitude/direction/vector
loss에는 적용하지 않는다.

같은 sub-batch를 physical tilt, body height, angular speed, bounded teacher
correction 크기를 결합한 recovery difficulty 순위로도 25/25/50으로 나눈다.
구현은 recovery 3구간과 magnitude 3구간의 3×3 joint cell에서 표본을 뽑으므로,
각 marginal의 easy/boundary/hard 및 low/boundary/saturated 비율이 동시에
2/2/4가 된다. 각 구간에 표본이 없는 작은 dataset에서는 가장 가까운 recovery
또는 magnitude pool로 fallback한다.

포화 target의 magnitude 과소예측은 방향이 맞기 전에는 강제로 키우지 않는다.
최종 update cosine을 $c$라 할 때 stop-gradient gate를

$$
q=\operatorname{stopgrad}\left[\sigma\left(\frac{c-0.7}{0.1}\right)\right]
$$

로 정의하고, 다음 항을 추가한다.

$$
\mathcal L_{\mathrm{under}}
=q\,\mathbf 1[m^*\ge0.8r]
\left(\frac{\max(0,m^*-m_\theta)}{r}\right)^2.
$$

따라서 final direction loss가 먼저 방향을 정렬하고, 정렬된 hard query에 대해서만
`--conditional-magnitude-weight`가 복구 update 크기를 teacher 쪽으로 올린다.

## Multi-step closed-loop supervision

각 DAgger round는 기존 `dagger_round_N.npz` 외에
`dagger_closed_loop_round_N.npz`를 저장한다. 이 파일에는 episode/reset 경계를
넘지 않는 연속 observation, 최초 warm-start plan, objective weight, DIAL teacher
plan이 들어간다.

기본 temporal horizon은 round별 `[4,4,6,8,12]`로 증가한다. 서로 다른 horizon의
round dataset은 padding으로 강제로 합치지 않고 학습 중 round별로 균등 순환한다.

학습은 기록된 observation에는 dynamics gradient를 전파하지 않지만, 학생 plan은

$$
U_t^\theta=\pi_\theta(o_t,\widetilde U_t,\omega),\qquad
\widetilde U_{t+1}=\operatorname{shift}(U_t^\theta)
$$

로 재귀 전개한다. 각 step의 plan을 그 state에서 수집한 DIAL teacher plan과
비교하므로, 독립적인 one-step query뿐 아니라 배포 중 warm-start 오차가 다음
step으로 누적되는 현상도 직접 supervision한다.

마지막 beta=0 round에서는 별도 낮은 learning rate를 사용한다. `10,000` 학습
step과 `400` collection state/mode를 기본 evaluation interval `2,500`에 맞춰
4개의 cycle로 나눈다. 각 cycle은 다음 순서를 따른다.

1. 현재 rollout-best 학생으로 100 state/mode를 실행한다.
2. 그 학생이 실제 생성한 query를 exact DIAL로 새로 relabel한다.
3. 누적된 현재 round 데이터로 2,500 step 학습한다.
4. rollout을 평가하고 지금까지의 best checkpoint를 복원한다.
5. 복원된 best 학생에서 다음 cycle 데이터를 다시 수집한다.

연속 두 cycle에서 개선되지 않으면 남은 수집과 학습을 중단한다. Cycle별 dataset은
`dagger_round_N_cycle_C.npz`와 `dagger_closed_loop_round_N_cycle_C.npz`로
보존한다. 실제 완료 step, horizon, cycle score와 복원 checkpoint는
`run_config.json`의 `dagger_training_history`에 기록된다.

## Hard recovery relabel fine-tuning

일반 DAgger가 끝나면 마지막 rollout-best 학생을 anchor별로 500 step 실행한다.
각 episode의 최근 24개 query를 ring buffer에 유지하고 fall이 발생하면 실패에
가까운 query일수록 recovery difficulty에 최대 2.0의 proximity bonus를 더한다.
이미 물리적으로 넘어진 state는 제외하며 base height가 0.20 m보다 높고
roll/pitch tilt가 1.25 rad보다 작은 복구 가능한 state만 후보로 유지한다.

각 anchor에서 difficulty가 높은 64개 query를 골라 현재 학생 plan에서
DIAL-TC-MPPI update를 16회 다시 계산해 평균한다. 이 데이터는
`hard_recovery_relabel.npz`에 저장된다. 이후 기존 base/DAgger round와 recovery
dataset을 각각 하나의 balanced group으로 두고 learning rate `1e-5`로 최대
5,000 step fine-tuning한다.

매 1,000 step마다 `500 step × 10 common seeds × training anchors` rollout을
수행한다. worst-anchor score가 연속 두 번 개선되지 않으면 조기 종료하고 가장
좋았던 recovery checkpoint를 복원한다. 상세 결과는 `run_config.json`의
`hard_recovery_training_history`에 저장된다. Unseen weight의 state나 rollout은
수집, early stopping, checkpoint 선택 어디에도 사용하지 않는다.

첫 state에서는 initial diffusion 10개, 이후에는 regular diffusion 2개를 쓰기
때문에 query 수가 단순히 state 수와 같지는 않다. exact MJX rollout Jacobian과
8회 MPPI 평균 때문에 collection이 학습보다 훨씬 오래 걸릴 수 있다.

## Randomized checkpoint selection

`--selection-track-command`는 기존 고정 0.8 m/s 대신 각 randomized rollout의
실제 $(v_x^*,v_y^*,\omega_z^*)$를 사용한다.

모든 학습 anchor는 동일한 seed $s_j$로 reset되므로 mode별 초기 자세와 command
randomization이 직접 비교 가능하다. 먼저 모든 checkpoint를 기본
`300 step × 5 seeds`로 screening한다. 상위 5개 checkpoint만 실제 배포 horizon인
`500 step × 10 seeds`로 다시 평가해 최종 정책을 고른다. 두 단계 모두 세 anchor
평균이 아니라 가장 낮은 anchor score를 사용한다.

$$
S_{\mathrm{select}}=\min_{k\in\{1,2,3\}}S_k.
$$

각 $S_k$ 내부의 생존, tracking, tilt, jerk 식은 기존과 같다. 이 방식은 잘 걷는
anchor가 약한 anchor를 평균으로 가리는 것을 막는다. Unseen weight는 selection에
사용하지 않고 학습 완료 후 zero-shot holdout으로만 평가한다.

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

두 단계 결과와 최종 선택은 `checkpoint_selection.json`의 `screening`,
`finalists`, `selected` 필드에 각각 기록된다.

## 먼저 실행할 검증

대규모 작업 전에 같은 randomization, 63차원 observation, mixed-episode,
student-only DAgger 경로를 최소 설정으로 확인한다.

```bash
dial-csm --example unitree_go2_trot_randomized \
  --output csm_runs/randomized-smoke \
  --collection-episode-lengths 1,2 \
  --selection-track-command --smoke
```

완료된 run의 `run_config.json`에는 실제 environment randomization 범위가
`environment_randomization`으로 저장된다.
