# 최고 성능 Compositional Energy 정책 프레임워크

이 문서는 Anchor-Factorized Gibbs Score(AFGS) 실험 이전에 가장 좋은 실제
rollout 결과를 낸 학습 파이프라인을 재현하기 위한 기준 문서다. 코드의 기준
프레임워크 이름은 `objective_compositional_energy`이며, 기준 run은 다음과 같다.

```text
csm_runs/compositional-energy-sobolev/unitree_go2_walk-20260805-065102
```

배포용 `policy.pkl`은 학습의 마지막 체크포인트가 아니다. 모든 저장
체크포인트를 실제 환경에서 평가해 선택한 다음 파일의 복사본이다.

```text
checkpoints/step_065000_dagger_3/policy.pkl
```

## 1. 목표와 compositional 구조

action-node trajectory를 \(U=(u_0,\ldots,u_H)\), observation을 \(o\)라고
하고 세 rollout objective를

\[
C(U,o)=[C_{\mathrm{tracking}},C_{\mathrm{stability}},C_{\mathrm{gait}}]^\top
\]

라고 한다. 네트워크는 weight-conditioned scalar 하나가 아니라 objective별
scalar energy 세 개를 동시에 예측한다.

\[
\hat E_\theta(U,o)
=[\hat E_{\theta,1},\hat E_{\theta,2},\hat E_{\theta,3}]^\top.
\]

임의의 preference weight \(\omega\ge 0\)에 대한 energy와 gradient는 추론 시
정확히 선형 결합한다.

\[
\hat E_\omega=\sum_i\omega_i\hat E_{\theta,i},\qquad
\nabla_U\hat E_\omega=\sum_i\omega_i\nabla_U\hat E_{\theta,i}.
\]

따라서 unseen weight를 별도 입력으로 학습하거나 simplex 전체에서 수집하지
않아도 objective basis가 정확하다면 같은 세 objective의 임의 조합을 zero-shot
으로 구성할 수 있다. 양의 상수배 weight는 같은 preference이므로 추론 시
weight 합을 학습 mode들의 기준 합으로 맞춘다.

이 모델은 score regression 계열이지만 noised score vector를 직접 출력하지
않는다. scalar energy를 학습한 뒤 autodiff로
\(-\nabla_U\hat E_\omega\)를 얻는 conservative vector-field 모델이다.

## 2. objective와 학습 mode

Go2 objective 순서는 다음과 같다.

1. `tracking`: 목표 평면 속도 및 yaw angular velocity 추종 오차
2. `stability`: upright, yaw, torso height 관련 비용
3. `gait`: 목표 foot-height trajectory와 실제 foot height의 오차

학습에 사용한 서로 선형 독립인 세 mode는 `csm/go2_modes.json`에 있다.

\[
W=\begin{bmatrix}2&2&1\\2&1&2\\1&2&2\end{bmatrix},
\qquad \operatorname{rank}(W)=3.
\]

각 행의 합은 5다. 이 mode들은 energy head가 아니라 데이터가 방문할
state/query 분포를 정한다. head는 항상 tracking, stability, gait 각각에
대응한다.

## 3. DIAL-MPC teacher와 데이터 수집

teacher는 실제 `MBDPI.reverse_once`다. 각 query \(U_q\)에서 독립적인 MPPI
refinement를 8회 실행해 평균한다.

\[
\Delta U_{\mathrm{DIAL}}
=\frac{1}{8}\sum_{r=1}^{8}U_r^{\mathrm{refined}}-U_q.
\]

이는 Monte-Carlo target 분산을 낮추면서 DIAL의 bounded update를 보존한다.
첫 MPPI call의 proposal 중 64개 trajectory를 고른다. 절반은 Gibbs weight가
큰 elite, 절반은 uniform random sample이다. elite는 좋은 basin을, random
sample은 그 주변 landscape를 식별하게 한다.

선택 trajectory는 Brax/MJX 환경에서 직접 rollout하며, preference를 곱하기
전 objective별 horizon 합을 label로 저장한다.

\[
C_i(U,o)=\sum_{t=1}^{T}c_i(x_t,u_t).
\]

따라서 한 trajectory가 세 energy head를 모두 감독한다. query에서는
objective별 exact rollout Jacobian도 계산한다.

\[
J_i^*(U_q,o)=\frac{\partial C_i(U_q,o)}{\partial U_q}.
\]

MJX constraint solver의 dynamic loop는 reverse mode를 지원하지 않으므로
`jax.jacfwd`를 사용한다. 저장 데이터는 value sample, DIAL update anchor,
objective별 exact Jacobian을 함께 포함한다.

## 4. 모델 구조

입력은 55차원 Go2 observation과 전체 action-node trajectory를 이어 붙인
벡터다. 실패한 gait-phase observation 실험의 sin/cos 두 항은 없다.

공유 encoder는 `512-512-512` MLP이고 objective마다 독립적인 `256-256-1`
scalar head가 있다.

\[
z=f_{\mathrm{enc}}([\operatorname{vec}(U),o]),\qquad
\hat e_i=f_i(z).
\]

추가 trainable scalar `log_update_scale`은 물리적 cost-gradient 단위와
bounded DIAL control-update 단위 사이의 양의 변환 계수
\(s=\exp(\alpha)\)만 학습하며 objective energy 자체는 변형하지 않는다.

## 5. 정규화와 네 가지 학습 신호

objective별 평균과 표준편차를 \(\mu_i,\sigma_i\)라 하면 head는

\[
y_i=(C_i-\mu_i)/\sigma_i
\]

를 예측한다. 기준 run의 최종 통계는 다음과 같다.

```text
mean = [0.7735277414, 0.6342990398, 3.7650680542]
std  = [1.0641529560, 1.5603061914, 5.7700304985]
```

전체 loss는

\[
\mathcal L=\mathcal L_{\mathrm{value}}
+0.2\mathcal L_{\mathrm{dir}}
+0.1\mathcal L_{\mathrm{cal}}
+0.1\mathcal L_{\mathrm{sob}}
\]

이다.

### 5.1 Value regression

\[
\mathcal L_{\mathrm{value}}
=\operatorname{mean}_{b,i}\operatorname{Huber}(\hat e_{b,i}-y_{b,i}).
\]

### 5.2 DIAL update 방향 guidance

raw-unit objective Jacobian과 composed descent direction은

\[
\hat J_i^{\mathrm{raw}}=\sigma_i\nabla_U\hat e_i,
\qquad d_\omega=-\sum_i\omega_i\hat J_i^{\mathrm{raw}}
\]

이다. 실행 직전 고정 node인 첫 node는 \(d_{\omega,0}=0\)으로 만든다.

\[
\mathcal L_{\mathrm{dir}}
=1-\frac{\langle d_\omega,\Delta U_{\mathrm{DIAL}}\rangle}
{\|d_\omega\|_2\|\Delta U_{\mathrm{DIAL}}\|_2+\epsilon}.
\]

### 5.3 Update 크기 calibration

방향 학습이 energy gradient를 결정하고, 별도 scalar \(s\)만 MPPI update의
크기를 맞춘다. calibration에서는 energy gradient를 stop-gradient한다.

\[
\mathcal L_{\mathrm{cal}}
=\operatorname{Huber}\left(
\log\operatorname{RMS}(s\,\operatorname{sg}[d_\omega])
-\log\operatorname{RMS}(\Delta U_{\mathrm{DIAL}})\right).
\]

### 5.4 Objective별 Sobolev supervision

exact raw cost Jacobian을 normalized-energy 단위로 바꾼 target은

\[
J_i^{\mathrm{target}}=J_i^*/\sigma_i
\]

이고 다음처럼 직접 회귀한다.

\[
\mathcal L_{\mathrm{sob}}
=\operatorname{mean}_{b,i,h,a}\operatorname{Huber}\left(
\frac{\partial\hat e_{b,i}}{\partial U_{h,a}}
-\frac{J^*_{b,i,h,a}}{\sigma_i}\right).
\]

이 최고 버전은 objective gradient를 방향/크기 loss로 다시 분해하지 않는다.
cost normalization과 별개의 gradient rescaling, hard-gradient clipping도 넣지
않는다. 단일 pointwise Sobolev 항이 기준이다.

## 6. Query-level DAgger

base data로 30,000 gradient step을 학습한 뒤 DAgger를 3 round 수행한다. 각
round는 mode마다 200 state를 방문하고 15,000 gradient step을 추가한다. 각
state에서 learner query를 실제 DIAL teacher로 다시 labeling한다. 실행 plan은
고정 beta 0.5로 섞는다.

\[
U_{\mathrm{execute}}=\operatorname{clip}
(0.5U_{\mathrm{student}}+0.5U_{\mathrm{teacher}},-1,1).
\]

base와 각 DAgger round는 dataset 크기와 무관하게 round-robin으로 같은 빈도로
sample된다. 새 round가 추가되면 각 group에 같은 질량을 주어 cost mean/std를
다시 계산한다. head의 마지막 affine layer를 정확히 재매개화해 raw energy
예측을 보존하고, 좌표계가 바뀐 뒤 Adam state를 새로 시작한다.

총 gradient step은 \(30{,}000+3\times15{,}000=75{,}000\)이다. 그러나 배포
정책은 rollout 성능이 더 좋았던 65,000 step을 선택했다.

## 7. Student-only 추론

배포 시 DIAL-MPC나 simulator rollout은 호출하지 않는다. 이전 plan을 spline
shift한 warm start에서 시작해 learned energy만 8회 미분한다.

\[
U^{k+1}=\Pi_{[-1,1]}(U^k+\eta v^k),\qquad
v^k=0.5v^{k-1}+0.5(-s\nabla_U\hat E_\omega(U^k,o)).
\]

기준값은 \(\eta=1.0\)이다. 한 inference call의 최초 warm-start plan
\(U^{\mathrm{anchor}}\)에서 누적 displacement RMS가 0.05를 넘지 않도록
trust region으로 투영한다.

\[
\operatorname{RMS}(U^k-U^{\mathrm{anchor}})\le 0.05.
\]

첫 action node는 gradient update에서 고정한다. 다음 control tick에는 저장된
exact spline shift matrix로 plan을 warm-start한다.

## 8. Rollout 기반 체크포인트 선택

5,000 gradient step마다 checkpoint를 저장하고 모든 checkpoint를 각 mode와
두 reset seed에서 300 step 평가한다. 평균 전진 속도 0.5 m/s 미만인 rollout은
생존 점수를 인정하지 않는다. 선택 점수는

\[
S=\overline T_{\mathrm{qualified}}+300R_{\mathrm{qualified}}
-10\,\mathrm{tracking\ RMSE}-\mathrm{mean\ tilt}-\mathrm{action\ jerk}
\]

이다. 학습 loss나 마지막 checkpoint가 아니라 이 점수가 가장 큰 checkpoint를
`policy.pkl`로 배포한다.

| weight | 생존 step (seed 0/1) | mean vx | tracking RMSE | mean tilt | action jerk |
|---|---:|---:|---:|---:|---:|
| `(2,2,1)` | `300 / 300` | 0.7157 | 0.2003 | 0.0314 | 0.001702 |
| `(2,1,2)` | `114 / 114` | 0.5731 | 0.3426 | 0.1106 | 0.001404 |
| `(1,2,2)` | `140 / 140` | 0.5647 | 0.3253 | 0.1051 | 0.001434 |

전체 평균 생존은 184.67 step, 300-step 생존율은 1/3이다. 두 seed 결과가
동일하므로 `(2,2,1)`의 두 번의 300-step은 서로 다른 두 mode가 아니라 같은
mode의 두 reset 평가다. 현재 기준 중 최고지만 세 mode 전체가 해결됐다는
의미는 아니다.

## 9. 정확한 재현 및 시각화 명령

```bash
cd /workspace/dial-mpc
source .venv/bin/activate

dial-csm --example unitree_go2_trot \
  --output csm_runs/compositional-energy-sobolev \
  --mode-weights csm/go2_modes.json \
  --samples 2048 --collect-steps 400 \
  --energy-candidates 64 --teacher-repeats 8 \
  --train-iters 30000 --dagger-rounds 3 \
  --dagger-steps 200 --dagger-train-iters 15000 --dagger-beta 0.5 \
  --batch-size 256 --learning-rate 1e-4 \
  --guidance-weight 0.2 --calibration-weight 0.1 --sobolev-weight 0.1 \
  --energy-steps 8 --energy-step-size 1.0 --trust-radius 0.05 \
  --minimum-mean-vx 0.5 --checkpoint-every 5000 \
  --selection-steps 300 --selection-seeds 2 --eval-steps 500
```

기준 checkpoint의 실시간 viewer:

```bash
dial-csm-eval \
  --policy csm_runs/compositional-energy-sobolev/unitree_go2_walk-20260805-065102/policy.pkl \
  --example unitree_go2_trot \
  --omega 2,2,1 --steps 300 --episodes 0 \
  --web-viewer-host 0.0.0.0 --web-viewer-port 8080
```

local PC에서 서버 viewer로 접속하려면 local terminal에서 다음 tunnel을 열고
`http://127.0.0.1:8080`을 방문한다.

```bash
ssh -N -L 8080:127.0.0.1:8080 <user>@<server>
```

## 10. 실패 실험과의 경계

이 기준에는 AFGS model/data/training/policy/CLI, hard deployment gate, anchor 및
unseen validation-weight 전수 통과 조건, gait-phase sin/cos observation이 없다.
또한 objective Sobolev gradient의 방향/크기 분해, DAgger hard-gradient 제한,
beta curriculum도 사용하지 않는다. 실험 결과 디렉터리는 비교와 감사를 위해
남기되 해당 실험 코드는 배포 경로에서 제거한다.

DIAL-TC-MPPI 자체는 expert/controller 기능으로 유지한다. 제거 대상은 그 위에
구축했다가 성능이 악화된 AFGS student framework이며, TC-MPPI의 독립 planner
구현은 아니다.
