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

이 문서에서 **energy는 cost의 rollout sum**을 뜻한다. 즉 네트워크가 임의의
추상적인 latent energy를 만드는 것이 아니라, 현재 상태에서 주어진 action
trajectory를 실제 dynamics에 넣어 미래 horizon 동안 실행했을 때 누적되는
objective별 cost를 근사한다.

현재 simulator state를 $x_0$, 여기서 얻은 observation을 $o=g(x_0)$라고 하자.
action-node trajectory는 $U=(u_0,\ldots,u_H)$이다. Go2 설정에서는 5개 node를
사용하며, DIAL의 spline 변환 $\mathcal S$가 이들을 16개의 simulation action으로
변환한다.

$$
(a_0,\ldots,a_{T-1})=\mathcal S(U),\qquad T=16.
$$

환경 dynamics를 $F$라 쓰면 rollout state와 objective $i$의 원래 energy는

$$
x_{t+1}=F(x_t,a_t),
\qquad
E_i^*(x_0,U)=\sum_{t=0}^{T-1}c_i(x_{t+1},a_t)
$$

로 정의된다. 여기서
$i\in\{\mathrm{tracking},\mathrm{stability},\mathrm{gait}\}$이다. 따라서

$$
E^*(x_0,U)=
\begin{bmatrix}
\sum_t c_{\mathrm{tracking}}(x_{t+1},a_t)\\
\sum_t c_{\mathrm{stability}}(x_{t+1},a_t)\\
\sum_t c_{\mathrm{gait}}(x_{t+1},a_t)
\end{bmatrix}.
$$

학습 데이터의 `costs[n, i]`가 바로 이 raw rollout sum이다. MPPI temperature나
Gibbs normalization이 들어간 값이 아니며, reward와의 관계는
$R_\omega=-\omega^\top E^*$이다.

네트워크는 완전한 simulator state $x_0$ 대신 observation $o$를 입력받아 이
세 rollout sum을 예측한다.

$$
\hat E_\theta(U,o)
=[\hat E_{\theta,1},\hat E_{\theta,2},\hat E_{\theta,3}]^\top
\approx E^*(x_0,U).
$$

임의의 preference weight $\omega\ge 0$에 대한 energy와 gradient는 추론 시
정확히 선형 결합한다.

$$
\hat E_\omega=\sum_i\omega_i\hat E_{\theta,i},\qquad
\nabla_U\hat E_\omega=\sum_i\omega_i\nabla_U\hat E_{\theta,i}.
$$

따라서 unseen weight를 별도 입력으로 학습하거나 simplex 전체에서 수집하지
않아도 objective basis가 정확하다면 같은 세 objective의 임의 조합을 zero-shot
으로 구성할 수 있다. 양의 상수배 weight는 같은 preference이므로 추론 시
weight 합을 학습 mode들의 기준 합으로 맞춘다.

이 모델은 score regression 계열이지만 noised score vector를 직접 출력하지
않는다. scalar energy를 학습한 뒤 autodiff로
$-\nabla_U\hat E_\omega$를 얻는 conservative vector-field 모델이다.

## 2. objective와 학습 mode

Go2 objective 순서는 다음과 같다.

1. `tracking`: 목표 평면 속도 및 yaw angular velocity 추종 오차
2. `stability`: upright, yaw, torso height 관련 비용
3. `gait`: 목표 foot-height trajectory와 실제 foot height의 오차

학습에 사용한 서로 선형 독립인 세 mode는 `csm/go2_modes.json`에 있다.

$$
W=\begin{bmatrix}2&2&1\\2&1&2\\1&2&2\end{bmatrix},
\qquad \operatorname{rank}(W)=3.
$$

각 행의 합은 5다. 이 mode들은 energy head가 아니라 데이터가 방문할
state/query 분포를 정한다. head는 항상 tracking, stability, gait 각각에
대응한다.

## 3. DIAL-MPC teacher와 데이터 수집

teacher는 실제 `MBDPI.reverse_once`다. 각 query $U_q$에서 독립적인 MPPI
refinement를 8회 실행해 평균한다.

$$
\Delta U_{\mathrm{DIAL}}
=\frac{1}{8}\sum_{r=1}^{8}U_r^{\mathrm{refined}}-U_q.
$$

이는 Monte-Carlo target 분산을 낮추면서 DIAL의 bounded update를 보존한다.
첫 MPPI call의 proposal 중 64개 trajectory를 고른다. 절반은 Gibbs weight가
큰 elite, 절반은 uniform random sample이다. elite는 좋은 basin을, random
sample은 그 주변 landscape를 식별하게 한다.

선택 trajectory는 Brax/MJX 환경에서 직접 rollout하며, preference를 곱하기
전 objective별 horizon 합을 label로 저장한다.

$$
C_i(U,o)=\sum_{t=1}^{T}c_i(x_t,u_t).
$$

따라서 한 trajectory가 세 energy head를 모두 감독한다. query에서는
objective별 exact rollout Jacobian도 계산한다.

$$
J_i^*(U_q,o)=\frac{\partial C_i(U_q,o)}{\partial U_q}.
$$

MJX constraint solver의 dynamic loop는 reverse mode를 지원하지 않으므로
`jax.jacfwd`를 사용한다. 저장 데이터는 value sample, DIAL update anchor,
objective별 exact Jacobian을 함께 포함한다.

## 4. 모델 구조

### 4.1 Observation의 의미와 구성

observation $o=g(x)$는 controller가 매 control tick에 받는 55차원 벡터다.

$$
o=\left[
v^{\mathrm{target}},\
\varpi^{\mathrm{target}},\
q^{\mathrm{ctrl}},\
q^{\mathrm{pos}},\
v^b,\
\varpi^b,\
\dot q^{\mathrm{joint}}
\right].
$$

| 항 | 차원 | 의미 |
|---|---:|---|
| $v^{\mathrm{target}}$ | 3 | 목표 $x,y,z$ 선속도 |
| $\varpi^{\mathrm{target}}$ | 3 | 목표 roll, pitch, yaw 각속도 |
| $q^{\mathrm{ctrl}}$ | 12 | 현재 12개 joint actuator control 값 |
| $q^{\mathrm{pos}}$ | 19 | base position 3, base quaternion 4, joint angle 12 |
| $v^b$ | 3 | body frame에서의 base 선속도 |
| $\varpi^b$ | 3 | body frame에서의 base 각속도 |
| $\dot q^{\mathrm{joint}}$ | 12 | 12개 joint velocity |
| 합계 | 55 | 네트워크에 입력되는 observation 크기 |

현재 최고 버전에는 gait phase의 $\sin\phi,\cos\phi$, 목표/실제 foot height,
contact state가 observation에 직접 들어가지 않는다. 따라서 네트워크가 보는
정보는 완전한 simulator state보다 작다. 동일한 $(o,U)$에 서로 다른 숨은
상태가 대응할 수 있다면 value regression이 학습하는 최적 함수는 엄밀히

$$
\hat E_i(o,U)\approx
\mathbb E\!\left[E_i^*(x_0,U)\mid g(x_0)=o,U\right]
$$

라는 조건부 평균이다. 이는 특히 gait energy의 식별이 tracking이나
stability보다 어려울 수 있는 이유이기도 하다.

### 4.2 Network architecture

네트워크 입력은 이 55차원 observation과 전체 action-node trajectory를 이어
붙인 벡터다.

공유 encoder는 `512-512-512` MLP이고 objective마다 독립적인 `256-256-1`
scalar head가 있다.

$$
z=f_{\mathrm{enc}}([\operatorname{vec}(U),o]),\qquad
\hat e_i=f_i(z).
$$

추가 trainable scalar `log_update_scale`은 물리적 cost-gradient 단위와
bounded DIAL control-update 단위 사이의 양의 변환 계수
$s=\exp(\alpha)$만 학습하며 objective energy 자체는 변형하지 않는다.

## 5. 정규화와 네 가지 학습 신호

학습 batch 크기를 $B$, objective 수를 $K=3$, action-node 수를 $N=5$,
action 차원을 $A=12$라고 쓰자. value sample $b$의 raw rollout cost는
$C_{b,i}=E_i^*(x_b,U_b)$이다.

objective별 평균과 표준편차를 $\mu_i,\sigma_i$라 하면 정규화 target은

$$
y_{b,i}=\frac{C_{b,i}-\mu_i}{\sigma_i}
$$

이다. head는 $\hat e_{b,i}=f_{\theta,i}(U_b,o_b)$를 출력하며, raw cost 단위의
예측값은

$$
\hat E_{b,i}=\sigma_i\hat e_{b,i}+\mu_i
$$

로 복원한다. 기준 run의 최종 통계는 다음과 같다.

```text
mean = [0.7735277414, 0.6342990398, 3.7650680542]
std  = [1.0641529560, 1.5603061914, 5.7700304985]
```

전체 loss는

$$
\mathcal L=\mathcal L_{\mathrm{value}}
+0.2\mathcal L_{\mathrm{dir}}
+0.1\mathcal L_{\mathrm{cal}}
+0.1\mathcal L_{\mathrm{sob}}
$$

이다.

네 loss는 서로 다른 정보를 준다.

- value loss는 후보 trajectory의 누적 cost **값**을 맞춘다.
- direction loss는 objective를 조합한 gradient가 DIAL update와 같은
  **방향**을 가리키게 한다.
- calibration loss는 그 방향을 control update로 바꾸는 전역 **크기**를
  맞춘다.
- Sobolev loss는 각 objective surface의 action별 **미분값**을 exact rollout
  gradient와 맞춘다.

### 5.1 Value regression

잔차를 $r_{b,i}=\hat e_{b,i}-y_{b,i}$라고 하자. 코드에서 사용하는 Huber
함수의 threshold는 1이다.

$$
h(r)=
\begin{cases}
\frac{1}{2}r^2,& |r|\le 1,\\
|r|-\frac{1}{2},& |r|>1.
\end{cases}
$$

value loss는 모든 batch와 objective에 대한 평균이다.

$$
\mathcal L_{\mathrm{value}}
=\frac{1}{BK}\sum_{b=1}^{B}\sum_{i=1}^{K}h(r_{b,i}).
$$

작은 오차에서는 제곱 오차처럼 정밀하게 학습하고, 큰 outlier에서는 절댓값에
가깝게 증가한다. 따라서 매우 큰 rollout cost 하나가 batch 전체를 지배하는
현상을 줄인다. 이 항만으로 함수값은 학습할 수 있지만, 제한된 candidate
사이에서 energy의 기울기가 올바르다는 보장은 없다. 나머지 항들이 그 미분
구조를 추가로 제한한다.

### 5.2 DIAL update 방향 guidance

guidance query $b$에서 objective head $i$의 normalized Jacobian을

$$
\hat G_{b,i,h,a}
=\frac{\partial\hat e_i(o_b,U_b)}{\partial U_{b,h,a}}
$$

라고 하자. 정규화된 energy를 raw cost 단위로 되돌리면 $\mu_i$의 미분은
0이고 $\sigma_i$만 곱해진다.

$$
\hat J_{b,i,h,a}^{\mathrm{raw}}
=\frac{\partial\hat E_i}{\partial U_{b,h,a}}
=\sigma_i\hat G_{b,i,h,a}.
$$

weight $\omega_b$에 대한 cost-descent 방향은

$$
d_{b,h,a}
=-\sum_{i=1}^{K}\omega_{b,i}\hat J_{b,i,h,a}^{\mathrm{raw}}
=-\sum_{i=1}^{K}\omega_{b,i}\sigma_i
\frac{\partial\hat e_i}{\partial U_{b,h,a}}
$$

이다. 실행 직전 고정 node인 첫 node는 $d_{b,0,a}=0$으로 만든다. 같은
query에서 teacher를 8회 평균해 얻은 target을
$\Delta_{b,h,a}^{\mathrm{DIAL}}$이라 하면 sample별 cosine loss는

$$
\ell_{\mathrm{dir},b}
=1-
\frac{\sum_{h,a}d_{b,h,a}\Delta_{b,h,a}^{\mathrm{DIAL}}}
{\sqrt{\sum_{h,a}d_{b,h,a}^2}
 \sqrt{\sum_{h,a}(\Delta_{b,h,a}^{\mathrm{DIAL}})^2}+\epsilon}
$$

이고, guidance batch 크기를 $B_g$라 하면

$$
\mathcal L_{\mathrm{dir}}
=\frac{1}{B_g}\sum_{b=1}^{B_g}\ell_{\mathrm{dir},b}.
$$

이 loss는 두 벡터의 각도만 비교한다. $d_b$가 DIAL update보다 너무 크거나
작아도 방향이 같으면 loss가 작다. 크기는 다음 calibration 항이 담당한다.

### 5.3 Update 크기 calibration

방향 학습이 energy gradient를 결정하고, 별도 scalar $s$만 MPPI update의
크기를 맞춘다. $M=NA$라 하고 trajectory update의 RMS를

$$
\operatorname{RMS}(z_b)
=\sqrt{\frac{1}{M}\sum_{h=1}^{N}\sum_{a=1}^{A}z_{b,h,a}^2+\epsilon}
$$

로 정의한다. 학습되는 양의 scale은 $s=\exp(\alpha)>0$이다. sample별
log-RMS 잔차는

$$
q_b=
\log\operatorname{RMS}\!\left(s\,\operatorname{sg}[d_b]\right)
-\log\operatorname{RMS}\!\left(\Delta_b^{\mathrm{DIAL}}\right)
$$

이다. `sg`는 stop-gradient다. 따라서 이 항의 gradient는 energy head로
흐르지 않고 $s$만 조정한다. 최종 loss는

$$
\mathcal L_{\mathrm{cal}}
=\frac{1}{B_g}\sum_{b=1}^{B_g}h(q_b).
$$

log를 사용하므로 절대 차이보다 배율 차이를 비교한다. 예를 들어 target보다
2배 큰 update와 2배 작은 update가 대칭적인 크기의 오차를 갖는다.

### 5.4 Objective별 Sobolev supervision

simulator rollout을 직접 미분한 exact raw-cost Jacobian은

$$
J_{b,i,h,a}^*
=\frac{\partial C_{b,i}}{\partial U_{b,h,a}}
=\frac{\partial}{\partial U_{b,h,a}}
\sum_{t=0}^{T-1}c_i(x_{b,t+1},a_{b,t})
$$

이다. 이는 action node가 dynamics를 통해 모든 미래 state와 cost에 미치는
영향까지 포함한 total derivative다. head는 normalized energy를 출력하므로
target도

$$
G_{b,i,h,a}^{\mathrm{target}}=\frac{J_{b,i,h,a}^*}{\sigma_i}
$$

로 변환한다. 유효한 exact-gradient sample 수를 $B_v$라 하면

$$
\mathcal L_{\mathrm{sob}}
=\frac{1}{B_vKNA}
\sum_{b=1}^{B_v}\sum_{i=1}^{K}\sum_{h=1}^{N}\sum_{a=1}^{A}
h\!\left(
\frac{\partial\hat e_i(o_b,U_b)}{\partial U_{b,h,a}}
-\frac{J_{b,i,h,a}^*}{\sigma_i}
\right).
$$

value loss가 함수값을 맞춘다면 Sobolev loss는 그 함수의 국소 기울기까지
맞춘다. direction loss와 달리 weight로 합치기 전 objective별 Jacobian을 각각
감독하므로, unseen $\omega$에서도
$\sum_i\omega_i\nabla_U\hat E_i$를 구성할 근거를 제공한다.

이 최고 버전은 objective gradient를 방향/크기 loss로 다시 분해하지 않는다.
cost normalization과 별개의 gradient rescaling, hard-gradient clipping도 넣지
않는다. 단일 pointwise Sobolev 항이 기준이다.

## 6. Query-level DAgger

base data로 30,000 gradient step을 학습한 뒤 DAgger를 3 round 수행한다. 각
round는 mode마다 200 state를 방문하고 15,000 gradient step을 추가한다. 각
state에서 learner query를 실제 DIAL teacher로 다시 labeling한다. 실행 plan은
고정 beta 0.5로 섞는다.

$$
U_{\mathrm{execute}}=\operatorname{clip}
(0.5U_{\mathrm{student}}+0.5U_{\mathrm{teacher}},-1,1).
$$

base와 각 DAgger round는 dataset 크기와 무관하게 round-robin으로 같은 빈도로
sample된다. 새 round가 추가되면 각 group에 같은 질량을 주어 cost mean/std를
다시 계산한다. head의 마지막 affine layer를 정확히 재매개화해 raw energy
예측을 보존하고, 좌표계가 바뀐 뒤 Adam state를 새로 시작한다.

총 gradient step은 $30{,}000+3\times15{,}000=75{,}000$이다. 그러나 배포
정책은 rollout 성능이 더 좋았던 65,000 step을 선택했다.

## 7. Student-only 추론

배포 시 DIAL-MPC나 simulator rollout은 호출하지 않는다. 이전 plan을 spline
shift한 warm start에서 시작해 learned energy만 8회 미분한다.

$$
U^{k+1}=\Pi_{[-1,1]}(U^k+\eta v^k),\qquad
v^k=0.5v^{k-1}+0.5(-s\nabla_U\hat E_\omega(U^k,o)).
$$

기준값은 $\eta=1.0$이다. 한 inference call의 최초 warm-start plan
$U^{\mathrm{anchor}}$에서 누적 displacement RMS가 0.05를 넘지 않도록
trust region으로 투영한다.

$$
\operatorname{RMS}(U^k-U^{\mathrm{anchor}})\le 0.05.
$$

첫 action node는 gradient update에서 고정한다. 다음 control tick에는 저장된
exact spline shift matrix로 plan을 warm-start한다.

## 8. Rollout 기반 체크포인트 선택

5,000 gradient step마다 checkpoint를 저장하고 모든 checkpoint를 각 mode와
두 reset seed에서 300 step 평가한다. 평균 전진 속도 0.5 m/s 미만인 rollout은
생존 점수를 인정하지 않는다. 선택 점수는

$$
S=\overline T_{\mathrm{qualified}}+300R_{\mathrm{qualified}}
-10\,\mathrm{tracking\ RMSE}-\mathrm{mean\ tilt}-\mathrm{action\ jerk}
$$

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
