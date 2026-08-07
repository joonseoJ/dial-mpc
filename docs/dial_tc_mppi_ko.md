# DIAL-TC-MPPI 설계와 구현

이 구현은 DIAL-MPC의 diffusion-style annealing을 유지하면서, 각 annealing
단계의 독립 가우시안 샘플을 TC-MPPI의 조건부 시간상관 가우시안 샘플로
교체한다. 결과 action을 나중에 필터링하는 방식이 아니라, 최적화가 평가하는
후보 자체가 시간적으로 매끄럽다.

## 1. DIAL의 최적화 변수

DIAL은 매 simulator step의 action을 직접 최적화하지 않는다. 대신

\[
Y=[y_0,\ldots,y_{M-1}],\qquad M=H_{node}+1
\]

이라는 적은 수의 action node를 최적화하고, quadratic spline
\(U=\mathcal P(Y)\)으로 전체 rollout action을 만든다. 따라서 본 구현은
TC-MPPI를 node 공간에 적용한다. 이는 전체 action 공간에 TC precision을
구성하는 것보다 작고, DIAL의 저차원 탐색과 spline 구조를 그대로 보존한다.

## 2. 시간 미분 비용에서 precision 만들기

현재 시점 이전의 실행 action node를
\(Y_h=[y_{-d},\ldots,y_{-1}]\)라 하고, 미래 node를 \(Y_t=Y\)라 한다.
논문 식 (13)과 같이 유한차분 행렬을 구성한다.

\[
D^{(0)}=I,\qquad
D^{(1)}=\frac{1}{\Delta t}
\begin{bmatrix}
-1&1&&\\
&-1&1&\\
&&\ddots&\ddots
\end{bmatrix},\qquad
D^{(i)}=D^{(1)}D^{(i-1)}.
\]

설정값 `tc_derivative_weights`를 \(r_0,\ldots,r_d\)라 하면 논문 식
(15)의 Hessian/precision은

\[
H=\sum_{i=0}^{d}r_i(D^{(i)})^T D^{(i)}
=\begin{bmatrix}H_{hh}&H_{ht}\\H_{th}&H_{tt}\end{bmatrix}.
\]

고차 미분 weight가 커질수록 급격히 굽거나 진동하는 node sequence의
확률이 작아진다. 구현은 모든 actuator에 같은 시간 precision을 적용하며,
이는 \(H\otimes I_{n_u}\)를 메모리에 직접 만들지 않고 계산한 것과 같다.

## 3. 평균을 결합하는 두 모드

TC-MPPI에는 prior 평균 \(\bar Y\)와 실제 sampling 평균 \(\hat Y\)가 있다.
`tc_mean_mode: conditional`은 논문의 평균을 그대로 사용한다.
논문에서 선택한 gradient operator와 동일하게

\[
G_{ref}=-\sum_{i=0}^{d-1}r_i(D^{(i)})^TD^{(i)},\qquad
G_{est}=-r_0I
\]

를 사용한다. 현재 구현의 미래 reference action은 0이며, 직전 DIAL 결과를
estimated optimum으로 사용한다. 논문 식 (18), (19)에 의해

\[
\bar Y=-H_{tt}^{-1}
 (H_{th}Y_h+G_{ref,t}Y_{ref}),
\]

\[
\hat Y=-H_{tt}^{-1}
 (H_{th}Y_h+G_{est,t}[Y_h;Y_{est}]).
\]

첫 항 때문에 새 계획의 시작이 실제 과거 action과 인과적으로 연결된다.
동기 실행에서는 매 제어 step에 실행한 첫 action을, 비동기 planner에서는
직전 publish plan의 첫 node를 history FIFO에 넣는다. reset 시 history도
0으로 초기화한다.

그러나 DIAL은 한 MPC update 안에서 proposal 평균과 분산을 여러 번 바꾸는
annealing 알고리즘이다. 매 annealing 단계마다 위의 고정-prior 평균으로
DIAL 평균을 다시 투영하면 proposal/prior 간격이 커지고 importance weight가
퇴화한다. 실제 Go2 기본값에서는 importance log-ratio가 reward logit보다
약 80배 커졌다.

따라서 기본 `tc_mean_mode: dial`은

\[
\bar Y=\hat Y=Y_{DIAL}
\]

로 두어 DIAL이 최적화한 평균을 유지하고, TC-MPPI의 조건부 covariance만
사용한다. 첫 node도 원래 DIAL처럼 현재 평균에 고정한다. 논문식 conditional
mean은 비교 실험을 위해 계속 선택할 수 있다.

## 4. DIAL annealing과 상관 샘플링 결합

DIAL의 각 annealing 단계가 주는 node별 표준편차를
\(S_i=\operatorname{diag}(\sigma_i)\)라고 한다. 표준 정규 난수
\(\epsilon_k\)로 후보를

\[
Y_k=Y_{DIAL}+S_iH_{tt}^{-1/2}\epsilon_k,
\qquad
Y_k\sim\mathcal N(Y_{DIAL},S_iH_{tt}^{-1}S_i)
\]

처럼 만든다. 즉 TC-MPPI는 시간 방향 covariance의 **모양**을 정하고,
DIAL은 iteration과 horizon에 따른 탐색 **크기**를 정한다. 이후 기존 DIAL과
동일하게 spline 변환, Brax rollout, reward 평가를 수행한다.

## 5. importance weight와 DIAL update

`conditional` 모드에서는 prior와 sampling 분포의 평균이 다르므로 reward만으로
가중하면 편향된다.
논문 식 (23)의 log density ratio를 node별 scale까지 포함해

\[
\rho_k=(\bar Y-\hat Y)^T
S_i^{-1}H_{tt}S_i^{-1}Y_k
\]

로 계산한다. DIAL이 쓰던 표준화 reward logit과 합친 실제 구현의 logit은

\[
L_k=
\frac{R_k-R_{\hat Y}}
{\max(\operatorname{std}(R),10^{-6})\,\tau}
+\beta\rho_k,
\qquad
w_k=\operatorname{softmax}(L_k).
\]

여기서 \(\tau\)는 `temp_sample`, \(\beta\)는 `tc_importance_scale`이다.
기본 `dial` 모드에서는 \(\bar Y=\hat Y\)이므로 \(\rho_k=0\)이며 DIAL의
reward weight만 사용한다.
마지막으로 원래 DIAL/MPPI update를 그대로 사용한다.

\[
Y\leftarrow\sum_k w_kY_k.
\]

이 과정을 `Ndiffuse_init` 또는 `Ndiffuse`번 반복하며 매 단계의
`traj_diffuse_factor`가 covariance 크기를 줄인다.

## 6. 설정과 실행

```bash
dial-mpc --example unitree_go2_trot_tc
```

주요 설정은 다음과 같다.

- `time_correlated`: 새 sampling 경로 사용 여부. 기본값은 `false`다.
- `tc_history_length`: \(d\), 조건화할 과거 node 수이자 최고 미분 차수다.
- `tc_mean_mode`: 안정적인 `dial` 또는 논문 평균을 쓰는 `conditional`.
- `tc_derivative_weights`: 정확히 \(d+1\)개인 \([r_0,\ldots,r_d]\).
- `tc_importance_scale`: density-ratio 항 \(\beta\). 논문식 그대로는 1이다.

기본 예제는 \(d=2\), \([1,0,3\times10^{-5}]\)를 사용한다. node 간격은 0.08 s라서
2차 차분에 이미 \(\Delta t^{-4}\) scale이 들어간다. 따라서 고차 weight는
작게 시작해야 한다. action이 너무 둔하면 마지막 weight를 낮추고, 여전히
진동하면 높인다.

action bound를 지키기 위해 후보를 `[-1, 1]`로 clip한다. 이 경우 엄밀히는
Gaussian이 아니라 잘린 분포가 되므로 경계 근처의 importance ratio는
근사값이다. 이는 기존 DIAL의 clipping 동작을 보존하기 위한 선택이다.

## 7. deploy에서 관절 목표를 실행하는 방법

planner의 action은 직접적인 토크가 아니라 정규화된 **관절 위치 목표**다.
Brax/MJX rollout은 미래의 각 step마다 그때의 상태 $q_k,\dot q_k$를 사용해

\[
q_k^*=\operatorname{act2joint}(u_k),\qquad
\tau_k=\operatorname{clip}\left(
K_p(q_k^*-q_k)-K_d\dot q_k,\tau_{min},\tau_{max}
\right)
\]

를 다시 계산한다. 이전 deploy 구현은 planner가 계획 시작 상태

\[
\tau_k^{old}=K_p(q_k^*-q_0)-K_d\dot q_0
\]

로 horizon 전체 토크를 한 번에 계산한 뒤 이를 그대로 재생했다. 따라서 실제
로봇이 예측 궤적에서 조금만 벗어나도 오차를 되돌리는 피드백이 없었고, plan
buffer가 끝나면 마지막의 낡은 토크가 계속 반복됐다. 시작 직후의 random
jitter, 쓰러짐, 무릎이 끝까지 접히는 현상을 크게 만드는 원인이었다.

현재 sim과 real의 torque deploy 경로는 planner로부터 $q_k^*$를 받고, 각각
최신 MuJoCo 상태 또는 최신 Unitree motor state로 위 PD 식을 제어 주기마다
평가한다. 이는 home 자세를 강제로 넣거나 무릎 각도를 고정하는 보정이 아니다.
position 모드는 기존 동작을 그대로 사용하며 reward weight도 바꾸지 않는다.
