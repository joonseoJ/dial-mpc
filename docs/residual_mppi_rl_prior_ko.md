# Residual-MPPI용 Go2 RL prior 학습

이 파이프라인은 Residual-MPPI의 basic-task prior로 사용할 stochastic
command-conditioned locomotion policy를 Soft Actor-Critic(SAC)으로 학습한다.
정책은 normalized Go2 action $a\in[-1,1]^{12}$에 대한 tanh-Gaussian 분포이며
다음 세 연산을 제공한다.

```python
policy.mode(observation)             # deterministic action
policy.sample(observation, rng)      # stochastic action
policy.log_prob(observation, action) # exact tanh-Jacobian log likelihood
```

`log_prob`는 이후 Residual-DIAL-TC-MPPI에서 prior cost

$$
C_{\mathrm{prior}}(U)
=-\sum_{t=0}^{H-1}\gamma^t\log\pi_0(u_t\mid o_t)
$$

를 계산하는 데 사용한다.

## Reward 설계

Prior는 단순 survival policy가 아니다. Alive/upright/height/fall 항을 가장 강하게
두되 randomized command tracking과 trot gait reward를 함께 사용해 정지 정책으로
수렴하지 않도록 한다.

$$
r_{\mathrm{prior}}
=4r_{\mathrm{alive}}
+r_{\mathrm{tracking}}
+r_{\mathrm{upright}}
+0.5r_{\mathrm{height}}
+0.25r_{\mathrm{gait}}
-0.05c_{\mathrm{smooth}}
-0.01c_{\mathrm{action}}
-10c_{\mathrm{fall}}.
$$

Tracking, upright, height와 gait reward는 각각 bounded exponential reward로
구현되어 하나의 큰 squared error가 전체 gradient를 지배하지 않는다. 환경은
`unitree_go2_trot_randomized`의 command 및 초기 상태 randomization을 그대로
사용한다. 두 randomization 중 하나라도 꺼져 있으면 학습을 거부한다.

## 권장 대규모 학습

```bash
cd /workspace/dial-mpc
source .venv/bin/activate

dial-rl-prior --example unitree_go2_trot_randomized \
  --output csm_runs/residual-mppi-rl-prior \
  --num-timesteps 20000000 \
  --episode-length 1000 \
  --num-envs 512 --num-eval-envs 64 --num-evals 21 \
  --batch-size 1024 \
  --min-replay-size 32768 --max-replay-size 1000000 \
  --grad-updates-per-step 1 \
  --learning-rate 3e-4 --discounting 0.99 --reward-scaling 0.1 \
  --hidden-sizes 512,512,256 --init-noise-std 0.3 \
  --eval-steps 1000 --eval-seeds 10
```

학습 결과 폴더에는 다음 파일이 생성된다.

- `prior_policy.pkl`: Residual-MPPI에서 직접 로드할 stochastic actor
- `training_history.json`: SAC 학습 및 중간 evaluation metric
- `checkpoints/`: Brax SAC 중간 checkpoint
- `evaluation.json`: deterministic prior의 장기 생존/tracking/tilt/jerk
- `visualization.html`: 첫 evaluation seed의 rollout
- `run_config.json`: 학습 설정, reward scale, 환경 randomization

## Smoke 검증

```bash
dial-rl-prior --example unitree_go2_trot_randomized \
  --output csm_runs/rl-prior-smoke --smoke
```

Smoke 결과는 성능 평가용이 아니라 environment wrapper, SAC update, checkpoint
직렬화 및 정확한 `log_prob` 인터페이스를 확인하는 용도다.

## 학습 후 최소 통과 조건

Residual-MPPI teacher를 만들기 전에 prior 단독으로 다음을 확인한다.

- randomized command에서 1,000-step 완주율 90% 이상
- seed별 특정 command에서 반복적으로 넘어지는 mode가 없을 것
- deterministic action의 tracking RMSE가 지나치게 크지 않을 것
- stochastic sample과 deterministic mode 모두 finite `log_prob`를 가질 것
- action jerk가 높아 안정성을 해치지 않을 것

Prior가 이 조건을 만족하지 못하면 Residual-MPPI나 CSM distillation을 먼저 진행하지
않는다. 약한 prior의 log likelihood는 잘못된 장기 행동을 teacher 목적함수에 직접
주입하기 때문이다.

## Any-state recovery prior

`--recovery-prior`는 정상 자세 생존과 완전 낙상 후 일어나기를 구분해서 학습한다.

- reset은 nominal/tilted/fallen mode를 명시적인 확률로 혼합한다.
- fallen transition에서 `done=1`로 episode를 끝내지 않는다.
- 몸통 건강도 $h(s)$와 한 step 개선량 $h(s_{t+1})-h(s_t)$를 보상한다.
- 일정 시간 upright/height 기준을 동시에 만족해야 recovery 성공으로 인정한다.
- recovery 후에만 velocity tracking과 gait reward가 활성화된다.
- 학습 중 주기적인 선속도/각속도 외란을 가한다.
- recovery 환경에서는 몸통 collision이 있는 별도 torque MuJoCo 모델을 사용한다.

기본 recovery reward는 다음 형태다.

$$
r_t = 2h(s_{t+1})
      + 8\big(h(s_{t+1})-h(s_t)\big)
      + 4I_{\mathrm{upright}}
      + 12I_{\mathrm{new\ recovery}}
      + I_{\mathrm{upright}}(2r_{\mathrm{tracking}}+0.5r_{\mathrm{gait}})
      - 0.03c_{\mathrm{smooth}}-0.005c_{\mathrm{action}}.
$$

예시는 다음과 같다.

```bash
dial-rl-prior --example unitree_go2_trot_randomized \
  --output csm_runs/recovery-prior \
  --recovery-prior \
  --num-timesteps 100000000 --episode-length 1000 \
  --num-envs 1024 --num-eval-envs 96 --num-evals 21 \
  --batch-size 2048 --min-replay-size 65536 --max-replay-size 2000000 \
  --discounting 0.995 --eval-steps 1000 --eval-seeds 20
```

단일 stochastic actor는 학습 분포에 대한 경험적 recovery 확률을 제공할 뿐,
임의의 상태와 임의의 add-on cost에 대한 안전을 수학적으로 보장하지 않는다.
배포 보장이 필요하면 recovery/locomotion option 분리와 safety shield 또는 viability
constraint를 함께 사용해야 한다.
