# 연구 방향 검토: CSM의 novelty는 어디에 남아 있나

2026-09-26 측정. 세 Go2 과제(외란 극복·보행·보행 스타일)에서 CSM은 동작하고
weight 변화에 실시간으로 반응하지만, RL보다 데이터 수집·학습이 훨씬 비싸다.
이 문서는 그 격차를 뒤집을 수 있는 방향을 찾기 위해 하루 동안 돌린 검증
실험과, 그 과정에서 드러난 기준선 오류, 그리고 결론을 기록한다.

## 요약

1. **걷기 PPO 기준선에 gait 시계가 없었다.** 시계를 붙이면 PPO는 8개 weight
   전부, 64칸 전부에서 CSM을 이기고 DIAL보다 7–10배 싸다(중앙값 0.116 대 CSM
   1.296). ω를 입력으로 받는 PPO 하나도 cone 전체에서 0.092다(18분 학습). 이로써 세 과제
   모두에서 PPO가 DIAL보다 강한 최적화기가 되고, DIAL을 증류하는 CSM은 품질에서
   RL을 이길 수 없다.
2. **효율을 싸게 고칠 방법은 없었다.** 비용 모델 학습(RaMP식), 반복↔상태 배분,
   쿼리 수 축소 모두 측정으로 막혔다.
3. **남은 구조적 장점 후보 — 학습에 없던 제약의 실행 중 합성 — 도 사전 등록한
   판정에서 기각됐다.** 단일 관절 제약 족으로 학습한 PPO가 본 적 없는 결합 제약과
   관절 고정까지 0회 붕괴로 처리했고, 비용은 CSM의 1/10–1/20이다. CSM에 coverage
   DAgger를 더한 판은 오히려 나빠졌다.
4. 부수적으로 얻은 이론적 사실: 개루프 plan 비용은 σ 규모에서 매끄럽지 않고
   (MPPI가 날카로운 온도에서 비싸고, 비용 모델이 편향되는 이유), 앞 몇 스텝만
   바꾸고 나머지를 피드백 정책에 맡기면 같은 지형이 훨씬 잘 조건화된다.

**결론: Go2 보행 계열에서 "DIAL을 증류한 합성형 제어기"로 RL과 경쟁하는 라인은
닫는 것이 맞다.** §6에 다음 방향 후보를 적었다.

## 1. 기준선 정정: gait 시계

걷기 목적함수의 gait 행은 `get_foot_step(..., info["step"] * dt)` — 시간으로
정해진 발 높이 스케줄이다. `unitree_go2_walk`의 관측에는 시계가 없다. DIAL은
롤아웃 안에서 step을 읽고 CSM 학생은 warm-start된 plan에 위상을 싣지만, 반응형
정책은 위상을 알 수 없다. `csm.rl_baseline --clock`이 `sin`, `cos`(2π·cadence·t)를
붙인다(평가기 `compose_walk_eval`은 policy에 기록된 `clock_cadence`로 같은 관측을
다시 만든다).

| | CSM | PPO (시계 없음) | **PPO + 시계** | 조건부 PPO + 시계 (200M) |
|---|---|---|---|---|
| 30초 중앙값 | 1.296 | 1.772 | **0.116** | 0.093 |
| 30초 평균 | 1.436 | 1.904 | **0.119** | 0.092 |
| 30초 최악 | 2.416 | 2.791 | **0.159** | 0.121 |
| 3초 중앙값 | 1.093 | 1.772 | **0.207** | 0.142 |
| CSM 대비 칸 승 | — | 22/32(CSM 승) | **PPO 32/32** | — |

DIAL 대비 비용 비율, 8 weight × 4 명령 × 2 seed, 1500 step. 조건부 PPO는 망 하나로
8 target 전부를 채점한 값이다(시계 없는 조건부 200M은 평균 1.847이었다). 같은 평가 코드로 기존 수치(boost2: PPO 2.68, CSM 1.13)를
재현해 harness 오류가 아님을 확인했다. 시계 없는 PPO도 tracking·stability 행에서는
DIAL보다 나았고 gait 행에서만 졌다. 보행 스타일 과제(`unitree_go2_gait`)는 관측에
이미 위상이 있어 그 비교는 공정했고, 외란 극복은 시간 인덱스 목적이 없다.

정정한 문서: `docs/baseline_comparison_ko.md` §0, `docs/push_recovery_baselines_ko.md`
§2·§4의 주석.

## 2. 효율 개선 후보 — 모두 막힘

### 2.1 update 대신 비용을 배운다 (plan-cost surrogate)

`g(obs, U, V) → 행별 비용`을 저장된 cloud의 **모든** 샘플로 학습하고, 실행 시
g 위에서 MPPI를 돈다(RaMP, NeurIPS'23와 같은 구조). 걷기 held-out shard, 노이즈
보정 cosine(기대 update 대비):

| 조건 | field (update 회귀) | surrogate |
|---|---|---|
| 수집 온도, 32k 쿼리 | 0.976 | **0.983** |
| 학습 범위 밖 ω=(1,0,0) | 0.933 | **0.994** |
| 온도 ×0.25 (ESS≈1%), 32k | **0.95** | 0.74 |
| 같은 조건, 8k → 32k | — | 0.655 → 0.663 (데이터로 안 줄어듦) |

폐루프(1500 step): surrogate 8k 쿼리는 넘어지고, 32k는 비용비 1.6–2.6(CSM 821k는
1.2–2.4). 보행 스타일(ESS 4–6%)에서도 8k에서 field 0.65 대 surrogate 0.45.

**원인:** 날카로운 온도의 MPPI는 cloud 최상위 소수 샘플에 좌우되는데, 접촉 때문에
개루프 비용이 σ 규모에서 매끄럽지 않다(쿼리별 선형 근사가 상위 10% 샘플에서 약
2 logit 잔차). update 회귀의 오차는 분산(상태 간 평균됨)이고 비용 모델의 오차는
편향(데이터로 안 사라짐)이다. **"왜 Q/비용 모델이 아니라 update를 배우나"의
측정된 답**이지만, 연구 방향은 아니다.

### 2.2 수집량 줄이기

- 같은 롤아웃 예산에서 반복↔쿼리 배분은 중립(보행 스타일: 4000×8 ≈ 8000×4 ≈
  16000×2 ≈ 32000×1, 반복 쪽이 약간 낫다).
- 걷기 field를 처음 2k/8k 쿼리로만 적합하면 오프라인 cosine 0.94–0.96인데 폐루프
  24/24 넘어짐. 32k는 중앙값 2.9·5/24 넘어짐, 821k는 1.31·0/24.

## 3. 사전 등록한 가설: 실행 중 제약 합성 — 기각

**가설.** plan-space 학생은 plan을 투영하면 학습 때 없던 제약에도 나머지 좌표를
다시 최적화한다. coverage(무작위 제약 아래에서 수집)만 있으면 제약 족 밖의
제약(결합·고정)도 합성하고, 반응형 정책은 못 한다.

**설계** (`csm/constraint_compose.py` 문서 문자열에 판정 규칙까지 사전 기록):

- 학습 족: 관절 하나에 한쪽 경계(정상 사용 범위의 중앙값~5/95 백분위 사이),
  25%는 제약 없음, 250 step마다 다시 뽑음.
- 시험: knee(족 안, FR 종아리 굴곡 제한) / front(두 앞 종아리) / crouch(네
  종아리 상한 0.1) / lock(FR 종아리 고정, 폭 0) — 뒤 셋이 족 밖.
- arm: DIAL(플래너 안 투영, 참값), csm-prod(기존 field + 투영), csm-cov(족 아래
  학생이 모는 DAgger 164k 쿼리를 v5에 더해 같은 레시피로 재적합), ppo(PPO+시계,
  안전 필터), bppo(PPO+시계를 같은 족으로 학습, 경계를 관측, env가 행동을 자름).
- 붕괴: 몸통 0.15 m 미만 100 step 연속 또는 종료 시(환경 `done`은 웅크림에도
  켜지므로 쓰지 않음). uniform, box_fast/box_turn × 3 seed, 1500 step.

**결과** (비용, 괄호는 붕괴 수 /6):

| arm | none | knee | front | crouch | lock |
|---|---|---|---|---|---|
| DIAL | 0.0356 | 0.0488 | 0.0734 | 0.0771 | 0.0586 |
| csm-prod | 0.0646 | 0.0726 | 0.0985 | 5.21 (6) | 0.149 |
| csm-cov | 0.0924 | 0.0982 | 0.1269 | 4.52 (6) | 1.02 (2) |
| ppo + 시계 + 필터 | 0.0046 | 0.0112 | 0.0152 | 0.0289 | 5.34 (6) |
| **bppo** | **0.0032** | **0.0041** | **0.0047** | **0.0239** | **0.0081** |

판정(규칙 그대로): 족 밖 붕괴 csm-cov 8, bppo 0, csm-prod 6 → **기각.**

- 단일 관절 한쪽 경계만 보고 학습한 PPO가 4관절 결합과 양쪽 고정까지 붕괴 없이
  처리했다. 조건부 RL의 조합 일반화가 예상보다 강하다.
- 시계를 준 PPO는 필터만으로도 crouch를 걷는다(시계 없는 PPO는 6/6 뒤집혔다).
  필터가 실패한 것은 lock 하나이고, 거기서는 plan-space 학생이 살아남는다 —
  CSM에 남은 유일한 우위지만 bppo가 같은 일을 1/18 비용으로 한다.
- coverage 라운드는 모든 조건에서 학생을 나쁘게 만들었다. 학생이 모는 라운드의
  라벨이 더 시끄럽다는 기존 관찰(`fit_from_clouds` 문서의 min-height 경고)과 같은
  방향이다.
- 무릎 대역(knee)에서 학생+필터 +111%, 학생+투영 +12~26%, DIAL +37%였던 앞선
  관찰은 유지된다 — 투영이 필터보다 훨씬 낫다는 것 자체는 사실이다.

## 4. 개루프 대 피드백 완성 비용의 조건화

`csm/smoothness_probe.py`: 학생이 도달한 16개 상태에서, 학생 plan을 지나는 무작위
직선(미세 수준 σ 단위, ±2σ) 위의 16-step 비용을 비교했다.

| 비용 | 2차식으로 설명 안 되는 분산 비율 (중앙값) | 잔차 (T=0.02 logit) | 2049-샘플 cloud ESS |
|---|---|---|---|
| 개루프 (MPPI가 채점하는 것) | 0.062 | 1.13 | 0.19 |
| 앞 4 step만 plan, 나머지 PPO | 0.037 | **0.08** | **0.87** |
| 앞 8 step만 plan, 나머지 PPO | 0.045 | 0.35 | 0.48 |

형태로 본 거칠기 차이는 크지 않다(6.2% → 3.7%). 큰 차이는 **크기**다: 피드백이
섭동의 결과를 흡수해 비용의 비매끄러운 부분이 logit 단위로 14분의 1이 되고, 같은
온도에서 ESS가 0.19 → 0.87이 된다. 개루프 plan 비용을 채점하는 플래너가 샘플을
많이 쓰고, 그 비용을 회귀하는 모델이 편향되는 이유와 같은 원인이다.

## 5. 선행연구 지형 (웹 확인)

- 실행 중 목적·제약 주입: **Flexible Locomotion Learning with Diffusion MPC**
  (Huang et al., ICLR 2026, arXiv 2510.04234) — 같은 Go2, 같은 제약 목록.
- 행동 시퀀스 위 ψ + MPPI: **RaMP** (Chen et al., NeurIPS 2023, 2305.17250).
- Boolean 합성: Todorov 2009, van Niekerk ICML 2019, Nangue Tasse NeurIPS 2020;
  Decision Diffuser가 이미 Go에서 gait 합성.
- 빈자리: 샘플링 MPC를 증류하면서 실행 중 재가중을 유지하는 논문은 없다 — 그러나
  §1·§3의 결과로 이 빈자리의 실용 가치가 사라졌다.

## 6. 다음 방향 후보

CSM의 원래 질문 — "목적을 실행 중에 바꿀 수 있는 제어기" — 은 여전히 유효하다.
바뀐 것은 **누가 최적화기여야 하는가**이다.

1. **RL을 최적화기로, 플래너는 실행 중 보정층으로 — 채택, 후속 문서
   `docs/runtime_customization_ko.md`.** RL 정책이 못 하는 것(학습에 없던 제약·
   목적)만 짧은 계획으로 메운다: 정책을 4 step만 샘플링하고 나머지를 정책(+계획된
   상수 잔차)이 폐루프로 완성한다. 재학습 없이 무릎 고정 6/6 붕괴 → 0/6, 대역 제약에서
   제약 족으로 학습한 PPO와 비기거나 이기고, 몸통 높이 목적에서 5.6분 재학습한 PPO와
   같은 자리, 제약 없는 과제에서도 정책을 24% 개선. 2 step마다 계획하면 실시간
   (계획 한 번 약 27 ms). 개루프 잔차 샘플링(Residual-MPPI식)과 긴 chunk는 무릎
   고정에서 6/6 붕괴한다.
2. **조건부 RL의 조합 일반화.** bppo가 보여 준 것 — 단일 제약으로 학습해 결합·고정에
   일반화 — 이 어디까지 가는지(비상자 제약, 목적 행의 결합, ω 외삽)와 언제
   깨지는지. 원래의 "합성" 질문을 RL 쪽에서 다시 묻는 것.
3. **분석 논문.** "샘플링 MPC 증류는 언제 가치가 있나": 입력 완전성 함정(시계),
   feedback-as-hedge(CVaR·belief 게이트), update 대 비용 모델 추정, coverage 한계,
   개루프 조건화. 모두 측정이 있다.

## 재현

```bash
# §1
csm_runs/rl_sweep_clock.sh
python -m csm.baseline_table 1500 rl-sweep-clock
# §3
python -m csm.collect_cli --example unitree_go2_trot_csm --out csm_runs/walk_collect_band \
  --basis boost0 boost1 boost2 --steps 1000 --num-envs 16 --repeats 2 --temperature 0.020 \
  --level-scales 2.625 1.0 --student-policy csm_runs/clouds-fit-20260905-083613/policy.pkl \
  --band-family --seed 1
csm_runs/band_chain.sh
python -m csm.constraint_compose train-ppo --out csm_runs/band-ppo/200M
python -m csm.constraint_compose eval --arm csm-prod csm:csm_runs/clouds-fit-20260905-083613/policy.pkl \
  --arm ppo ppo:csm_runs/rl-sweep-clock/uniform/policy.pkl --arm dial dial --out csm_runs/constraint_eval.json
python -m csm.constraint_compose eval --arm bppo ppo:csm_runs/band-ppo/200M/policy.pkl \
  --out csm_runs/constraint_eval_bppo.json
python -m csm.constraint_compose verdict csm_runs/constraint_eval*.json
# §4, §6.1
python -m csm.smoothness_probe --out csm_runs/smoothness_probe.json
python -m csm.policy_completed_mppi --out csm_runs/constraint_eval_pmppi.json
```

§2의 탐색 스크립트(surrogate, 반복 배분, 소량 데이터 field)는 세션 scratchpad에만
있다.
