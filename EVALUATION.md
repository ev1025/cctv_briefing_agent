# 평가 결과 보고서 — 열화상 오경보 필터

대상: 열화상 × 실화상 멀티모달 오경보 필터 (Qwen3-VL-2B + 온도 게이트 융합)

## 1. 평가 방법

| 항목 | 내용 |
|---|---|
| 데이터 | AIHub 514 산업시설 열화상 CCTV (라벨 status: normal / danger, 태양광 설비) |
| 구성 | calib 30프레임(정상 15·위험 15) + held-out 20프레임(정상 10·위험 10) |
| 분리 | calib·held-out 은 **서로 겹치지 않는 프레임**(임계 보정용 / 일반화 검증용) |
| 정답 매핑 | danger → DANGER(위험), normal → FALSE_ALARM(오경보) |
| 도구 | `scripts/eval_filter.py` (라벨 정답과 예측 대조, 혼동행렬) |

## 2. 핵심 결과

| 구성 | 정확도 |
|---|---|
| 베이스라인 (VLM 단독 판정) | **50%** (전부 DANGER 오판) |
| 온도 게이트 + VLM 융합 (calib 30) | **100%** (30/30) |
| 일반화 검증 (held-out 20) | **100%** (20/20) |

> 실측 출력: `scripts/eval_filter.py` 실행 결과는 `outputs/eval_<set>.txt` 에 저장된다.

- **베이스라인 50%**: 소형 2B VLM 이 같은 설비의 온도 등급(정상↔과열)을 사진만으로 못 가려 모든 프레임을 위험으로 오판.
- **융합 100%**: 온도 차이(ΔT)로 1차 판정 → 정상/위험 정확 분류.
- **과적합 아님**: calib 에서 정한 임계(+8℃)가 미지의 held-out 에도 그대로 적용돼 100% (서로 겹치지 않는 프레임).

### 혼동행렬 (calib 30)
```
            예측 DANGER   예측 FALSE_ALARM
실제 DANGER        15              0
실제 FALSE_ALARM    0             15
```

## 3. 온도 신호의 분리력 (무GPU 통계 분석)

| 특징 | 정상 평균 | 위험 평균 | 단일임계 정확도 |
|---|---|---|---|
| 핫스팟 ΔT (상위1% − 장면 중앙값) | +7.0℃ | +9.2℃ | **100%** |
| 단일 픽셀 최댓값 − 중앙값 | +10.8℃ | +11.4℃ | 83% (겹침) |

→ 단일 픽셀 최댓값은 반사 글린트 노이즈로 정상/위험이 겹친다. **상위 1%(p99)** 로 잡으면 완전 분리.

## 4. 처리 시간 (GPU warm, RTX 4060)

| 케이스 | VLM 호출 | 시간 |
|---|---|---|
| 정상 (낮은 온도차이) | 생략 | 즉시 (~0초) |
| 위험 후보 (ΔT ≥ 임계) | 호출 | 약 6초 |

## 5. 한계 및 주의

- **객체 식별**(태양광/배터리 등 명칭)은 2B 모델 특성상 불안정. 단, 최종 판정(status)은 **온도 게이트가 보장**한다.
- **양성 열원 강등**(사람·조명을 정상으로 거르기)은 타깃 시나리오(전기차·주차장 등 비설비 열원이 섞인 현장)용 기능. 현재 태양광 데이터엔 그런 양성 케이스가 없어 **미검증**.
- 임계 +8℃ 는 태양광 슬라이스 보정값. 설비 종류·사이트별 **재보정 필요**(`THERMAL_DANGER_DELTA_C`).

## 6. 재현

```bash
# 샘플 추출(라벨 GT 포함, calib/held-out 분리)
THERMAL_DATA_DIR="...\\2.Validation" python -m scripts.extract_samples --normal 15 --danger 15
THERMAL_DATA_DIR="...\\2.Validation" python -m scripts.extract_samples --skip 15 --normal 10 --danger 10 --out samples_test

# 평가
python -m scripts.eval_filter                         # calib
SAMPLE_DIR=samples_test python -m scripts.eval_filter  # held-out
```

원시 실행 로그: `eval_*.log` (gitignore).
