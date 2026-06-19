# 열화상 오경보 필터 (Thermal False-Alarm Filter)

열화상 카메라 알람이 울리면, **열화상 1장 + 실화상(RGB) 1장**을 멀티모달로 교차분석해
**오경보(FALSE_ALARM) / 위험(DANGER)** 을 판정하는 단일 마이크로서비스.

## 산출물

단일 API 호출(`POST /api/v1/verify-fire-alarm`) 1회로 아래 JSON을 반환한다.

```json
{
  "camera_id": "CAM_T1",
  "status": "DANGER",
  "identified_heat_source": "배터리",
  "reasoning": "...",
  "temp_summary": "핫스팟(상위1%) 34.7°C, 장면 중앙값 23.5°C, +11.2°C 높음 ...",
  "thermal_dt": 11.2,
  "decision_source": "thermal+vlm",
  "vlm_status": "DANGER",
  "timing": {"vlm_ms": 6000, "total_ms": 6000}
}
```

## 판정 파이프라인 (융합)

```
열화상 + 실화상 (+온도 CSV)
   │
   ▼ ① 열화상 온도 게이트 (싸다)
   │   핫스팟 ΔT(p99 - 장면 중앙값) < 임계  → 즉시 FALSE_ALARM (VLM 생략)
   │   ΔT ≥ 임계  → ② 로
   ▼ ② VLM 크로스체크 (Qwen3-VL-2B)
   │   핫스팟 정체를 실화상에서 식별 → 사람·조명·햇빛 등 양성 열원이면 FALSE_ALARM 강등
   │                                  그 외 과열 설비면 DANGER
   ▼ JSON 판정 (status enum 은 lm-format-enforcer 로 강제)
```

핵심: **정량 신호(온도 ΔT)가 알람을 게이트**하고, **VLM은 객체 식별 + 양성 열원 강등**을 맡는다.
2B VLM 단독 판정은 온도 등급을 못 가려 무조건 DANGER(50%)였으나, 융합으로 100%.

## 평가 (보유 슬라이스, 라벨 GT 대조)

| 구성 | calib(30) | held-out(20) |
|---|---|---|
| VLM 단독 | 50% (전부 DANGER) | 50% |
| **융합(온도 게이트 + VLM)** | **100%** | **100%** |

- 데이터: AIHub 514 산업시설 열화상(태양광), 프레임당 열화상 `01.jpg` + 정합 RGB `03.jpg` + 온도 `01.csv`
- 임계 `THERMAL_DANGER_DELTA_C`(기본 8°C)는 태양광 calib 값. 설비종류·사이트별 재보정 필요.
- 재현: `scripts/eval_filter.py`(라벨 status를 GT로 대조), `extract_samples.py --skip`(train/test 분리)

## 구조

```
cctv_briefing_agent/
├── requirements.txt
├── README.md
├── frontend/index.html        # 2화면(열화상|실화상) + 판정 배지
└── src/
    ├── config.py              # 모델/프롬프트/온도임계/JSON스키마
    ├── vlm_analyzer.py        # Qwen3-VL 2이미지 추론 + JSON 강제
    ├── api.py                 # run_verify(융합) + 엔드포인트 + 온도 CSV 파서
    └── main.py                # FastAPI 앱 (/ 프론트 서빙)
scripts/
    ├── extract_samples.py     # AIHub zip(CP949) → 페어 추출 (+manifest, --skip)
    ├── eval_filter.py         # 라벨 GT 정확도 평가
    └── test_filter.py         # API 스모크
```

## 실행

```bash
pip install -r requirements.txt          # torch 는 GPU CUDA 빌드 별도 설치 권장
SAMPLE_DIR=samples uvicorn src.main:app --host 0.0.0.0 --port 8011
# 브라우저로 http://localhost:8011  (또는 POST /api/v1/verify-fire-alarm)
```
