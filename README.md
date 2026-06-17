# CCTV 화재 전조 브리핑 에이전트

화재 알람 발생 시, **현재 영상 분석(VLM)** 과 **과거 이력 검색(RAG)** 을 한 번에 융합해
관제 요원용 '상황 브리핑 리포트' 를 생성하는 단일 마이크로서비스.

## 산출물

단일 API 호출(`POST /api/v1/generate-briefing`) 1회로 아래 JSON 리포트를 반환한다.

```json
{
  "camera_id": "CAM_03",
  "event_time": "2026-06-17T14:32:00",
  "current_status": "현재 영상에서 관찰된 불꽃·연기·인명 위험 묘사 (VLM)",
  "precursor_history": ["과거 24h 내 동일 카메라의 전조 이력 (RAG)"],
  "summary": "현재+과거를 종합한 관제 브리핑"
}
```

## 파이프라인 (3단계 융합)

| 단계 | 모듈 | 입력 | 출력 |
|---|---|---|---|
| 1. 현재 상황 분석 | `vlm_analyzer` (Qwen3-VL-2B) | 알람 시점 mp4 | 즉각 위험 묘사 텍스트 |
| 2. 전조 증상 검색 | `rag_retriever` (ChromaDB + bge-m3) | camera_id, event_time | 과거 전조 이력 리스트 |
| 3. 브리핑 합성 | `api` (LLM 프롬프트) | 위 둘 | 종합 브리핑 |

## 아키텍처

```
cctv_briefing_agent/
├── requirements.txt
├── README.md
└── src/
    ├── config.py          # env-before-torch, GPU/dtype, 경로, 시간창/임베딩/프롬프트
    ├── vlm_analyzer.py     # 현재 상황 분석 (Qwen2.5-VL-2B, OpenCV 프레임 입력)
    ├── rag_retriever.py    # 전조 검색 (ChromaDB + BAAI/bge-m3)
    ├── api.py             # /api/v1/generate-briefing 라우터 + 합성
    └── main.py            # FastAPI 앱 엔트리
```

설계 기준
- VLM: 기본 `Qwen/Qwen3-VL-2B-Instruct`. (명세의 `Qwen2.5-VL-2B` 는 미출시 ID 라 한국어 네이티브 2B 로 대체. `VLM_MODEL_ID` env 로 교체 가능.)
- 실행 환경: 로컬 RTX4060 8GB = 실운영. VLM fp16(2B), 임베딩 bge-m3 는 CPU(VRAM 충돌 회피).
- 레퍼런스 재사용: VLM 로드/추론 = `hailo_vlm`, ChromaDB/임베딩 = `cctv_memory`.
- 자체 Chroma 스키마(`camera_id`, `event_time` epoch 포함)로 "동일 카메라·과거 시간창" 필터 지원.

## 진행 상태

- [x] 단계 1: 프로젝트 초기화 (구조, requirements, config)
- [ ] 단계 2: 현재 상황 분석 모듈 (`vlm_analyzer.py`)
- [ ] 단계 3: 전조 증상 검색 모듈 (`rag_retriever.py`)
- [ ] 단계 4: 메인 API + 합성 로직 (`api.py`, `main.py`)
- [ ] 단계 5: GitHub 연동 및 초기 커밋

## 실행

```bash
pip install -r requirements.txt
# (torch 는 GPU CUDA 빌드에 맞춰 별도 설치 권장)
uvicorn src.main:app --host 0.0.0.0 --port 8000
```
