"""config.py - cctv_briefing_agent 공통 설정 (VLM + RAG 융합 브리핑)

  이 모듈은 torch/transformers 보다 *먼저* import 되어야 한다.
  GPU(CUDA_VISIBLE_DEVICES)와 HF 캐시(HF_HOME)는 torch import 순간 고정되므로
  관련 env 세팅이 torch import 보다 앞서야 한다. 그래서 모든 모듈의 첫 import 가 config 다.
  (config 자체는 torch 를 import 하지 않고 dtype 등은 lazy 로 해석한다.)

"""
import os

# === 1) GPU 고정 (선택) ===
#   로컬(RTX4060 8GB) 기본: 하드코딩 없음. 특정 GPU 만 쓰려면 '실행 전' env 로 지정.
#     예) CUDA_VISIBLE_DEVICES=2,3 python -m uvicorn src.main:app
#   (cctv_memory 는 서버 전용이라 2,3 을 setdefault 했지만 여기선 두지 않는다.)

# === 2) HF 캐시 위치 (선택) ===
_DATA2 = "/workspace/data2"
if os.path.isdir(_DATA2) and "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = os.path.join(_DATA2, "hf_cache")

# === 3) 경로 ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # .../cctv_briefing_agent/src
PROJECT_DIR = os.path.dirname(BASE_DIR)                     # .../cctv_briefing_agent
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
CHROMA_DIR = os.path.join(OUTPUT_DIR, os.environ.get("CHROMA_SUBDIR", "chroma"))
os.makedirs(CHROMA_DIR, exist_ok=True)

# === 4) VLM (현재 상황 분석) ===
#   주의: 'Qwen/Qwen2.5-VL-2B-Instruct' 는 실재하지 않는 ID 다(2.5-VL 은 3B/7B/72B 만 출시, 2B 없음).
#   한국어 네이티브 2B 로 hailo_vlm 에서 검증된 Qwen3-VL-2B 를 기본값으로 사용(Qwen2-VL-2B 는 한국어 거부).
#   Qwen2.5-VL 계열을 쓰려면 env 로 VLM_MODEL_ID=Qwen/Qwen2.5-VL-3B-Instruct 지정.
VLM_MODEL_ID = os.environ.get("VLM_MODEL_ID", "Qwen/Qwen3-VL-2B-Instruct")
VLM_DTYPE = os.environ.get("VLM_DTYPE", "float16")         # 8GB 로컬 기본 fp16 (2B 는 4bit 불필요)
VLM_DEVICE = os.environ.get("VLM_DEVICE", "cuda:0")        # 단일 GPU. CPU 강제 시 'cpu'
LOAD_IN_4BIT = os.environ.get("LOAD_IN_4BIT", "0") == "1"  # 더 작은 GPU 폴백용

# 비디오 입력: OpenCV 균일 프레임 샘플 -> processor(videos=[frames]). hailo_vlm 검증값.
VIDEO_NUM_FRAMES = int(os.environ.get("VIDEO_NUM_FRAMES", "16"))          # 짝수(temporal_patch_size=2)
VIDEO_FRAME_MAX_SIDE = int(os.environ.get("VIDEO_FRAME_MAX_SIDE", "560")) # 8GB VRAM 보호
IMAGE_MIN_PIXELS = int(os.environ.get("IMAGE_MIN_PIXELS", str(256 * 28 * 28)))
IMAGE_MAX_PIXELS = int(os.environ.get("IMAGE_MAX_PIXELS", str(1280 * 28 * 28)))

# 생성 파라미터
VLM_MAX_NEW_TOKENS = int(os.environ.get("VLM_MAX_NEW_TOKENS", "320"))
VLM_TEMPERATURE = float(os.environ.get("VLM_TEMPERATURE", "0.0"))         # 0 = greedy(결정적)
VLM_REPETITION_PENALTY = float(os.environ.get("VLM_REPETITION_PENALTY", "1.05"))
#   한자 토큰 차단(디코드 시 한국어 강제). 기본 off = 모델 원문 그대로 노출(프롬프트로 튜닝).
VLM_FORCE_KOREAN = os.environ.get("VLM_FORCE_KOREAN", "0") == "1"


def torch_dtype():
    """VLM_DTYPE 문자열 -> torch dtype (torch 는 호출 시점에 lazy import)."""
    import torch
    return {
        "float16": torch.float16, "fp16": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        "float32": torch.float32, "fp32": torch.float32,
    }.get(VLM_DTYPE.lower(), torch.float16)


def build_quant_config():
    """LOAD_IN_4BIT=1 일 때 bitsandbytes 4bit 설정(아니면 None)."""
    if not LOAD_IN_4BIT:
        return None
    import torch
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)


# === 5) 임베딩 (BAAI/bge-m3) ===
EMBED_MODEL_ID = os.environ.get("EMBED_MODEL_ID", "BAAI/bge-m3")
EMBED_DEVICE = os.environ.get("EMBED_DEVICE", "cpu")       # VLM 과 VRAM 충돌 회피 위해 CPU 기본

# === 6) RAG (전조 증상 검색) ===
CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "cctv_history")
#   과거 시간창: event_time 기준 [event_time - MAX_HOURS, event_time - MIN_HOURS] 를 조회.
#   기본은 알람 직전부터 24h 과거까지(12~24h 밴드를 포함하는 안전한 상위집합).
LOOKBACK_MIN_HOURS = float(os.environ.get("LOOKBACK_MIN_HOURS", "0"))
LOOKBACK_MAX_HOURS = float(os.environ.get("LOOKBACK_MAX_HOURS", "24"))
RAG_TOP_K = int(os.environ.get("RAG_TOP_K", "5"))
SEARCH_MIN_SCORE = float(os.environ.get("SEARCH_MIN_SCORE", "0.45"))   # cosine 유사도 하한

# === 6b) 온도 판정 보정 ===
#   핫스팟 ΔT(p99 - 장면 중앙값)가 이 값 이상이면 과열 위험으로 안내(프롬프트에 주입).
#   주의: 보유 슬라이스(태양광)에서 calib 한 값(normal p99-med ≤7.2 / danger ≥9.2 → 중간 ~8).
#   설비 종류·사이트마다 재보정 필요. env THERMAL_DANGER_DELTA_C 로 조정.
THERMAL_DANGER_DELTA_C = float(os.environ.get("THERMAL_DANGER_DELTA_C", "8.0"))

# === 7) 프론트엔드 / 샘플 ===
FRONTEND_DIR = os.path.join(PROJECT_DIR, "frontend")
#   데모용 열화상/실화상 페어 폴더. 개인 절대경로는 커밋하지 않음(기본 samples/, env SAMPLE_DIR 로 지정).
SAMPLE_DIR = os.environ.get("SAMPLE_DIR", os.path.join(PROJECT_DIR, "samples"))

# === 8) 프롬프트 (오경보 필터) ===
#   열화상 핫스팟 정체를 실화상에서 식별 -> 정상 발열(오경보) vs 과열 이상(위험). 도메인 중립. JSON 강제.
VERIFY_SYSTEM_PROMPT = (
    "너는 산업 현장의 열화상 기반 화재 조기경보를 판정하는 AI 관제사다. "
    "첫 번째 이미지는 열화상 카메라 영상으로, 뜨거운 곳일수록 밝거나 붉게 표시된다. "
    "두 번째 이미지는 같은 시각·같은 위치를 찍은 실화상(RGB) 영상이다. "
    "열화상에서 가장 뜨거운 부분(핫스팟)의 위치를 실화상에서 찾아, 그 자리에 있는 객체가 무엇인지 식별하라. "
    "햇빛 반사, 조명, 사람, 정상 가동 중인 표면 발열 등 일상적이고 정상적인 열원이면 오경보로 판단하고, "
    "주변보다 비정상적으로 과열된 설비(예: 과열된 배터리·전기 패널)처럼 화재로 번질 수 있는 열원이면 위험으로 판단하라."
)
VERIFY_USER_PROMPT = (
    "위 두 이미지를 분석해 핫스팟의 정체를 식별하고 화재 위험 여부를 판정하라.\n"
    "반드시 아래 JSON 형식으로만 답하라(다른 텍스트·설명·코드펜스 금지):\n"
    '{"status": "FALSE_ALARM 또는 DANGER", "identified_heat_source": "식별된 객체명", "reasoning": "판단 근거"}'
)
#   JSON 출력 강제: lm-format-enforcer 로 디코딩을 스키마에 묶어 '항상 유효한 JSON' 보장.
#   미설치/비활성 시 프롬프트 + 정규식 파서로 폴백(vlm_analyzer._parse_verdict).
VLM_ENFORCE_JSON = os.environ.get("VLM_ENFORCE_JSON", "1") == "1"
VERIFY_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["FALSE_ALARM", "DANGER"]},
        "identified_heat_source": {"type": "string"},
        "reasoning": {"type": "string"},
    },
    "required": ["status", "identified_heat_source", "reasoning"],
}

# === 9) (구) 브리핑 프롬프트 — RAG(rag_retriever) 보존용, verify 플로우 미사용 ===
#   현재 상황 분석(VLM): 보이는 위험 요소만 객관적으로 묘사(추측/부정 echo 금지).
VLM_PROMPT = (
    "당신은 화재 알람이 울린 CCTV 현장을 분석하는 AI 입니다.\n"
    "이 영상에서 관찰되는 불꽃, 연기, 스파크, 인명 피해(쓰러진 사람·대피) 등 "
    "즉각적인 위험 요소를 한국어로 객관적으로 묘사하세요.\n"
    "보이지 않는 위험을 추측하거나 '없음'으로 나열하지 말고, 화면에 실제로 보이는 것만 사실대로 적으세요.\n"
    "불꽃이나 연기가 보이면 그 위치와 규모를 먼저 적으세요."
)

#   전조 시맨틱 질의(RAG 고정 쿼리). camera_id/시간창 필터와 함께 사용.
PRECURSOR_QUERY = os.environ.get(
    "PRECURSOR_QUERY",
    "화재, 연기, 불꽃, 스파크, 전기 합선, 과열, 타는 냄새, 매캐한 연기, "
    "사람의 배회나 침입 등 화재 전조로 의심되는 특이사항")

#   브리핑 합성에 LLM(Qwen 텍스트 전용) 사용 여부. False 면 결정적 템플릿 요약만.
BRIEFING_USE_LLM = os.environ.get("BRIEFING_USE_LLM", "1") == "1"
#   브리핑 합성(LLM): 현재 + 과거 전조 -> 관제 브리핑. (단계 4에서 사용)
BRIEFING_PROMPT = (
    "다음은 화재 알람이 발생한 CCTV 현장의 '현재 상황 분석' 과 '과거 전조 이력' 입니다.\n\n"
    "현재 상황 :\n{current_status}\n\n"
    "과거 전조 이력 : \n{precursor_history}\n\n"
    "위 두 정보를 종합하여, 관제 요원이 즉시 읽을 수 있는 한국어 '상황 브리핑' 을 작성하세요. "
    "현재 위험 요약, 과거 전조와의 연관성, 권고 조치를 3~5문장으로 간결하게 적으세요."
)
