"""config.py - cctv_briefing_agent 공통 설정 (열화상×실화상 멀티모달 오경보 필터).

[중요] 이 모듈은 torch/transformers 보다 *먼저* import 되어야 한다.
  GPU(CUDA_VISIBLE_DEVICES)와 HF 캐시(HF_HOME)는 torch import 순간 고정되므로
  관련 env 세팅이 torch import 보다 앞서야 한다. 그래서 모든 모듈의 첫 import 가 config 다.
  (config 자체는 torch 를 import 하지 않고 dtype 등은 lazy 로 해석한다.)
"""
import os

# === 1) GPU 고정 (선택) ===
#   로컬(RTX4060 8GB) 기본: 하드코딩 없음. 특정 GPU 만 쓰려면 '실행 전' env 로 지정.
#     예) CUDA_VISIBLE_DEVICES=2,3 python -m uvicorn src.main:app

# === 2) HF 캐시 위치 (선택) ===
_DATA2 = "/workspace/data2"
if os.path.isdir(_DATA2) and "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = os.path.join(_DATA2, "hf_cache")

# === 3) 경로 ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # .../cctv_briefing_agent/src
PROJECT_DIR = os.path.dirname(BASE_DIR)                     # .../cctv_briefing_agent
FRONTEND_DIR = os.path.join(PROJECT_DIR, "frontend")
#   데모용 열화상/실화상 페어 폴더. 개인 절대경로는 커밋하지 않음(기본 samples/, env SAMPLE_DIR).
SAMPLE_DIR = os.environ.get("SAMPLE_DIR", os.path.join(PROJECT_DIR, "samples"))

# === 4) VLM (Qwen3-VL-2B) ===
#   주의: 'Qwen/Qwen2.5-VL-2B-Instruct' 는 실재하지 않는 ID 다(2.5-VL 은 3B/7B/.. 만, 2B 는 Qwen3-VL).
#   Qwen2.5-VL 계열을 쓰려면 env 로 VLM_MODEL_ID=Qwen/Qwen2.5-VL-3B-Instruct 지정.
VLM_MODEL_ID = os.environ.get("VLM_MODEL_ID", "Qwen/Qwen3-VL-2B-Instruct")
VLM_DTYPE = os.environ.get("VLM_DTYPE", "float16")         # 8GB 로컬 기본 fp16
VLM_DEVICE = os.environ.get("VLM_DEVICE", "cuda:0")        # 단일 GPU. CPU 강제 시 'cpu'
LOAD_IN_4BIT = os.environ.get("LOAD_IN_4BIT", "0") == "1"  # 더 작은 GPU 폴백용
IMAGE_MIN_PIXELS = int(os.environ.get("IMAGE_MIN_PIXELS", str(256 * 28 * 28)))
IMAGE_MAX_PIXELS = int(os.environ.get("IMAGE_MAX_PIXELS", str(1280 * 28 * 28)))
VLM_MAX_NEW_TOKENS = int(os.environ.get("VLM_MAX_NEW_TOKENS", "320"))
VLM_REPETITION_PENALTY = float(os.environ.get("VLM_REPETITION_PENALTY", "1.05"))


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


# === 5) 온도 판정 보정 ===
#   핫스팟 ΔT(p99 - 장면 중앙값)가 이 값 이상이면 과열 위험으로 게이트(run_verify 1차 게이트).
#   주의: 보유 슬라이스(태양광)에서 calib 한 값(normal p99-med ≤7.2 / danger ≥9.2 → 중간 ~8).
#   설비 종류·사이트마다 재보정 필요. env THERMAL_DANGER_DELTA_C 로 조정.
THERMAL_DANGER_DELTA_C = float(os.environ.get("THERMAL_DANGER_DELTA_C", "8.0"))

# === 6) 프롬프트 / JSON 강제 ===
#   SYSTEM = 역할·입력설명·과제·판정기준·출력형식(고정). USER = 트리거만(중복 제거).
#   온도 측정값은 vlm_analyzer 에서 USER 텍스트에 덧붙는다(판정 문구 없이 측정값만).
VERIFY_SYSTEM_PROMPT = (
    "너는 산업 현장의 열화상 기반 화재 조기경보를 판정하는 AI 관제사다. "
    "입력은 같은 시각·같은 위치의 이미지 2장이다: ① 열화상(뜨거울수록 밝거나 붉음), ② 실화상(RGB). "
    "열화상에서 가장 뜨거운 핫스팟의 위치를 실화상에서 찾아 그 자리의 객체를 식별하라. "
    "햇빛 반사·조명·사람·정상 가동 중 표면 발열 등 정상적인 열원이면 FALSE_ALARM, "
    "주변보다 비정상적으로 과열된 설비(예: 과열된 배터리·전기 패널)처럼 화재로 번질 수 있는 열원이면 DANGER 로 판정하라. "
    "반드시 다음 JSON 형식으로만 답한다(다른 텍스트·코드펜스 금지): "
    '{"status": "FALSE_ALARM 또는 DANGER", "identified_heat_source": "식별된 객체명", "reasoning": "판단 근거"}'
)
VERIFY_USER_PROMPT = "위 두 이미지를 판정하라."
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
