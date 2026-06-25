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
#   데모용 열화상/실화상 페어 폴더. 개인 절대경로는 커밋하지 않음(기본 LLM/DATA/images/thermal/samples, env SAMPLE_DIR).
#   data/ 아래: samples=보정셋(임계 calib), samples_test=검증셋(held-out). 둘 다 AIHub라 gitignore.
SAMPLE_DIR = os.environ.get("SAMPLE_DIR", os.path.join(os.path.dirname(PROJECT_DIR), "DATA", "images", "thermal", "samples"))

# === 4) VLM (Qwen3-VL-2B) ===
#   주의: 'Qwen/Qwen2.5-VL-2B-Instruct' 는 실재하지 않는 ID 다(2.5-VL 은 3B/7B/.. 만, 2B 는 Qwen3-VL).
#   Qwen2.5-VL 계열을 쓰려면 env 로 VLM_MODEL_ID=Qwen/Qwen2.5-VL-3B-Instruct 지정.
VLM_MODEL_ID = os.environ.get("VLM_MODEL_ID", "Qwen/Qwen3-VL-2B-Instruct")
VLM_DTYPE = os.environ.get("VLM_DTYPE", "float16")         # 8GB 로컬 기본 fp16
VLM_DEVICE = os.environ.get("VLM_DEVICE", "cuda:0")        # 단일 GPU. CPU 강제 시 'cpu'
LOAD_IN_4BIT = os.environ.get("LOAD_IN_4BIT", "0") == "1"  # 더 작은 GPU 폴백용
IMAGE_MIN_PIXELS = int(os.environ.get("IMAGE_MIN_PIXELS", str(256 * 28 * 28)))
IMAGE_MAX_PIXELS = int(os.environ.get("IMAGE_MAX_PIXELS", str(1280 * 28 * 28)))
VLM_MAX_NEW_TOKENS = int(os.environ.get("VLM_MAX_NEW_TOKENS", "640"))  # CoT reasoning 후 뒤 필드까지 완성되게 넉넉히
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


# === 5) 설비별 온도 판정 규칙 (라우터) ===
# 설비별 위험 판별(온도 게이트) 방식이 다르기 때문에 라우터를 통해 설비별 분류
# 태양광   = 온도 상위 1%(p99)와 장면 중앙값의 차이(상대 ΔT) ΔT < 0.8이면 normal, ΔT >= 8.0 -> 임계 ~8.
# 배터리팩 = 균일 과열(열폭주) -> 상대 ΔT 안 큼(둘 다 ~2°C). 대신 절대 핫스팟(p99) 온도로 갈림:
# normal 28.0~28.4 / danger 29.3~30.0 -> 임계 ~28.8.
# run_verify 가 VLM 으로 식별한 설비에 맞는 규칙으로 라우팅. 사이트·계절(절대온도는 환경 의존)마다 재보정 필요.
THERMAL_DANGER_DELTA_C = float(os.environ.get("THERMAL_DANGER_DELTA_C", "8.0"))   # 태양광: 상대 ΔT
BATTERY_DANGER_TEMP_C = float(os.environ.get("BATTERY_DANGER_TEMP_C", "28.8"))    # 배터리팩: 절대 핫스팟(p99)


def thermal_rule_for(heat_source):
    """식별된 설비 -> (mode, threshold, label). 배터리팩=절대온도, 그 외(태양광 등)=상대 ΔT 기본."""
    if heat_source and "배터리" in heat_source:
        return ("abs", BATTERY_DANGER_TEMP_C, "절대온도")
    return ("delta", THERMAL_DANGER_DELTA_C, "상대ΔT")

# === 6) 프롬프트 (체인 룰: 객체식별 -> 위험판정 -> 근거, 각각 별도 호출) ===
#   2B 모델엔 JSON 다필드 강제(긴 출력 -> 루프/잘림)보다 '한 번에 하나씩' 묻는 체인이 빠르고 정확.
#   공용 시스템 프롬프트는 '셋업'만, 각 단계 질문은 USER 에서 한 가지만 묻는다.
VERIFY_SYSTEM_PROMPT = (
    "너는 산업 현장의 화재 조기경보를 판정하는 AI 관제사다. "
    "1번 이미지는 열화상(뜨거울수록 밝거나 붉음), 2번은 같은 위치의 실화상(RGB)이다."
)
#   1단계 — 핫스팟 객체 식별. 2B 가 평서문으로 풀어쓰는 걸 막으려고 '후보 목록 객관식'(구분자 없이 평문).
#   sLLM 은 <> [] 같은 특수 구분자를 잘 못 따르므로, 후보 단어를 주고 하나만 고르게 한다(코드가 매칭).
VERIFY_OBJECT_CHOICES = (
    "태양광", "배터리팩", "배전반", "변압기", "전동기", "배관", "케이블", "사람", "조명", "차량", "햇빛", "기타",
)
VERIFY_OBJECT_PROMPT = (
    "열화상에서 가장 뜨거운 지점이 실화상에서 무슨 물체인지, 아래 목록 중 가장 가까운 것 하나만 그 단어 그대로 답하라.\n"
    + " / ".join(VERIFY_OBJECT_CHOICES)
    + "\n목록에 없으면 기타. 설명·문장 없이 단어만."
)
#   2단계 — 위험/정상 판정 (1단계 객체를 맥락으로). 한 단어만.
VERIFY_STATUS_PROMPT = (
    "핫스팟의 객체는 '{object}' 이다. "
    "사람·조명·차량·햇빛 등 화재와 무관한 열원이거나 정상 가동 중 발열이면 FALSE_ALARM, "
    "비정상적으로 과열된 설비나 화재 징후면 DANGER. FALSE_ALARM 또는 DANGER 한 단어만 답하라."
)
#   3단계 — 판단 근거 (객체+판정을 맥락으로). 한 문장만.
VERIFY_REASON_PROMPT = (
    "핫스팟 객체는 '{object}', 판정은 '{status}' 다. 그렇게 판단한 근거를 한 문장으로만 답하라."
)
