"""vlm_analyzer.py - 멀티모달 오경보 필터 (열화상 + 실화상 크로스체크).

Qwen3-VL-2B 로 열화상 1장 + 정합 실화상(RGB) 1장을 보고, 체인 룰로 3단계 추론:
  1) 핫스팟 객체 식별  2) 위험/정상 판정  3) 판단 근거.
2B 모델엔 JSON 다필드 강제(긴 출력 -> 루프/잘림)보다 '한 번에 하나씩 묻는' 체인이 빠르고 정확.

  - 모델/프로세서는 모듈 싱글톤(1회 로드), GPU 는 Lock 으로 직렬 사용.
  - 이미지 2장은 qwen-vl-utils(process_vision_info) 로 추출, 실패 시 PIL 직접 전달 폴백.

CLI 스모크: python -m src.vlm_analyzer <thermal.jpg> <rgb.jpg>
"""
import json
import threading

from . import config  # noqa: F401  (env-before-torch: 최상단)

_MODEL = None
_PROCESSOR = None
_LOAD_LOCK = threading.Lock()  # 모델 로드 직렬화
_GEN_LOCK = threading.Lock()   # GPU 추론 직렬화


# ── 모델 로드 (싱글톤) ────────────────────────────────────────────────────────
def _load_model():
    """(processor, model) 싱글톤 반환. 최초 1회만 로드."""
    global _MODEL, _PROCESSOR
    if _MODEL is not None:
        return _PROCESSOR, _MODEL
    with _LOAD_LOCK:
        if _MODEL is not None:
            return _PROCESSOR, _MODEL
        import torch
        from transformers import AutoProcessor

        hf_id = config.VLM_MODEL_ID
        cuda_ok = torch.cuda.is_available()
        device = config.VLM_DEVICE if cuda_ok else "cpu"
        dtype = config.torch_dtype() if cuda_ok else torch.float32
        quant = config.build_quant_config() if cuda_ok else None

        processor = AutoProcessor.from_pretrained(
            hf_id, trust_remote_code=True,
            min_pixels=config.IMAGE_MIN_PIXELS, max_pixels=config.IMAGE_MAX_PIXELS)

        load_kwargs = dict(trust_remote_code=True)
        if quant is not None:
            load_kwargs.update(quantization_config=quant, device_map=device)
        else:
            load_kwargs.update(torch_dtype=dtype, device_map=device)

        # 전용 클래스 하드코딩 금지: Auto 가 체크포인트에 맞는 클래스를 고른다
        #   (하드코딩하면 다른 아키텍처를 '오류 없이' 잘못 로드해 generate 시 깨짐).
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(hf_id, **load_kwargs).eval()

        _PROCESSOR, _MODEL = processor, model
        print(f"[VLM] loaded {hf_id} device={device} dtype={dtype} 4bit={quant is not None}", flush=True)
        return _PROCESSOR, _MODEL


# ── 유틸 ─────────────────────────────────────────────────────────────────────
def _load_image(path):
    from PIL import Image
    return Image.open(path).convert("RGB")


def _norm_status(s):
    s = str(s).upper()
    return "DANGER" if "DANGER" in s else ("FALSE_ALARM" if "FALSE" in s else "UNKNOWN")


def _extract_object(text, choices):
    """객관식 출력에서 후보 단어 매칭. 출력에 후보가 보이면 그 단어로, 없으면 '기타'.

    공백 무시 부분일치(평서문으로 풀어써도 후보 단어만 들어있으면 추출). 긴 후보 우선(부분 겹침 방지).
    """
    if not text:
        return "기타"
    t = text.replace(" ", "")
    for c in sorted(choices, key=len, reverse=True):
        if c.replace(" ", "") in t:
            return c
    return "기타"


def _ask(processor, model, user_text, images, max_new_tokens):
    """이미지 2장 + 한 가지 질문 -> 답변 텍스트(JSON 강제 없음). 체인 1스텝."""
    import torch

    messages = [
        {"role": "system", "content": config.VERIFY_SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": images[0]},
            {"type": "image", "image": images[1]},
            {"type": "text", "text": user_text},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    vis = images
    try:  # qwen-vl-utils 로 비전 입력 추출, 실패 시 PIL 직접 전달
        from qwen_vl_utils import process_vision_info
        pi, _ = process_vision_info(messages)
        if pi:
            vis = pi
    except Exception:
        pass

    inputs = processor(text=[text], images=vis, padding=True, return_tensors="pt").to(model.device)
    in_len = inputs.input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             repetition_penalty=config.VLM_REPETITION_PENALTY)
    return processor.batch_decode(
        out[:, in_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


# ── 공개 API ─────────────────────────────────────────────────────────────────
def verify_fire_alarm(thermal_image_path, rgb_image_path, temp_summary=None):
    """체인 룰(객체 식별 -> 위험 판정 -> 근거) -> {status, identified_heat_source, reasoning, timing}.

    timing: 체인 3단계별 소요(ms) {object_ms, status_ms, reason_ms}.
    temp_summary 는 받지만 VLM 엔 주지 않는다(온도는 게이트 전담, VLM 은 순수 시각 판단).
    """
    import time

    processor, model = _load_model()
    images = [_load_image(thermal_image_path), _load_image(rgb_image_path)]

    with _GEN_LOCK:  # 체인 3콜을 한 번에 GPU 점유
        # 1) 핫스팟 객체 식별 (후보 목록 객관식 -> 코드가 후보 매칭)
        t0 = time.time()
        obj = _extract_object(_ask(processor, model, config.VERIFY_OBJECT_PROMPT, images, max_new_tokens=24),
                              config.VERIFY_OBJECT_CHOICES)
        t1 = time.time()
        # 2) 위험/정상 판정 (1단계 객체를 맥락으로, 한 단어)
        st_raw = _ask(processor, model, config.VERIFY_STATUS_PROMPT.format(object=obj),
                      images, max_new_tokens=12)
        status = _norm_status(st_raw)
        t2 = time.time()
        # 3) 판단 근거 (객체+판정을 맥락으로, 한 문장)
        reason = _ask(processor, model, config.VERIFY_REASON_PROMPT.format(object=obj, status=status),
                      images, max_new_tokens=80)
        t3 = time.time()

    return {
        "status": status,
        "identified_heat_source": obj,
        "reasoning": reason,
        "timing": {  # 체인 단계별 소요(ms)
            "object_ms": round((t1 - t0) * 1000),
            "status_ms": round((t2 - t1) * 1000),
            "reason_ms": round((t3 - t2) * 1000),
        },
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("usage: python -m src.vlm_analyzer <thermal.jpg> <rgb.jpg>")
        raise SystemExit(1)
    print(json.dumps(verify_fire_alarm(sys.argv[1], sys.argv[2]), ensure_ascii=False, indent=2))
