"""vlm_analyzer.py - 멀티모달 오경보 필터 (열화상 + 실화상 크로스체크).

Qwen3-VL-2B 로 열화상 1장 + 정합 실화상(RGB) 1장을 동시에 보고,
열화상 핫스팟의 정체를 실화상에서 식별해 오경보(FALSE_ALARM) / 위험(DANGER)을 판정한다.
출력은 JSON {status, identified_heat_source, reasoning}.

  - 모델/프로세서는 모듈 싱글톤(1회 로드), GPU 는 Lock 으로 직렬 사용.
  - 이미지 2장은 qwen-vl-utils(process_vision_info) 로 추출, 실패 시 PIL 직접 전달 폴백.

CLI 스모크: python -m src.vlm_analyzer <thermal.jpg> <rgb.jpg> [thermal.csv]
"""
import json
import re
import threading

from . import config  # noqa: F401  (env-before-torch: 최상단)

_MODEL = None
_PROCESSOR = None
_KO_LP = None                 # 한국어 강제 LogitsProcessor 캐시(옵션)
_JSON_FN = None               # lm-format-enforcer prefix_allowed_tokens_fn 캐시
_JSON_FN_BUILT = False
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

        # AutoModelForImageTextToText 는 체크포인트 config.architectures 를 보고 올바른 클래스를
        #   고른다(Qwen3-VL -> Qwen3VL..., Qwen2.5-VL -> Qwen2_5_VL...). 전용 클래스를 하드코딩하면
        #   Qwen3 체크포인트를 Qwen2.5 클래스로 '오류 없이' 잘못 로드해 generate 시 깨진다.
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(hf_id, **load_kwargs)
        model.eval()

        _PROCESSOR, _MODEL = processor, model
        print(f"[VLM] loaded {hf_id} device={device} dtype={dtype} 4bit={quant is not None}", flush=True)
        return _PROCESSOR, _MODEL


# ── 한국어 강제 (옵션, 기본 off) ──────────────────────────────────────────────
def _get_korean_lp(processor, device):
    """한자 포함 토큰의 logits 를 -inf 로 막는 LogitsProcessorList(1회 계산 후 캐시)."""
    global _KO_LP
    if _KO_LP is not None:
        return _KO_LP
    import torch
    from transformers import LogitsProcessor, LogitsProcessorList

    han = re.compile(r"[㐀-䶿一-鿿豈-﫿\U00020000-\U0002fa1f]")
    tok = getattr(processor, "tokenizer", None) or processor
    ban = [i for i in range(len(tok)) if han.search(tok.decode([i]))]
    if not ban:
        _KO_LP = LogitsProcessorList([])
        return _KO_LP
    ban_t = torch.tensor(sorted(ban), dtype=torch.long, device=device)

    class _BanHan(LogitsProcessor):
        def __call__(self, input_ids, scores):
            scores[:, ban_t[ban_t < scores.shape[-1]]] = float("-inf")
            return scores

    _KO_LP = LogitsProcessorList([_BanHan()])
    return _KO_LP


# ── JSON 출력 강제 (lm-format-enforcer, 1회 빌드 후 캐시) ──────────────────────
def _json_prefix_fn(processor):
    """디코딩을 VERIFY_JSON_SCHEMA 에 묶는 prefix_allowed_tokens_fn. 미설치/비활성 시 None."""
    global _JSON_FN, _JSON_FN_BUILT
    if _JSON_FN_BUILT:
        return _JSON_FN
    _JSON_FN_BUILT = True
    if not config.VLM_ENFORCE_JSON:
        _JSON_FN = None
        return None
    try:
        # transformers 5.x 에서 PreTrainedTokenizerBase 위치가 바뀌어 lm-format-enforcer
        #   integration import 가 깨진다(타입힌트용일 뿐) -> tokenization_utils 에 심볼 주입(shim).
        import transformers
        import transformers.tokenization_utils as _tu
        if not hasattr(_tu, "PreTrainedTokenizerBase"):
            _tu.PreTrainedTokenizerBase = transformers.PreTrainedTokenizerBase
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import (
            build_transformers_prefix_allowed_tokens_fn)
        tok = getattr(processor, "tokenizer", None) or processor
        _JSON_FN = build_transformers_prefix_allowed_tokens_fn(
            tok, JsonSchemaParser(config.VERIFY_JSON_SCHEMA))
        print("[VLM] JSON 출력 강제 ON (lm-format-enforcer)", flush=True)
    except Exception as e:
        print(f"[VLM] JSON 강제 비활성(lm-format-enforcer 없음/실패) -> 정규식 폴백: {e}", flush=True)
        _JSON_FN = None
    return _JSON_FN


# ── 입력/출력 유틸 ────────────────────────────────────────────────────────────
def _load_image(path):
    from PIL import Image
    return Image.open(path).convert("RGB")


def _parse_verdict(text):
    """모델 출력에서 {status, identified_heat_source, reasoning} JSON 을 견고하게 추출."""
    m = re.search(r"\{.*\}", text.strip(), re.DOTALL)
    obj = None
    if m:
        try:
            obj = json.loads(m.group(0))
        except Exception:
            obj = None
    if not isinstance(obj, dict):
        return {"status": "UNKNOWN", "identified_heat_source": "",
                "reasoning": text.strip(), "parse_error": True}
    status = str(obj.get("status", "")).upper()
    status = "DANGER" if "DANGER" in status else ("FALSE_ALARM" if "FALSE" in status else (status or "UNKNOWN"))
    return {
        "status": status,
        "identified_heat_source": str(obj.get("identified_heat_source", "")),
        "reasoning": str(obj.get("reasoning", "")),
    }


# ── 공개 API ─────────────────────────────────────────────────────────────────
def verify_fire_alarm(thermal_image_path, rgb_image_path, temp_summary=None):
    """열화상 + 실화상 크로스체크 -> {status, identified_heat_source, reasoning, raw}.

    Args:
        thermal_image_path: 열화상(의사색) 이미지 경로.
        rgb_image_path: 같은 시각·위치의 정합 실화상(RGB) 경로.
        temp_summary: (선택) 온도 CSV 요약 텍스트(핫스팟 °C·ΔT). 정량 근거로 주입.
    """
    import torch

    processor, model = _load_model()
    thermal = _load_image(thermal_image_path)
    rgb = _load_image(rgb_image_path)

    user_text = config.VERIFY_USER_PROMPT
    if temp_summary:
        user_text += f"\n\n[열화상 측정 온도] {temp_summary}"

    messages = [
        {"role": "system", "content": config.VERIFY_SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": thermal},
            {"type": "image", "image": rgb},
            {"type": "text", "text": user_text},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    images = [thermal, rgb]
    try:  # qwen-vl-utils 로 비전 입력 추출(스펙 권장), 실패 시 PIL 직접 전달
        from qwen_vl_utils import process_vision_info
        imgs, _ = process_vision_info(messages)
        if imgs:
            images = imgs
    except Exception:
        pass

    with _GEN_LOCK:  # GPU 직렬 사용
        inputs = processor(text=[text], images=images,
                           padding=True, return_tensors="pt").to(model.device)
        in_len = inputs.input_ids.shape[1]
        gen_kwargs = dict(max_new_tokens=config.VLM_MAX_NEW_TOKENS,
                          repetition_penalty=config.VLM_REPETITION_PENALTY, do_sample=False)
        if config.VLM_FORCE_KOREAN:
            lp = _get_korean_lp(processor, model.device)
            if len(lp) > 0:
                gen_kwargs["logits_processor"] = lp
        prefix_fn = _json_prefix_fn(processor)   # lm-format-enforcer 로 JSON 스키마 강제
        if prefix_fn is not None:
            gen_kwargs["prefix_allowed_tokens_fn"] = prefix_fn
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        raw = processor.batch_decode(
            out[:, in_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

    verdict = _parse_verdict(raw)
    verdict["raw"] = raw
    return verdict


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("usage: python -m src.vlm_analyzer <thermal.jpg> <rgb.jpg> [thermal.csv]")
        raise SystemExit(1)
    ts = None
    if len(sys.argv) > 3:
        from .api import _thermal_summary
        ts = _thermal_summary(sys.argv[3])
    print(json.dumps(verify_fire_alarm(sys.argv[1], sys.argv[2], ts), ensure_ascii=False, indent=2))
