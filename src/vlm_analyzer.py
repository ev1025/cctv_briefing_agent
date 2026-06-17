"""vlm_analyzer.py - 현재 상황 분석 모듈 (단계 2).

Qwen/Qwen2.5-VL-2B-Instruct 로 알람 시점 mp4 를 분석해 즉각 위험 요소를 묘사한다.
hailo_vlm 의 검증된 경로를 이식: OpenCV 균일 프레임 샘플 -> processor(videos=[frames]).
  - decord/av/qwen-vl-utils 없이 OpenCV 로만 프레임을 뽑아 Windows 친화·의존성 최소.
  - 모델/프로세서는 모듈 싱글톤으로 1회 로드(요청마다 재로드 방지), GPU 는 Lock 으로 직렬 사용.

CLI 스모크: python -m src.vlm_analyzer <video.mp4>
"""
import threading

from . import config  # noqa: F401  (env-before-torch: 최상단)

_MODEL = None
_PROCESSOR = None
_KO_LP = None                 # 한국어 강제 LogitsProcessor 캐시(옵션)
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
        #   Qwen3 체크포인트를 Qwen2.5 클래스로 '오류 없이' 잘못 로드해 generate 시 깨진다(get_rope_index).
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(hf_id, **load_kwargs)
        model.eval()

        _PROCESSOR, _MODEL = processor, model
        print(f"[VLM] loaded {hf_id} device={device} dtype={dtype} 4bit={quant is not None}", flush=True)
        return _PROCESSOR, _MODEL


# ── 프레임 샘플링 (OpenCV) ────────────────────────────────────────────────────
def _to_pil(bgr, max_side):
    import cv2
    from PIL import Image
    h, w = bgr.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _sample_frames(path, num_frames=None, max_side=None):
    """영상 -> 균일 간격 num_frames 장(PIL). seek 실패 시 순차 디코드 폴백. 짝수 보정."""
    import cv2
    import numpy as np

    num_frames = num_frames or config.VIDEO_NUM_FRAMES
    max_side = max_side or config.VIDEO_FRAME_MAX_SIDE

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"영상을 열 수 없습니다: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    frames, idx_used = [], []
    if total > 0:
        for idx in np.linspace(0, total - 1, num=num_frames).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, fr = cap.read()
            if ok:
                frames.append(_to_pil(fr, max_side))
                idx_used.append(int(idx))

    if len(frames) < 2:  # 폴백: 전체 순차 디코드 후 균일 선택
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        allf = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            allf.append(fr)
        frames, idx_used = [], []
        if allf:
            for idx in np.linspace(0, len(allf) - 1, num=num_frames).astype(int):
                frames.append(_to_pil(allf[int(idx)], max_side))
                idx_used.append(int(idx))
    cap.release()

    if not frames:
        raise RuntimeError(f"프레임 추출 실패: {path}")
    if len(frames) % 2 == 1:  # temporal_patch_size=2 정렬: 마지막 프레임 복제
        frames.append(frames[-1])
        idx_used.append(idx_used[-1])

    meta = {"total_num_frames": total or len(frames),
            "fps": fps if fps > 0 else None,
            "frames_indices": idx_used}
    return frames, meta


# ── 한국어 강제 (옵션, 기본 off) ──────────────────────────────────────────────
def _get_korean_lp(processor, device):
    """한자 포함 토큰의 logits 를 -inf 로 막는 LogitsProcessorList(1회 계산 후 캐시)."""
    global _KO_LP
    if _KO_LP is not None:
        return _KO_LP
    import re
    import torch
    from transformers import LogitsProcessor, LogitsProcessorList

    han = re.compile(r"[㐀-䶿一-鿿豈-﫿\U00020000-\U0002fa1f]")
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


# ── 공개 API ─────────────────────────────────────────────────────────────────
def analyze_current_status(video_path, prompt=None):
    """알람 영상(mp4) -> 현재 위험 상황 텍스트(한국어).

    Args:
        video_path: 분석할 mp4 경로.
        prompt: 분석 프롬프트(기본 config.VLM_PROMPT).
    Returns:
        VLM 이 묘사한 즉각 위험 상황 텍스트.
    """
    import torch

    prompt = prompt or config.VLM_PROMPT
    processor, model = _load_model()
    frames, meta = _sample_frames(video_path)

    messages = [{"role": "user", "content": [
        {"type": "video"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    with _GEN_LOCK:  # GPU 직렬 사용
        proc_kwargs = dict(text=[text], videos=[frames], padding=True, return_tensors="pt")
        try:  # Qwen3-VL 계열: 사전 샘플 프레임을 그대로 사용(내부 재샘플 방지) + 타임스탬프 메타
            inputs = processor(**proc_kwargs, do_sample_frames=False,
                               video_metadata=[meta]).to(model.device)
        except TypeError:  # 해당 kwargs 미지원 프로세서(예: Qwen2.5-VL) 폴백
            inputs = processor(**proc_kwargs).to(model.device)
        in_len = inputs.input_ids.shape[1]

        gen_kwargs = dict(max_new_tokens=config.VLM_MAX_NEW_TOKENS,
                          repetition_penalty=config.VLM_REPETITION_PENALTY)
        if config.VLM_TEMPERATURE and config.VLM_TEMPERATURE > 0:
            gen_kwargs.update(do_sample=True, temperature=config.VLM_TEMPERATURE)
        else:
            gen_kwargs.update(do_sample=False)
        if config.VLM_FORCE_KOREAN:
            lp = _get_korean_lp(processor, model.device)
            if len(lp) > 0:
                gen_kwargs["logits_processor"] = lp

        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        gen = out[:, in_len:]
        result = processor.batch_decode(
            gen, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    return result


def generate_text(prompt, max_new_tokens=None):
    """텍스트 전용 생성. 이미 로드된 VLM 을 LLM 으로 재사용(브리핑 합성용)."""
    import torch

    processor, model = _load_model()
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    with _GEN_LOCK:
        inputs = processor(text=[text], padding=True, return_tensors="pt").to(model.device)
        in_len = inputs.input_ids.shape[1]
        gen_kwargs = dict(max_new_tokens=max_new_tokens or config.VLM_MAX_NEW_TOKENS,
                          do_sample=False, repetition_penalty=config.VLM_REPETITION_PENALTY)
        if config.VLM_FORCE_KOREAN:
            lp = _get_korean_lp(processor, model.device)
            if len(lp) > 0:
                gen_kwargs["logits_processor"] = lp
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        gen = out[:, in_len:]
        return processor.batch_decode(
            gen, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python -m src.vlm_analyzer <video.mp4>")
        raise SystemExit(1)
    print("=" * 60)
    print(analyze_current_status(sys.argv[1]))
    print("=" * 60)
