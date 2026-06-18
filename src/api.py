"""api.py - 멀티모달 오경보 필터 API.

POST /api/v1/verify-fire-alarm :
  입력  {camera_id, thermal_image_path, rgb_image_path, [thermal_csv_path]}
  동작  Qwen3-VL 로 열화상 + 실화상 크로스체크 -> 핫스팟 정체 식별 -> 오경보/위험 판정
  출력  {camera_id, status, identified_heat_source, reasoning, temp_summary, timing}

데모 보조: GET /api/v1/image(이미지 서빙), GET /api/v1/samples(열화상/실화상 페어 목록).
"""
import os
import time
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import config

router = APIRouter(prefix="/api/v1", tags=["false-alarm-filter"])


class VerifyRequest(BaseModel):
    camera_id: str
    thermal_image_path: str                  # 열화상(의사색) 이미지 경로
    rgb_image_path: str                      # 정합 실화상(RGB) 경로
    thermal_csv_path: Optional[str] = None   # (선택) 온도행렬 CSV -> 정량 근거 주입


@router.get("/health")
def health():
    return {"status": "ok"}


def _thermal_summary(csv_path):
    """온도행렬 CSV -> 핫스팟 요약 텍스트. 1~5행 측정메타 스킵, 6행부터 512x640 섭씨 행렬."""
    try:
        import numpy as np
        rows = []
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split(";")
                if len(parts) < 50:           # 메타행(키;값)은 컬럼 수 적음 -> 스킵
                    continue
                try:
                    rows.append([float(x) for x in parts if x != ""])
                except ValueError:
                    continue
        if not rows:
            return None
        w = min(len(r) for r in rows)
        arr = np.array([r[:w] for r in rows], dtype=float)
        mx, med = float(arr.max()), float(np.median(arr))
        return f"최고 {mx:.1f}°C, 장면 중앙값 {med:.1f}°C, 핫스팟이 주변보다 +{mx - med:.1f}°C 높음"
    except Exception as e:
        print(f"[thermal] CSV 파싱 실패: {e}", flush=True)
        return None


@router.post("/verify-fire-alarm")
def verify_fire_alarm(req: VerifyRequest):
    """열화상 + 실화상 크로스체크로 오경보/위험 판정."""
    if not os.path.isfile(req.thermal_image_path):
        raise HTTPException(400, f"thermal_image_path 를 찾을 수 없습니다: {req.thermal_image_path}")
    if not os.path.isfile(req.rgb_image_path):
        raise HTTPException(400, f"rgb_image_path 를 찾을 수 없습니다: {req.rgb_image_path}")

    temp_summary = None
    if req.thermal_csv_path and os.path.isfile(req.thermal_csv_path):
        temp_summary = _thermal_summary(req.thermal_csv_path)

    from . import vlm_analyzer
    t0 = time.time()
    verdict = vlm_analyzer.verify_fire_alarm(
        req.thermal_image_path, req.rgb_image_path, temp_summary)
    vlm_ms = round((time.time() - t0) * 1000)

    return {
        "camera_id": req.camera_id,
        "status": verdict.get("status"),
        "identified_heat_source": verdict.get("identified_heat_source"),
        "reasoning": verdict.get("reasoning"),
        "temp_summary": temp_summary,
        "raw": verdict.get("raw"),
        "timing": {"vlm_ms": vlm_ms, "total_ms": vlm_ms},
    }


# ── 프론트 데모 보조 ──────────────────────────────────────────────────────────
@router.get("/image")
def image(path: str):
    """이미지 서빙(열화상/실화상 미리보기용)."""
    if not os.path.isfile(path):
        raise HTTPException(404, f"이미지를 찾을 수 없습니다: {path}")
    return FileResponse(path)


@router.get("/samples")
def samples():
    """SAMPLE_DIR 의 열화상/실화상/CSV 페어 목록(파일명 ID 매칭). 프론트 드롭다운용.

    파일명 규칙: <ID>01xx.jpg(열화상) / <ID>03xx.jpg(정합 RGB) / <ID>01xx.csv(온도).
    끝 4자리 = 모달리티(앞2: 01=열화상,02=광각,03=정합RGB) + 변형(뒤2). ID = 그 앞부분.
    """
    d = config.SAMPLE_DIR
    pairs = {}
    if d and os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            stem, ext = os.path.splitext(f)
            ext = ext.lower()
            if len(stem) < 4:
                continue
            modality, idbase, full = stem[-4:-2], stem[:-4], os.path.join(d, f)
            if ext == ".jpg" and modality == "01":
                pairs.setdefault(idbase, {})["thermal"] = full
            elif ext == ".jpg" and modality == "03":
                pairs.setdefault(idbase, {})["rgb"] = full
            elif ext == ".csv" and modality == "01":
                pairs.setdefault(idbase, {})["csv"] = full
    out = [{"id": k, **v} for k, v in sorted(pairs.items()) if "thermal" in v and "rgb" in v]
    return {"dir": d, "pairs": out}
