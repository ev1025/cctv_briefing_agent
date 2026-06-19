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


def _thermal_features(csv_path):
    """온도행렬 CSV -> {hot, med, dt, summary} 또는 None.

    1~5행 측정메타 스킵, 6행부터 512x640 섭씨 행렬.
    핫스팟은 단일 픽셀 max(반사 글린트 노이즈) 대신 상위1%(p99) 로 robust 하게 잡는다.
    핫스팟 ΔT(p99 - 장면 중앙값)가 정상/위험을 가르는 핵심 신호(이 슬라이스 p99-med AUC≈1.0).
    """
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
        med = float(np.median(arr))
        hot = float(np.percentile(arr, 99))    # robust 핫스팟(단일 픽셀 max 노이즈 회피)
        dt, thr = hot - med, config.THERMAL_DANGER_DELTA_C
        summary = (f"핫스팟(상위1%) {hot:.1f}°C, 장면 중앙값 {med:.1f}°C, 핫스팟이 주변보다 +{dt:.1f}°C 높음. "
                   f"[판정 기준: 이 핫스팟 ΔT 가 약 +{thr:.0f}°C 이상이면 과열 위험, 그 미만이면 정상 작동 발열]")
        return {"hot": hot, "med": med, "dt": dt, "summary": summary}
    except Exception as e:
        print(f"[thermal] CSV 파싱 실패: {e}", flush=True)
        return None


def _thermal_summary(csv_path):
    """온도 요약 텍스트만 반환(하위호환). 상세는 _thermal_features."""
    tf = _thermal_features(csv_path)
    return tf["summary"] if tf else None


# 온도가 명백히 낮은데도 알람이 뜬 경우/양성 열원 키워드(실화상 식별로 오경보 강등)
_BENIGN_KEYWORDS = ("사람", "행인", "작업자", "조명", "전등", "햇빛", "햇볕", "햇살", "빛 반사", "반사광", "태양광 반사")


def run_verify(thermal_path, rgb_path, csv_path=None):
    """열화상+실화상+온도 융합 판정. eval/endpoint 공용.

    결정 규칙(온도 CSV 있을 때):
      - 핫스팟 ΔT < 임계  -> FALSE_ALARM (정량적으로 과열 아님; 열화상 알람의 1차 게이트)
      - ΔT >= 임계        -> DANGER, 단 VLM 이 명백한 양성 열원(사람·조명·햇빛 반사)으로 식별하면 FALSE_ALARM 으로 강등
    CSV 없으면 VLM 판정을 그대로 사용.
    """
    import time
    from . import vlm_analyzer

    tf = _thermal_features(csv_path) if (csv_path and os.path.isfile(csv_path)) else None
    temp_summary = tf["summary"] if tf else None

    t0 = time.time()
    v = vlm_analyzer.verify_fire_alarm(thermal_path, rgb_path, temp_summary)
    vlm_ms = round((time.time() - t0) * 1000)

    status, source = v.get("status"), "vlm"
    if tf is not None:
        text = f"{v.get('identified_heat_source', '')} {v.get('reasoning', '')}"
        benign = any(k in text for k in _BENIGN_KEYWORDS)
        if tf["dt"] < config.THERMAL_DANGER_DELTA_C:
            status, source = "FALSE_ALARM", "thermal_gate(저ΔT)"
        elif benign:
            status, source = "FALSE_ALARM", "vlm_override(양성열원)"
        else:
            status, source = "DANGER", "thermal+vlm"

    return {
        "status": status,
        "vlm_status": v.get("status"),
        "identified_heat_source": v.get("identified_heat_source"),
        "reasoning": v.get("reasoning"),
        "temp_summary": temp_summary,
        "thermal_dt": round(tf["dt"], 1) if tf else None,
        "decision_source": source,
        "raw": v.get("raw"),
        "timing": {"vlm_ms": vlm_ms, "total_ms": vlm_ms},
    }


@router.post("/verify-fire-alarm")
def verify_fire_alarm(req: VerifyRequest):
    """열화상 + 실화상 + 온도 융합으로 오경보/위험 판정."""
    if not os.path.isfile(req.thermal_image_path):
        raise HTTPException(400, f"thermal_image_path 를 찾을 수 없습니다: {req.thermal_image_path}")
    if not os.path.isfile(req.rgb_image_path):
        raise HTTPException(400, f"rgb_image_path 를 찾을 수 없습니다: {req.rgb_image_path}")

    res = run_verify(req.thermal_image_path, req.rgb_image_path, req.thermal_csv_path)
    return {"camera_id": req.camera_id, **res}


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
