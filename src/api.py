"""api.py - 멀티모달 오경보 필터 API.

POST /api/v1/verify-fire-alarm :
  입력  {camera_id, thermal_image_path, rgb_image_path, [thermal_csv_path]}
  동작  Qwen3-VL 로 열화상 + 실화상 크로스체크 -> 핫스팟 정체 식별 -> 오경보/위험 판정
  출력  {camera_id, status, identified_heat_source, reasoning, temp_summary, timing}

데모 보조: GET /api/v1/image(이미지 서빙), GET /api/v1/samples(열화상/실화상 페어 목록).
"""
import os
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from . import config

router = APIRouter(prefix="/api/v1", tags=["false-alarm-filter"])


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
        dt = hot - med
        # VLM 에 줄 '측정값'만(판정 문구 없음). 소형 모델 위해 특수기호 없이 평문(℃->도).
        summary = f"핫스팟(상위1%) {hot:.1f}도, 장면 중앙값 {med:.1f}도, 핫스팟이 주변보다 {dt:.1f}도 높음"
        return {"hot": hot, "med": med, "dt": dt, "summary": summary}
    except Exception as e:
        print(f"[thermal] CSV 파싱 실패: {e}", flush=True)
        return None


def _thermal_summary(csv_path):
    """온도 요약 텍스트만 반환(하위호환). 상세는 _thermal_features."""
    tf = _thermal_features(csv_path)
    return tf["summary"] if tf else None


# 온도가 명백히 낮은데도 알람이 뜬 경우/양성 열원 키워드(실화상 식별로 오경보 강등)
#   양성 열원 '객체' 키워드. '반사' 계열은 설비 표면 묘사에도 흔히 섞여 오강등을 유발하므로 제외.
_BENIGN_KEYWORDS = ("사람", "행인", "작업자", "조명", "전등", "햇빛", "햇볕", "햇살")


def run_verify(thermal_path, rgb_path, csv_path=None):
    """열화상 알람 검증. 온도 1차 게이트 -> 위험 후보만 VLM 2차 크로스체크.

    순서(온도 CSV 있을 때):
      1) 핫스팟 ΔT < 임계  -> 즉시 FALSE_ALARM (VLM 생략, ~0초; 과열 아님)
      2) ΔT >= 임계        -> VLM 실행 -> 양성 열원(사람·조명·햇빛 반사)이면 FALSE_ALARM, 아니면 DANGER
    CSV 없으면 게이트 없이 VLM 판정을 그대로 사용.
    """
    import time
    from . import vlm_analyzer

    tf = _thermal_features(csv_path) if (csv_path and os.path.isfile(csv_path)) else None

    # 1차 게이트: 온도가 임계 미만이면 VLM 안 돌리고 즉시 오경보(비용 절감)
    if tf is not None and tf["dt"] < config.THERMAL_DANGER_DELTA_C:
        thr = config.THERMAL_DANGER_DELTA_C
        return {
            "status": "FALSE_ALARM",
            "vlm_status": None,
            "identified_heat_source": None,
            "reasoning": f"핫스팟 ΔT +{tf['dt']:.1f}°C 로 임계(+{thr:.0f}°C) 미만 → 정상 작동 발열(VLM 생략).",
            "temp_summary": tf["summary"],
            "thermal_dt": round(tf["dt"], 1),
            "decision_source": "thermal_gate(낮은 온도차이·VLM생략)",
            "raw": None,
            "timing": {"vlm_ms": 0, "total_ms": 0},
        }

    # 2차: 게이트 통과(또는 CSV 없음) -> VLM 크로스체크.
    #   VLM 에는 온도 '측정값'만 주고 '위험 기준 초과' 같은 판정 문구는 주지 않는다(_thermal_features).
    #   판정은 게이트가 전담; VLM 이 온도 verdict 에 휩쓸려 DANGER 쏠림/양성강등 약화되는 것 방지.
    temp_summary = tf["summary"] if tf else None
    t0 = time.time()
    v = vlm_analyzer.verify_fire_alarm(thermal_path, rgb_path, temp_summary)
    vlm_ms = round((time.time() - t0) * 1000)

    status, source = v.get("status"), "vlm"
    if tf is not None:
        # 양성 강등은 '식별된 객체' 가 양성 열원일 때만(추론 본문의 stray '햇빛' 언급 등은 무시).
        heat = v.get("identified_heat_source") or ""
        benign = any(k in heat for k in _BENIGN_KEYWORDS)
        status, source = ("FALSE_ALARM", "vlm_override(양성열원)") if benign else ("DANGER", "thermal+vlm")

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
def verify_fire_alarm(
    camera_id: str = Form(...),
    thermal: UploadFile = File(...),                       # 열화상 이미지 파일
    rgb: UploadFile = File(...),                           # 정합 실화상(RGB) 파일
    thermal_csv: Optional[UploadFile] = File(None),        # (선택) 온도 CSV 파일
):
    """열화상·실화상(+온도 CSV) 이미지를 업로드(multipart)로 받아 오경보/위험 판정. 임시 저장 후 처리·삭제."""
    import shutil
    import tempfile

    tmp = tempfile.mkdtemp(prefix="verify_")

    def _save(up, name):
        p = os.path.join(tmp, name)
        with open(p, "wb") as f:
            shutil.copyfileobj(up.file, f)
        return p

    try:
        tp = _save(thermal, "thermal.jpg")
        rp = _save(rgb, "rgb.jpg")
        cp = _save(thermal_csv, "thermal.csv") if thermal_csv is not None else None
        res = run_verify(tp, rp, cp)
        return {"camera_id": camera_id, **res}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


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
