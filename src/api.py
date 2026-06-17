"""api.py - 브리핑 API 라우터 (단계 4).

POST /api/v1/generate-briefing :
  입력  {camera_id, event_time, video_path}
  동작  VLM(현재 상황) + RAG(과거 전조) -> 종합 브리핑 합성
  출력  {camera_id, event_time, current_status, precursor_history[], summary}
"""
import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from . import config
from . import rag_retriever

router = APIRouter(prefix="/api/v1", tags=["briefing"])


class BriefingRequest(BaseModel):
    camera_id: str
    event_time: str          # ISO 문자열 (예: 2026-06-17T14:32:00) 또는 epoch 문자열
    video_path: str          # 서버에서 접근 가능한 알람 영상 경로


@router.get("/health")
def health():
    return {"status": "ok"}


def _format_precursors(precursors):
    if not precursors:
        return "과거 시간창 내 특이 전조 이력이 검색되지 않음."
    return "\n".join(
        f"- [{p['time']}] {p['caption']} (유사도 {p['score']})" for p in precursors)


def synthesize(current_status, precursors):
    """현재 + 전조 -> 종합 브리핑. 1차 VLM 텍스트 생성, 실패/비활성 시 템플릿 폴백."""
    hist = _format_precursors(precursors)
    if config.BRIEFING_USE_LLM:
        try:
            from . import vlm_analyzer
            prompt = config.BRIEFING_PROMPT.format(
                current_status=current_status, precursor_history=hist)
            text = vlm_analyzer.generate_text(prompt)
            if text:
                return text
        except Exception as e:  # 모델 미설치/생성 실패 시 결정적 폴백
            print(f"[briefing] LLM 합성 실패, 템플릿 폴백: {e}", flush=True)

    n = len(precursors)
    lead = "과거 전조 이력 없이" if n == 0 else f"과거 {n}건의 전조 이력과 함께"
    return (f"[현재] {current_status}\n"
            f"[전조]\n{hist}\n"
            f"[종합] {lead} 현재 위험이 감지되었습니다. 해당 구역 즉시 확인 및 초동 대응을 권고합니다.")


@router.post("/generate-briefing")
def generate_briefing(req: BriefingRequest):
    """알람 -> 현재 상황(VLM) + 과거 전조(RAG) 융합 브리핑 리포트."""
    if not os.path.isfile(req.video_path):
        raise HTTPException(status_code=400, detail=f"video_path 를 찾을 수 없습니다: {req.video_path}")

    from . import vlm_analyzer
    current_status = vlm_analyzer.analyze_current_status(req.video_path)
    precursors = rag_retriever.retrieve_precursors(req.camera_id, req.event_time)
    summary = synthesize(current_status, precursors)

    return {
        "camera_id": req.camera_id,
        "event_time": req.event_time,
        "current_status": current_status,
        "precursor_history": precursors,
        "summary": summary,
    }
