"""main.py - FastAPI 앱 엔트리.

실행: uvicorn src.main:app --host 0.0.0.0 --port 8000   (프로젝트 루트에서)
"""
from . import config  # noqa: F401  (env-before-torch: 반드시 최상단)

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .api import router

app = FastAPI(title="CCTV 화재 전조 브리핑 에이전트", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_methods=["*"], allow_headers=["*"])
app.include_router(router)


@app.get("/")
def index():
    """프론트엔드(단일 HTML) 서빙. cctv_memory 다크 관제 테마 차용."""
    p = os.path.join(config.FRONTEND_DIR, "index.html")
    if os.path.isfile(p):
        return FileResponse(p)
    return {"service": "cctv_briefing_agent", "docs": "/docs"}


@app.get("/console")
def console():
    """작업자 관제 콘솔(React 단일 HTML): 경보 수신 -> 클릭 -> 이미지+분석 -> 최종 승인."""
    p = os.path.join(config.FRONTEND_DIR, "console.html")
    if os.path.isfile(p):
        return FileResponse(p)
    return {"error": "console.html 없음"}


@app.get("/health")
def health():
    return {"status": "ok", "service": "cctv_briefing_agent"}
