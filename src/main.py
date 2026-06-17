"""main.py - FastAPI 앱 엔트리.

실행: uvicorn src.main:app --host 0.0.0.0 --port 8000   (프로젝트 루트에서)
"""
from . import config  # noqa: F401  (env-before-torch: 반드시 최상단)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import router

app = FastAPI(title="CCTV 화재 전조 브리핑 에이전트", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_methods=["*"], allow_headers=["*"])
app.include_router(router)


@app.get("/health")
def health():
    return {"status": "ok", "service": "cctv_briefing_agent"}
