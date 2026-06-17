"""seed_history.py - 전조 이력 합성 시드 (단계 4 E2E 검증용).

특정 camera_id 에 대해 '기준 시각(event_time)' 으로부터 과거 1~23시간대에
화재 전조성 묘사(연기·스파크·과열·배회)와 정상 묘사를 섞어 ChromaDB 에 주입한다.
다른 카메라 로그도 1건 넣어 camera_id 필터가 동작하는지 확인한다.

실행(프로젝트 루트):
  python -m scripts.seed_history --camera CAM_03 --reset
  python -m scripts.seed_history --camera CAM_03 --event 2026-06-17T14:30:00
"""
import argparse
import os
import sys
from datetime import datetime, timedelta

# 프로젝트 루트를 sys.path 에 추가(`python scripts/seed_history.py` 직접 실행도 지원)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import rag_retriever  # noqa: E402

# (과거 hours, 캡션) - 전조성/정상 혼합
_LOGS = [
    (23, "배전반 주변에서 옅은 연기가 잠깐 피어올랐다가 사라짐"),
    (20, "기계 과열로 추정되는 김이 설비 상단에서 올라옴"),
    (16, "정상 작업. 지게차가 통로에서 화물을 운반함"),
    (12, "전기 패널에서 스파크가 튀는 장면이 짧게 포착됨"),
    (8, "야간에 사람이 설비 주변을 반복적으로 배회함"),
    (5, "정상. 순찰자가 통로를 지나가며 점검함"),
    (2, "작업자가 타는 냄새를 맡은 듯 코를 막고 환기팬을 가동함"),
    (1, "배전반 온도 경고등이 점멸하고 옅은 연기가 재발함"),
]


def seed(camera_id, event_time, reset=False):
    store = rag_retriever.get_store()
    if reset:
        store.reset()
        print("[seed] 컬렉션 초기화")

    ev = rag_retriever._to_epoch(event_time)
    records = []
    for h, cap in _LOGS:
        records.append({
            "camera_id": camera_id,
            "event_time": ev - h * 3600.0,
            "caption": cap,
            "source": "seed",
        })
    # 다른 카메라(필터 확인용) - 전조성이지만 camera_id 가 다름
    records.append({
        "camera_id": "CAM_99",
        "event_time": ev - 3 * 3600.0,
        "caption": "다른 구역에서 큰 불꽃과 짙은 연기가 관측됨",
        "source": "seed",
    })

    n = store.add_logs(records)
    print(f"[seed] camera={camera_id} 기준시각={rag_retriever._fmt_time(ev)} 주입={n}건 "
          f"(현재 컬렉션 총 {store.count()}건)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="CAM_03")
    ap.add_argument("--event", default=None, help="기준 시각 ISO (기본: 현재)")
    ap.add_argument("--reset", action="store_true", help="컬렉션 초기화 후 주입")
    args = ap.parse_args()
    event_time = args.event or datetime.now().isoformat(timespec="seconds")
    seed(args.camera, event_time, reset=args.reset)
