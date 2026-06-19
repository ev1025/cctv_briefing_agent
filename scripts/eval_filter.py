"""eval_filter.py - 오경보 필터 정확도 평가 (라벨 GT 대조).

samples/manifest.json(extract_samples 생성)의 GT status 와 verify_fire_alarm 예측을 비교.
GT 매핑: danger -> DANGER(실제 과열=위험), normal -> FALSE_ALARM(정상 발열=오경보).

실행(프로젝트 루트):
  python -m scripts.eval_filter            # 온도 CSV 주입 ON
  python -m scripts.eval_filter --no-csv   # 이미지만(온도 미주입)
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config          # noqa: E402
from src.api import run_verify  # noqa: E402

GT_MAP = {"danger": "DANGER", "normal": "FALSE_ALARM"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-csv", action="store_true", help="온도 CSV 주입 끄고 이미지만")
    ap.add_argument("--limit", type=int, default=0, help="앞 N개만(0=전체)")
    args = ap.parse_args()

    manifest = json.load(open(os.path.join(config.SAMPLE_DIR, "manifest.json"), encoding="utf-8"))
    if args.limit:
        manifest = manifest[:args.limit]

    cm = {"DANGER": {"DANGER": 0, "FALSE_ALARM": 0, "OTHER": 0},
          "FALSE_ALARM": {"DANGER": 0, "FALSE_ALARM": 0, "OTHER": 0}}
    correct, records = 0, []
    t0 = time.time()
    for m in manifest:
        gt = GT_MAP.get(m["status"])
        if gt is None:
            continue
        thermal = os.path.join(config.SAMPLE_DIR, m["thermal"])
        rgb = os.path.join(config.SAMPLE_DIR, m["rgb"])
        csv = None if (args.no_csv or not m.get("csv")) else os.path.join(config.SAMPLE_DIR, m["csv"])
        v = run_verify(thermal, rgb, csv)
        pred = v["status"]
        cm[gt][pred if pred in ("DANGER", "FALSE_ALARM") else "OTHER"] += 1
        ok = (pred == gt)
        correct += ok
        records.append({"id": m["id"], "standard": m["standard"], "gt": gt, "pred": pred,
                        "ok": bool(ok), "identified_heat_source": v.get("identified_heat_source"),
                        "decision_source": v.get("decision_source"), "thermal_dt": v.get("thermal_dt")})

    n = len(records)
    setname = os.path.basename(config.SAMPLE_DIR.rstrip("/\\")) or "samples"
    summary = {
        "set": setname, "n": n, "correct": correct,
        "accuracy": round(correct / max(n, 1), 4),
        "csv": (not args.no_csv),
        "confusion_matrix": cm,
        "errors": [r for r in records if not r["ok"]],
        "results": records,
    }

    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit(f"=== 평가: {n}프레임 · csv={'off' if args.no_csv else 'on'} · {time.time()-t0:.0f}s ===")
    emit(f"정확도: {correct}/{n} = {summary['accuracy']*100:.1f}%")
    emit("혼동행렬 (행=GT, 열=예측):")
    emit(f"  {'GT/Pred':14s} DANGER  FALSE_ALARM  OTHER")
    for gt in ("DANGER", "FALSE_ALARM"):
        emit(f"  {gt:14s} {cm[gt]['DANGER']:6d}  {cm[gt]['FALSE_ALARM']:11d}  {cm[gt]['OTHER']:5d}")
    emit("오답:")
    for r in summary["errors"]:
        emit(f"  {r['gt']:11s} -> {r['pred']:11s} [{r['standard']}] {r['id']} "
             f"(heat={r['identified_heat_source']!r} src={r['decision_source']} dt={r['thermal_dt']})")

    # outputs/ 에 저장(gitignore): 사람용 .txt + 구조화 .json
    odir = os.path.join(config.PROJECT_DIR, "outputs")
    os.makedirs(odir, exist_ok=True)
    with open(os.path.join(odir, f"eval_{setname}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(odir, f"eval_{setname}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[saved] outputs/eval_{setname}.txt, outputs/eval_{setname}.json")


if __name__ == "__main__":
    main()
