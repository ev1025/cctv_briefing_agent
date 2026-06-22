"""eval_filter.py - 오경보 필터 정확도 평가

samples/manifest.json(라벨 데이터 샘플)과 verify_fire_alarm 예측을 비교
GT 매핑: danger -> DANGER(위험, 실제 과열), normal -> FALSE_ALARM(오경보, 정상 발열).

실행(프로젝트 루트):
  python -m scripts.eval_filter            # 온도 CSV 주입 ON
  python -m scripts.eval_filter --no-csv   # 이미지만(온도 미주입)
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config          # noqa: E402
from src.api import run_verify  # noqa: E402

# 라벨데이터 - 예상값 매핑
GT_MAP = {"danger": "DANGER", "normal": "FALSE_ALARM"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-csv", action="store_true", help="온도 CSV 주입 끄고 이미지만")
    ap.add_argument("--limit", type=int, default=0, help="앞 N개만(0=전체)")
    args = ap.parse_args()

    manifest = json.load(open(os.path.join(config.SAMPLE_DIR, "manifest.json"), encoding="utf-8"))
    if args.limit:
        manifest = manifest[:args.limit]

    # 혼동행렬
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
        vstatus = v.get("vlm_status")
        records.append({
            # --- 1. 메타 데이터 (기본 정보) ---
            "id": m["id"],                  # 이미지 파일 이름
            "standard": m["standard"],      # 설비 종류

            # --- 2. 최종 결과 요약 (분석할 때 가장 먼저 보는 값) ---
            "gt": gt,                                         # 실제 정답 (위험 여부)
            "final_pred": pred,                               # 최종 모델 파이프라인 예측값
            "final_ok": bool(ok),                             # 최종 정오 (gt == final_pred)
            "decision_source": v.get("decision_source"),      # 최종 판정 주체 (Gate / VLM)

            # --- 3. 온도 게이트 상세 (1차 필터) ---
            "gate_thermal_dt": v.get("thermal_dt"),           # 온도 차이

            # --- 4. VLM 상세 (2차 필터) ---
            "vlm_ok": (None if vstatus is None else bool(vstatus == gt)), # VLM 단독 정오
            "vlm_status": vstatus,                            # VLM 판정 값
            "vlm_identified_heat_source": v.get("identified_heat_source"), # 설비 종류 판정
            "vlm_reasoning": v.get("reasoning"),              # 판단 근거
            "vlm_timing_ms": v.get("timing"),                 # 체인 단계별 소요(ms): object/status/reason
        })

    # 테스트결과 json 생성
    n = len(records)
    setname = os.path.basename(config.SAMPLE_DIR.rstrip("/\\")) or "samples"
    # "누가 결정했나"로 분리: 게이트 단독 vs VLM 호출. VLM 단독 실력은 vlm_status vs gt.
    gate_recs = [r for r in records if r["vlm_status"] is None]   # 온도 게이트가 결정(VLM 미호출)
    vlm_recs = [r for r in records if r["vlm_status"] is not None]  # VLM 호출됨
    gate_correct = sum(r["final_pred"] == r["gt"] for r in gate_recs)
    vlm_correct = sum(r["vlm_status"] == r["gt"] for r in vlm_recs)

    # VLM 체인 단계별 평균 소요(ms): VLM 호출 프레임만 대상
    def _avg_ms(key):
        vals = [r["vlm_timing_ms"].get(key) for r in vlm_recs
                if isinstance(r.get("vlm_timing_ms"), dict) and isinstance(r["vlm_timing_ms"].get(key), (int, float))]
        return round(sum(vals) / len(vals)) if vals else None
    timing_avg = ({k: _avg_ms(k) for k in ("object_ms", "status_ms", "reason_ms", "vlm_total_ms")}
                  if vlm_recs else None)

    summary = {
        "set": setname,
        "n": n,                                            # 총 샘플 수
        "correct": correct,                                # 시스템 정답 수
        "accuracy": round(correct / max(n, 1), 4),         # 시스템(융합) 정확도 = pred vs gt
        "gate_n": len(gate_recs),                          # 게이트 단독 결정 프레임 수
        "gate_accuracy": round(gate_correct / len(gate_recs), 4) if gate_recs else None,
        "vlm_n": len(vlm_recs),                            # VLM 호출 프레임 수
        "vlm_accuracy": round(vlm_correct / len(vlm_recs), 4) if vlm_recs else None,  # VLM 단독(vlm_status vs gt)
        "vlm_timing_avg_ms": timing_avg,                   # VLM 체인 단계별 평균 소요(ms)
        "csv": (not args.no_csv),                          # 온도 csv 파일 사용여부
        "confusion_matrix": cm,                            # confusion matrix
        "errors": [r for r in records if not r["final_ok"]],
        "results": records,
    }

    def emit(s=""):
        print(s)

    emit(f"=== 평가: {n}프레임 · csv={'off' if args.no_csv else 'on'} · {time.time()-t0:.0f}s ===")
    emit(f"정확도(시스템 융합): {correct}/{n} = {summary['accuracy']*100:.1f}%")
    if gate_recs:
        emit(f"  게이트 단독(VLM 미호출): {gate_correct}/{len(gate_recs)} = {summary['gate_accuracy']*100:.0f}%")
    if vlm_recs:
        emit(f"  VLM 단독(vlm_status vs gt): {vlm_correct}/{len(vlm_recs)} = {summary['vlm_accuracy']*100:.0f}%  <- VLM 진짜 실력")
        if timing_avg and timing_avg.get("vlm_total_ms") is not None:
            emit(f"  VLM 체인 평균(ms): 객체 {timing_avg['object_ms']} · 상태 {timing_avg['status_ms']} · "
                 f"근거 {timing_avg['reason_ms']} (합 {timing_avg['vlm_total_ms']})")
    emit("혼동행렬 (행=GT, 열=예측):")
    emit(f"  {'GT/Pred':14s} DANGER  FALSE_ALARM  OTHER")
    for gt in ("DANGER", "FALSE_ALARM"):
        emit(f"  {gt:14s} {cm[gt]['DANGER']:6d}  {cm[gt]['FALSE_ALARM']:11d}  {cm[gt]['OTHER']:5d}")
    emit("오답:")
    for r in summary["errors"]:
        emit(f"  {r['gt']:11s} -> {r['final_pred']:11s} [{r['standard']}] {r['id']} "
             f"(heat={r['vlm_identified_heat_source']!r} src={r['decision_source']} dt={r['gate_thermal_dt']})")

    # outputs/eval/ 에 타임스탬프(년월일시분) JSON 으로만 저장 -> 실행마다 따로 남음(덮어쓰기 방지). gitignore.
    #   요약은 콘솔(emit)에 그대로 출력되고, summary 필드에 다 들어가므로 .txt 별도 저장은 생략.
    ts = datetime.now().strftime("%Y%m%d%H%M")
    odir = os.path.join(config.PROJECT_DIR, "outputs", "eval")
    os.makedirs(odir, exist_ok=True)
    base = f"eval_{setname}_{ts}"
    with open(os.path.join(odir, base + ".json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[saved] outputs/eval/{base}.json")


if __name__ == "__main__":
    main()
