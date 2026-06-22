"""extract_samples.py - AIHub 514 VS3/VL3 zip 에서 열화상+실화상+CSV 페어 추출.

AIHub 514 산업시설 열화상 CCTV 데이터(https://aihub.or.kr/aidata/105)에서
용량이 가장 적은 Validation를 샘플로 라벨링데이터(VL3.zip)와 원천데이터(VS3.zip)
열화상 이미지, 실화상 이미지, CSV 정보를 하나의 세트(페어)로 묶어서 추출

압축파일이 CP949인코딩 되어있어 폴더명 글자가 깨져보이는 현상(Mojibake) 발생
-> 실제 파일 이름(basename)은 깨지지 않는 영문과 숫자(ASCII ID)로 되어 있어, 파일 이름으로 매핑


실행(프로젝트 루트):
  THERMAL_DATA_DIR="...\\117.산업시설 열화상 CCTV 데이터\\01.데이터\\2.Validation" \
  python -m scripts.extract_samples --normal 3 --danger 3
"""

import argparse
import json
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("THERMAL_DATA_DIR", ""),
                    help="2.Validation 폴더 경로(라벨링데이터/원천데이터 포함)")
    ap.add_argument("--out", default=config.SAMPLE_DIR)
    ap.add_argument("--normal", type=int, default=3)
    ap.add_argument("--danger", type=int, default=3)
    ap.add_argument("--skip", type=int, default=0, help="각 status 앞 K개 건너뛰고 추출(held-out 분리용)")
    args = ap.parse_args()
    if not args.data or not os.path.isdir(args.data):
        print("THERMAL_DATA_DIR(또는 --data)로 2.Validation 경로를 지정하세요.")
        raise SystemExit(1)

    vl3 = os.path.join(args.data, "라벨링데이터", "VL3.zip")
    vs3 = os.path.join(args.data, "원천데이터", "VS3.zip")
    os.makedirs(args.out, exist_ok=True)

    # 1) 라벨 -> frame(status, standard, thermal/rgb/csv 파일명)
    frames = []
    with zipfile.ZipFile(vl3) as z:
        for n in z.namelist():
            if not n.lower().endswith(".json"):
                continue
            try:
                d = json.loads(z.read(n).decode("utf-8"))
            except Exception:
                continue
            img = d.get("image", {})
            th, rgb = img.get("filename"), img.get("filename_rgb")
            if not th or not rgb:
                continue
            anns = d.get("annotations", [{}]) or [{}]
            frames.append({
                "thermal": th, "rgb": rgb, "csv": os.path.splitext(th)[0] + ".csv",
                "status": d.get("metadata", {}).get("status"),
                "standard": anns[0].get("attributes", {}).get("standard", ""),
            })

    # 2) VS3 basename -> 내부 멤버명 매핑 (CP949 폴더 무시, ASCII basename 으로)
    with zipfile.ZipFile(vs3) as zsrc:
        member = {m.split("/")[-1]: m for m in zsrc.namelist()}

        def have_all(fr):
            return all(fr[k] in member for k in ("thermal", "rgb", "csv"))

        picked = []
        for want, cnt in (("normal", args.normal), ("danger", args.danger)):
            c, skipped = 0, 0
            for fr in frames:
                if fr["status"] != want or not have_all(fr):
                    continue
                if skipped < args.skip:        # held-out 분리: 앞 K개 건너뜀
                    skipped += 1
                    continue
                picked.append(fr)
                c += 1
                if c >= cnt:
                    break

        for fr in picked:
            for k in ("thermal", "rgb", "csv"):
                with open(os.path.join(args.out, fr[k]), "wb") as f:
                    f.write(zsrc.read(member[fr[k]]))

    # 평가용 GT manifest 저장(eval_filter 가 읽음)
    manifest = [{"id": fr["thermal"][:-8], "status": fr["status"], "standard": fr["standard"],
                 "thermal": fr["thermal"], "rgb": fr["rgb"], "csv": fr["csv"]} for fr in picked]
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, ensure_ascii=False, indent=2)

    print(f"추출 완료: {len(picked)}프레임 -> {args.out} (manifest.json 포함)")
    for fr in picked:
        print(f"  [{fr['status'] or '?':6s}] {fr['standard']:8s} id={fr['thermal'][:-8]}")


if __name__ == "__main__":
    main()
