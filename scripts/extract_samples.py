"""extract_samples.py - AIHub 514 VS3/VL3 zip 에서 열화상+실화상+CSV 페어 추출.

zip 내부 한글 폴더명은 CP949 라 mojibake 로 보이지만 파일 basename 은 ASCII ID 라
basename 으로 매칭해 추출한다(원문서 7장 권장 방식). 라벨(VL3)의 status 로 normal/danger 균형 추출.

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
            c = 0
            for fr in frames:
                if fr["status"] != want or not have_all(fr):
                    continue
                picked.append(fr)
                c += 1
                if c >= cnt:
                    break

        for fr in picked:
            for k in ("thermal", "rgb", "csv"):
                with open(os.path.join(args.out, fr[k]), "wb") as f:
                    f.write(zsrc.read(member[fr[k]]))

    print(f"추출 완료: {len(picked)}프레임 -> {args.out}")
    for fr in picked:
        print(f"  [{fr['status'] or '?':6s}] {fr['standard']:8s} id={fr['thermal'][:-8]}")


if __name__ == "__main__":
    main()
