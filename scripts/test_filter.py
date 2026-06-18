"""test_filter.py - /api/v1/verify-fire-alarm 호출 테스트.

로컬 열화상/실화상 이미지 경로를 넣어 오경보 필터 API 를 호출하고 응답(JSON)을 출력한다.
서버 먼저 기동: uvicorn src.main:app --host 127.0.0.1 --port 8011

실행(프로젝트 루트):
  python scripts/test_filter.py                                  # 기본 더미 경로
  python scripts/test_filter.py --thermal dummy_thermal.jpg --rgb dummy_rgb.jpg --csv dummy.csv
"""
import argparse
import json
import urllib.error
import urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8011/api/v1/verify-fire-alarm")
    ap.add_argument("--camera", default="CAM_T1")
    ap.add_argument("--thermal", default="dummy_thermal.jpg", help="열화상 이미지 경로")
    ap.add_argument("--rgb", default="dummy_rgb.jpg", help="정합 실화상(RGB) 경로")
    ap.add_argument("--csv", default=None, help="(선택) 온도 CSV 경로")
    args = ap.parse_args()

    payload = {"camera_id": args.camera,
               "thermal_image_path": args.thermal, "rgb_image_path": args.rgb}
    if args.csv:
        payload["thermal_csv_path"] = args.csv

    req = urllib.request.Request(
        args.url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            out = json.loads(r.read().decode("utf-8"))
        print(json.dumps(out, ensure_ascii=False, indent=2))
    except urllib.error.HTTPError as e:
        print("HTTP", e.code, e.read().decode("utf-8", "ignore"))
    except Exception as e:
        print("요청 실패(서버 기동 확인):", e)


if __name__ == "__main__":
    main()
