"""test_filter.py - /api/v1/verify-fire-alarm 호출 테스트 (이미지 업로드).

로컬 열화상/실화상 이미지를 multipart 로 업로드해 오경보 필터 API 를 호출하고 응답을 출력한다.
서버 먼저 기동: uvicorn src.main:app --host 127.0.0.1 --port 8011

실행(프로젝트 루트):
  python scripts/test_filter.py --thermal a.jpg --rgb b.jpg --csv c.csv
"""
import argparse
import json
import mimetypes
import os
import urllib.error
import urllib.request


def _multipart(fields, files):
    """fields: {name: 값}, files: {name: 경로} -> (Content-Type, body bytes)."""
    boundary = "----cctvbriefing" + os.urandom(8).hex()
    parts = []
    for k, v in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    for name, path in files.items():
        fn = os.path.basename(path)
        ct = mimetypes.guess_type(path)[0] or "application/octet-stream"
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{fn}"\r\n'
            f'Content-Type: {ct}\r\n\r\n'.encode())
        with open(path, "rb") as f:
            parts.append(f.read())
        parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return f"multipart/form-data; boundary={boundary}", b"".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8011/api/v1/verify-fire-alarm")
    ap.add_argument("--camera", default="CAM_T1")
    ap.add_argument("--thermal", default="dummy_thermal.jpg", help="열화상 이미지 경로")
    ap.add_argument("--rgb", default="dummy_rgb.jpg", help="정합 실화상(RGB) 경로")
    ap.add_argument("--csv", default=None, help="(선택) 온도 CSV 경로")
    args = ap.parse_args()

    files = {"thermal": args.thermal, "rgb": args.rgb}
    if args.csv:
        files["thermal_csv"] = args.csv
    ct, body = _multipart({"camera_id": args.camera}, files)

    req = urllib.request.Request(args.url, data=body, headers={"Content-Type": ct}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            print(json.dumps(json.loads(r.read().decode("utf-8")), ensure_ascii=False, indent=2))
    except urllib.error.HTTPError as e:
        print("HTTP", e.code, e.read().decode("utf-8", "ignore"))
    except Exception as e:
        print("요청 실패(서버 기동 확인):", e)


if __name__ == "__main__":
    main()
