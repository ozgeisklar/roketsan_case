"""
Yan yana demo videosu: solda ham tespitler, sağda tracking sonrası.

Sağ panelde interpolasyonla gelen kutular farklı renkte çizilir, böylece
dedektörün kaçırdığı ama track'in taşıdığı kareler gözle görülebiliyor. Metrik
tablosu bu farkı sayıyla veriyor; bu video onun neye benzediğini gösteriyor.
"""

import argparse
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detect_cache import VIDEOS, cache_path  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACKED_DIR = os.path.join(ROOT, "results", "tracking", "tracked")
OUT_DIR = os.path.join(ROOT, "results", "tracking", "demo")

DETECTED_COLOR = (80, 220, 80)
INTERPOLATED_COLOR = (0, 165, 255)
RAW_COLOR = (200, 200, 200)


def draw(frame, boxes, title, thickness, font_scale):
    for x1, y1, x2, y2, label, color in boxes:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        if label:
            cv2.putText(frame, label, (int(x1), max(int(y1) - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(frame, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def main():
    parser = argparse.ArgumentParser(description="Ham ve tracking ciktisini yan yana videoya yazar")
    parser.add_argument("--slug", default="Stockflue_Flyaround", choices=list(VIDEOS))
    parser.add_argument("--model", default="yolo11x_1536")
    parser.add_argument("--conf", type=float, default=0.25, help="Cizim esigi")
    parser.add_argument("--width", type=int, default=960, help="Panel basina genislik")
    args = parser.parse_args()

    raw = json.load(open(cache_path(args.slug, args.model)))["frames"]
    tracked = json.load(open(os.path.join(
        TRACKED_DIR, f"{args.slug}__{args.model}__botsort.json")))["frames"]

    cap = cv2.VideoCapture(os.path.join(ROOT, VIDEOS[args.slug]))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    scale = args.width / cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{args.slug}__{args.model}__demo.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (args.width * 2, height))

    thickness = max(1, round(args.width / 640))
    font_scale = max(0.35, thickness * 0.25)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (args.width, height), interpolation=cv2.INTER_AREA)

        left = [(x1 * scale, y1 * scale, x2 * scale, y2 * scale, "", RAW_COLOR)
                for x1, y1, x2, y2, score in raw.get(str(idx), []) if score >= args.conf]
        right = [(x1 * scale, y1 * scale, x2 * scale, y2 * scale, f"#{int(tid)}",
                  INTERPOLATED_COLOR if flag else DETECTED_COLOR)
                 for x1, y1, x2, y2, score, tid, flag in tracked.get(str(idx), []) if score >= args.conf]

        panels = (draw(frame.copy(), left, f"ham tespit ({len(left)})", thickness, font_scale),
                  draw(frame.copy(), right, f"BoT-SORT ({len(right)}, turuncu=interpolasyon)",
                       thickness, font_scale))
        writer.write(cv2.hconcat(panels))
        idx += 1

    cap.release()
    writer.release()
    print(f"{idx} kare -> {out_path}")


if __name__ == "__main__":
    main()
