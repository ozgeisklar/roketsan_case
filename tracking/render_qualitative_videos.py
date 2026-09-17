import argparse
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detect_cache import VIDEOS, cache_path  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACKED_DIR = os.path.join(ROOT, "results", "tracking", "tracked")
OUT_DIR = os.path.join(ROOT, "results", "qualitative", "stockflue")

COLORS = {
    "baseline": (0, 0, 255),
    "tuned": (0, 0, 255),
    "tracking": (0, 0, 255),
    "interpolated": (0, 165, 255),
}

DEFAULT_ITEMS = [
    ("baseline", "yolo11x_1536_baseline", "YOLO11x baseline"),
    ("baseline", "yolo26x_1536_baseline", "YOLO26x baseline"),
    ("baseline", "dino_full_baseline", "DINO full baseline"),
    ("baseline", "dino_tiled_baseline", "DINO tiled baseline"),
    ("tuned", "yolo11x_1536_tuned", "YOLO11x fine-tuned"),
    ("tuned", "yolo26x_1536_tuned", "YOLO26x fine-tuned"),
    ("tuned", "dino_full_tuned", "DINO full fine-tuned"),
    ("tuned", "dino_tiled_tuned", "DINO tiled fine-tuned"),
    ("tracking", "yolo11x_1536", "YOLO11x + BoT-SORT"),
    ("tracking", "yolo26x_1536", "YOLO26x + BoT-SORT"),
    ("tracking", "dino_full", "DINO full + BoT-SORT"),
    ("tracking", "dino_tiled", "DINO tiled + BoT-SORT"),
]


def slug_name(text):
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


def load_item(slug, stage, model):
    if stage == "tracking":
        path = os.path.join(TRACKED_DIR, f"{slug}__{model}__botsort.json")
    else:
        path = cache_path(slug, model)
        if not os.path.exists(path) and model.endswith("_tuned"):
            legacy_model = model[:-len("_tuned")]
            path = cache_path(slug, legacy_model)
    if not os.path.exists(path):
        return None, path
    return json.load(open(path))["frames"], path


def draw_header(frame, title):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 38), (0, 0, 0), -1)
    cv2.putText(frame, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (255, 255, 255), 2, cv2.LINE_AA)


def render_video(slug, stage, model, title, conf, width, max_frames=None):
    frames, path = load_item(slug, stage, model)
    if frames is None:
        print(f"Eksik JSON: {path}")
        return None

    cap = cv2.VideoCapture(os.path.join(ROOT, VIDEOS[slug]))
    if not cap.isOpened():
        raise RuntimeError(f"Video acilamadi: {VIDEOS[slug]}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    src_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    scale = width / src_w
    height = int(src_h * scale)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_name = f"{stage}__{slug_name(title)}.mp4"
    out_path = os.path.join(OUT_DIR, out_name)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    thickness = max(1, round(width / 640))
    font_scale = max(0.4, thickness * 0.3)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames is not None and idx >= max_frames):
            break

        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        rows = frames.get(str(idx), [])
        for row in rows:
            x1, y1, x2, y2, score = row[:5]
            if score < conf:
                continue

            is_interp = len(row) >= 7 and row[6] == 1
            color = COLORS["interpolated"] if is_interp else COLORS[stage]
            x1, y1, x2, y2 = [int(v * scale) for v in (x1, y1, x2, y2)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            label = f"{score:.2f}"
            if stage == "tracking" and len(row) >= 7:
                label = f"#{int(row[5])} {label}"
                if is_interp:
                    label += " interp"
            cv2.putText(frame, label, (x1, max(y1 - 5, 50)), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, color, thickness, cv2.LINE_AA)

        draw_header(frame, title)
        writer.write(frame)
        idx += 1

    cap.release()
    writer.release()
    print(f"{idx:>4} kare -> {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Stockflue icin bbox cizimli nitel video seti uretir")
    parser.add_argument("--slug", default="Stockflue_Flyaround", choices=list(VIDEOS))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Hizli kontrol icin ilk N frame ile sinirla")
    args = parser.parse_args()

    for stage, model, title in DEFAULT_ITEMS:
        render_video(args.slug, stage, model, title, args.conf, args.width, args.max_frames)

    print(f"\nCiktilar: {OUT_DIR}")


if __name__ == "__main__":
    main()
