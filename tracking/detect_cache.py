import argparse
import json
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT_IMAGES = os.path.join(ROOT, "dataset", "test", "images")
OUT_DIR = os.path.join(ROOT, "results", "tracking", "detections")

# slug -> video yolu. Slug'lar prepare_test_set.slugify() çıktısıyla aynı olmalı,
# çünkü GT dosya adları (Stockflue_Flyaround_f000030.jpg) o fonksiyonla üretildi.
VIDEOS = {
    "Stockflue_Flyaround": "archive/test/Stockflue Flyaround.mp4",
    "Surenen_Pass_Trail_Running": "archive/test/Surenen Pass Trail Running.mp4",
}


def yolo_predictor(weights, imgsz):
    model = YOLO(os.path.join(ROOT, weights))

    def predict(frame):
        result = model.predict(frame, imgsz=imgsz, classes=[0], verbose=False, conf=0.01)[0]
        return result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()

    return predict


def dino_predictor(tiled):
    # Import burada: tiled_dino transformers'i cekiyor, YOLO kosusunun buna ihtiyaci yok.
    from tiled_dino import TiledGroundingDino

    dino = TiledGroundingDino(model_name=os.path.join(ROOT, "dino_tuned_round2"))

    def predict(frame):
        if tiled:
            return dino.detect_tiled(frame, threshold=0.05, text_threshold=0.05)
        return dino.detect(frame, threshold=0.05, text_threshold=0.05)

    return predict


CONFIGS = {
    "yolo11x_1536": lambda: yolo_predictor("yolo11x_tuned_round2.pt", 1536),
    "yolo26x_1536": lambda: yolo_predictor("yolo26x_tuned_round2.pt", 1536),
    "dino_full": lambda: dino_predictor(tiled=False),
    "dino_tiled": lambda: dino_predictor(tiled=True),
}


def cache_path(slug, model):
    return os.path.join(OUT_DIR, f"{slug}__{model}.json")


def run_video(predict, slug, model, use_gt_jpeg):
    video_path = os.path.join(ROOT, VIDEOS[slug])
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"{video_path} açılamadı")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    frames, idx, swapped = {}, 0, 0
    start = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if use_gt_jpeg:
            gt_path = os.path.join(GT_IMAGES, f"{slug}_f{idx:06d}.jpg")
            if os.path.exists(gt_path):
                gt_frame = cv2.imread(gt_path)
                if gt_frame is not None:
                    frame, swapped = gt_frame, swapped + 1

        boxes, scores = predict(frame)
        frames[idx] = [
            [round(float(v), 2) for v in box] + [round(float(score), 5)]
            for box, score in zip(boxes, scores)
        ]
        idx += 1

        if idx % 50 == 0:
            per_frame = (time.time() - start) / idx
            remaining = (total - idx) * per_frame / 60 if total else 0
            print(f"    {idx}/{total or '?'} kare | {per_frame:.2f} s/kare | "
                  f"kalan ~{remaining:.1f} dk", flush=True)

    cap.release()

    os.makedirs(OUT_DIR, exist_ok=True)
    payload = {
        "video": VIDEOS[slug],
        "slug": slug,
        "model": model,
        "width": width,
        "height": height,
        "fps": fps,
        "num_frames": idx,
        "gt_jpeg_frames": swapped,
        "seconds_per_frame": round((time.time() - start) / max(idx, 1), 4),
        "frames": frames,
    }
    with open(cache_path(slug, model), "w") as f:
        json.dump(payload, f)

    boxes_total = sum(len(v) for v in frames.values())
    print(f"    bitti: {idx} kare, {boxes_total} kutu, {swapped} karede GT JPEG kullanıldı")


def main():
    parser = argparse.ArgumentParser(description="Her karede dedektör çalıştırıp tespitleri önbelleğe alır")
    parser.add_argument("--models", default="yolo11x_1536",
                        help=f"Virgülle ayrılmış: {','.join(CONFIGS)}")
    parser.add_argument("--videos", default="Stockflue_Flyaround,Surenen_Pass_Trail_Running",
                        help=f"Virgülle ayrılmış: {','.join(VIDEOS)}")
    parser.add_argument("--no-gt-jpeg", action="store_true",
                        help="GT karelerinde de video decode'unu kullan (birebir tekrar üretimi bozar)")
    parser.add_argument("--force", action="store_true", help="Var olan önbelleği yeniden üret")
    args = parser.parse_args()

    slugs = args.videos.split(",")
    for model in args.models.split(","):
        pending = [s for s in slugs if args.force or not os.path.exists(cache_path(s, model))]
        if not pending:
            print(f"{model}: tüm önbellekler mevcut, atlanıyor")
            continue

        print(f"\n{'=' * 70}\n{model}\n{'=' * 70}")
        predict = CONFIGS[model]()
        for slug in pending:
            print(f"  {slug}")
            run_video(predict, slug, model, use_gt_jpeg=not args.no_gt_jpeg)


if __name__ == "__main__":
    main()
