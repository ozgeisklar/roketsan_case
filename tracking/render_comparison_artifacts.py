import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detect_cache import VIDEOS, cache_path  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT_PATH = os.path.join(ROOT, "dataset", "test", "annotations", "instances_filtered.json")
TRACKED_DIR = os.path.join(ROOT, "results", "tracking", "tracked")
OUT_DIR = os.path.join(ROOT, "results", "qualitative", "comparison")

SLUG = "Stockflue_Flyaround"
CONF = 0.25
PANEL_W = 640

PANELS = [
    ("YOLO11x baseline", "cache", "yolo11x_1536_baseline", (0, 0, 255)),
    ("YOLO11x fine-tuned", "cache", "yolo11x_1536", (0, 0, 255)),
    ("YOLO11x + BoT-SORT", "tracked", "yolo11x_1536", (0, 0, 255)),
    ("DINO tiled + BoT-SORT", "tracked", "dino_tiled", (0, 0, 255)),
]

GT_COLOR = (255, 80, 80)
INTERP_COLOR = (0, 165, 255)


def parse_frame(file_name):
    return int(os.path.basename(file_name).rsplit("_f", 1)[1].split(".")[0])


def load_predictions(kind, model):
    if kind == "cache":
        path = cache_path(SLUG, model)
    else:
        path = os.path.join(TRACKED_DIR, f"{SLUG}__{model}__botsort.json")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return json.load(open(path))["frames"]


def load_gt():
    coco = json.load(open(GT_PATH))
    images = {im["id"]: im for im in coco["images"] if os.path.basename(im["file_name"]).startswith(SLUG)}
    anns = defaultdict(list)
    for ann in coco["annotations"]:
        if ann["image_id"] not in images:
            continue
        x, y, w, h = ann["bbox"]
        anns[parse_frame(images[ann["image_id"]]["file_name"])].append([x, y, x + w, y + h])
    return dict(anns)


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def rows_to_boxes(rows):
    return [r[:4] for r in rows if r[4] >= CONF]


def get_rows(preds, frame_idx):
    return preds.get(str(frame_idx), [])


def match_count(gt, rows):
    boxes = rows_to_boxes(rows)
    if not gt:
        return 0
    if not boxes:
        return 0
    return int((iou_matrix(gt, boxes).max(axis=1) >= 0.5).sum())


def draw_panel(frame, title, rows, gt_boxes=None, show_gt=False):
    panel = cv2.resize(frame, (PANEL_W, int(frame.shape[0] * PANEL_W / frame.shape[1])),
                       interpolation=cv2.INTER_AREA)
    scale = PANEL_W / frame.shape[1]
    thickness = max(1, round(PANEL_W / 640))
    font_scale = 0.45

    if show_gt and gt_boxes:
        for x1, y1, x2, y2 in gt_boxes:
            x1, y1, x2, y2 = [int(v * scale) for v in (x1, y1, x2, y2)]
            cv2.rectangle(panel, (x1, y1), (x2, y2), GT_COLOR, thickness + 1)
            cv2.putText(panel, "GT", (x1, max(y1 - 5, 48)), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, GT_COLOR, thickness, cv2.LINE_AA)

    for row in rows:
        if row[4] < CONF:
            continue
        x1, y1, x2, y2 = [int(v * scale) for v in row[:4]]
        is_interp = len(row) >= 7 and row[6] == 1
        color = INTERP_COLOR if is_interp else (0, 0, 255)
        label = f"{row[4]:.2f}"
        if len(row) >= 7:
            label = f"#{int(row[5])} {label}"
            if is_interp:
                label += " interp"
        cv2.rectangle(panel, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(panel, label, (x1, max(y1 - 5, 64)), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness, cv2.LINE_AA)

    cv2.rectangle(panel, (0, 0), (panel.shape[1], 38), (0, 0, 0), -1)
    cv2.putText(panel, title, (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def make_grid(panels):
    top = cv2.hconcat(panels[:2])
    bottom = cv2.hconcat(panels[2:])
    return cv2.vconcat([top, bottom])


def render_comparison_video(preds_by_panel):
    cap = cv2.VideoCapture(os.path.join(ROOT, VIDEOS[SLUG]))
    if not cap.isOpened():
        raise RuntimeError(f"Video acilamadi: {VIDEOS[SLUG]}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    panel_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * PANEL_W / cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_path = os.path.join(OUT_DIR, "stockflue_2x2_qualitative_comparison.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (PANEL_W * 2, panel_h * 2))

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        panels = [
            draw_panel(frame, title, get_rows(preds_by_panel[title], idx))
            for title, _kind, _model, _color in PANELS
        ]
        writer.write(make_grid(panels))
        idx += 1

    cap.release()
    writer.release()
    print(f"{idx} kare -> {out_path}")
    return out_path


def select_examples(gt_by_frame, preds_by_panel):
    yolo_base = preds_by_panel["YOLO11x baseline"]
    yolo_track = preds_by_panel["YOLO11x + BoT-SORT"]
    dino_track = preds_by_panel["DINO tiled + BoT-SORT"]

    candidates = []
    for frame_idx, gt in gt_by_frame.items():
        base_hit = match_count(gt, get_rows(yolo_base, frame_idx))
        yolo_hit = match_count(gt, get_rows(yolo_track, frame_idx))
        dino_hit = match_count(gt, get_rows(dino_track, frame_idx))
        total = len(gt)
        candidates.append({
            "frame": frame_idx,
            "gt": total,
            "baseline_hit": base_hit,
            "yolo_track_hit": yolo_hit,
            "dino_track_hit": dino_hit,
            "disagreement": abs(dino_hit - base_hit) + abs(dino_hit - yolo_hit),
        })

    success = max(candidates, key=lambda c: (c["dino_track_hit"] == c["gt"], c["gt"], c["yolo_track_hit"]))
    failure = max(candidates, key=lambda c: (c["gt"] - c["dino_track_hit"], c["gt"]))
    uncertain = max(candidates, key=lambda c: (c["disagreement"], c["gt"]))
    return {
        "success": success,
        "failure": failure,
        "uncertain": uncertain,
    }


def render_example_images(examples, gt_by_frame, preds_by_panel):
    cap = cv2.VideoCapture(os.path.join(ROOT, VIDEOS[SLUG]))
    frame_cache = {}
    for label, info in examples.items():
        frame_idx = info["frame"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        frame_cache[frame_idx] = frame

        panels = []
        for title, _kind, _model, _color in PANELS:
            panel_title = f"{title} | GT {info['gt']}"
            panels.append(draw_panel(frame, panel_title, get_rows(preds_by_panel[title], frame_idx),
                                     gt_by_frame.get(frame_idx, []), show_gt=True))
        grid = make_grid(panels)
        out_path = os.path.join(OUT_DIR, f"{label}_stockflue_f{frame_idx:06d}.jpg")
        cv2.imwrite(out_path, grid, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"{label:<9} frame {frame_idx:>6} -> {out_path}")
    cap.release()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    preds_by_panel = {
        title: load_predictions(kind, model)
        for title, kind, model, _color in PANELS
    }
    gt_by_frame = load_gt()

    video_path = render_comparison_video(preds_by_panel)
    examples = select_examples(gt_by_frame, preds_by_panel)
    render_example_images(examples, gt_by_frame, preds_by_panel)

    meta_path = os.path.join(OUT_DIR, "example_selection.json")
    with open(meta_path, "w") as f:
        json.dump({"video": video_path, "examples": examples}, f, indent=2)
    print(f"metadata -> {meta_path}")


if __name__ == "__main__":
    main()
