import argparse
import contextlib
import io
import json
import os
import sys

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detect_cache import cache_path  # noqa: E402
from offline_tracker import DEFAULTS, build_stage, iou_matrix, to_rows  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACKED_DIR = os.path.join(ROOT, "results", "tracking", "tracked")
GT_PATH = os.path.join(ROOT, "dataset", "test", "annotations", "instances_filtered.json")
OUT_PATH = os.path.join(ROOT, "results", "evaluation", "summary_tracking.md")

# raw aşamasının yeniden üretmesi gereken değerler (results/evaluation/summary_tuned_round2.md)
BASELINE_MAP50 = {"yolo11x_1536": 0.760, "yolo26x_1536": 0.749,
                  "dino_full": 0.906, "dino_tiled": 0.914}


def parse_name(file_name):
    """Stockflue_Flyaround_f000030.jpg -> (Stockflue_Flyaround, 30)"""
    slug, frame = file_name.rsplit("/", 1)[-1].rsplit("_f", 1)
    return slug, int(frame.split(".")[0])


def load_frames(slug, model, stage):
    path = os.path.join(TRACKED_DIR, f"{slug}__{model}__{stage}.json")
    return json.load(open(path))["frames"]


def to_coco(coco_gt, frames_by_slug, image_ids=None):
    """Kare bazlı kutuları COCO tahmin listesine çevirir."""
    detections = []
    for image in coco_gt.dataset["images"]:
        if image_ids is not None and image["id"] not in image_ids:
            continue
        slug, frame_idx = parse_name(image["file_name"])
        for x1, y1, x2, y2, score, _tid, _flag in frames_by_slug[slug].get(str(frame_idx), []):
            detections.append({"image_id": image["id"], "category_id": 1,
                               "bbox": [x1, y1, x2 - x1, y2 - y1], "score": score})
    return detections


def evaluate(coco_gt, detections, image_ids=None):
    """evaluate_on_gt.py ile aynı metrikler: mAP, AP_S/AP_M ve F1'i maksimize eden P/R."""
    if not detections:
        return None
    with contextlib.redirect_stdout(io.StringIO()):
        coco_eval = COCOeval(coco_gt, coco_gt.loadRes(detections), "bbox")
        if image_ids is not None:
            coco_eval.params.imgIds = list(image_ids)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    # precision boyutu [T, R, K, A, M]; T=0 -> IoU 0.50, A=0 -> tum alanlar, M=2 -> maxDets 100
    precisions = coco_eval.eval["precision"][0, :, 0, 0, 2]
    recalls = coco_eval.eval["params"].recThrs
    best_f1, best_p, best_r = 0.0, 0.0, 0.0
    for p, r in zip(precisions, recalls):
        if p > -1:
            f1 = 2 * p * r / (p + r + 1e-16)
            if f1 > best_f1:
                best_f1, best_p, best_r = f1, p, r

    stats = coco_eval.stats
    return {"mAP_50": stats[1], "mAP_50_95": stats[0], "AP_small": stats[3],
            "AP_medium": stats[4], "P": best_p, "R": best_r, "F1": best_f1}


def interpolation_gain(coco_gt, frames_by_slug, op_conf):
    """Yalnızca interpolasyon sayesinde yakalanan GT kutusu sayısını sayar.

    Sıralama metriği olan mAP, "kaç kutu geri geldi" sorusunu doğrudan
    cevaplamıyor. Burada pratik bir çalışma noktası (op_conf) sabitlenip, gerçek
    tespitlerle eşleşmeyen ama interpolasyon kutusuyla IoU>=0.5 eşleşen GT
    kutuları sayılıyor: tracking'in recall katkısının en doğrudan kanıtı.
    """
    recovered = missed = 0
    for image in coco_gt.dataset["images"]:
        annotations = coco_gt.imgToAnns.get(image["id"], [])
        if not annotations:
            continue
        gt = np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2],
                        a["bbox"][1] + a["bbox"][3]] for a in annotations], dtype=np.float32)

        slug, frame_idx = parse_name(image["file_name"])
        rows = [r for r in frames_by_slug[slug].get(str(frame_idx), []) if r[4] >= op_conf]
        detected = np.array([r[:4] for r in rows if r[6] == 0], dtype=np.float32).reshape(-1, 4)
        interpolated = np.array([r[:4] for r in rows if r[6] == 1], dtype=np.float32).reshape(-1, 4)

        hit_det = iou_matrix(gt, detected).max(axis=1) >= 0.5 if len(detected) else np.zeros(len(gt), bool)
        hit_int = iou_matrix(gt, interpolated).max(axis=1) >= 0.5 if len(interpolated) else np.zeros(len(gt), bool)
        recovered += int((~hit_det & hit_int).sum())
        missed += int((~hit_det).sum())
    return recovered, missed


def row(label, metrics):
    m = metrics
    return (f"| {label:<26} | {m['mAP_50']:<8.3f} | {m['mAP_50_95']:<9.3f} | {m['AP_small']:<8.3f} | "
            f"{m['AP_medium']:<8.3f} | {m['P']:<8.3f} | {m['R']:<8.3f} | {m['F1']:<8.3f} |")


HEADER = (f"| {'Model / asama':<26} | {'mAP@50':<8} | {'mAP@50-95':<9} | {'AP_Small':<8} | "
          f"{'AP_Med':<8} | {'Opt. P':<8} | {'Opt. R':<8} | {'Opt. F1':<8} |")
SEPARATOR = "|" + "-" * 28 + "|" + ("-" * 10 + "|") * 3 + ("-" * 10 + "|") * 4




def main():
    parser = argparse.ArgumentParser(description="Tracking asamalarini GT uzerinde degerlendirir")
    parser.add_argument("--models", default="yolo11x_1536")
    parser.add_argument("--videos", default="Stockflue_Flyaround,Surenen_Pass_Trail_Running")
    parser.add_argument("--stages", default="raw,nms,botsort")
    parser.add_argument("--op-conf", type=float, default=0.25,
                        help="Interpolasyon kazanci sayilirken kullanilan calisma noktasi")
    args = parser.parse_args()

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(GT_PATH)
    slugs = args.videos.split(",")
    by_video = {slug: [i["id"] for i in coco_gt.dataset["images"]
                       if parse_name(i["file_name"])[0] == slug] for slug in slugs}

    lines = ["# Tracking ile Zamansal Iyilestirme", "",
             f"GT: {len(coco_gt.dataset['images'])} kare, "
             f"{len(coco_gt.dataset['annotations'])} kutu ({', '.join(slugs)})", "",
             HEADER, SEPARATOR]
    notes = []

    for model in args.models.split(","):
        for stage in args.stages.split(","):
            frames_by_slug = {slug: load_frames(slug, model, stage) for slug in slugs}
            metrics = evaluate(coco_gt, to_coco(coco_gt, frames_by_slug))
            lines.append(row(f"{model} / {stage}", metrics))

            if stage == "raw" and model in BASELINE_MAP50:
                expected = BASELINE_MAP50[model]
                delta = abs(metrics["mAP_50"] - expected)
                status = "GECTI" if delta < 0.001 else "KALDI"
                notes.append(f"- Kapi kontrolu {model}: raw mAP@50 {metrics['mAP_50']:.3f}, "
                             f"beklenen {expected:.3f} -> {status}")
            if stage == "botsort":
                recovered, missed = interpolation_gain(coco_gt, frames_by_slug, args.op_conf)
                notes.append(f"- {model}: conf>={args.op_conf} calisma noktasinda dedektorun "
                             f"kacirdigi {missed} GT kutusunun {recovered} tanesi interpolasyonla geldi")

        # Video bazinda kirilim: Surenen 406x720 oldugu icin auto_grid (1,1) donduruyor,
        # yani dosemeli DINO orada tam kare DINO ile ayni; ortalamalar bunu gizliyor.
        lines.append(SEPARATOR)
        for slug in slugs:
            for stage in args.stages.split(","):
                frames_by_slug = {s: load_frames(s, model, stage) for s in slugs}
                metrics = evaluate(coco_gt, to_coco(coco_gt, frames_by_slug, set(by_video[slug])),
                                   by_video[slug])
                lines.append(row(f"{slug[:14]} / {stage}", metrics))
        lines.append(SEPARATOR)

    lines += ["", "## Notlar", ""] + notes
    lines.append(f"- Orneklem kucuk ({len(coco_gt.dataset['annotations'])} kutu); "
                 f"birkac kutuluk degisim oranlarda buyuk gorunuyor.")
    lines.append("- Interpolasyon cift yonlu, yani nedensel degil: gercek zamanli bir "
                 "sistemde yalnizca track_buffer ile geriye donuk doldurma mumkun olurdu.")
    lines.append("- Eslesen kutularda koordinat ham tespitten aliniyor, Kalman tahmininden "
                 "degil. Kalman kutusu kullanildiginda mAP@50 ayni kaliyor ama DINO'nun "
                 "mAP@50-95'i 0.641'den 0.628'e dusuyordu: sureklilik kazanci lokalizasyon "
                 "kaybina deger degil. YOLO'da tersi gecerli (gurultulu kutulari duzeltiyor), "
                 "yani bu takas dedektorun lokalizasyon kalitesine bagli.")


    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
