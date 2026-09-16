
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import ultralytics
from torchvision.ops import nms
from ultralytics.trackers import BOTSORT
from ultralytics.utils import YAML, IterableSimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detect_cache import VIDEOS, cache_path  # noqa: E402
from tiled_dino import suppress_contained  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACKS_DIR = os.path.join(ROOT, "results", "tracking", "tracks")
OUT_DIR = os.path.join(ROOT, "results", "tracking", "tracked")

DEFAULTS = {
    "max_gap": 15,        # yarım saniye; daha uzun boşlukta kutuyu uydurmuş oluruz
    "alpha": 0.5,         # kendi skoru ile track kanıtı arasındaki denge
    "min_track_len": 3,
    "interp_factor": 0.9,
    "nms_iou": 0.55,      # tiled_dino.detect_tiled ile aynı: dino_tiled'da işlemsiz kalır
    "track_low_thresh": 0.03,
}


class Detections:

    def __init__(self, xyxy, conf):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = np.zeros(len(conf), dtype=np.float32)

    @property
    def xywh(self):
        x1, y1, x2, y2 = self.xyxy.T
        return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, mask):
        return Detections(self.xyxy[mask], self.conf[mask])


def iou_matrix(a, b):
    """(N,4) ve (M,4) xyxy dizileri arasında (N,M) IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def frame_arrays(cache, idx):
    rows = cache["frames"].get(str(idx), [])
    if not rows:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    arr = np.asarray(rows, dtype=np.float32)
    return arr[:, :4], arr[:, 4]


def deduplicate(boxes, scores, iou_thresh):
    
    if len(boxes) == 0:
        return boxes, scores
    keep = nms(torch.from_numpy(boxes), torch.from_numpy(scores), iou_thresh).numpy()
    return suppress_contained(boxes[keep], scores[keep])


def build_tracker(track_low_thresh):
   
    cfg = YAML.load(Path(ultralytics.__file__).parent / "cfg" / "trackers" / "botsort.yaml")
    cfg["with_reid"] = False
    # Uzak insanlar 0.17-0.25 skor alıyor (bkz. tiled_dino.py); varsayılan 0.1'lik
    # alt eşik bu zayıf tespitleri ikinci eşleştirme turundan tamamen dışlıyor.
    cfg["track_low_thresh"] = track_low_thresh
    return BOTSORT(IterableSimpleNamespace(**cfg))


def associate(cache, slug, nms_iou, track_low_thresh):
    
    cap = cv2.VideoCapture(os.path.join(ROOT, VIDEOS[slug]))
    tracker = build_tracker(track_low_thresh)

    tracks = defaultdict(list)
    untracked = {}
    for idx in range(cache["num_frames"]):
        ok, frame = cap.read()
        if not ok:
            break

        boxes, scores = deduplicate(*frame_arrays(cache, idx), nms_iou)
        output = tracker.update(Detections(boxes, scores), frame)

        out_boxes = output[:, :4] if len(output) else np.zeros((0, 4), dtype=np.float32)
        ious = iou_matrix(boxes, out_boxes)

        for j, track_row in enumerate(output):
            # Koordinatı tracker'ın Kalman kutusundan değil, eşleşen ham tespitten
            # alıyoruz: tracker'ın işi kanıt üretmek (süreklilik ve kimlik), konum
            # belirlemek değil. Kalman kutusu IoU 0.5'te fark yaratmıyor ama sıkı
            # eşiklerde (mAP@50-95, AP_Small) lokalizasyonu gözle görülür bozuyor.
            # Eşleşme yoksa track savruluyor demektir, o zaman Kalman kutusu kalır.
            box = [float(v) for v in track_row[:4]]
            if len(boxes):
                best = int(np.argmax(ious[:, j]))
                if ious[best, j] >= 0.5:
                    box = boxes[best].tolist()
            tracks[int(track_row[4])].append([idx, box, float(track_row[5])])

        # Tracker yalnızca onaylanmış track'leri döndürüyor, yani hiçbir track'e
        # girmeyen tespitler çıktıdan tamamen kayboluyor. Bunları atmak recall
        # tavanını düşürür; tekil track olarak saklıyoruz, aşağıdaki uzunluk
        # katsayısı zaten skorlarını kırpacak.
        if len(boxes):
            loose = ious.max(axis=1) < 0.5 if len(out_boxes) else np.ones(len(boxes), bool)
            untracked[idx] = [[boxes[i].tolist(), float(scores[i])] for i in np.nonzero(loose)[0]]

    cap.release()
    return tracks, untracked


def load_or_associate(cache, slug, model, params, force):
    """İlişkilendirme sonucunu önbellekten okur, yoksa üretir.

    Parametre taraması yalnızca interpolasyon ve skorlamayı değiştiriyor;
    ilişkilendirme değişmediği için pahalı decode adımını tekrarlamıyoruz.
    """
    path = os.path.join(TRACKS_DIR, f"{slug}__{model}.json")
    if os.path.exists(path) and not force:
        data = json.load(open(path))
        return ({int(k): v for k, v in data["tracks"].items()},
                {int(k): v for k, v in data["untracked"].items()})

    tracks, untracked = associate(cache, slug, params["nms_iou"], params["track_low_thresh"])
    os.makedirs(TRACKS_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"tracks": tracks, "untracked": untracked}, f)
    return tracks, untracked


def refine_track(observations, max_gap, alpha, min_track_len, interp_factor):
    """Bir track'in gözlemlerini boşluk doldurma ve yeniden skorlamadan geçirir.

    Track skoru ile uzunluk katsayısı YALNIZCA gerçek tespitlerden hesaplanır;
    interpolasyon kutuları kanıt değil sonuçtur, onları da saymak kendi kendini
    doğrulayan bir döngü olurdu.

    Tek formül dört davranışı birden kapsıyor: istikrarlı bir track kendi skoru
    ile track kanıtının harmanını alır, kısa track ve hiç takip edilmemiş tespit
    (uzunluk 1) uzunluk katsayısıyla aşağı çekilir, interpolasyon kutusu ise zaten
    kırpılmış bir skorla aynı formülden geçer. Silmek yerine ağırlık düşürmek
    tercih edildi: COCO AP bir eşik değil sıralama metriği, silmek recall tavanını
    gereksiz yere düşürürdü.
    """
    scores = sorted(s for _, _, s in observations)
    track_score = sum(scores[-3:]) / len(scores[-3:])
    length_factor = min(1.0, len(observations) / min_track_len)

    entries = [[frame, box, score, 0] for frame, box, score in observations]
    for (f0, b0, s0), (f1, b1, s1) in zip(observations, observations[1:]):
        gap = f1 - f0
        if 1 < gap <= max_gap:
            for frame in range(f0 + 1, f1):
                t = (frame - f0) / gap
                box = [a + (b - a) * t for a, b in zip(b0, b1)]
                entries.append([frame, box, interp_factor * min(s0, s1), 1])

    for entry in entries:
        entry[2] = (alpha * entry[2] + (1 - alpha) * track_score) * length_factor
    return entries


def to_rows(frames):
    """{kare: [[x1,y1,x2,y2,skor,track_id,interpolasyon_mu], ...]} biçimine getirir."""
    return {
        str(frame): [[*[round(v, 2) for v in box], round(score, 5), tid, flag]
                     for box, score, tid, flag in items]
        for frame, items in frames.items()
    }


def build_stage(cache, slug, model, stage, params, force):
    frames = defaultdict(list)

    if stage in ("raw", "nms"):
        for idx in range(cache["num_frames"]):
            boxes, scores = frame_arrays(cache, idx)
            if stage == "nms":
                boxes, scores = deduplicate(boxes, scores, params["nms_iou"])
            frames[idx] = [(box.tolist(), float(score), -1, 0) for box, score in zip(boxes, scores)]
        return frames

    tracks, untracked = load_or_associate(cache, slug, model, params, force)

    # Takip edilmeyen tespitler uzunluğu 1 olan track'ler sayılır, böylece
    # aşağıdaki iyileştirme tek bir kod yolundan geçer.
    singles, next_id = {}, max(tracks, default=0) + 1
    for idx, items in untracked.items():
        for box, score in items:
            singles[next_id] = [[idx, box, score]]
            next_id += 1

    refine_args = {k: params[k] for k in ("max_gap", "alpha", "min_track_len", "interp_factor")}
    for tid, observations in list(tracks.items()) + list(singles.items()):
        for frame, box, score, flag in refine_track(observations, **refine_args):
            frames[frame].append((box, score, tid, flag))
    return frames


def main():
    parser = argparse.ArgumentParser(description="Önbellekli tespitlere çevrimdışı tracking uygular")
    parser.add_argument("--models", default="yolo11x_1536")
    parser.add_argument("--videos", default="Stockflue_Flyaround,Surenen_Pass_Trail_Running")
    parser.add_argument("--stages", default="raw,nms,botsort")
    parser.add_argument("--force", action="store_true", help="İlişkilendirmeyi yeniden hesapla")
    for key, value in DEFAULTS.items():
        parser.add_argument(f"--{key.replace('_', '-')}", type=type(value), default=value)
    args = parser.parse_args()

    params = {k: getattr(args, k) for k in DEFAULTS}
    os.makedirs(OUT_DIR, exist_ok=True)

    for model in args.models.split(","):
        for slug in args.videos.split(","):
            cache = json.load(open(cache_path(slug, model)))
            for stage in args.stages.split(","):
                frames = build_stage(cache, slug, model, stage, params, args.force)
                out_path = os.path.join(OUT_DIR, f"{slug}__{model}__{stage}.json")
                with open(out_path, "w") as f:
                    json.dump({"slug": slug, "model": model, "stage": stage,
                               "params": params if stage == "botsort" else {},
                               "frames": to_rows(frames)}, f)

                boxes = sum(len(v) for v in frames.values())
                interp = sum(1 for v in frames.values() for e in v if e[3])
                tids = {e[2] for v in frames.values() for e in v if e[2] >= 0}
                print(f"{slug:<28} {model:<14} {stage:<8} kutu {boxes:>6} | "
                      f"interp {interp:>5} | track {len(tids):>4}")


if __name__ == "__main__":
    main()
