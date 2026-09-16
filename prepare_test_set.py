import argparse
import json
import os
import time
import zipfile
from datetime import datetime, timezone
from xml.etree import ElementTree as ET

import cv2

from tiled_dino import TiledGroundingDino, draw_detections

# Envanter analizine göre test setine alınan videolar ve seçim gerekçeleri
# (results/video_inventory_summary.md). Kalan 7 video eğitimde kullanılır.
SELECTION_RATIONALE = {
    "Stockflue Flyaround.mp4": "En yuksek kucuk-nesne orani (COCO small %60)",
    "Surenen Pass Trail Running.mp4": "Dinamik takip - patikada kosan sporcu (%49 small)",
}

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")
CVAT_LABEL = "person"
CVAT_LABEL_COLOR = "#33ddff"


def slugify(video_name):
    stem = os.path.splitext(video_name)[0]
    return "".join(ch if ch.isalnum() else "_" for ch in stem).strip("_")


def sample_frames(video_path, interval, out_dir, slug, jpeg_quality=95):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"{video_path} açılamadı")

    saved = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            file_name = f"{slug}_f{frame_idx:06d}.jpg"
            cv2.imwrite(os.path.join(out_dir, file_name),
                        frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            saved.append((file_name, frame_idx, frame.shape[1], frame.shape[0]))
        frame_idx += 1
    cap.release()
    return saved


def build_cvat_xml(images, task_name):
    """CVAT for images 1.1 XML'i üretir. CVAT eşleştirmeyi <image name> ile yapar."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f+00:00")
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"

    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "id").text = "1"
    ET.SubElement(task, "name").text = task_name
    ET.SubElement(task, "size").text = str(len(images))
    ET.SubElement(task, "mode").text = "annotation"
    ET.SubElement(task, "overlap").text = "0"
    ET.SubElement(task, "bugtracker")
    ET.SubElement(task, "created").text = now
    ET.SubElement(task, "updated").text = now
    ET.SubElement(task, "start_frame").text = "0"
    ET.SubElement(task, "stop_frame").text = str(max(0, len(images) - 1))
    ET.SubElement(task, "frame_filter")

    labels = ET.SubElement(task, "labels")
    label = ET.SubElement(labels, "label")
    ET.SubElement(label, "name").text = CVAT_LABEL
    ET.SubElement(label, "color").text = CVAT_LABEL_COLOR
    ET.SubElement(label, "type").text = "rectangle"
    ET.SubElement(label, "attributes")

    segments = ET.SubElement(task, "segments")
    segment = ET.SubElement(segments, "segment")
    ET.SubElement(segment, "id").text = "1"
    ET.SubElement(segment, "start").text = "0"
    ET.SubElement(segment, "stop").text = str(max(0, len(images) - 1))
    ET.SubElement(segment, "url")

    owner = ET.SubElement(task, "owner")
    ET.SubElement(owner, "username")
    ET.SubElement(owner, "email")
    ET.SubElement(meta, "dumped").text = now

    for image_id, item in enumerate(images):
        img_el = ET.SubElement(root, "image", {
            "id": str(image_id),
            "name": item["file_name"],
            "width": str(item["width"]),
            "height": str(item["height"]),
        })
        for (x1, y1, x2, y2), score in zip(item["boxes"], item["scores"]):
            ET.SubElement(img_el, "box", {
                "label": CVAT_LABEL,
                # source="auto" -> CVAT bu kutuları otomatik uretilmis olarak isaretler
                "source": "auto",
                "occluded": "0",
                "xtl": f"{x1:.2f}",
                "ytl": f"{y1:.2f}",
                "xbr": f"{x2:.2f}",
                "ybr": f"{y2:.2f}",
                "z_order": "0",
                "score": f"{score:.4f}",
            })

    ET.indent(root, space="  ")
    return ET.ElementTree(root)


def build_coco(images, description):
    coco = {
        "info": {
            "description": description,
            "date_created": datetime.now(timezone.utc).isoformat(),
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": CVAT_LABEL, "supercategory": "person"}],
    }
    ann_id = 1
    for image_id, item in enumerate(images):
        coco["images"].append({
            "id": image_id,
            "file_name": item["file_name"],
            "width": item["width"],
            "height": item["height"],
            "video": item["video"],
            "frame_index": item["frame_index"],
        })
        for (x1, y1, x2, y2), score in zip(item["boxes"], item["scores"]):
            w, h = float(x2 - x1), float(y2 - y1)
            coco["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [round(float(x1), 2), round(float(y1), 2), round(w, 2), round(h, 2)],
                "area": round(w * h, 2),
                "iscrowd": 0,
                "score": round(float(score), 4),
            })
            ann_id += 1
    return coco


def main():
    parser = argparse.ArgumentParser(description="Test seti kareleri ve DINO on-etiketleri")
    parser.add_argument("--input-dir", default="archive/test")
    parser.add_argument("--output-dir", default="dataset/test")
    parser.add_argument("--interval", type=int, default=30,
                        help="Kac karede bir ornek alinacak (30 = saniyede 1 kare)")
    parser.add_argument("--threshold", type=float, default=0.15,
                        help="On-etiketleme icin DUSUK esik (recall onceligi)")
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--no-tiling", action="store_true",
                        help="Sadece tam kare cikarimi yap (karsilastirma icin)")
    parser.add_argument("--preview-count", type=int, default=8)
    args = parser.parse_args()

    images_dir = os.path.join(args.output_dir, "images")
    ann_dir = os.path.join(args.output_dir, "annotations")
    preview_dir = os.path.join(args.output_dir, "preview")
    for d in (images_dir, ann_dir, preview_dir):
        os.makedirs(d, exist_ok=True)

    print("=" * 78)
    print("ADIM 1/3: Test videolarindan seyrek kare ornekleme")
    print("=" * 78)
    video_names = sorted(f for f in os.listdir(args.input_dir)
                         if f.lower().endswith(VIDEO_EXTS))
    if not video_names:
        raise SystemExit(f"{args.input_dir} icinde video bulunamadi.")

    frame_records = []
    for video_name in video_names:
        video_path = os.path.join(args.input_dir, video_name)
        slug = slugify(video_name)
        reason = SELECTION_RATIONALE.get(video_name, "-")
        print(f"\n{video_name}\n  Secim gerekcesi: {reason}")
        saved = sample_frames(video_path, args.interval, images_dir, slug)
        for file_name, frame_idx, w, h in saved:
            frame_records.append({
                "file_name": file_name, "video": video_name,
                "frame_index": frame_idx, "width": w, "height": h,
            })

    # CVAT bir gorsel klasorunu alfabetik siralar; ayni sirayi kullaniyoruz
    frame_records.sort(key=lambda r: r["file_name"])
    print(f"\nToplam {len(frame_records)} kare -> {images_dir}")

    print("\n" + "=" * 78)
    print("ADIM 2/3: Yuksek recall on-etiketleme (dosemeli Grounding DINO ogretmeni)")
    print("=" * 78)
    teacher = TiledGroundingDino()
    total_boxes = 0
    start = time.time()

    for i, record in enumerate(frame_records, 1):
        frame = cv2.imread(os.path.join(images_dir, record["file_name"]))
        # Tam kare çıkarımı da yapılır: döşemenin kaç insan kazandırdığını ölçmek için
        full_boxes, _ = teacher.detect(frame, args.threshold, args.threshold)
        if args.no_tiling:
            boxes, scores = full_boxes, _
        else:
            boxes, scores = teacher.detect_tiled(
                frame, overlap=args.overlap,
                threshold=args.threshold, text_threshold=args.threshold,
            )
        record["boxes"] = boxes.tolist()
        record["scores"] = scores.tolist()
        record["fullframe_count"] = int(len(full_boxes))
        total_boxes += len(boxes)

        if i % 10 == 0 or i == len(frame_records):
            elapsed = time.time() - start
            print(f"  {i}/{len(frame_records)} kare | {total_boxes} kutu | "
                  f"{elapsed / i:.2f} s/kare | kalan ~{(len(frame_records) - i) * elapsed / i / 60:.1f} dk")

    print("\n" + "=" * 78)
    print("ADIM 3/3: CVAT ve COCO formatlarina aktarim")
    print("=" * 78)

    xml_path = os.path.join(ann_dir, "test_preannot_cvat.xml")
    build_cvat_xml(frame_records, "drone_person_test_gt").write(
        xml_path, encoding="utf-8", xml_declaration=True
    )
    zip_path = os.path.join(ann_dir, "test_preannot_cvat.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(xml_path, "annotations.xml")

    coco_path = os.path.join(ann_dir, "test_preannot_coco.json")
    with open(coco_path, "w", encoding="utf-8") as f:
        json.dump(build_coco(frame_records, "Drone person detection - DINO on-etiketleri"),
                  f, indent=2, ensure_ascii=False)

    index_path = os.path.join(args.output_dir, "frame_index.json")
    per_video = {}
    for r in frame_records:
        stats = per_video.setdefault(r["video"], {"frames": 0, "boxes": 0, "fullframe": 0})
        stats["frames"] += 1
        stats["boxes"] += len(r["boxes"])
        stats["fullframe"] += r["fullframe_count"]
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({
            "split": "test",
            "note": "Bu kareler egitimde kullanilmaz.",
            "sample_interval": args.interval,
            "preannotation": {
                "model": teacher.model_name,
                "text_prompt": teacher.text_prompt,
                "threshold": args.threshold,
                "tiled": not args.no_tiling,
                "overlap": args.overlap,
            },
            "per_video": per_video,
            "frames": [{k: v for k, v in r.items() if k != "boxes" and k != "scores"}
                       for r in frame_records],
        }, f, indent=2, ensure_ascii=False)

    # Gorsel kontrol icin en kalabalik kareleri onizle
    busiest = sorted(frame_records, key=lambda r: -len(r["boxes"]))[:args.preview_count]
    for record in busiest:
        frame = cv2.imread(os.path.join(images_dir, record["file_name"]))
        annotated = draw_detections(frame, record["boxes"], record["scores"])
        scale = 1600 / max(annotated.shape[1], annotated.shape[0])
        if scale < 1:
            annotated = cv2.resize(annotated, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(preview_dir, record["file_name"]), annotated)

    header = (f"\n{'Video':<34} | {'Kare':>5} | {'Tam kare':>8} | {'Dosemeli':>8} | "
              f"{'Kazanc':>7} | {'Kutu/kare':>9}")
    print(header)
    print("-" * (len(header) - 1))
    total_full = sum(s["fullframe"] for s in per_video.values())
    for video, stats in per_video.items():
        gain = stats["boxes"] / stats["fullframe"] if stats["fullframe"] else float("inf")
        print(f"{video:<34} | {stats['frames']:>5} | {stats['fullframe']:>8} | "
              f"{stats['boxes']:>8} | {gain:>6.2f}x | {stats['boxes'] / stats['frames']:>9.2f}")
    print("-" * (len(header) - 1))
    print(f"{'TOPLAM':<34} | {len(frame_records):>5} | {total_full:>8} | {total_boxes:>8} | "
          f"{total_boxes / max(total_full, 1):>6.2f}x | {total_boxes / len(frame_records):>9.2f}")
    print(f"\nDoseme sayesinde {total_boxes - total_full} ek aday kutu bulundu.")
    print("Bu kutular DOGRULANMAMISTIR; bir kismi yanlis pozitiftir ve CVAT'ta silinecektir.")

    print(f"\nCiktilar:")
    print(f"  Gorseller (CVAT'a yuklenecek) : {images_dir}")
    print(f"  CVAT on-etiket (zip)          : {zip_path}")
    print(f"  CVAT on-etiket (xml)          : {xml_path}")
    print(f"  COCO on-etiket                : {coco_path}")
    print(f"  Kare indeksi                  : {index_path}")
    print(f"  Onizleme (gorsel kontrol)     : {preview_dir}")
    print(f"\nCVAT akisi: yeni task olustur -> '{images_dir}' klasorundeki gorselleri yukle")
    print(f"            -> Actions > Upload annotations > 'CVAT 1.1' -> {zip_path}")


if __name__ == "__main__":
    main()
