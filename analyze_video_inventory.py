import os
import json
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

def analyze_videos(input_dir="archive", output_dir="results", model_name="IDEA-Research/grounding-dino-base", sample_interval=30, conf_thresh=0.25, text_prompt="person."):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"dino yükleniyor: {model_name} (Cihaz: {device})...")

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(device)
    model.eval()
    
    video_exts = ('.mp4', '.avi', '.mov', '.mkv')
    video_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(video_exts)])
    
    if not video_files:
        print(f"{input_dir} içinde video bulunamadı!")
        return

    print(f"Toplam {len(video_files)} video incelenecek. Örnekleme aralığı: her {sample_interval}. kare (saniyede ~1 kare)\n")
    
    inventory_data = []
    
    for video_name in video_files:
        video_path = os.path.join(input_dir, video_name)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Hata: {video_name} açılamadı.")
            continue
            
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if fps > 0 else 0
        
        frame_idx = 0
        sampled_frames = 0
        frames_with_det = 0
        box_heights = []
        box_widths = []
        box_areas = []
        rel_heights = []
        confidences = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % sample_interval == 0:
                sampled_frames += 1
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image_rgb)

                # Modele girdi hazırla
                inputs = processor(images=pil_image, text=text_prompt, return_tensors="pt").to(device)

                # DINO ile tahmin yap
                with torch.no_grad():
                   outputs = model(**inputs)

                # Tahminleri orijinal çözünürlüğe göre filtrele
                results = processor.post_process_grounded_object_detection(
                  outputs,
                  input_ids=inputs.input_ids,
                  threshold=conf_thresh,
                  text_threshold=conf_thresh,
                  target_sizes=[(h, w)]
                )[0]

                boxes = results["boxes"].cpu().numpy()
                scores = results["scores"].cpu().numpy()
                
                if len(boxes) > 0:
                    frames_with_det += 1
                    for (x1, y1, x2, y2), cf in zip(boxes, scores):
                        bw = max(0.0, float(x2 - x1))
                        bh = max(0.0, float(y2 - y1))
                        area = bw * bh
                        box_widths.append(bw)
                        box_heights.append(bh)
                        box_areas.append(area)
                        rel_heights.append((bh / h) * 100.0)
                        confidences.append(float(cf))
                        
            frame_idx += 1
            
        cap.release()
        
        total_dets = len(box_heights)
        small_count = sum(1 for a in box_areas if a < 32 * 32)
        med_count = sum(1 for a in box_areas if 32 * 32 <= a <= 96 * 96)
        large_count = sum(1 for a in box_areas if a > 96 * 96)
        
        small_pct = (small_count / total_dets * 100.0) if total_dets > 0 else 0.0
        med_pct = (med_count / total_dets * 100.0) if total_dets > 0 else 0.0
        large_pct = (large_count / total_dets * 100.0) if total_dets > 0 else 0.0
        
        active_ratio = (frames_with_det / sampled_frames * 100.0) if sampled_frames > 0 else 0.0
        
        summary = {
            "video_name": video_name,
            "resolution": f"{w}x{h}",
            "width": w,
            "height": h,
            "fps": round(fps, 1),
            "total_frames": total_frames,
            "duration_sec": round(duration_sec, 1),
            "sampled_frames": sampled_frames,
            "frames_with_det": frames_with_det,
            "active_ratio_pct": round(active_ratio, 1),
            "total_detections": total_dets,
            "avg_dets_per_frame": round(total_dets / sampled_frames, 2) if sampled_frames > 0 else 0,
            "median_box_height_px": round(float(np.median(box_heights)), 1) if box_heights else 0,
            "mean_box_height_px": round(float(np.mean(box_heights)), 1) if box_heights else 0,
            "min_box_height_px": round(float(np.min(box_heights)), 1) if box_heights else 0,
            "max_box_height_px": round(float(np.max(box_heights)), 1) if box_heights else 0,
            "median_rel_height_pct": round(float(np.median(rel_heights)), 2) if rel_heights else 0,
            "coco_small_pct": round(small_pct, 1),
            "coco_med_pct": round(med_pct, 1),
            "coco_large_pct": round(large_pct, 1),
            "mean_conf": round(float(np.mean(confidences)), 2) if confidences else 0
        }
        
        inventory_data.append(summary)
        print(f"[{video_name}] Çözünürlük: {w}x{h} | Süre: {duration_sec:.1f}s | "
              f"Örnek Kare: {sampled_frames} | İnsan Bulunan Kare: %{active_ratio:.1f} | "
              f"Toplam İnsan: {total_dets} | Medyan Yükseklik: {summary['median_box_height_px']}px "
              f"(Small: %{small_pct:.0f}, Med: %{med_pct:.0f}, Large: %{large_pct:.0f})")

    # JSON olarak kaydet
    json_path = os.path.join(output_dir, "video_inventory.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(inventory_data, f, indent=2, ensure_ascii=False)
        
    # Markdown tablosu oluştur
    md_path = os.path.join(output_dir, "video_inventory_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Drone Video Envanteri ve Ölçek Analizi\n\n")
        f.write("| Video Adı | Çözünürlük | Süre (s) | Toplam Kare | Örneklenen | İnsanlı Kare % | Toplam Tespit | Medyan Yükseklik | COCO Small % | COCO Med % | COCO Large % |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
        for d in inventory_data:
            f.write(f"| `{d['video_name']}` | {d['resolution']} | {d['duration_sec']}s | {d['total_frames']} | "
                    f"{d['sampled_frames']} | %{d['active_ratio_pct']} | {d['total_detections']} | "
                    f"{d['median_box_height_px']} px ({d['median_rel_height_pct']}%) | "
                    f"%{d['coco_small_pct']} | %{d['coco_med_pct']} | %{d['coco_large_pct']} |\n")
                    
    print(f"\nSonuçlar kaydedildi:\n- {json_path}\n- {md_path}\n")

if __name__ == "__main__":
    analyze_videos()
