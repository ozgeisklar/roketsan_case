import argparse
import json
import os
import time
from collections import defaultdict

import cv2
import numpy as np

# Modeller
from ultralytics import YOLO
from tiled_dino import TiledGroundingDino

def get_yolo_predictor(model_path, imgsz):
    model = YOLO(model_path)
    def predict(image):
        start_time = time.time()
        # AP hesabi icin dusuk conf (0.01) kullaniyoruz
        results = model.predict(image, imgsz=imgsz, classes=[0], verbose=False, conf=0.01)
        inference_time = time.time() - start_time
        
        boxes = []
        scores = []
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
        return boxes, scores, inference_time
    return predict

def get_dino_predictor(tiled=False):
    # fp16 sessiz hata verdigi icin fp32 kullaniyoruz
    dino = TiledGroundingDino(use_fp16=False)
    def predict(image):
        start_time = time.time()
        # AP hesabi icin dusuk esik
        if tiled:
            boxes, scores = dino.detect_tiled(image, threshold=0.05, text_threshold=0.05)
        else:
            boxes, scores = dino.detect(image, threshold=0.05, text_threshold=0.05)
        inference_time = time.time() - start_time
        return boxes, scores, inference_time
    return predict

CONFIGS = {
    "yolo11x_640": lambda: get_yolo_predictor("yolo11x.pt", 640),
    "yolo11x_1536": lambda: get_yolo_predictor("yolo11x.pt", 1536),
    "yolo26x_640": lambda: get_yolo_predictor("yolo26x.pt", 640),
    "yolo26x_1536": lambda: get_yolo_predictor("yolo26x.pt", 1536),
    "dino_full": lambda: get_dino_predictor(tiled=False),
    "dino_tiled": lambda: get_dino_predictor(tiled=True),
}

def main():
    parser = argparse.ArgumentParser(description="Modellerin GT uzerinde performans olcumu")
    parser.add_argument("--gt", default="dataset/test/annotations/instances_default.json", help="COCO GT JSON")
    parser.add_argument("--images", default="dataset/test/images", help="Test gorselleri klasoru")
    parser.add_argument("--out", default="results/evaluation", help="Cikti klasoru")
    parser.add_argument("--models", default="yolo11x_640,yolo11x_1536,yolo26x_640,yolo26x_1536,dino_full,dino_tiled", help="Virgulle ayrilmis model listesi")
    args = parser.parse_args()
    
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        raise SystemExit("pycocotools kurulu degil. Lutfen 'pip install pycocotools' calistirin.")
        
    os.makedirs(args.out, exist_ok=True)
    
    print(f"GT yukleniyor: {args.gt}")
    coco_gt = COCO(args.gt)
    image_ids = coco_gt.getImgIds()
    print(f"Toplam {len(image_ids)} test karesi bulundu.")
    
    models_to_run = args.models.split(",")
    results_summary = {}
    
    for model_name in models_to_run:
        print(f"\n{'='*60}\nModel Degerlendiriliyor: {model_name}\n{'='*60}")
        if model_name not in CONFIGS:
            print(f"HATA: Bilinmeyen model '{model_name}'. Atlaniyor.")
            continue
            
        predictor = CONFIGS[model_name]()
        coco_dt = []
        total_time = 0
        
        for i, img_id in enumerate(image_ids):
            img_info = coco_gt.loadImgs(img_id)[0]
            # CVAT export path handling
            img_filename = img_info['file_name'].split('/')[-1] 
            img_path = os.path.join(args.images, img_filename)
            
            if not os.path.exists(img_path):
                # Fallback
                img_path = os.path.join(args.images, img_info['file_name'])
                
            img = cv2.imread(img_path)
            if img is None:
                print(f"UYARI: Gorsel okunamadi {img_path}")
                continue
                
            boxes, scores, inf_time = predictor(img)
            total_time += inf_time
            
            for box, score in zip(boxes, scores):
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                coco_dt.append({
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(score)
                })
                
            if (i + 1) % 10 == 0 or (i + 1) == len(image_ids):
                print(f"  Islenen: {i+1}/{len(image_ids)} | Ort. Sure: {total_time/(i+1):.3f}s")
                
        avg_time = total_time / len(image_ids)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        print(f"\nCikarim tamamlandi. Hız: {avg_time:.3f}s/kare ({fps:.1f} FPS)")
        
        dt_path = os.path.join(args.out, f"{model_name}_predictions.json")
        with open(dt_path, "w") as f:
            json.dump(coco_dt, f)
            
        if len(coco_dt) == 0:
            print("UYARI: Hic tespit yapilmadi!")
            continue
            
        print("COCO Metrikleri hesaplaniyor...")
        coco_dt_obj = coco_gt.loadRes(dt_path)
        coco_eval = COCOeval(coco_gt, coco_dt_obj, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        stats = coco_eval.stats
        
        # Optimal Precision ve Recall degerlerini cikar (IoU=0.50 icin)
        # eval['precision'] boyutu: [T, R, K, A, M] -> [10, 101, 1, 4, 3]
        # T=0 (IoU=0.50), K=0 (Category=person), A=0 (All areas), M=2 (MaxDets=100)
        precisions = coco_eval.eval['precision'][0, :, 0, 0, 2]
        recalls = coco_eval.eval['params'].recThrs
        
        # F1 skoru maksimize eden noktayi bul
        best_f1 = 0
        best_p = 0
        best_r = 0
        for p, r in zip(precisions, recalls):
            if p > -1:
                f1 = 2 * p * r / (p + r + 1e-16)
                if f1 > best_f1:
                    best_f1 = f1
                    best_p = p
                    best_r = r
                    
        results_summary[model_name] = {
            "mAP_50_95": float(stats[0]),
            "mAP_50": float(stats[1]),
            "mAP_small": float(stats[3]),
            "mAP_medium": float(stats[4]),
            "mAP_large": float(stats[5]),
            "AR_100": float(stats[8]),
            "Optimal_Precision_50": float(best_p),
            "Optimal_Recall_50": float(best_r),
            "Optimal_F1_50": float(best_f1),
            "Inference_Time_s": float(avg_time),
            "FPS": float(fps)
        }
        
    summary_path = os.path.join(args.out, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=4)
        
    print("\n\n" + "="*100)
    print("DEGERLENDIRME OZETI")
    print("="*100)
    header = f"| {'Model':<15} | {'mAP@50':<8} | {'mAP@50-95':<9} | {'AP_Small':<8} | {'AP_Med':<8} | {'Opt. P':<8} | {'Opt. R':<8} | {'Opt. F1':<8} | {'Sure (s)':<8} | {'FPS':<5} |"
    print(header)
    print("|" + "-"*17 + "|" + "-"*10 + "|" + "-"*11 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*7 + "|")
    
    for m, s in results_summary.items():
        print(f"| {m:<15} | {s['mAP_50']:<8.3f} | {s['mAP_50_95']:<9.3f} | {s['mAP_small']:<8.3f} | {s['mAP_medium']:<8.3f} | {s['Optimal_Precision_50']:<8.3f} | {s['Optimal_Recall_50']:<8.3f} | {s['Optimal_F1_50']:<8.3f} | {s['Inference_Time_s']:<8.3f} | {s['FPS']:<5.1f} |")
    
    md_path = os.path.join(args.out, "summary.md")
    with open(md_path, "w") as f:
        f.write("# Baseline Model Performans Karsilastirmasi\n\n")
        f.write(header + "\n")
        f.write("|" + "-"*17 + "|" + "-"*10 + "|" + "-"*11 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*7 + "|\n")
        for m, s in results_summary.items():
            f.write(f"| {m:<15} | {s['mAP_50']:<8.3f} | {s['mAP_50_95']:<9.3f} | {s['mAP_small']:<8.3f} | {s['mAP_medium']:<8.3f} | {s['Optimal_Precision_50']:<8.3f} | {s['Optimal_Recall_50']:<8.3f} | {s['Optimal_F1_50']:<8.3f} | {s['Inference_Time_s']:<8.3f} | {s['FPS']:<5.1f} |\n")
            
    print(f"\nSonuclar {args.out} klasorune kaydedildi.")

if __name__ == "__main__":
    main()
