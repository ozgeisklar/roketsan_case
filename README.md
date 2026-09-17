# Drone Videolarinda Insan Tespiti

Bu repo, Kaggle Drone Videos veri setinde **insan (person)** sinifini tespit etmek ve lokalize etmek icin hazirlanan prototipi icerir. Calisma; baseline modeller, dosemeli Grounding DINO, aktif ogrenme/fine-tune ve BoT-SORT tabanli zamansal iyilestirme adimlarindan olusur.

## 1. Kurulum

Python ortamini aktiflestirin:

```bash
conda activate roketsan
```

Gerekli temel paketler:

```bash
pip install ultralytics transformers torch torchvision opencv-python pycocotools pillow numpy matplotlib
```

Not: CUDA destekli PyTorch kurulumu sisteminize gore degisebilir. Lokal GPU kullanilacaksa PyTorch'un CUDA ile uyumlu kuruldugundan emin olun.

## 2. Veri ve Agirlik Dizini

Beklenen ana dizin yapisi:

```text
roketsan_case/
  dataset/
    archive/
      test/
        Stockflue Flyaround.mp4
        Surenen Pass Trail Running.mp4
    test/
      images/
      annotations/
        instances_filtered.json
  weights/
    yolo11x.pt
    yolo26x.pt
    yolo11x_tuned_round2.pt
    yolo26x_tuned_round2.pt
    dino_tuned_round2/
  tracking/
  results/
```

Kaggle uzerinde egitim icin kullanilan notebook'lar:

- `kaggle_active_learning_selection.ipynb`
- `kaggle_finetune_yolo.ipynb`
- `kaggle_finetune_dino_active.ipynb`

## 3. Envanter Analizi

Videolarin cozunurluk, frame sayisi, tespit yogunlugu ve kutu boyutu dagilimini cikarmak icin:

```bash
python3 analyze_video_inventory.py
```

Cikti:

```text
results/video_inventory.json
results/video_inventory_summary.md
```

## 4. Test Seti Hazirlama

Test videolarindan seyrek frame orneklemek ve DINO ile on-etiket uretmek icin:

```bash
python3 prepare_test_set.py
```

Bu adim:

- test frame'lerini `dataset/test/images/` altina yazar,
- CVAT icin on-etiket XML/ZIP uretir,
- COCO formatinda annotation dosyasi olusturur.

CVAT uzerinde elle duzeltilen final annotation:

```text
dataset/test/annotations/instances_filtered.json
```

## 5. Baseline ve Fine-tuned Model Degerlendirmesi

Mevcut ground-truth uzerinde modelleri degerlendirmek icin:

```bash
python3 evaluate_on_gt.py \
  --models yolo11x_640,yolo11x_1536,yolo26x_640,yolo26x_640,dino_full,dino_tiled
```

Ciktilar:

```text
results/evaluation/summary.md
results/evaluation/summary_filtered.md
results/evaluation/summary_tuned.md
results/evaluation/summary_tuned_round2.md
```

## 6. Tracking Icin Detection Cache Uretimi

Tracking adiminda detector tekrar tekrar calistirilmesin diye once her frame icin detection cache uretilir.

Fine-tuned modeller icin:

```bash
python3 tracking/detect_cache.py \
  --models yolo11x_1536,yolo26x_1536,dino_full,dino_tiled \
  --videos Stockflue_Flyaround,Surenen_Pass_Trail_Running
```

Stockflue nitel baseline videolari icin baseline cache uretmek isterseniz:

```bash
python3 tracking/detect_cache.py \
  --models yolo11x_1536_baseline,yolo26x_1536_baseline,dino_full_baseline,dino_tiled_baseline \
  --videos Stockflue_Flyaround
```

Cikti:

```text
results/tracking/detections/
```

## 7. BoT-SORT Tracking

Cache'lenmis detection ciktisini BoT-SORT ile zamansal olarak iyilestirmek icin:

```bash
python3 tracking/offline_tracker.py \
  --models yolo11x_1536,yolo26x_1536,dino_full,dino_tiled \
  --videos Stockflue_Flyaround,Surenen_Pass_Trail_Running \
  --stages raw,nms,botsort
```

Cikti:

```text
results/tracking/tracked/
```

BoT-SORT burada yeni bir detector olarak kullanilmaz. Detector kutulari cache'ten okunur, BoT-SORT bu kutulari ardışık frame'lerde ayni kisilere ait track'lere baglar. Kisa bosluklar interpolasyonla doldurulur ve skorlar track guvenine gore yeniden duzenlenir.

## 8. Tracking Degerlendirmesi

Tracking sonrasi sonuclari COCO metrikleriyle degerlendirmek icin:

```bash
python3 tracking/evaluate_tracking.py \
  --models yolo11x_1536,yolo26x_1536,dino_full,dino_tiled \
  --videos Stockflue_Flyaround,Surenen_Pass_Trail_Running \
  --stages botsort
```

Cikti:

```text
results/evaluation/summary_tracking.md
```

## 9. Nitel Karsilastirma Videolari

Stockflue Flyaround videosu icin baseline, fine-tuned ve tracking ciktisi uretmek icin:

```bash
python3 tracking/render_qualitative_videos.py --width 1280
```

Cikti klasoru:

```text
results/qualitative/stockflue/
```

Ornek ciktilar:

```text
baseline__yolo11x_baseline.mp4
baseline__yolo26x_baseline.mp4
baseline__dino_full_baseline.mp4
baseline__dino_tiled_baseline.mp4
tuned__yolo11x_fine_tuned.mp4
tuned__yolo26x_fine_tuned.mp4
tuned__dino_full_fine_tuned.mp4
tuned__dino_tiled_fine_tuned.mp4
tracking__yolo11x___bot_sort.mp4
tracking__yolo26x___bot_sort.mp4
tracking__dino_full___bot_sort.mp4
tracking__dino_tiled___bot_sort.mp4
```

## 10. Raporlar

Ana rapor:

```text
RAPOR_DETAYLI.md
```

Daha kisa ozet rapor:

```text
RAPOR_KISA.md
```

Onemli sonuc dosyalari:

```text
results/evaluation/summary.md
results/evaluation/summary_filtered.md
results/evaluation/summary_tuned.md
results/evaluation/summary_tuned_round2.md
results/evaluation/summary_tracking.md
```


## 11. Notlar

- YOLO 640 varyantlari fine-tune edilmedi; baseline asamasinda 1536 varyantlari acik sekilde daha iyi performans verdi.
- DINO tiled daha dogru sonuclar uretir ancak hesaplama maliyeti yuksektir.
- Tracking metrikleri MOT metrikleriyle degil COCO detection metrikleriyle raporlandi; cunku ground-truth icinde track ID bilgisi yoktur.


