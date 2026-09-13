import cv2
import os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


def process_drone_videos(input_dir, output_dir, model_name="IDEA-Research/grounding-dino-base",
                         conf_thresh=0.25, text_prompt="person."):
    """
    Belirtilen klasördeki videoları Grounding DINO modeli ile kare kare (tek tek) işler,
    sadece insanları tespit eder ve çıktı klasörüne kaydeder.
    """
    # Çıktı klasörünü oluştur
    os.makedirs(output_dir, exist_ok=True)

    # Cihaz seçimi (GPU varsa CUDA, yoksa CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{model_name} modeli yükleniyor... (Cihaz: {device})")

    # Modeli ve işlemciyi Hugging Face'ten yükle
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(device)
    model.eval()

    # Desteklenen video formatları
    video_exts = ('.mp4', '.avi', '.mov', '.mkv')

    # Girdi klasöründeki videoları bul
    video_files = [f for f in os.listdir(input_dir) if f.lower().endswith(video_exts)]

    if not video_files:
        print(f"{input_dir} dizininde video bulunamadı.")
        return

    print(f"Toplam {len(video_files)} video bulundu. İşlem başlıyor...\n")

    for video_file in video_files:
        input_path = os.path.join(input_dir, video_file)
        output_path = os.path.join(output_dir, f"dino_out_{video_file}")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Hata: {video_file} açılamadı.")
            continue

        # Video özelliklerini al
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Video kaydediciyi ayarla
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"İşleniyor: {video_file} | Çözünürlük: {width}x{height} | FPS: {fps:.2f} | Toplam Frame: {total_frames}")

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # BGR frame'i RGB PIL Image'a dönüştür
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
                target_sizes=[(height, width)]
            )[0]

            # Bounding box'ları frame üzerine çiz
            annotated_frame = frame.copy()
            boxes = results["boxes"].cpu().numpy()
            scores = results["scores"].cpu().numpy()

            for box, score in zip(boxes, scores):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"person {score:.2f}"
                cv2.putText(annotated_frame, label, (x1, max(y1 - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Çıktı videosuna yaz
            out.write(annotated_frame)

            # İlerleme durumunu konsola yazdır (Her 50 frame'de bir)
            if frame_count % 50 == 0:
                print(f"  [{video_file}] Frame: {frame_count}/{total_frames} işlendi...")

        # Kaynakları serbest bırak
        cap.release()
        out.release()
        print(f"Tamamlandı: {video_file} -> Çıktı: {output_path}\n")

    cv2.destroyAllWindows()
    print("Tüm videoların işlenmesi başarıyla tamamlandı!")


if __name__ == "__main__":
    # Girdi ve çıktı dizinleri
    INPUT_DIRECTORY = "archive/"  # Drone videolarının olduğu klasör
    OUTPUT_DIRECTORY = "results/baseline_dino/"  # DINO çıktılarının kaydedileceği klasör

    process_drone_videos(
        input_dir=INPUT_DIRECTORY,
        output_dir=OUTPUT_DIRECTORY,
        model_name="IDEA-Research/grounding-dino-base",  # Dilerseniz daha hızlı olan "IDEA-Research/grounding-dino-tiny" de kullanabilirsiniz
        conf_thresh=0.25,  # YOLO ile birebir aynı güven eşiği
        text_prompt="person."  # Tespit edilmek istenen sınıf
    )
