import cv2
import os
from ultralytics import YOLO

def process_drone_videos(input_dir, output_dir, model_name="yolo11x.pt", conf_thresh=0.25):
    """
    Belirtilen klasördeki videoları YOLO modeli ile işler ve sadece insanları (class 0)
    tespit ederek çıktı klasörüne kaydeder.
    """
    # Çıktı klasörünü oluştur
    os.makedirs(output_dir, exist_ok=True)
    
    # Modeli yükle (İlk çalışmada yolo11x.pt dosyasını otomatik indirecektir)
    print(f"{model_name} modeli yükleniyor...")
    model = YOLO(model_name)
    
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
        output_path = os.path.join(output_dir, f"yolo11x_out_{video_file}")
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Hata: {video_file} açılamadı.")
            continue
            
        # Video özelliklerini al
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Video kaydediciyi ayarla (MP4V codec genel uyumludur)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"İşleniyor: {video_file} | Çözünürlük: {width}x{height} | FPS: {fps:.2f} | Toplam Frame: {total_frames}")
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # YOLO ile tahmin yap
            # classes=[0] -> Sadece 0. sınıfı (Person) tespit et
            # conf=conf_thresh -> Güven skoru eşiği
            results = model.predict(frame, imgsz=1536, classes=[0], conf=conf_thresh, verbose=False)
            
            # Bounding box'ları frame üzerine çiz (plot)
            annotated_frame = results[0].plot()
            
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
    # Kendi dizin yollarınızı buraya girin
    INPUT_DIRECTORY = "archive/test/"  # Drone videolarınızın olduğu klasör
    OUTPUT_DIRECTORY = "results/baseline_yolo26x1536/"  # Çıktıların kaydedileceği klasör
    
    process_drone_videos(
        input_dir=INPUT_DIRECTORY,
        output_dir=OUTPUT_DIRECTORY,
        model_name="yolo26x.pt", # En büyük ve en performanslı model
        conf_thresh=0.25 # Güven eşiğini başlangıç için 0.25 tuttuk, duruma göre artırıp azaltabilirsiniz.
    )