"""
Döşemeli (tiled) Grounding DINO çıkarım modülü.

Grounding DINO'nun image processor'ı girdiyi shortest_edge=800 / longest_edge=1333'e
küçültür. 4K (3840x2160) bir drone karesi bu yüzden modele girmeden önce 1333x750'ye,
yani 2.88 kat küçülerek girer: 60 piksellik bir insan 21 piksele iner ve kaybolur.

Bu modül kareyi örtüşen parçalara bölüp her parçayı ayrı ayrı modele vererek etkin
çözünürlük kaybını azaltır, sonra parça sonuçlarını global koordinatlara taşıyıp
NMS + kapsama (containment) bastırması ile birleştirir.

Etiketleme çevrimdışı yapıldığı için burada hız değil recall önemlidir; pahalı
öğretmenin kalitesi daha sonra ucuz bir öğrenci modele damıtılacaktır.
"""

import inspect

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

DEFAULT_MODEL = "IDEA-Research/grounding-dino-base"

# Grounding DINO image processor'ının hedef boyutları (etkin çözünürlük hesabı için)
PROCESSOR_SHORTEST_EDGE = 800
PROCESSOR_LONGEST_EDGE = 1333


def auto_grid(width, height):
    """
    Çözünürlüğe göre döşeme ızgarasını seçer.

    4K kareler 3x3'e bölünür (parça ~1536x864, yani modele neredeyse tam çözünürlükte
    girer). 720p kareler 2x2'ye bölünür: parçalar processor tarafından bu kez
    büyütülür, bu da küçük insanları belirginleştirir. 406x720 gibi zaten küçük
    kareler tek parça işlenir, bölmek fayda sağlamaz.
    """
    long_side = max(width, height)
    if long_side >= 3000:
        return 3, 3
    if long_side >= 1200:
        return 2, 2
    return 1, 1


def effective_scale(width, height):
    """Bir karenin processor tarafından uygulanan yeniden boyutlandırma oranı."""
    short_side, long_side = min(width, height), max(width, height)
    scale = PROCESSOR_SHORTEST_EDGE / short_side
    if long_side * scale > PROCESSOR_LONGEST_EDGE:
        scale = PROCESSOR_LONGEST_EDGE / long_side
    return scale


def tile_windows(width, height, rows, cols, overlap=0.2):
    """
    Örtüşen ve tümü eşit boyutta olan parça pencereleri üretir.

    Parçaların eşit boyutta olması hem batch'lemeyi güvenli kılar hem de processor'ın
    her parçaya aynı ölçeklemeyi uygulamasını garanti eder. Örtüşme, parça sınırından
    ikiye bölünen bir insanın komşu parçada bütün olarak görünmesini sağlar.
    """
    if rows == 1 and cols == 1:
        return [(0, 0, width, height)]

    tile_w = min(width, int(round(width / cols * (1 + overlap))))
    tile_h = min(height, int(round(height / rows * (1 + overlap))))
    step_x = (width - tile_w) / (cols - 1) if cols > 1 else 0
    step_y = (height - tile_h) / (rows - 1) if rows > 1 else 0

    windows = []
    for r in range(rows):
        for c in range(cols):
            x1 = int(round(c * step_x))
            y1 = int(round(r * step_y))
            windows.append((x1, y1, x1 + tile_w, y1 + tile_h))
    return windows


def geometric_filter(boxes, scores, max_area_px, min_area_px=16.0, max_aspect=4.0):
    """
    Geometrik olarak insan olamayacak kutuları atar.

    Grounding DINO, düz su/gökyüzü/kar gibi dokusuz bölgelerde kesitin tamamını "person"
    olarak kutulama eğiliminde. Bu halüsinasyonlar 0.30+ skor alabildiği, gerçek uzak
    insanlar ise 0.17-0.25 aldığı için eşik yükseltmek işe yaramaz: önce gerçek insanları
    kaybedersin, çöp kalır. Ayırt edici özellik skor değil geometridir.

    max_area_px: mutlak alan üst sınırı. Kasıtlı olarak oran değil piksel: sınır tek bir
        ölçek varsayımından (bir insan bir döşeme parçasının belirli bir oranından büyük
        olamaz) türetilip TÜM geçişlere aynı şekilde uygulanmalı. Tam kare geçişine kare
        oranı uygulanırsa bütçe çok geniş kalır ve tüm sahneyi saran kutular kaçar.
    min_area_px: birkaç piksellik gürültü kutuları atılır
    max_aspect:  insan bu kadar kat "geniş" olamaz (w/h üst sınırı)
    """
    if len(boxes) == 0:
        return boxes, scores

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    areas = widths * heights

    keep = (
        (areas <= max_area_px)
        & (areas >= min_area_px)
        & (widths > 0)
        & (heights > 0)
        & (widths <= heights * max_aspect)
    )
    return boxes[keep], scores[keep]


def suppress_contained(boxes, scores, containment_thresh=0.80):
    """
    Düşük skorlu bir kutu, yüksek skorlu bir kutunun içinde büyük oranda kalıyorsa atar.

    NMS bunu yakalayamaz: parça sınırında kesilen bir insanın yarım kutusu ile tam
    kutusunun IoU'su düşük kalır, ama yarım kutu tam kutunun içindedir.
    """
    if len(boxes) == 0:
        return boxes, scores

    order = np.argsort(-scores)
    boxes, scores = boxes[order], scores[order]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = np.ones(len(boxes), dtype=bool)

    for i in range(len(boxes)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(boxes)):
            if not keep[j] or areas[j] <= 0:
                continue
            ix1 = max(boxes[i, 0], boxes[j, 0])
            iy1 = max(boxes[i, 1], boxes[j, 1])
            ix2 = min(boxes[i, 2], boxes[j, 2])
            iy2 = min(boxes[i, 3], boxes[j, 3])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter / areas[j] >= containment_thresh:
                keep[j] = False

    return boxes[keep], scores[keep]


class TiledGroundingDino:
    """Donmuş Grounding DINO öğretmeni: tam kare ve döşemeli çıkarım sağlar."""

    def __init__(self, model_name=DEFAULT_MODEL, device=None, use_fp16=False,
                 text_prompt="person."):
        """
        use_fp16 varsayılan olarak kapalıdır. GTX 1650 (Turing TU117) üzerinde fp16,
        406x720 gibi sıra dışı en-boy oranlarında hata vermeden SIFIR tespit döndürdü
        (fp32 aynı karelerde 0.77 skorla insan buluyordu). Sessiz başarısızlık veri
        üretim hattında en tehlikeli hata türü olduğu için varsayılan fp32'dir; fp16'yı
        açmadan önce iki dtype'ı aynı kareler üzerinde karşılaştırıp doğrulayın.
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if (use_fp16 and self.device == "cuda") else torch.float32
        self.text_prompt = text_prompt

        print(f"{model_name} yükleniyor... (cihaz: {self.device}, dtype: {self.dtype})")
        self.processor = AutoProcessor.from_pretrained(model_name)
        # transformers v5 `dtype`, v4 `torch_dtype` bekliyor; Kaggle/Colab sürümleri farklı
        try:
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                model_name, dtype=self.dtype
            )
        except TypeError:
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                model_name, torch_dtype=self.dtype
            )
        self.model = self.model.to(self.device).eval()

        # v5 `threshold`, v4 `box_threshold` kullanıyor
        post_process_params = inspect.signature(
            self.processor.post_process_grounded_object_detection
        ).parameters
        self._box_thresh_kwarg = "threshold" if "threshold" in post_process_params else "box_threshold"

    def _forward(self, crops_bgr, threshold, text_threshold):
        """Bir grup BGR kesiti modelden geçirip her biri için (boxes, scores) döndürür."""
        pil_images = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops_bgr]
        inputs = self.processor(
            images=pil_images,
            text=[self.text_prompt] * len(pil_images),
            return_tensors="pt",
        ).to(self.device)
        if self.dtype == torch.float16:
            inputs["pixel_values"] = inputs["pixel_values"].half()

        with torch.inference_mode():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs["input_ids"],
            text_threshold=text_threshold,
            target_sizes=[(c.shape[0], c.shape[1]) for c in crops_bgr],
            **{self._box_thresh_kwarg: threshold},
        )
        return [
            (
                r["boxes"].float().cpu().numpy().reshape(-1, 4),
                r["scores"].float().cpu().numpy().reshape(-1),
            )
            for r in results
        ]

    def detect(self, frame_bgr, threshold=0.25, text_threshold=0.25):
        """Baseline davranışı: tek geçişte tam kare çıkarımı."""
        return self._forward([frame_bgr], threshold, text_threshold)[0]

    def detect_tiled(self, frame_bgr, grid=None, overlap=0.2, threshold=0.15,
                     text_threshold=0.15, iou_thresh=0.55, include_full_frame=True,
                     max_area_ratio=0.25, batch_size=1):
        """
        Döşemeli çıkarım: parçalar + (opsiyonel) tam kare, ardından NMS ile birleştirme.

        Tam kare de dahil edilir çünkü parçalara sığmayan büyük/yakın insanları yakalar;
        parçalar ise küçük/uzak insanları yakalar. İkisi birbirini tamamlar.

        Geometrik filtre BİRLEŞTİRMEDEN ÖNCE, her geçişe aynı mutlak alan sınırıyla
        uygulanır. Sınır parça boyutundan türetilir: bir insan bir parçanın
        max_area_ratio oranından büyük olamaz. Aynı sınır tam kare geçişine de uygulanır,
        aksi halde tüm sahneyi saran halüsinasyon kutuları süzgeçten kaçar.
        """
        h, w = frame_bgr.shape[:2]
        rows, cols = grid or auto_grid(w, h)
        windows = tile_windows(w, h, rows, cols, overlap)

        tile_w = windows[0][2] - windows[0][0]
        tile_h = windows[0][3] - windows[0][1]
        max_area_px = tile_w * tile_h * max_area_ratio

        crops, offsets = [], []
        for (x1, y1, x2, y2) in windows:
            crops.append(frame_bgr[y1:y2, x1:x2])
            offsets.append((x1, y1))
        if include_full_frame and (rows, cols) != (1, 1):
            crops.append(frame_bgr)
            offsets.append((0, 0))

        all_boxes, all_scores = [], []
        for start in range(0, len(crops), batch_size):
            chunk = crops[start:start + batch_size]
            chunk_offsets = offsets[start:start + batch_size]
            for (boxes, scores), (ox, oy) in zip(
                self._forward(chunk, threshold, text_threshold), chunk_offsets
            ):
                boxes, scores = geometric_filter(boxes, scores, max_area_px)
                if len(boxes) == 0:
                    continue
                boxes = boxes.copy()
                boxes[:, [0, 2]] += ox
                boxes[:, [1, 3]] += oy
                all_boxes.append(boxes)
                all_scores.append(scores)

        if not all_boxes:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

        boxes = np.clip(np.concatenate(all_boxes), [0, 0, 0, 0], [w, h, w, h]).astype(np.float32)
        scores = np.concatenate(all_scores).astype(np.float32)

        keep = nms(torch.from_numpy(boxes), torch.from_numpy(scores), iou_thresh).numpy()
        boxes, scores = boxes[keep], scores[keep]
        return suppress_contained(boxes, scores)


def tiling_report(width, height, overlap=0.2):
    """Bir çözünürlük için döşemenin etkin çözünürlük kazancını raporlar."""
    rows, cols = auto_grid(width, height)
    windows = tile_windows(width, height, rows, cols, overlap)
    tw, th = windows[0][2] - windows[0][0], windows[0][3] - windows[0][1]
    return {
        "resolution": f"{width}x{height}",
        "grid": f"{rows}x{cols}",
        "tile_size": f"{tw}x{th}",
        "forward_passes": len(windows) + (1 if (rows, cols) != (1, 1) else 0),
        "fullframe_scale": round(effective_scale(width, height), 3),
        "tile_scale": round(effective_scale(tw, th), 3),
        "resolution_gain": round(effective_scale(tw, th) / effective_scale(width, height), 2),
    }


def draw_detections(frame, boxes, scores, color=(0, 255, 0), label_prefix="person"):
    """Kutuları kare üzerine çizer; çözünürlükle ölçeklenen kalınlık/font kullanır."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]
    thickness = max(1, round((h + w) / 1600))
    font_scale = max(0.4, thickness * 0.4)

    for (x1, y1, x2, y2), score in zip(boxes, scores):
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(annotated, p1, p2, color, thickness)
        label = f"{label_prefix} {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        top = p1[1] - th - baseline
        if top < 0:
            top = p1[1] + th + baseline
        cv2.rectangle(annotated, (p1[0], top - th - baseline), (p1[0] + tw, top), color, -1)
        cv2.putText(annotated, label, (p1[0], top - baseline), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return annotated


if __name__ == "__main__":
    # Arşivdeki çözünürlükler için döşeme planını ve çözünürlük kazancını yazdır
    print(f"{'Çözünürlük':>12} | {'Izgara':>6} | {'Parça':>10} | {'Geçiş':>5} | "
          f"{'Tam kare':>8} | {'Parça':>6} | {'Kazanç':>6}")
    print("-" * 72)
    for width, height in [(3840, 2160), (1280, 720), (720, 1280), (406, 720)]:
        r = tiling_report(width, height)
        print(f"{r['resolution']:>12} | {r['grid']:>6} | {r['tile_size']:>10} | "
              f"{r['forward_passes']:>5} | {r['fullframe_scale']:>8} | {r['tile_scale']:>6} | "
              f"{r['resolution_gain']:>5}x")
