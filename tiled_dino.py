import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

DEFAULT_MODEL = "IDEA-Research/grounding-dino-base"


def auto_grid(width, height):
    long_side = max(width, height)
    if long_side >= 3000:
        return 3, 3
    if long_side >= 1200:
        return 2, 2
    return 1, 1



def tile_windows(width, height, rows, cols, overlap=0.2):
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

    def __init__(self, model_name=DEFAULT_MODEL, device=None, text_prompt="person."):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        self.text_prompt = text_prompt

        print(f"{model_name} yükleniyor... (cihaz: {self.device}, dtype: {self.dtype})")
        self.processor = AutoProcessor.from_pretrained(model_name)
        try:
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                model_name, dtype=self.dtype
            )
        except TypeError:
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                model_name, torch_dtype=self.dtype
            )
        self.model = self.model.to(self.device).eval()


    def _forward(self, crops_bgr, threshold, text_threshold):
        pil_images = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops_bgr]
        inputs = self.processor(
            images=pil_images,
            text=[self.text_prompt] * len(pil_images),
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs["input_ids"],
            text_threshold=text_threshold,
            target_sizes=[(c.shape[0], c.shape[1]) for c in crops_bgr],
            threshold=threshold,
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


