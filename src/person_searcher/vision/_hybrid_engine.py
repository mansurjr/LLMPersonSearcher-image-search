"""Hybrid face, clothing, and body Re-ID feature extraction and matching."""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

MODELS_DIR = Path(os.environ.get("LLM_PERSON_MODELS_DIR", str(Path.home() / ".cache" / "llm-person-searcher" / "models")))

MODEL_URLS: Dict[str, str] = {
    "face_detection_yunet_2023mar.onnx": (
        "https://huggingface.co/opencv/face_detection_yunet/resolve/main/"
        "face_detection_yunet_2023mar.onnx"
    ),
    "w600k_mbf.onnx": (
        "https://huggingface.co/deepghs/insightface/resolve/main/"
        "buffalo_s/w600k_mbf.onnx"
    ),
    "face_recognition_sface_2021dec.onnx": (
        "https://huggingface.co/opencv/opencv_zoo/resolve/main/"
        "models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
    ),
    "person_reid_youtu_2021nov.onnx": (
        "https://huggingface.co/opencv/opencv_zoo/resolve/main/"
        "models/person_reid_youtureid/person_reid_youtu_2021nov.onnx"
    ),
    "object_detection_nanodet_2022nov.onnx": (
        "https://huggingface.co/opencv/opencv_zoo/resolve/main/"
        "models/object_detection_nanodet/object_detection_nanodet_2022nov.onnx"
    ),
}

# Standard ArcFace 112x112 5-point landmark template
ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def ensure_models(models_dir: Path = MODELS_DIR) -> Dict[str, Path]:
    """Download any missing ONNX model weights into `models/`."""
    models_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    for filename, url in MODEL_URLS.items():
        fpath = models_dir / filename
        if not fpath.exists() or fpath.stat().st_size < 10_000:
            print(f"[ModelLoader] Downloading {filename} ...")
            urllib.request.urlretrieve(url, str(fpath))
        paths[filename] = fpath
    return paths


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).flatten()
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        return vec
    return vec / norm


def cosine_sim(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None:
        return None
    a_flat = np.asarray(a, dtype=np.float32).flatten()
    b_flat = np.asarray(b, dtype=np.float32).flatten()
    if a_flat.size == 0 or b_flat.size == 0 or a_flat.size != b_flat.size:
        return None
    na = float(np.linalg.norm(a_flat))
    nb = float(np.linalg.norm(b_flat))
    if na < 1e-12 or nb < 1e-12:
        return None
    return float(np.dot(a_flat, b_flat) / (na * nb))


@dataclass
class PersonSignature:
    """Extracted multi-modal features for a single person image (any variant)."""

    variant: str  # "face_only" | "face_and_clothes" | "clothes_only"
    has_face: bool
    has_clothes: bool
    face_bbox: Optional[List[int]] = None          # [x1, y1, x2, y2]
    face_score: float = 0.0
    body_bbox: Optional[List[int]] = None          # [x1, y1, x2, y2]
    clothes_bbox: Optional[List[int]] = None       # [x1, y1, x2, y2]
    arcface_emb: Optional[np.ndarray] = None       # 512-D L2-normalized
    sface_emb: Optional[np.ndarray] = None         # 128-D L2-normalized
    reid_emb: Optional[np.ndarray] = None          # 768-D L2-normalized
    cloth_emb: Optional[np.ndarray] = None         # 218-D L2-normalized
    dominant_colors: List[str] = field(default_factory=list)
    image_size: Tuple[int, int] = (0, 0)           # (width, height)


class NanoDetPersonDetector:
    """Lightweight ONNX person detector (COCO class 0 = person) from OpenCV Zoo."""

    def __init__(self, model_path: str, prob_threshold: float = 0.35, iou_threshold: float = 0.5):
        self.strides = (8, 16, 32, 64)
        self.image_shape = (416, 416)
        self.reg_max = 7
        self.prob_threshold = prob_threshold
        self.iou_threshold = iou_threshold
        self.project = np.arange(self.reg_max + 1, dtype=np.float32)
        self.mean = np.array([103.53, 116.28, 123.675], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([57.375, 57.12, 58.395], dtype=np.float32).reshape(1, 1, 3)
        self.net = cv2.dnn.readNet(model_path)

        self.anchors_mlvl = []
        for stride in self.strides:
            feat_h = int(self.image_shape[0] / stride)
            feat_w = int(self.image_shape[1] / stride)
            shift_x = np.arange(0, feat_w) * stride
            shift_y = np.arange(0, feat_h) * stride
            xv, yv = np.meshgrid(shift_x, shift_y)
            cx = xv.flatten() + 0.5 * (stride - 1)
            cy = yv.flatten() + 0.5 * (stride - 1)
            self.anchors_mlvl.append(np.column_stack((cx, cy)))

    def detect_persons(self, image_bgr: np.ndarray) -> List[Tuple[List[int], float]]:
        """Return list of ([x1, y1, x2, y2], confidence) for detected persons (class 0)."""
        orig_h, orig_w = image_bgr.shape[:2]
        resized = cv2.resize(image_bgr, self.image_shape)
        img = (resized.astype(np.float32) - self.mean) / self.std
        blob = cv2.dnn.blobFromImage(img)
        self.net.setInput(blob)
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())

        cls_scores, bbox_preds = outs[::2], outs[1::2]
        bboxes_mlvl, scores_mlvl = [], []

        for stride, cls_score, bbox_pred, anchors in zip(
            self.strides, cls_scores, bbox_preds, self.anchors_mlvl
        ):
            if cls_score.ndim == 3:
                cls_score = cls_score.squeeze(axis=0)
            if bbox_pred.ndim == 3:
                bbox_pred = bbox_pred.squeeze(axis=0)

            # Person is class 0 in COCO
            person_scores = cls_score[:, 0]
            x_exp = np.exp(np.clip(bbox_pred.reshape(-1, self.reg_max + 1), -20, 20))
            bbox_pred_dist = (x_exp / np.sum(x_exp, axis=1, keepdims=True)) @ self.project
            bbox_pred_dist = bbox_pred_dist.reshape(-1, 4) * stride

            x1 = np.clip(anchors[:, 0] - bbox_pred_dist[:, 0], 0, self.image_shape[1])
            y1 = np.clip(anchors[:, 1] - bbox_pred_dist[:, 1], 0, self.image_shape[0])
            x2 = np.clip(anchors[:, 0] + bbox_pred_dist[:, 2], 0, self.image_shape[1])
            y2 = np.clip(anchors[:, 1] + bbox_pred_dist[:, 3], 0, self.image_shape[0])

            bboxes_mlvl.append(np.column_stack([x1, y1, x2, y2]))
            scores_mlvl.append(person_scores)

        bboxes = np.concatenate(bboxes_mlvl, axis=0)
        scores = np.concatenate(scores_mlvl, axis=0)

        mask = scores >= self.prob_threshold
        if not np.any(mask):
            return []

        bboxes = bboxes[mask]
        scores = scores[mask]

        scale_x = orig_w / float(self.image_shape[1])
        scale_y = orig_h / float(self.image_shape[0])
        bboxes[:, [0, 2]] *= scale_x
        bboxes[:, [1, 3]] *= scale_y

        bboxes_wh = bboxes.copy()
        bboxes_wh[:, 2] = bboxes[:, 2] - bboxes[:, 0]
        bboxes_wh[:, 3] = bboxes[:, 3] - bboxes[:, 1]

        indices = cv2.dnn.NMSBoxes(
            bboxes_wh.tolist(), scores.tolist(), self.prob_threshold, self.iou_threshold
        )
        results: List[Tuple[List[int], float]] = []
        if len(indices) > 0:
            for idx in np.array(indices).flatten():
                bx1, by1, bx2, by2 = bboxes[idx]
                box = [
                    max(0, int(round(bx1))),
                    max(0, int(round(by1))),
                    min(orig_w, int(round(bx2))),
                    min(orig_h, int(round(by2))),
                ]
                if (box[2] - box[0]) >= 24 and (box[3] - box[1]) >= 24:
                    results.append((box, float(scores[idx])))
        # Sort by area * score descending so primary subject comes first
        results.sort(key=lambda item: ((item[0][2] - item[0][0]) * (item[0][3] - item[0][1])) * item[1], reverse=True)
        return results


class HybridPersonEngine:
    """
    Unified Face + Body Re-ID + Dress/Clothing Feature Extraction & Search Engine.
    """

    def __init__(self, models_dir: Path = MODELS_DIR):
        paths = ensure_models(models_dir)
        providers = ["CPUExecutionProvider"]

        self.face_detector = cv2.FaceDetectorYN.create(
            model=str(paths["face_detection_yunet_2023mar.onnx"]),
            config="",
            input_size=(320, 320),
            score_threshold=0.60,
            nms_threshold=0.3,
            top_k=50,
        )

        self.arcface_sess = ort.InferenceSession(
            str(paths["w600k_mbf.onnx"]), providers=providers
        )
        self.arcface_in_name = self.arcface_sess.get_inputs()[0].name

        self.sface_recognizer = cv2.FaceRecognizerSF.create(
            model=str(paths["face_recognition_sface_2021dec.onnx"]),
            config="",
        )

        self.reid_sess = ort.InferenceSession(
            str(paths["person_reid_youtu_2021nov.onnx"]), providers=providers
        )
        self.reid_in_name = self.reid_sess.get_inputs()[0].name

        self.person_detector = NanoDetPersonDetector(
            str(paths["object_detection_nanodet_2022nov.onnx"]), prob_threshold=0.35
        )

    def _detect_faces_multiscale(self, image_bgr: np.ndarray) -> List[np.ndarray]:
        """
        Detect faces using YuNet. If no face is found directly (e.g. a very tight
        face-only crop filling 100% of the frame), pad the border and retry.
        """
        h, w = image_bgr.shape[:2]

        # Resize large images for reliable detection speed & scale
        max_dim = max(h, w)
        scale = 1.0
        work_img = image_bgr
        if max_dim > 1024:
            scale = 1024.0 / max_dim
            work_img = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))
        elif max_dim < 160:
            scale = 240.0 / max_dim
            work_img = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))

        wh, ww = work_img.shape[:2]
        self.face_detector.setInputSize((ww, wh))
        _, faces = self.face_detector.detect(work_img)

        if faces is not None and len(faces) > 0:
            faces = faces.copy()
            faces[:, :14] /= scale
            return [f for f in faces]

        # Fallback: pad image with neutral border (handles tight face-only crops!)
        for pad_ratio, border_mode in [(0.45, cv2.BORDER_CONSTANT), (0.35, cv2.BORDER_REPLICATE)]:
            pad_y = int(wh * pad_ratio)
            pad_x = int(ww * pad_ratio)
            padded = cv2.copyMakeBorder(
                work_img, pad_y, pad_y, pad_x, pad_x, border_mode, value=(128, 128, 128)
            )
            ph, pw = padded.shape[:2]
            self.face_detector.setInputSize((pw, ph))
            _, faces_pad = self.face_detector.detect(padded)
            if faces_pad is not None and len(faces_pad) > 0:
                faces_pad = faces_pad.copy()
                faces_pad[:, 0] -= pad_x
                faces_pad[:, 1] -= pad_y
                for k in range(5):
                    faces_pad[:, 4 + 2 * k] -= pad_x
                    faces_pad[:, 4 + 2 * k + 1] -= pad_y
                faces_pad[:, :14] /= scale
                return [f for f in faces_pad]

        return []

    def _align_arcface_112(self, image_bgr: np.ndarray, face_row: np.ndarray) -> np.ndarray:
        """Align face to 112x112 using 5 landmarks via similarity transform."""
        kps = face_row[4:14].reshape(5, 2).astype(np.float32)
        tform, _ = cv2.estimateAffinePartial2D(kps, ARCFACE_DST, method=cv2.LMEDS)
        if tform is None:
            x, y, bw, bh = face_row[:4].astype(int)
            h, w = image_bgr.shape[:2]
            crop = image_bgr[max(0, y):min(h, y + bh), max(0, x):min(w, x + bw)]
            if crop.size == 0:
                crop = image_bgr
            return cv2.resize(crop, (112, 112))
        aligned = cv2.warpAffine(image_bgr, tform, (112, 112), borderValue=0.0)
        return aligned

    def _extract_face_embeddings(
        self, image_bgr: np.ndarray, face_row: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract both 512-D ArcFace and 128-D SFace L2-normalized embeddings."""
        aligned_112 = self._align_arcface_112(image_bgr, face_row)
        rgb = cv2.cvtColor(aligned_112, cv2.COLOR_BGR2RGB).astype(np.float32)
        blob = ((rgb - 127.5) / 127.5).transpose(2, 0, 1)[np.newaxis, ...]
        arc_out = self.arcface_sess.run(None, {self.arcface_in_name: blob})[0]
        arc_emb = l2_normalize(arc_out)

        aligned_sface = self.sface_recognizer.alignCrop(image_bgr, face_row)
        sface_out = self.sface_recognizer.feature(aligned_sface)
        sface_emb = l2_normalize(sface_out)

        return arc_emb, sface_emb

    def _extract_reid_embedding(self, body_crop_bgr: np.ndarray) -> np.ndarray:
        """Extract 768-D YouTu Person Re-ID embedding from a body/clothing crop."""
        resized = cv2.resize(body_crop_bgr, (128, 256))
        rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normed = (rgb - mean) / std
        blob = normed.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
        feat = self.reid_sess.run(None, {self.reid_in_name: blob})[0]
        return l2_normalize(feat)

    def _extract_clothing_descriptor(
        self, clothes_crop_bgr: np.ndarray
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract a 218-D Spatial Upper/Mid/Lower Clothing Color + CIE-LAB + Texture
        descriptor focused on the central torso/dress region, plus human-readable
        dominant color tags.
        """
        resized = cv2.resize(clothes_crop_bgr, (128, 256))
        # Focus horizontally on the central 65% to minimize background interference
        core = resized[:, 22:106]  # shape (256, 84, 3)

        hsv = cv2.cvtColor(core, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(core, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(core, cv2.COLOR_BGR2GRAY)

        # Divide vertically into 3 clothing zones:
        # Zone 0: Upper chest / collar / upper dress
        # Zone 1: Mid torso / waist
        # Zone 2: Lower body / skirt / pants
        zones_hsv = [hsv[0:90, :], hsv[75:175, :], hsv[160:256, :]]
        zones_lab = [lab[0:90, :], lab[75:175, :], lab[160:256, :]]

        feats: List[np.ndarray] = []
        for zh, zl in zip(zones_hsv, zones_lab):
            # Chromatic vs achromatic separation
            # Pixels with reasonable saturation & value -> 2D Hue-Sat histogram (12 x 4 = 48 bins)
            h_hist = cv2.calcHist([zh], [0, 1], None, [12, 4], [0, 180, 0, 256]).flatten()
            h_hist = l2_normalize(h_hist)

            # Value / brightness & saturation distribution (8 + 8 = 16 bins)
            v_hist = l2_normalize(cv2.calcHist([zh], [2], None, [8], [0, 256]).flatten())
            s_hist = l2_normalize(cv2.calcHist([zh], [1], None, [8], [0, 256]).flatten())

            # CIE-LAB mean & std (6 values, scaled)
            lab_mean = np.mean(zl.reshape(-1, 3), axis=0) / 255.0
            lab_std = np.std(zl.reshape(-1, 3), axis=0) / 128.0

            feats.extend([h_hist * 1.4, v_hist * 0.6, s_hist * 0.8, lab_mean, lab_std])

        # Texture / pattern orientation & edge density (stripes, plaid, floral vs solid)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        orient_hist, _ = np.histogram(ang, bins=8, range=(0, 360), weights=mag)
        orient_hist = l2_normalize(orient_hist.astype(np.float32)) * 0.5

        combined = np.concatenate(feats + [orient_hist]).astype(np.float32)
        dominant_colors = self._classify_dominant_colors(hsv)
        return l2_normalize(combined), dominant_colors

    @staticmethod
    def _classify_dominant_colors(hsv_patch: np.ndarray) -> List[str]:
        """Identify top 2 human-readable clothing colors from an HSV torso patch."""
        pixels = hsv_patch.reshape(-1, 3)
        H, S, V = pixels[:, 0], pixels[:, 1], pixels[:, 2]
        total = max(1, len(pixels))

        counts: Dict[str, int] = {
            "black": int(np.sum(V < 48)),
            "white": int(np.sum((S < 38) & (V > 195))),
            "gray": int(np.sum((S < 42) & (V >= 48) & (V <= 195))),
        }
        chromatic = (S >= 42) & (V >= 48)
        hc = H[chromatic]
        counts["red"] = int(np.sum((hc < 10) | (hc >= 165)))
        counts["orange"] = int(np.sum((hc >= 10) & (hc < 22)))
        counts["yellow"] = int(np.sum((hc >= 22) & (hc < 35)))
        counts["green"] = int(np.sum((hc >= 35) & (hc < 85)))
        counts["cyan"] = int(np.sum((hc >= 85) & (hc < 102)))
        counts["blue"] = int(np.sum((hc >= 102) & (hc < 132)))
        counts["purple"] = int(np.sum((hc >= 132) & (hc < 150)))
        counts["pink"] = int(np.sum((hc >= 150) & (hc < 165)))

        sorted_colors = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        top = [name for name, cnt in sorted_colors[:2] if (cnt / total) >= 0.12]
        return top if top else [sorted_colors[0][0]]

    def extract_signature(self, image_or_path: Any) -> PersonSignature:
        """
        Analyze an image (path or BGR numpy array) and extract:
        - Variant classification (`face_only`, `face_and_clothes`, `clothes_only`)
        - Face embeddings (512-D ArcFace + 128-D SFace) if face is present
        - Body Re-ID embedding (768-D YouTu Re-ID) and Clothing descriptor (218-D)
          if clothing/body is present
        """
        if isinstance(image_or_path, (str, Path)):
            img_bgr = cv2.imread(str(image_or_path))
            if img_bgr is None:
                raise ValueError(f"Could not read image at: {image_or_path}")
        elif isinstance(image_or_path, np.ndarray):
            img_bgr = image_or_path
        else:
            raise TypeError("Expected file path or numpy BGR image array")

        h, w = img_bgr.shape[:2]

        faces = self._detect_faces_multiscale(img_bgr)
        primary_face: Optional[np.ndarray] = None
        face_bbox: Optional[List[int]] = None
        face_conf = 0.0
        arcface_emb: Optional[np.ndarray] = None
        sface_emb: Optional[np.ndarray] = None

        if len(faces) > 0:
            # Choose largest high-confidence face
            faces.sort(
                key=lambda f: max(1.0, f[2]) * max(1.0, f[3]) * float(f[14]),
                reverse=True,
            )
            primary_face = faces[0]
            fx, fy, fw, fh = primary_face[:4]
            face_conf = float(primary_face[14])
            face_bbox = [
                max(0, int(round(fx))),
                max(0, int(round(fy))),
                min(w, int(round(fx + fw))),
                min(h, int(round(fy + fh))),
            ]
            arcface_emb, sface_emb = self._extract_face_embeddings(img_bgr, primary_face)

        person_detections = self.person_detector.detect_persons(img_bgr)
        body_bbox: Optional[List[int]] = None

        if primary_face is not None:
            fx1, fy1, fx2, fy2 = face_bbox  # type: ignore[misc]
            fcx = 0.5 * (fx1 + fx2)
            fcy = 0.5 * (fy1 + fy2)
            # Find person box containing the detected face center
            for pbox, _ in person_detections:
                if pbox[0] <= fcx <= pbox[2] and pbox[1] <= fcy <= pbox[3]:
                    body_bbox = pbox
                    break
        elif len(person_detections) > 0:
            body_bbox = person_detections[0][0]

        has_face = primary_face is not None
        has_clothes = False
        clothes_bbox: Optional[List[int]] = None

        if has_face and face_bbox is not None:
            fx1, fy1, fx2, fy2 = face_bbox
            fw = max(1, fx2 - fx1)
            fh = max(1, fy2 - fy1)
            space_below_face = h - fy2

            # Is there meaningful torso/clothing space below the face?
            # If the face takes up most of the image height or space below face < 0.55 * face_height,
            # it is a `face_only` crop!
            if space_below_face >= max(32, int(0.55 * fh)) and (fh / float(h)) < 0.62:
                has_clothes = True
                # Build clothing crop strictly BELOW the face chin so skin does not pollute dress features
                cx_center = 0.5 * (fx1 + fx2)
                half_torso_w = int(max(fw * 1.65, (body_bbox[2] - body_bbox[0]) * 0.5 if body_bbox else 0))
                cx1 = max(0, int(cx_center - half_torso_w))
                cx2 = min(w, int(cx_center + half_torso_w))
                cy1 = min(h - 16, max(0, int(fy2 + 0.05 * fh)))
                cy2 = body_bbox[3] if (body_bbox and body_bbox[3] > cy1 + 24) else h
                clothes_bbox = [cx1, cy1, cx2, cy2]
                if body_bbox is None:
                    body_bbox = [cx1, max(0, fy1), cx2, cy2]
            else:
                has_clothes = False
        else:
            # No face detected -> image is `clothes_only` (full body, torso crop, dress crop, back view)
            has_clothes = True
            if body_bbox is not None:
                bx1, by1, bx2, by2 = body_bbox
                bh = by2 - by1
                # Exclude top 15% (head/hair region if full body) for pure clothing descriptor
                clothes_bbox = [bx1, min(by2 - 16, by1 + int(0.15 * bh)), bx2, by2]
            else:
                body_bbox = [0, 0, w, h]
                clothes_bbox = [
                    int(w * 0.08),
                    int(h * 0.10),
                    max(int(w * 0.08) + 16, int(w * 0.92)),
                    max(int(h * 0.10) + 16, int(h * 0.95)),
                ]

        reid_emb: Optional[np.ndarray] = None
        cloth_emb: Optional[np.ndarray] = None
        dominant_colors: List[str] = []

        if has_clothes:
            bx1, by1, bx2, by2 = body_bbox if body_bbox else [0, 0, w, h]
            body_crop = img_bgr[max(0, by1):min(h, by2), max(0, bx1):min(w, bx2)]
            if body_crop.size == 0:
                body_crop = img_bgr
            reid_emb = self._extract_reid_embedding(body_crop)

            cx1, cy1, cx2, cy2 = clothes_bbox if clothes_bbox else [0, 0, w, h]
            clothes_crop = img_bgr[max(0, cy1):min(h, cy2), max(0, cx1):min(w, cx2)]
            if clothes_crop.size == 0 or clothes_crop.shape[0] < 12 or clothes_crop.shape[1] < 12:
                clothes_crop = body_crop
            cloth_emb, dominant_colors = self._extract_clothing_descriptor(clothes_crop)

        if has_face and has_clothes:
            variant = "face_and_clothes"
        elif has_face:
            variant = "face_only"
        else:
            variant = "clothes_only"

        return PersonSignature(
            variant=variant,
            has_face=has_face,
            has_clothes=has_clothes,
            face_bbox=face_bbox,
            face_score=round(face_conf, 4),
            body_bbox=body_bbox,
            clothes_bbox=clothes_bbox,
            arcface_emb=arcface_emb,
            sface_emb=sface_emb,
            reid_emb=reid_emb,
            cloth_emb=cloth_emb,
            dominant_colors=dominant_colors,
            image_size=(w, h),
        )

    @staticmethod
    def _calibrate_face_pct(arc_cos: Optional[float], sface_cos: Optional[float]) -> Optional[float]:
        """
        Convert raw ArcFace + SFace cosine similarities into an intuitive 0..100%
        confidence score.
        For ArcFace (w600k_mbf) & SFace:
          - Same person typically has cosine >= 0.38 (strong match >= 0.50)
          - Different people typically have cosine in [-0.15, 0.22]
        """
        if arc_cos is None and sface_cos is None:
            return None
        if arc_cos is not None and sface_cos is not None:
            raw = 0.70 * arc_cos + 0.30 * sface_cos
        elif arc_cos is not None:
            raw = arc_cos
        else:
            raw = sface_cos  # type: ignore[assignment]

        # Piecewise linear calibration to [0, 100]
        if raw <= 0.10:
            pct = max(0.0, (raw + 0.10) * 75.0)  # 0..15%
        elif raw < 0.36:
            # 0.10 -> 15%, 0.36 -> 65%
            pct = 15.0 + (raw - 0.10) / (0.36 - 0.10) * 50.0
        elif raw < 0.55:
            # 0.36 -> 65%, 0.55 -> 90%
            pct = 65.0 + (raw - 0.36) / (0.55 - 0.36) * 25.0
        else:
            # 0.55 -> 90%, 0.85+ -> 99.8%
            pct = 90.0 + min(9.9, (raw - 0.55) / 0.35 * 9.9)
        return round(float(np.clip(pct, 0.0, 99.9)), 2)

    @staticmethod
    def _calibrate_reid_pct(reid_cos: Optional[float]) -> Optional[float]:
        """Calibrate 768-D YouTu Re-ID cosine similarity to 0..100%."""
        if reid_cos is None:
            return None
        if reid_cos <= 0.25:
            pct = max(0.0, reid_cos / 0.25 * 20.0)
        elif reid_cos < 0.55:
            pct = 20.0 + (reid_cos - 0.25) / 0.30 * 48.0
        elif reid_cos < 0.78:
            pct = 68.0 + (reid_cos - 0.55) / 0.23 * 22.0
        else:
            pct = 90.0 + min(9.9, (reid_cos - 0.78) / 0.22 * 9.9)
        return round(float(np.clip(pct, 0.0, 99.9)), 2)

    @staticmethod
    def _calibrate_cloth_pct(cloth_cos: Optional[float]) -> Optional[float]:
        """Calibrate 218-D Spatial Dress/Clothing descriptor similarity to 0..100%."""
        if cloth_cos is None:
            return None
        if cloth_cos <= 0.35:
            pct = max(0.0, cloth_cos / 0.35 * 25.0)
        elif cloth_cos < 0.68:
            pct = 25.0 + (cloth_cos - 0.35) / 0.33 * 45.0
        elif cloth_cos < 0.86:
            pct = 70.0 + (cloth_cos - 0.68) / 0.18 * 20.0
        else:
            pct = 90.0 + min(9.9, (cloth_cos - 0.86) / 0.14 * 9.9)
        return round(float(np.clip(pct, 0.0, 99.9)), 2)

    @classmethod
    def compare_signatures(
        cls,
        query_sig: PersonSignature,
        db_sig: PersonSignature,
        mode: str = "auto",
    ) -> Dict[str, Any]:
        """
        Compare a query PersonSignature against a DB PersonSignature.

        `mode`:
          - "auto"   : Adaptive fusion across all available modalities (Face + Body ReID + Dress)
                       with a synergy bonus when BOTH face and clothing match, plus smart fallback
                       when a person changed clothes or when one image is face-only / clothes-only.
          - "strict" : Require BOTH face AND clothing/dress to match if both are present in query.
          - "face"   : Prioritize face identity.
          - "clothes": Prioritize clothing/dress & body Re-ID.
        """
        arc_cos = cosine_sim(query_sig.arcface_emb, db_sig.arcface_emb)
        sface_cos = cosine_sim(query_sig.sface_emb, db_sig.sface_emb)
        reid_cos = cosine_sim(query_sig.reid_emb, db_sig.reid_emb)
        cloth_cos = cosine_sim(query_sig.cloth_emb, db_sig.cloth_emb)

        face_pct = cls._calibrate_face_pct(arc_cos, sface_cos)
        reid_pct = cls._calibrate_reid_pct(reid_cos)
        cloth_pct = cls._calibrate_cloth_pct(cloth_cos)

        # Combined clothing & body appearance score (when clothes exist in both)
        appearance_pct: Optional[float] = None
        if reid_pct is not None and cloth_pct is not None:
            appearance_pct = round(0.55 * reid_pct + 0.45 * cloth_pct, 2)
        elif reid_pct is not None:
            appearance_pct = reid_pct
        elif cloth_pct is not None:
            appearance_pct = cloth_pct

        matched_modalities: List[str] = []
        if face_pct is not None and face_pct >= 65.0:
            matched_modalities.append("face")
        if appearance_pct is not None and appearance_pct >= 65.0:
            matched_modalities.append("clothes_and_body")

        if mode == "face":
            combined_pct = face_pct if face_pct is not None else (appearance_pct or 0.0) * 0.5
            match_type = "face_priority"
        elif mode == "clothes":
            combined_pct = appearance_pct if appearance_pct is not None else (face_pct or 0.0) * 0.5
            match_type = "clothes_priority"
        elif mode == "strict" and face_pct is not None and appearance_pct is not None:
            combined_pct = round(0.52 * face_pct + 0.48 * appearance_pct, 2)
            match_type = "strict_face_and_clothes"
        else:
            if face_pct is not None and appearance_pct is not None:
                exact_variant_score = 0.55 * face_pct + 0.45 * appearance_pct
                # Synergy bonus when BOTH face AND clothing match strongly
                if face_pct >= 65.0 and appearance_pct >= 65.0:
                    bonus = min(6.5, ((face_pct - 65.0) / 35.0) * ((appearance_pct - 65.0) / 35.0) * 10.0 + 2.5)
                    combined_pct = min(99.9, exact_variant_score + bonus)
                    match_type = "face_and_clothes_match"
                elif face_pct >= 72.0:
                    # Same person face, different clothes or partial clothing match:
                    # Keep score high enough to identify the person, while still ranking
                    # below an exact face+clothes match!
                    combined_pct = max(exact_variant_score, face_pct * 0.88 + appearance_pct * 0.12)
                    match_type = "face_match_different_clothes"
                elif appearance_pct >= 75.0 and face_pct < 55.0:
                    # Different face but similar clothes -> penalize because face disagrees
                    combined_pct = 0.45 * face_pct + 0.55 * appearance_pct
                    match_type = "clothes_similar_face_mismatch"
                else:
                    combined_pct = exact_variant_score
                    match_type = "hybrid_partial"
            elif face_pct is not None:
                # One or both images are `face_only`
                if query_sig.has_clothes and not db_sig.has_clothes:
                    # Query has clothes+face, DB image has only face -> slight discount vs full face+clothes match
                    combined_pct = face_pct * 0.93
                elif not query_sig.has_clothes and db_sig.has_clothes:
                    combined_pct = face_pct * 0.96
                else:
                    combined_pct = face_pct
                match_type = "face_only_match"
            elif appearance_pct is not None:
                # One or both images are `clothes_only` (no face visible)
                combined_pct = appearance_pct
                match_type = "clothes_and_reid_match"
            else:
                combined_pct = 0.0
                match_type = "no_shared_modality"

        combined_pct = round(float(np.clip(combined_pct, 0.0, 99.9)), 2)
        return {
            "combined_score": combined_pct,
            "face_score": face_pct,
            "reid_score": reid_pct,
            "clothing_score": cloth_pct,
            "appearance_score": appearance_pct,
            "match_type": match_type,
            "matched_modalities": matched_modalities,
            "raw_cosine": {
                "arcface": round(arc_cos, 4) if arc_cos is not None else None,
                "sface": round(sface_cos, 4) if sface_cos is not None else None,
                "reid": round(reid_cos, 4) if reid_cos is not None else None,
                "clothing": round(cloth_cos, 4) if cloth_cos is not None else None,
            },
        }
