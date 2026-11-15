"""Optional adapter for the existing face, clothing and body Re-ID engine."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path
import threading
from typing import Any

MODEL_FILES = (
    "face_detection_yunet_2023mar.onnx", "w600k_mbf.onnx",
    "face_recognition_sface_2021dec.onnx", "person_reid_youtu_2021nov.onnx",
    "object_detection_nanodet_2022nov.onnx",
)


class HybridMatcher:
    def __init__(self, models_dir: str | Path, *, mode: str = "auto"):
        if mode not in {"auto", "face", "clothes", "strict"}:
            raise ValueError("Hybrid mode must be auto, face, clothes, or strict")
        model_root = Path(models_dir).expanduser().resolve()
        missing = [name for name in MODEL_FILES if not (model_root / name).is_file()
                   or (model_root / name).stat().st_size < 10000]
        if missing:
            raise FileNotFoundError(f"Missing ONNX models in {model_root}: {', '.join(missing)}")
        try:
            from ._hybrid_engine import HybridPersonEngine
            import cv2
            import numpy as np
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("Hybrid search requires the optional dependencies: pip install -e '.[hybrid]'") from exc
        if cv2.__version__.split(".")[0] != "4":
            raise RuntimeError("Hybrid search requires OpenCV 4; reinstall with pip install -e '.[hybrid]'")
        self.mode = mode
        try:
            self.engine = HybridPersonEngine(model_root)
        except Exception as exc:
            raise RuntimeError(f"Could not load hybrid models from {model_root}: {exc}") from exc
        self._inference_lock = threading.RLock()
        digest = hashlib.sha256(Path(__file__).with_name("_hybrid_engine.py").read_bytes())
        digest.update(f"{cv2.__version__}:{np.__version__}:{ort.__version__}".encode("ascii"))
        for name in MODEL_FILES:
            with (model_root / name).open("rb") as model:
                for chunk in iter(lambda: model.read(1024 * 1024), b""):
                    digest.update(chunk)
        self.cache_key = "hybrid-v1:" + digest.hexdigest()

    def extract(self, image_path: Path) -> dict[str, Any]:
        import cv2
        import numpy as np

        raw = image_path.read_bytes()
        if not raw:
            raise ValueError(f"Cannot decode empty image: {image_path}")
        try:
            image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        except cv2.error as exc:
            raise ValueError(f"Cannot decode image: {image_path}") from exc
        if image is None:
            raise ValueError(f"Cannot decode image: {image_path}")
        try:
            with self._inference_lock:
                signature = self.engine.extract_signature(image)
        except Exception as exc:
            raise ValueError(f"Hybrid feature extraction failed for {image_path}: {exc}") from exc
        return {key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in asdict(signature).items()}

    @staticmethod
    def _restore(features: dict[str, Any]):
        import numpy as np
        from ._hybrid_engine import PersonSignature

        if not isinstance(features, dict):
            raise ValueError("Hybrid features must be an object")
        restored = dict(features)
        for name, size in {"arcface_emb": 512, "sface_emb": 128, "reid_emb": 768, "cloth_emb": 218}.items():
            value = restored.get(name)
            if value is not None:
                array = np.asarray(value, dtype=np.float32)
                if array.shape != (size,) or not np.isfinite(array).all():
                    raise ValueError(f"Invalid {name} feature vector")
                restored[name] = array
        try:
            return PersonSignature(**restored)
        except TypeError as exc:
            raise ValueError("Invalid cached hybrid signature") from exc

    def validate(self, features: dict[str, Any]) -> None:
        self._restore(features)

    def similarity(self, left: dict[str, Any], right: dict[str, Any]) -> float:
        result = self.engine.compare_signatures(self._restore(left), self._restore(right), mode=self.mode)
        return result["combined_score"] / 100.0
