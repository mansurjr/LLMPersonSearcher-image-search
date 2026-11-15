import importlib.util
import subprocess
import sys

import pytest


def test_base_package_does_not_import_optional_ml_dependencies():
    code = (
        "from person_searcher.image_search import ImageSearcher; import sys; "
        "assert not {'cv2', 'numpy', 'onnxruntime', 'ultralytics'} & sys.modules.keys()"
    )
    result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_hybrid_backend_reports_missing_models_without_downloading(tmp_path):
    assert importlib.util.find_spec("person_searcher.vision.hybrid"), "Hybrid adapter is missing"
    from person_searcher.vision.hybrid import HybridMatcher
    with pytest.raises(FileNotFoundError, match="model|Model"):
        HybridMatcher(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_empty_hybrid_image_is_a_recoverable_input_error(tmp_path):
    pytest.importorskip("cv2")
    from person_searcher.vision.hybrid import HybridMatcher
    image = tmp_path / "empty.jpg"
    image.write_bytes(b"")
    with pytest.raises(ValueError, match="image|Image|decode"):
        HybridMatcher.__new__(HybridMatcher).extract(image)


def test_hybrid_cache_accepts_actual_218_dimension_clothing_descriptor():
    pytest.importorskip("cv2")
    from person_searcher.vision.hybrid import HybridMatcher
    features = {"variant": "clothes_only", "has_face": False, "has_clothes": True, "cloth_emb": [0.1] * 218}
    HybridMatcher.__new__(HybridMatcher).validate(features)


def test_invalid_model_files_have_a_clear_loading_error(tmp_path):
    pytest.importorskip("cv2")
    from person_searcher.vision.hybrid import HybridMatcher, MODEL_FILES
    for name in MODEL_FILES:
        (tmp_path / name).write_bytes(b"invalid" * 2000)
    with pytest.raises(RuntimeError, match="load|Load"):
        HybridMatcher(tmp_path)


def test_incompatible_opencv_major_reports_supported_version(tmp_path, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    from person_searcher.vision.hybrid import HybridMatcher, MODEL_FILES
    for name in MODEL_FILES:
        (tmp_path / name).write_bytes(b"invalid" * 2000)
    monkeypatch.setattr(cv2, "__version__", "5.0.0")
    with pytest.raises(RuntimeError, match="OpenCV 4"):
        HybridMatcher(tmp_path)
