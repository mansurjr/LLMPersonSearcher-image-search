# Image-based person search

This feature works inside LLMPersonSearcher on `feature/image-based-search`, based on `develop`. It searches the existing `person_detection` records and keeps their IDs, video/frame references, category, source image, crop path and bounding boxes. Annotation ingestion, YOLO ingestion and natural-language attribute search continue to use their existing storage contracts.

The default backend is a lightweight RGB histogram implemented with Pillow. An optional hybrid backend uses YuNet/NanoDet detection, ArcFace/SFace face embeddings, YouTu body Re-ID and a 218-dimensional clothing descriptor. Both backends use the same search service and result format.

## Windows setup

Run PowerShell in the repository root. Activation is optional; these commands use the virtual environment directly and require no execution-policy change.

```powershell
cd C:\Users\mirov\Desktop\LLMPersonSearcher
git branch --show-current
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider -q
.\.venv\Scripts\person-image-search.exe --help
```

Python 3.10+ is required. The histogram backend needs no model downloads, Ollama, OpenCV, NumPy or ONNX Runtime. The optional dependencies are loaded only when their feature is selected.

## 1. Create the detection database

Use the existing ingestion script with extracted dataset images and JSON annotations:

```powershell
.\.venv\Scripts\python.exe scripts/ingest_annotations.py `
  --annotations "D:\dataset\Training" `
  --images "D:\dataset\Training" `
  --database artifacts/search.db `
  --crops artifacts/crops `
  --limit 20
```

Replace both dataset paths with actual folders. `--limit` here limits ingested detections; use a small number to check the dataset layout first. Existing ingestion expects `annotation_<video-id>.json` files and extracted frame files under their video-ID folders. The labels and source images must be available locally after ZIP extraction. A Drive folder URL is not a local image path.

For unlabeled images, the repository's existing `scripts/ingest_yolo.py` can create detections after installing `.[yolo]`. No changes to that workflow are required by image search.

Run commands from the repository root when the database contains relative crop paths. Absolute crop paths work from other directories too.

## 2. Search by a person crop

```powershell
.\.venv\Scripts\person-image-search.exe `
  --reference "D:\queries\person.jpg" `
  --database artifacts/search.db `
  --limit 5 `
  --json
```

Use a crop containing the person of interest as the reference. Searching a full CCTV frame containing several people compares that entire frame's appearance with each stored crop.

The service considers **all matching detections before selecting the top results**. `--limit 5` means five best matches, not the first five database rows. A tie is resolved by lower detection ID. Rounding is applied only after ranking.

Optional filters:

```powershell
.\.venv\Scripts\person-image-search.exe `
  --reference "D:\queries\person.jpg" `
  --database artifacts/search.db `
  --video-id 2969641 `
  --split training `
  --min-similarity 0.7 `
  --limit 5 `
  --json
```

Use the `video_id` stored by this repo's ingestion, such as `2969641`, rather than assuming it equals the original MP4 filename. Other filters are `--event-class` and `--category person|child|all`. Image-only search defaults to `person`. With `--query`, the CLI keeps all matching categories (including `child`) by default, preserving the previous text-filter behavior; an explicit category overrides this.

`--json` returns one JSON object with `results`, `scanned_count`, `cached_count`, `skipped_count`, `errors` and backend ID. Each result preserves the detection fields and adds `similarity` on a 0..1 scale. Missing/corrupt candidate files are skipped and reported by detection ID. An unreadable reference or invalid arguments fail with exit code 2 and an explanatory message. A successful partial search returns exit code 0 with its skipped entries in the report.

## 3. Persistent indexing

Features are built automatically during search. To prepare them in advance:

```powershell
.\.venv\Scripts\person-image-search.exe --database artifacts/search.db --index-only --json
```

The additive `person_visual_feature` table stores one feature record per detection/backend. Full-file SHA256 fingerprints detect changed images. Backend versions keep feature formats separate; the hybrid key also includes engine code, model contents and runtime versions. Corrupt cached features are recomputed. A repeated index uses cached features, while distinct detection IDs remain distinct even if their pixels match.

The database must be writable to maintain this cache. Removing `person_visual_feature` removes only derived features; subsequent searches regenerate them. Existing `person_detection` and `person_attribute` records are not replaced by another schema.

Retrieval scans the filtered candidates linearly. The heap retains only the requested top results, while the repository currently materializes the candidate metadata. This is suitable for the current prototype; a large deployment may need paginated candidate retrieval and a dedicated vector index.

## 4. Optional hybrid matching

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,hybrid]"
```

This extra requires OpenCV 4.x. The OpenCV 5 default DNN engine produced incompatible NanoDet outputs during validation, so the dependency is constrained below 5 and the adapter reports an unsupported runtime clearly.

Provide these five ONNX files in a model directory:

```text
face_detection_yunet_2023mar.onnx
face_recognition_sface_2021dec.onnx
w600k_mbf.onnx
person_reid_youtu_2021nov.onnx
object_detection_nanodet_2022nov.onnx
```

Existing weights can be reused. For an explicit first-time download using the preserved engine's model URLs:

```powershell
.\.venv\Scripts\python.exe -c "from pathlib import Path; from person_searcher.vision._hybrid_engine import ensure_models; ensure_models(Path('artifacts/models'))"
```

This setup command downloads model files; normal search requires an explicit model directory and checks all files before loading. Model/data files and generated artifacts are ignored by Git.

```powershell
.\.venv\Scripts\person-image-search.exe `
  --reference "D:\queries\person.jpg" `
  --database artifacts/search.db `
  --backend hybrid `
  --models-dir artifacts/models `
  --mode clothes `
  --limit 5 `
  --json
```

Hybrid modes are `auto`, `face`, `clothes` and `strict`. The current customer's CCTV images have blurred/small faces, so `clothes` is the useful sample mode. Scores represent the preserved engine's visual-similarity scale, not calibrated identity probabilities. Scores from different backends should not be interpreted as the same accuracy measure.

## 5. Optional text conditions

The existing Ollama candidate filter is preserved, including the database's saved few-shot examples:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[ollama]"
.\.venv\Scripts\person-image-search.exe `
  --reference "D:\queries\person.jpg" `
  --database artifacts/search.db `
  --query "파란 상의에 검은 하의를 입은 사람" `
  --limit 5 `
  --json
```

Ollama must be running with the selected model, and matching detections must already have `person_attribute` rows. The JSON report includes `parsed_condition` so its interpretation can be checked. Automated tests use a local fake response rather than contacting Ollama.

The existing `python scripts/search_reid.py --reference ...` command delegates to the same service. It supports the new options while retaining the previous flags.

## Python API

```python
from pathlib import Path
from person_searcher import ImageSearcher

searcher = ImageSearcher(Path("artifacts/search.db"))
report = searcher.search(Path("query.jpg"), video_id="2969641", limit=5)
for match in report["results"]:
    print(match["id"], match["similarity"], match["video_id"], match["frame_number"])
```

Use `person_searcher.vision.hybrid.HybridMatcher(models_dir, mode="clothes")` as the optional `matcher=` argument for the hybrid backend. Custom matchers implement the `ImageMatcher` protocol: a versioned `cache_key`, JSON-compatible `extract`, `validate`, and `similarity` methods.

## Checked local customer demo

The local verification database is `artifacts/customer_demo/search.db`, with 21 person crops from alternate Training/Validation frames. Both reference frames were excluded from the gallery. The sample files and weights are local artifacts, not part of the Git change.

```powershell
# Lightweight search: no model/service dependency
.\.venv\Scripts\person-image-search.exe --reference artifacts/customer_demo/training_query.jpg --database artifacts/customer_demo/search.db --limit 3 --json

# Reuse the existing local ONNX weights
.\.venv\Scripts\person-image-search.exe --reference artifacts/customer_demo/training_query.jpg --database artifacts/customer_demo/search.db --backend hybrid --models-dir C:\Users\mirov\Desktop\face\models --mode clothes --video-id 2969641 --limit 1 --json
```

Verified hybrid retrieval selected frame 4025 for held-out Training frame 4010 (similarity 0.8275), and frame 3966 for held-out Validation frame 3967 (0.9655). These are bounded adjacent-frame checks, not a cross-camera accuracy benchmark. The customer's source archives contained extracted JPG frames; no standalone original videos were found in that audit.

Machine-readable local results are in `artifacts/customer_demo/verification.json`. A fresh clone requires ingesting its own dataset; these local artifact paths will not exist on another computer.

## Before handing off

```powershell
.\.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider -q
.\.venv\Scripts\python.exe -m pip check
git diff --check
git status --short --branch
```

The original `tests/test_query_builder.py` remains unchanged. New tests cover complete-catalog ranking, source/attribute filters, persistent cache reuse/invalidation, malformed images, cosine normalization, optional imports, hybrid errors, saved parsing examples and CLI behavior.

After reviewing the diff, commit and push only the feature branch:

```powershell
git add .gitignore pyproject.toml README.md docs/image-search.md scripts/search_reid.py src/person_searcher tests/test_image_search.py tests/test_hybrid_backend.py
git commit -m "feat: add image-based search with optional hybrid matching"
git push -u origin feature/image-based-search
```

Open a pull request with **base `develop`** and **compare `feature/image-based-search`**. Keep models, databases, images and `.venv` out of the commit.
