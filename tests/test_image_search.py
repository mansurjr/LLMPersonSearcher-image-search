import json
import sqlite3

from PIL import Image
import pytest

from person_searcher.domain.attribute_mapping import condition_to_attributes
from person_searcher.domain.models import ItemCondition, PersonSearchCondition
from person_searcher.storage.sqlite import SQLiteStore
from person_searcher.vision.reid import cosine_similarity


@pytest.fixture
def catalog(tmp_path):
    database = tmp_path / "search.db"
    store = SQLiteStore(database)
    store.initialize()
    for number, (color, video) in enumerate([("blue", "cam-a"), ("green", "cam-b"), ("red", "cam-a")], 1):
        crop = tmp_path / f"person-{number}.png"
        Image.new("RGB", (40, 80), color).save(crop)
        record = {"split": "training", "event_class": "gate", "video_id": video,
                  "source_video": video + ".mp4", "frame_number": number * 30,
                  "global_id": str(number), "category": "person", "annotation_index": 0,
                  "image_path": "source.jpg", "x": 10, "y": 20, "width": 40, "height": 80}
        store.add_detection(record, crop)
        condition = PersonSearchCondition(clothing=[ItemCondition(category="upper_body", colors=[color])])
        store.add_attributes(condition_to_attributes(detection_id=number, condition=condition, source="manual"))
    reference = tmp_path / "query.png"
    Image.new("RGB", (40, 80), "red").save(reference)
    return database, reference, store


def searcher(database, **kwargs):
    import importlib.util
    assert importlib.util.find_spec("person_searcher.image_search"), "Image search service is missing"
    from person_searcher.image_search import ImageSearcher
    return ImageSearcher(database, **kwargs)


def test_limit_applies_after_ranking_all_detection_candidates(catalog):
    database, query, _ = catalog
    report = searcher(database).search(query, limit=1)
    assert report["scanned_count"] == 3
    assert len(report["results"]) == 1
    match = report["results"][0]
    assert match["id"] == 3
    assert match["similarity"] == 1.0
    assert match["video_id"] == "cam-a"
    assert match["frame_number"] == 90
    assert [match[k] for k in ["x", "y", "width", "height"]] == [10, 20, 40, 80]


def test_source_and_attribute_filters_are_applied_before_ranking(catalog):
    database, query, _ = catalog
    service = searcher(database)
    report = service.search(query, video_id="cam-b")
    assert [r["id"] for r in report["results"]] == [2]
    condition = PersonSearchCondition(clothing=[ItemCondition(category="upper_body", colors=["red"])])
    report = service.search(query, condition=condition)
    assert [r["id"] for r in report["results"]] == [3]
    assert service.search(query, condition=condition, video_id="cam-b")["results"] == []


def test_features_are_persisted_reused_and_invalidated_when_crop_changes(catalog, monkeypatch):
    from person_searcher.vision.reid import ColorHistogramEmbedder

    database, query, _ = catalog
    service = searcher(database)
    service.index()
    with sqlite3.connect(database) as connection:
        rows = connection.execute("SELECT detection_id, fingerprint, features_json FROM person_visual_feature").fetchall()
    assert len(rows) == 3
    assert all(len(row[1]) == 64 for row in rows)
    assert all(json.loads(row[2]) for row in rows)
    original = ColorHistogramEmbedder.embed

    def reference_only(self, path):
        assert path == query, "An unchanged cached crop should not be re-embedded"
        return original(self, path)

    with monkeypatch.context() as patch:
        patch.setattr(ColorHistogramEmbedder, "embed", reference_only)
        assert service.search(query)["results"][0]["id"] == 3
    Image.new("RGB", (40, 80), "blue").save(query.parent / "person-3.png")
    assert service.search(query, min_similarity=0.9)["results"] == []
    with sqlite3.connect(database) as connection:
        new_hash = connection.execute("SELECT fingerprint FROM person_visual_feature WHERE detection_id=3").fetchone()[0]
    assert new_hash != next(r[1] for r in rows if r[0] == 3)


def test_bad_candidate_is_reported_without_discarding_valid_results(catalog):
    database, query, _ = catalog
    (query.parent / "person-1.png").unlink()
    (query.parent / "person-2.png").write_text("not an image")
    report = searcher(database).search(query)
    assert [r["id"] for r in report["results"]] == [3]
    assert report["skipped_count"] == 2
    assert {e["detection_id"] for e in report["errors"]} == {1, 2}


def test_corrupt_cached_features_are_recomputed(catalog):
    database, query, _ = catalog
    service = searcher(database)
    service.index()
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE person_visual_feature SET features_json='broken'")
    assert service.search(query)["results"][0]["id"] == 3


def test_query_is_validated_and_missing_database_is_not_created(tmp_path, catalog):
    with pytest.raises(FileNotFoundError):
        searcher(tmp_path / "absent.db")
    assert not (tmp_path / "absent.db").exists()
    database, query, _ = catalog
    query.write_text("broken")
    with pytest.raises(ValueError, match="image|Image|decode"):
        searcher(database).search(query)


@pytest.mark.parametrize("arguments", [{"limit": 0}, {"min_similarity": float("nan")}, {"min_similarity": 1.1}])
def test_invalid_search_parameters_fail_before_database_changes(catalog, arguments):
    database, query, _ = catalog
    with pytest.raises(ValueError):
        searcher(database).search(query, **arguments)


def test_cosine_normalizes_general_vectors():
    assert cosine_similarity([3, 0], [4, 0]) == pytest.approx(1.0)
    assert cosine_similarity([3, 4], [4, -3]) == pytest.approx(0.0)


@pytest.mark.parametrize("left,right", [([0, 0], [1, 0]), ([float("nan")], [1]), ([float("inf")], [1])])
def test_cosine_rejects_nonfinite_and_zero_vectors(left, right):
    with pytest.raises(ValueError):
        cosine_similarity(left, right)


def run_cli(*arguments):
    import subprocess
    import sys
    return subprocess.run([sys.executable, "-B", "-m", "person_searcher.image_cli", *map(str, arguments)],
                          capture_output=True, text=True)


def test_cli_ranks_full_catalog_and_outputs_json(catalog):
    database, query, _ = catalog
    result = run_cli("--database", database, "--reference", query, "--limit", 1, "--json")
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["results"][0]["id"] == 3
    assert report["scanned_count"] == 3


def test_cli_can_index_without_a_query_and_reuse_features(catalog):
    database, _, _ = catalog
    first = run_cli("--database", database, "--index-only", "--json")
    assert first.returncode == 0, first.stderr
    assert json.loads(first.stdout)["indexed_count"] == 3
    second = run_cli("--database", database, "--index-only", "--json")
    assert json.loads(second.stdout)["cached_count"] == 3


@pytest.mark.parametrize("arguments", [("--limit", "0"), ("--min-similarity", "nan"), ("--backend", "hybrid")])
def test_cli_bad_arguments_have_useful_errors_without_tracebacks(catalog, arguments):
    database, query, _ = catalog
    result = run_cli("--database", database, "--reference", query, *arguments)
    assert result.returncode == 2
    assert "error:" in result.stderr.lower()
    assert "Traceback" not in result.stderr


def test_cli_corrupt_reference_has_no_traceback(catalog):
    database, query, _ = catalog
    query.write_text("corrupt")
    result = run_cli("--database", database, "--reference", query)
    assert result.returncode == 2
    assert "decode image" in result.stderr.lower()
    assert "Traceback" not in result.stderr


def test_public_package_exposes_image_searcher(catalog):
    import person_searcher
    assert hasattr(person_searcher, "ImageSearcher")
    database, query, _ = catalog
    assert person_searcher.ImageSearcher(database).search(query, limit=1)["results"][0]["id"] == 3


def test_cli_query_preserves_saved_few_shot_examples(catalog, monkeypatch, capsys):
    from types import SimpleNamespace
    from person_searcher.image_cli import main
    from person_searcher.query.ollama_parser import OllamaQueryParser
    from person_searcher.storage.few_shot import FewShotStore

    database, query, _ = catalog
    condition = PersonSearchCondition(clothing=[ItemCondition(category="upper_body", colors=["red"])])
    examples = FewShotStore(database)
    examples.initialize()
    examples.add("custom saved red jacket example", condition)

    class LocalClient:
        def chat(self, **arguments):
            assert "custom saved red jacket example" in arguments["messages"][0]["content"]
            return SimpleNamespace(message=SimpleNamespace(content=condition.model_dump_json()))

    monkeypatch.setattr(OllamaQueryParser, "client", property(lambda self: LocalClient()))
    assert main(["--database", str(database), "--reference", str(query), "--query", "red top", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["results"][0]["id"] == 3


def test_top_k_uses_full_precision_before_rounding(catalog):
    database, query, _ = catalog

    class PreciseMatcher:
        cache_key = "test-precision"

        def extract(self, path):
            return {"file": path.name}

        def validate(self, features):
            pass

        def similarity(self, left, right):
            return {"person-1.png": 0.9000001, "person-2.png": 0.9000004, "person-3.png": 0.1}[right["file"]]

    assert searcher(database, matcher=PreciseMatcher()).search(query, limit=1)["results"][0]["id"] == 2


def test_existing_text_query_keeps_matching_child_detections(catalog, monkeypatch, capsys):
    from types import SimpleNamespace
    from person_searcher.image_cli import main
    from person_searcher.query.ollama_parser import OllamaQueryParser

    database, query, _ = catalog
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE person_detection SET category='child' WHERE id=3")
    condition = PersonSearchCondition(clothing=[ItemCondition(category="upper_body", colors=["red"])])

    class LocalClient:
        def chat(self, **arguments):
            return SimpleNamespace(message=SimpleNamespace(content=condition.model_dump_json()))

    monkeypatch.setattr(OllamaQueryParser, "client", property(lambda self: LocalClient()))
    assert main(["--database", str(database), "--reference", str(query), "--query", "red top", "--json"]) == 0
    results = json.loads(capsys.readouterr().out)["results"]
    assert [r["id"] for r in results] == [3]
