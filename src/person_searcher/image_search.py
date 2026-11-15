"""Image retrieval over the repository's existing person_detection records."""
from __future__ import annotations

from contextlib import closing
import hashlib
import heapq
import json
import math
from pathlib import Path
import sqlite3
from typing import Any, Protocol

from person_searcher.domain.models import PersonSearchCondition
from person_searcher.storage.detection_repository import DetectionRepository
from person_searcher.vision.reid import ColorHistogramEmbedder, cosine_similarity


class ImageMatcher(Protocol):
    """A versioned, JSON-serializable feature extractor and similarity scorer."""
    cache_key: str

    def extract(self, image_path: Path) -> dict[str, Any]: ...
    def validate(self, features: dict[str, Any]) -> None: ...
    def similarity(self, left: dict[str, Any], right: dict[str, Any]) -> float: ...


class HistogramMatcher:
    def __init__(self, bins_per_channel: int = 8):
        self.embedder = ColorHistogramEmbedder(bins_per_channel)
        self.cache_key = f"rgb-histogram-v2:{bins_per_channel}"

    def extract(self, image_path: Path) -> dict[str, Any]:
        return {"embedding": self.embedder.embed(image_path)}

    def validate(self, features: dict[str, Any]) -> None:
        values = features["embedding"]
        if not isinstance(values, list) or len(values) != 3 * self.embedder.bins_per_channel:
            raise ValueError("Invalid histogram feature dimensions")
        cosine_similarity(values, values)

    def similarity(self, left: dict[str, Any], right: dict[str, Any]) -> float:
        return cosine_similarity(left["embedding"], right["embedding"])


class ImageSearcher:
    def __init__(self, database_path: str | Path, *, matcher: ImageMatcher | None = None):
        self.database_path = Path(database_path).expanduser().resolve()
        if not self.database_path.is_file():
            raise FileNotFoundError(f"Detection database not found: {self.database_path}")
        with closing(sqlite3.connect(self.database_path.as_uri() + "?mode=ro", uri=True)) as connection:
            if not connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='person_detection'").fetchone():
                raise ValueError("Database has no person_detection table; run annotation/YOLO ingestion first")
        self.matcher = matcher or HistogramMatcher()
        self.repository = DetectionRepository(self.database_path)

    def _candidates(
        self, *, condition: PersonSearchCondition | None = None, video_id: str | None = None,
        event_class: str | None = None, split: str | None = None, category: str | None = "person",
    ) -> list[dict[str, object]]:
        if condition is None:
            return self.repository.find(video_id=video_id, event_class=event_class, split=split,
                                        category=category, limit=None)
        rows = self.repository.search_attributes(condition, limit=None)
        filters = {"video_id": video_id, "event_class": event_class, "split": split, "category": category}
        return [row for row in rows if all(value is None or row[key] == value for key, value in filters.items())]

    def _initialize_cache(self, connection: sqlite3.Connection) -> None:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("""
            CREATE TABLE IF NOT EXISTS person_visual_feature (
                detection_id INTEGER NOT NULL REFERENCES person_detection(id) ON DELETE CASCADE,
                backend TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                features_json TEXT NOT NULL,
                PRIMARY KEY (detection_id, backend)
            )
        """)
        connection.commit()

    @staticmethod
    def _fingerprint(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as image:
            for chunk in iter(lambda: image.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _features(self, connection: sqlite3.Connection, candidate: dict[str, object]) -> tuple[dict[str, Any], bool]:
        path = Path(str(candidate["crop_path"]))
        fingerprint = self._fingerprint(path)
        cached = connection.execute(
            "SELECT fingerprint, features_json FROM person_visual_feature WHERE detection_id=? AND backend=?",
            (candidate["id"], self.matcher.cache_key),
        ).fetchone()
        if cached and cached[0] == fingerprint:
            try:
                features = json.loads(cached[1])
                self.matcher.validate(features)
                return features, True
            except (ValueError, TypeError, KeyError):
                pass
        features = self.matcher.extract(path)
        self.matcher.validate(features)
        if self._fingerprint(path) != fingerprint:
            raise ValueError(f"Image changed while indexing; retry: {path}")
        connection.execute(
            "INSERT INTO person_visual_feature VALUES (?, ?, ?, ?) "
            "ON CONFLICT(detection_id, backend) DO UPDATE SET "
            "fingerprint=excluded.fingerprint, features_json=excluded.features_json",
            (candidate["id"], self.matcher.cache_key, fingerprint, json.dumps(features, allow_nan=False)),
        )
        # Do not hold a SQLite writer lock while the next image/model is processed.
        connection.commit()
        return features, False

    def index(
        self, *, video_id: str | None = None, event_class: str | None = None,
        split: str | None = None, category: str | None = "person",
    ) -> dict[str, Any]:
        report = {"backend": self.matcher.cache_key, "indexed_count": 0, "cached_count": 0, "errors": []}
        candidates = self._candidates(video_id=video_id, event_class=event_class, split=split, category=category)
        with closing(sqlite3.connect(self.database_path)) as connection:
            self._initialize_cache(connection)
            for candidate in candidates:
                try:
                    _, cached = self._features(connection, candidate)
                    report["cached_count" if cached else "indexed_count"] += 1
                except (OSError, ValueError) as exc:
                    report["errors"].append({"detection_id": candidate["id"], "error": str(exc)})
        report["skipped_count"] = len(report["errors"])
        return report

    def search(
        self, reference_path: str | Path, *, limit: int = 20, min_similarity: float = 0.0,
        video_id: str | None = None, event_class: str | None = None,
        split: str | None = None, category: str | None = "person",
        condition: PersonSearchCondition | None = None,
    ) -> dict[str, Any]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        if not math.isfinite(min_similarity) or not 0 <= min_similarity <= 1:
            raise ValueError("min_similarity must be between 0 and 1")
        reference = self.matcher.extract(Path(reference_path).expanduser())
        self.matcher.validate(reference)
        candidates = self._candidates(condition=condition, video_id=video_id, event_class=event_class,
                                      split=split, category=category)
        report = {"backend": self.matcher.cache_key, "scanned_count": 0, "cached_count": 0,
                  "skipped_count": 0, "results": [], "errors": []}

        def ranked_candidates(connection):
            for candidate in candidates:
                try:
                    features, cached = self._features(connection, candidate)
                    score = self.matcher.similarity(reference, features)
                    if not math.isfinite(score) or not -1 <= score <= 1:
                        raise ValueError("Matcher returned an invalid similarity")
                    report["scanned_count"] += 1
                    report["cached_count"] += int(cached)
                    if score >= min_similarity:
                        yield {**candidate, "similarity": score}
                except (OSError, ValueError) as exc:
                    report["errors"].append({"detection_id": candidate["id"], "error": str(exc)})

        with closing(sqlite3.connect(self.database_path)) as connection:
            self._initialize_cache(connection)
            report["results"] = heapq.nlargest(
                limit, ranked_candidates(connection), key=lambda row: (row["similarity"], -row["id"]),
            )
        for match in report["results"]:
            match["similarity"] = round(match["similarity"], 6)
        report["skipped_count"] = len(report["errors"])
        return report
