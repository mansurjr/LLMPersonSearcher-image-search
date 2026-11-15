"""Command-line image search using the shared detection database."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sqlite3
from typing import Sequence

from person_searcher.image_search import ImageSearcher


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Index and search stored person crops by visual similarity.")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--reference", type=Path, help="Query person-crop image")
    action.add_argument("--index-only", action="store_true", help="Build/reuse visual features without a query")
    parser.add_argument("--database", type=Path, default=Path("artifacts/search.db"))
    parser.add_argument("--limit", type=int, default=20, help="Maximum results after ranking all filtered detections")
    parser.add_argument("--min-similarity", type=float, default=0.0)
    parser.add_argument("--video-id")
    parser.add_argument("--event-class")
    parser.add_argument("--split")
    parser.add_argument("--category", choices=["person", "child", "all"],
                        help="Default: person for image-only search; all categories with --query")
    parser.add_argument("--query", help="Optional Ollama natural-language attribute filter")
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--backend", choices=["histogram", "hybrid"], default="histogram")
    parser.add_argument("--models-dir", type=Path, help="Directory containing all five ONNX models (hybrid only)")
    parser.add_argument("--mode", choices=["auto", "face", "clothes", "strict"], default="auto")
    parser.add_argument("--json", action="store_true", help="Print a machine-readable report without extra stdout text")
    args = parser.parse_args(argv)
    if args.limit <= 0:
        parser.error("--limit must be greater than zero")
    if not math.isfinite(args.min_similarity) or not 0 <= args.min_similarity <= 1:
        parser.error("--min-similarity must be between 0 and 1")
    if args.backend == "hybrid" and args.models_dir is None:
        parser.error("--models-dir is required with --backend hybrid")
    if args.backend == "histogram" and (args.mode != "auto" or args.models_dir):
        parser.error("--mode and --models-dir require --backend hybrid")
    if args.index_only and args.query:
        parser.error("--query requires --reference")

    try:
        matcher = None
        if args.backend == "hybrid":
            from person_searcher.vision.hybrid import HybridMatcher
            matcher = HybridMatcher(args.models_dir, mode=args.mode)
        searcher = ImageSearcher(args.database, matcher=matcher)
        category = args.category or ("all" if args.query else "person")
        filters = {"video_id": args.video_id, "event_class": args.event_class, "split": args.split,
                   "category": None if category == "all" else category}
        if args.index_only:
            report = searcher.index(**filters)
        else:
            condition = None
            if args.query:
                from person_searcher.query.ollama_parser import OllamaQueryParser
                from person_searcher.storage.few_shot import FewShotStore
                examples = FewShotStore(args.database)
                examples.initialize()
                condition = OllamaQueryParser(model=args.model, examples=examples).parse(args.query)
            report = searcher.search(args.reference, limit=args.limit, min_similarity=args.min_similarity,
                                     condition=condition, **filters)
            if condition is not None:
                report["parsed_condition"] = condition.model_dump()
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        elif args.index_only:
            print(f"Indexed: {report['indexed_count']}; cached: {report['cached_count']}; skipped: {report['skipped_count']}")
        else:
            if "parsed_condition" in report:
                print("Parsed condition:")
                print(json.dumps(report["parsed_condition"], ensure_ascii=False, indent=2))
            print("Ranked matches:")
            print(json.dumps(report["results"], ensure_ascii=False, indent=2, allow_nan=False))
            print(f"Scanned: {report['scanned_count']}; skipped: {report['skipped_count']}")
        if not args.json:
            for error in report["errors"]:
                print(f"Skipped detection {error['detection_id']}: {error['error']}")
    except (OSError, RuntimeError, ValueError, sqlite3.Error) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
