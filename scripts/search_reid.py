"""Compatibility entry point; prefer the installed person-image-search command."""
from person_searcher.image_cli import main


if __name__ == "__main__":
    raise SystemExit(main())
