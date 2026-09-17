"""Build or refresh the persistent Chroma index for every saved analysis JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR.parent) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR.parent))

from backend.core.chat import RagError, sync_saved_analyses


def main() -> int:
    parser = argparse.ArgumentParser(description="Index backend/saved_analysis/outputs into ChromaDB.")
    parser.add_argument("--force", action="store_true", help="Re-embed every JSON even when unchanged.")
    args = parser.parse_args()
    try:
        result = sync_saved_analyses(force=args.force)
    except RagError as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 1
    print(json.dumps(result, indent=2))
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
