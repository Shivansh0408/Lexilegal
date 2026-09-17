"""Durable upload and analysis-result storage for the PDF pipelines."""

from __future__ import annotations

import hashlib
import io
import json
import os
import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from werkzeug.utils import secure_filename

BACKEND_DIR = Path(__file__).resolve().parents[1]
_CONFIGURED_DATA_DIR = os.getenv("ANALYSIS_DATA_DIR", "").strip()
DATA_DIR = Path(_CONFIGURED_DATA_DIR or (BACKEND_DIR / "saved_analysis")).resolve()
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"
_LOCK = threading.RLock()
CACHE_VERSIONS = {"judge": 1, "lawyer": 2}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def load_saved_analysis(pipeline: str, document_id: str) -> dict:
    """Load one completed analysis by its exact pipeline and SHA-256 id."""

    if pipeline not in CACHE_VERSIONS:
        raise ValueError("Unsupported analysis pipeline.")
    if len(document_id) != 64 or any(character not in "0123456789abcdef" for character in document_id.lower()):
        raise ValueError("Invalid document id.")

    output_path = OUTPUT_DIR / pipeline / f"{document_id.lower()}.json"
    try:
        record = json.loads(output_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError("No saved analysis was found for this document.") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("The saved analysis could not be read.") from exc

    if record.get("pipeline") != pipeline or record.get("document_id") != document_id.lower():
        raise ValueError("The saved analysis identity does not match the request.")
    analysis = record.get("analysis")
    if not isinstance(analysis, dict):
        raise ValueError("The saved analysis is invalid.")
    return deepcopy(analysis)


def analyse_pdf_with_cache(
    pipeline: str,
    pdf_bytes: bytes,
    filename: str,
    analyser: Callable,
) -> tuple[dict, bool, dict]:
    """Save the PDF and reuse a prior pipeline result for identical bytes."""

    if pipeline not in {"judge", "lawyer"}:
        raise ValueError("Unsupported analysis pipeline.")
    safe_name = secure_filename(filename) or "uploaded.pdf"
    digest = hashlib.sha256(pdf_bytes).hexdigest()
    upload_path = UPLOAD_DIR / digest / safe_name
    output_path = OUTPUT_DIR / pipeline / f"{digest}.json"
    cache_version = CACHE_VERSIONS[pipeline]

    with _LOCK:
        upload_path.parent.mkdir(parents=True, exist_ok=True)
        if not upload_path.exists():
            upload_path.write_bytes(pdf_bytes)

        if output_path.exists():
            try:
                record = json.loads(output_path.read_text(encoding="utf-8"))
                if record.get("cache_version") != cache_version:
                    raise ValueError("Cached analysis was created by an older pipeline version.")
                saved_analysis = record.get("analysis")
                if not isinstance(saved_analysis, dict):
                    raise ValueError("Cached analysis is invalid.")
                filenames = record.setdefault("filenames", [])
                if safe_name not in filenames:
                    filenames.append(safe_name)
                    record["last_accessed_at"] = _utc_now()
                    _write_json(output_path, record)
                analysis = deepcopy(saved_analysis)
                analysis.setdefault("document", {})["filename"] = safe_name
                return analysis, True, {
                    "document_id": digest,
                    "saved_filename": safe_name,
                    "result_file": output_path.name,
                }
            except (OSError, ValueError, json.JSONDecodeError):
                # A damaged cache is safely replaced by a fresh completed analysis.
                pass

        analysis = analyser(io.BytesIO(pdf_bytes), safe_name)
        if not isinstance(analysis, dict):
            raise ValueError("The analysis pipeline did not return an object.")
        record = {
            "document_id": digest,
            "pipeline": pipeline,
            "cache_version": cache_version,
            "filenames": [safe_name],
            "saved_upload": str(upload_path.relative_to(DATA_DIR)),
            "created_at": _utc_now(),
            "analysis": analysis,
        }
        _write_json(output_path, record)
        return deepcopy(analysis), False, {
            "document_id": digest,
            "saved_filename": safe_name,
            "result_file": output_path.name,
        }
