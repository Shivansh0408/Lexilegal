"""Flask API entry point for LexBrief's PDF judicial-fact analysis."""

from __future__ import annotations

import logging
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR.parent) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR.parent))

load_dotenv(BACKEND_DIR / ".env")
load_dotenv(BACKEND_DIR / "collector" / ".env")

from backend.core.analysis import analyse_pdf  # noqa: E402
from backend.core.chat import (  # noqa: E402
    RagError,
    answer_saved_analysis_question,
    index_saved_analysis,
    sync_saved_analyses,
)
from backend.core.lawyer_analysis import analyse_pdf_for_lawyer  # noqa: E402
from backend.core.pdf_processing import PdfProcessingError  # noqa: E402
from backend.core.storage import analyse_pdf_with_cache  # noqa: E402


def _configure_logging() -> None:
    level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )


def create_app() -> Flask:
    _configure_logging()
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_UPLOAD_MB", "25")) * 1024 * 1024
    allowed_origins = [
        item.strip()
        for item in os.getenv("FRONTEND_ORIGINS", "http://localhost:3000").split(",")
        if item.strip()
    ]
    CORS(app, resources={r"/api/*": {"origins": allowed_origins}})
    logger = logging.getLogger("lexbrief.api")

    @app.get("/api/health")
    def health():
        return jsonify({
            "status": "ok",
            "service": "lexbrief-pdf-analysis",
            "pipelines": ["judge", "lawyer"],
            "saved_analysis_chat": True,
        })

    def run_pdf_pipeline(pipeline: str, analyser):
        request_id = uuid.uuid4().hex[:12]
        logger.info("Request %s: %s upload received", request_id, pipeline)
        uploaded = request.files.get("file")
        if uploaded is None:
            return jsonify({"error": "A PDF is required in multipart field 'file'.", "request_id": request_id}), 400

        filename = secure_filename(uploaded.filename or "")
        if not filename or not filename.lower().endswith(".pdf"):
            return jsonify({"error": "Only .pdf files are accepted.", "request_id": request_id}), 400

        pdf_bytes = uploaded.stream.read()
        if pdf_bytes[:5] != b"%PDF-":
            return jsonify({"error": "The uploaded file does not have a valid PDF signature.", "request_id": request_id}), 400

        try:
            result, cache_hit, storage = analyse_pdf_with_cache(pipeline, pdf_bytes, filename, analyser)
        except PdfProcessingError as exc:
            logger.warning("Request %s rejected: %s", request_id, exc)
            return jsonify({"error": str(exc), "request_id": request_id}), 422
        except Exception:
            logger.exception("Request %s failed unexpectedly", request_id)
            return jsonify({"error": "The document could not be analysed.", "request_id": request_id}), 500

        rag_indexed = False
        rag_error = None
        try:
            indexed_chunks = index_saved_analysis(pipeline, storage["document_id"])
            rag_indexed = indexed_chunks > 0
            logger.info(
                "Request %s: stored %s analysis in Chroma (%d chunks)",
                request_id,
                pipeline,
                indexed_chunks,
            )
        except (RagError, ValueError, FileNotFoundError) as exc:
            rag_error = str(exc)
            logger.warning("Request %s: analysis saved but RAG indexing is unavailable: %s", request_id, exc)

        logger.info("Request %s: sending validated %s analysis to frontend", request_id, pipeline)
        return jsonify({
            "request_id": request_id,
            "analysis": result,
            "cached": cache_hit,
            "document_id": storage["document_id"],
            "saved_filename": storage["saved_filename"],
            "rag_indexed": rag_indexed,
            "rag_error": rag_error,
        })

    @app.post("/api/judge/analyze")
    def judge_analyse():
        return run_pdf_pipeline("judge", analyse_pdf)

    @app.post("/api/lawyer/analyze")
    def lawyer_analyse():
        return run_pdf_pipeline("lawyer", analyse_pdf_for_lawyer)

    @app.post("/api/chat")
    def chat():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "A JSON request body is required."}), 400

        pipeline = str(payload.get("pipeline", "")).strip().lower()
        document_id = str(payload.get("document_id", "")).strip().lower()
        question = str(payload.get("question", "")).strip()
        history = payload.get("history", [])
        if pipeline not in {"judge", "lawyer"}:
            return jsonify({"error": "Pipeline must be 'judge' or 'lawyer'."}), 400
        if not question:
            return jsonify({"error": "Please enter a question."}), 400
        if len(question) > 2_000:
            return jsonify({"error": "The question is too long (maximum 2,000 characters)."}), 400
        if not isinstance(history, list):
            return jsonify({"error": "History must be an array."}), 400

        try:
            result = answer_saved_analysis_question(pipeline, document_id, question, history[:12])
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except RagError as exc:
            logger.warning("RAG chat request failed: %s", exc)
            return jsonify({"error": str(exc)}), 503
        except Exception:
            logger.exception("Chat request failed unexpectedly")
            return jsonify({"error": "The question could not be answered."}), 500
        return jsonify(result)

    @app.post("/api/chat/reindex")
    def reindex_chat_knowledge_base():
        payload = request.get_json(silent=True) or {}
        try:
            result = sync_saved_analyses(force=bool(payload.get("force", False)))
        except RagError as exc:
            return jsonify({"error": str(exc)}), 503
        status = 207 if result["errors"] else 200
        return jsonify(result), status

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_error):
        limit = os.getenv("MAX_UPLOAD_MB", "25")
        return jsonify({"error": f"The PDF exceeds the configured {limit} MB upload limit."}), 413

    return app


app = create_app()

if __name__ == "__main__":
    app.run(
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "false").lower() == "true",
    )
