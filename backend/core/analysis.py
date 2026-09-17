"""Orchestration layer for the independent judicial fact pipeline."""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path

from .LLM_judge import LlmJudgeError, analyse_with_llm
from .pdf_processing import extract_pdf_text
from .response_parser import extractive_analysis, format_for_frontend, parse_and_merge

LOGGER = logging.getLogger("lexbrief.analysis")
BACKEND_DIR = Path(__file__).resolve().parents[1]
DICTIONARY_DIR = BACKEND_DIR / "collector" / "dictionary"
STOPWORDS = {
    "about", "after", "against", "also", "been", "being", "between", "from", "have", "into",
    "shall", "that", "their", "there", "these", "this", "those", "under", "upon", "were", "where",
    "which", "with", "would", "person", "section", "court", "india", "law",
}


def _tokens(value: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-zA-Z]{4,}", value.lower())
        if token not in STOPWORDS
    }


@lru_cache(maxsize=1)
def load_dictionary_sections() -> tuple[dict, ...]:
    """Load, but never mutate, the collector's generated section dictionaries."""

    LOGGER.info("Dictionary step 1/2: loading section dictionaries from %s", DICTIONARY_DIR)
    entries: list[dict] = []
    if not DICTIONARY_DIR.exists():
        LOGGER.warning("Dictionary directory is missing")
        return tuple()
    for path in DICTIONARY_DIR.rglob("*.json"):
        if path.name in {"index.json", "discovery_manifest.json"}:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.warning("Skipping unreadable dictionary file %s: %s", path, exc)
            continue
        sections = payload.get("sections")
        if not isinstance(sections, dict):
            continue
        law = str(payload.get("law") or path.stem)
        for section_key, section_value in sections.items():
            if not isinstance(section_value, dict):
                continue
            entry = {
                "law": law,
                "section": str(section_value.get("section") or section_key),
                "title": str(section_value.get("title") or ""),
                "what": str(section_value.get("what") or ""),
                "chapter": str(section_value.get("chapter") or ""),
            }
            entry["_tokens"] = _tokens(" ".join(entry.values()))
            entries.append(entry)
    LOGGER.info("Dictionary step 2/2: loaded %d legal section record(s)", len(entries))
    return tuple(entries)


def find_relevant_sections(document_text: str, limit: int = 30) -> list[dict]:
    """Retrieve relevant law context by transparent lexical overlap."""

    doc_tokens = _tokens(document_text)
    scored: list[tuple[float, dict]] = []
    for entry in load_dictionary_sections():
        shared = doc_tokens.intersection(entry["_tokens"])
        if not shared:
            continue
        score = len(shared) / max(1, len(entry["_tokens"]) ** 0.5)
        if entry["title"] and entry["title"].lower() in document_text.lower():
            score += 5
        scored.append((score, entry))
    scored.sort(key=lambda item: item[0], reverse=True)
    selected = []
    for _, entry in scored[:limit]:
        public_entry = {key: value for key, value in entry.items() if key != "_tokens"}
        selected.append(public_entry)
    LOGGER.info("Selected %d dictionary section(s) as private LLM context", len(selected))
    return selected


def analyse_pdf(file_stream, filename: str) -> dict:
    """Run extraction, retrieval, fact analysis, validation, and formatting."""

    LOGGER.info("Pipeline step 1/5: processing PDF %s", filename)
    extraction = extract_pdf_text(file_stream)

    LOGGER.info("Pipeline step 2/5: retrieving relevant legal dictionary context")
    relevant_sections = find_relevant_sections(extraction.text)
    matched_laws = [entry["law"] for entry in relevant_sections]

    LOGGER.info("Pipeline step 3/5: extracting judicially useful facts")
    try:
        raw_outputs, failed_chunks = analyse_with_llm(extraction.text, relevant_sections)
        if failed_chunks:
            failed_text = "\n\n".join(chunk["text"] for chunk in failed_chunks)
            unique_errors = list(dict.fromkeys(chunk["error"] for chunk in failed_chunks))
            affected = ", ".join(str(chunk["chunk_index"]) for chunk in failed_chunks)
            failure_reason = f"affected chunk(s) {affected}; " + "; ".join(unique_errors)
            failure_reason = failure_reason[:1500]
            recovery = extractive_analysis(failed_text, failure_reason)
            raw_outputs.append(json.dumps(recovery, ensure_ascii=False))
        if not raw_outputs:
            analysis = extractive_analysis(extraction.text, "The LLM returned no completed chunks.")
        else:
            analysis = parse_and_merge(raw_outputs)
            if not analysis.get("facts"):
                analysis = extractive_analysis(extraction.text, "The LLM returned no fact items.")
    except (LlmJudgeError, ValueError, json.JSONDecodeError) as exc:
        LOGGER.exception("LLM analysis could not be completed")
        analysis = extractive_analysis(extraction.text, str(exc))

    LOGGER.info("Pipeline step 4/5: validating and parsing analysis")
    document = extraction.to_dict()
    document.pop("text", None)
    document["filename"] = filename
    result = format_for_frontend(analysis, document, matched_laws)

    LOGGER.info("Pipeline step 5/5: completed with %d fact(s)", result["fact_count"])
    return result
