"""LLM-based extraction of judicially useful facts from court-file text."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Iterable

import requests

LOGGER = logging.getLogger("lexbrief.llm")


class LlmJudgeError(RuntimeError):
    """Raised when the configured local LLM cannot return an analysis."""


SYSTEM_INSTRUCTIONS = """You are a neutral judicial fact-analysis assistant for Indian court files.
Extract only information supported by the supplied document. Do not decide guilt, liability, credibility,
sentence, or the final outcome. Do not invent missing details. Clearly separate allegations, admissions,
disputed assertions, testimony, documents, medical/forensic material, electronic material, and procedural
events. Track who asserts each important point and identify contradictions, corroboration, chain-of-custody
or authenticity concerns, and missing information. The legal dictionary context is retrieval assistance only;
do not claim that a provision applies merely because it was retrieved. Return JSON only.

Return this object. Arrays have no maximum length and must include every material item found:
{
  "case_title": "string or null",
  "case_overview": "neutral short overview",
  "parties": [{"name":"string","role":"string","description":"string"}],
  "facts": [{"fact":"string","status":"alleged|admitted|disputed|supported|unclear","asserted_by":"string or null","support":"string or null","page_reference":"string or null","legal_significance":"string","confidence":"high|medium|low"}],
  "chronology": [{"date":"string or unknown","event":"string","source":"string or null"}],
  "evidence": [{"item":"string","type":"testimonial|documentary|electronic|medical|forensic|physical|other","offered_by":"string or null","proves_or_rebuts":"string","reliability_note":"string","source_reference":"string or null"}],
  "admitted_or_undisputed_facts": ["string"],
  "disputed_facts": [{"issue":"string","positions":"string","supporting_material":"string"}],
  "credibility_and_reliability": ["neutral observation grounded in the file"],
  "procedural_history": ["string"],
  "questions_for_determination": ["fact-dependent question, not a conclusion"],
  "missing_information": ["string"],
  "legal_context": [{"law":"Act name only, no section number","relevance":"string"}],
  "analysis_limitations": ["string"]
}
"""


def _chunk_text(text: str, size: int) -> list[str]:
    """Split at paragraph boundaries without silently dropping any text."""

    if len(text) <= size:
        return [text]
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) > size:
            if current:
                chunks.append(current)
                current = ""
            for start in range(0, len(paragraph), size):
                chunks.append(paragraph[start : start + size])
            continue
        candidate = f"{current}\n\n{paragraph}".strip()
        if current and len(candidate) > size:
            chunks.append(current)
            current = paragraph
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _dictionary_context(entries: Iterable[dict]) -> str:
    lines = []
    # A small, high-quality context is much more reliable on CPU-hosted models
    # than repeating dozens of long bare-act summaries for every document chunk.
    for entry in list(entries)[:12]:
        explanation = str(entry.get("what", ""))[:500]
        lines.append(
            f"- {entry.get('law', 'Unknown law')} | section {entry.get('section', '?')} | "
            f"{entry.get('title', '')}: {explanation}"
        )
    return "\n".join(lines) or "No close dictionary match was found."


def _call_ollama(
    prompt: str,
    *,
    timeout_seconds: int | None = None,
    num_ctx: int | None = None,
    num_predict: int | None = None,
) -> str:
    """Call Ollama with optional per-pipeline performance limits."""

    api_url = os.getenv("LLAMA_API_URL", "http://localhost:11434/api/generate")
    model = os.getenv("LLAMA_MODEL", "llama3.2:3b")
    timeout = timeout_seconds or int(os.getenv("LLAMA_TIMEOUT_SECONDS", "300"))
    attempts = max(1, int(os.getenv("LLAMA_RETRIES", "1")))
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        LOGGER.info("Calling Ollama model %s (attempt %d/%d)", model, attempt, attempts)
        try:
            response = requests.post(
                api_url,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "keep_alive": "10m",
                    "options": {
                        "temperature": 0.1,
                        "num_ctx": num_ctx or int(os.getenv("LLAMA_NUM_CTX", "16384")),
                        "num_predict": num_predict or int(os.getenv("LLAMA_NUM_PREDICT", "4096")),
                    },
                },
                timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()
            result = payload.get("response")
            if not isinstance(result, str) or not result.strip():
                raise LlmJudgeError("The local LLM returned an empty response.")
            return result.strip()
        except (requests.RequestException, ValueError, LlmJudgeError) as exc:
            last_error = exc
            LOGGER.warning("Ollama attempt %d/%d failed: %s", attempt, attempts, exc)
            if attempt < attempts:
                time.sleep(min(2 * attempt, 5))
    raise LlmJudgeError(f"The local LLM request failed after {attempts} attempt(s): {last_error}")


def analyse_with_llm(document_text: str, legal_entries: list[dict]) -> tuple[list[str], list[dict]]:
    """Analyse every chunk while isolating failures to the affected chunk."""

    chunk_size = max(4_000, int(os.getenv("JUDGE_CHUNK_CHARACTERS", "8000")))
    chunks = _chunk_text(document_text, chunk_size)
    legal_context = _dictionary_context(legal_entries)
    LOGGER.info("LLM step 1/2: split the document into %d analysis chunk(s)", len(chunks))
    outputs: list[str] = []
    failures: list[dict] = []
    for index, chunk in enumerate(chunks, start=1):
        LOGGER.info("LLM step 2/2: analysing chunk %d/%d", index, len(chunks))
        prompt = (
            f"{SYSTEM_INSTRUCTIONS}\n\n"
            f"LEGAL DICTIONARY CONTEXT (do not copy section numbers to output):\n{legal_context}\n\n"
            f"DOCUMENT CHUNK {index} OF {len(chunks)}:\n{chunk}"
        )
        try:
            raw_output = _call_ollama(prompt)
            json_candidate = raw_output.strip()
            if json_candidate.startswith("```"):
                json_candidate = re.sub(r"^```(?:json)?\s*", "", json_candidate, flags=re.IGNORECASE)
                json_candidate = re.sub(r"\s*```$", "", json_candidate)
            try:
                json.loads(json_candidate)
            except json.JSONDecodeError as exc:
                raise LlmJudgeError(f"The local LLM returned invalid JSON: {exc}") from exc
            outputs.append(raw_output)
        except LlmJudgeError as exc:
            LOGGER.error("Chunk %d/%d failed and will use extractive recovery: %s", index, len(chunks), exc)
            failures.append({"chunk_index": index, "text": chunk, "error": str(exc)})
    return outputs, failures
