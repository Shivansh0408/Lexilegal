"""LLM extraction of source-grounded facts and strategy options for counsel."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Iterable

from .LLM_judge import LlmJudgeError, _call_ollama, _chunk_text

LOGGER = logging.getLogger("lexbrief.lawyer_llm")


class LlmLawyerError(RuntimeError):
    """Raised when the configured local LLM cannot return lawyer analysis."""


SYSTEM_INSTRUCTIONS = """You are a legal strategy analysis assistant for lawyers reviewing Indian case files.
Create a source-grounded working brief, not a judgment and not a prediction of guilt or liability. Work for the
client position actually described in the record; if the represented side is unclear, say so and present neutral
options for both sides. Never invent facts, authorities, deadlines, evidence, admissions, or procedural history.
Preserve allegations as allegations, client instructions as instructions, and conflicts as separate assertions.
Attach a source or page reference whenever the chunk permits it. The legal dictionary is retrieval assistance
only: recommend counsel verify the current authentic text and do not state that a provision applies merely
because it was retrieved. Strategy must remain lawful and ethical: never recommend destroying or altering
evidence, contacting represented or protected persons improperly, coaching witnesses, evading process, or
misleading a court. Return JSON only.

Return one JSON object using only the fields that have material information in this chunk. Omit empty fields and
empty arrays. Be concise, but capture every material fact in the chunk. The accepted fields and item shapes are:
{
  "case_title": "string or null",
  "case_overview": "short source-grounded overview",
  "client_position": "represented side and stated position, or unclear",
  "strategy_facts": [{"fact":"string","status":"alleged|admitted|disputed|supported|unclear","source":"string or null","page_reference":"string or null","strategy_use":"string","confidence":"high|medium|low"}],
  "objectives": ["client objective stated in the record or a clearly labelled counsel decision point"],
  "case_theories": [{"side":"string","theory":"string","support":"string","weakness":"string"}],
  "elements_and_issues": [{"issue":"string","supporting_facts":["string"],"contrary_facts":["string"],"research_needed":"string"}],
  "strengths": [{"point":"string","basis":"string","source_reference":"string or null"}],
  "vulnerabilities": [{"point":"string","impact":"string","response_option":"string","source_reference":"string or null"}],
  "contradictions": [{"issue":"string","assertion_one":"string","assertion_two":"string","sources":"string","follow_up":"string"}],
  "evidence_assessment": [{"item":"string","helps":"string","hurts":"string","authenticity_or_admissibility":"string","next_step":"string","source_reference":"string or null"}],
  "procedural_opportunities": [{"issue":"string","record_basis":"string","law_to_verify":"string","possible_step":"string","timing":"string or unknown"}],
  "legal_arguments": [{"argument":"string","supporting_record":"string","counterargument":"string","authority_to_verify":"string"}],
  "investigation_plan": [{"priority":"urgent|high|normal","action":"string","purpose":"string","source_basis":"string"}],
  "disclosure_requests": [{"material":"string","reason":"string","source_basis":"string"}],
  "witness_plan": [{"witness":"string","role":"string","helpful_points":["string"],"testing_points":["string"],"lawful_follow_up":"string"}],
  "applications_and_motions": [{"step":"string","factual_basis":"string","relief_or_purpose":"string","law_to_verify":"string"}],
  "negotiation_considerations": ["fact-grounded consideration; do not invent an offer"],
  "hearing_trial_plan": [{"stage":"string","theme":"string","record_material":["string"],"preparation":"string"}],
  "deadlines_and_preservation": [{"item":"string","date":"string or unknown","action":"string","basis":"string"}],
  "risk_register": [{"risk":"string","likelihood":"high|medium|low|unknown","impact":"high|medium|low|unknown","mitigation":"string"}],
  "unresolved_questions": ["string"],
  "missing_information": ["string"],
  "next_actions": [{"priority":"urgent|high|normal","action":"string","why":"string"}],
  "legal_context": [{"law":"Act name only, no section number","relevance":"string"}],
  "ethics_and_safety": ["lawful handling, preservation, confidentiality, or verification safeguard"],
  "analysis_limitations": ["string"]
}
"""


PAGE_MARKER = re.compile(r"(?m)^\[Page\s+(\d+)\]\s*")


def _chunk_lawyer_text(text: str, size: int) -> list[str]:
    """Split page-aware text into small chunks without losing page provenance."""

    matches = list(PAGE_MARKER.finditer(text))
    if not matches:
        return _chunk_text(text, size)

    chunks: list[str] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        page_number = match.group(1)
        body = text[match.end() : end].strip()
        if not body:
            continue
        marker = f"[Page {page_number}]\n"
        available = max(800, size - len(marker))
        for piece in _chunk_text(body, available):
            chunks.append(f"{marker}{piece}")
    return chunks or _chunk_text(text, size)


def _lawyer_dictionary_context(entries: Iterable[dict]) -> str:
    """Keep repeated per-chunk legal context small enough for CPU models."""

    lines = []
    for entry in list(entries)[:6]:
        explanation = re.sub(r"\s+", " ", str(entry.get("what", ""))).strip()[:220]
        lines.append(
            f"- {entry.get('law', 'Unknown law')} | section {entry.get('section', '?')} | "
            f"{entry.get('title', '')}: {explanation}"
        )
    return "\n".join(lines) or "No close dictionary match was found."


def analyse_with_lawyer_llm(document_text: str, legal_entries: list[dict]) -> tuple[list[str], list[dict]]:
    """Analyse every document chunk with the independent lawyer prompt."""

    # Keep an older .env value such as 8000 from reintroducing CPU timeouts.
    chunk_size = min(3_000, max(1_200, int(os.getenv("LAWYER_CHUNK_CHARACTERS", "3000"))))
    chunks = _chunk_lawyer_text(document_text, chunk_size)
    legal_context = _lawyer_dictionary_context(legal_entries)
    timeout_seconds = max(30, int(os.getenv("LAWYER_TIMEOUT_SECONDS", "180")))
    num_ctx = max(4_096, int(os.getenv("LAWYER_NUM_CTX", "8192")))
    num_predict = max(512, int(os.getenv("LAWYER_NUM_PREDICT", "1200")))
    max_consecutive_failures = max(1, int(os.getenv("LAWYER_MAX_CONSECUTIVE_FAILURES", "2")))
    LOGGER.info(
        "Lawyer LLM step 1/2: split the document into %d page-aware chunk(s), maximum %d characters each",
        len(chunks),
        chunk_size,
    )
    outputs: list[str] = []
    failures: list[dict] = []
    consecutive_failures = 0
    for index, chunk in enumerate(chunks, start=1):
        LOGGER.info("Lawyer LLM step 2/2: analysing chunk %d/%d", index, len(chunks))
        prompt = (
            f"{SYSTEM_INSTRUCTIONS}\n\n"
            f"LEGAL DICTIONARY CONTEXT (research leads only):\n{legal_context}\n\n"
            f"DOCUMENT CHUNK {index} OF {len(chunks)}:\n{chunk}"
        )
        try:
            raw_output = _call_ollama(
                prompt,
                timeout_seconds=timeout_seconds,
                num_ctx=num_ctx,
                num_predict=num_predict,
            )
            candidate = raw_output.strip()
            if candidate.startswith("```"):
                candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
                candidate = re.sub(r"\s*```$", "", candidate)
            json.loads(candidate)
            outputs.append(raw_output)
            consecutive_failures = 0
        except (LlmJudgeError, json.JSONDecodeError) as exc:
            message = f"The lawyer LLM returned invalid output: {exc}"
            LOGGER.error("Chunk %d/%d failed: %s", index, len(chunks), message)
            failures.append({"chunk_index": index, "text": chunk, "error": message})
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures and index < len(chunks):
                skip_reason = (
                    f"Skipped after {consecutive_failures} consecutive Lawyer LLM failures; "
                    "structured extractive recovery was used instead."
                )
                LOGGER.error("Stopping Ollama calls early: %s", skip_reason)
                failures.extend(
                    {"chunk_index": skipped_index, "text": skipped_chunk, "error": skip_reason}
                    for skipped_index, skipped_chunk in enumerate(chunks[index:], start=index + 1)
                )
                break
    return outputs, failures
