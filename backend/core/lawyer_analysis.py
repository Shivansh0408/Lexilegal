"""Orchestration for the independent lawyer strategy pipeline."""

from __future__ import annotations

import json
import logging

from .analysis import find_relevant_sections
from .lawyer_LLM import LlmLawyerError, analyse_with_lawyer_llm
from .pdf_processing import extract_pdf_text
from .response_parser import (
    extractive_lawyer_analysis,
    format_lawyer_for_frontend,
    parse_and_merge_lawyer,
)

LOGGER = logging.getLogger("lexbrief.lawyer_analysis")


def analyse_pdf_for_lawyer(file_stream, filename: str) -> dict:
    """Extract, retrieve, analyse, recover, validate, and format lawyer output."""

    LOGGER.info("Lawyer pipeline step 1/5: processing PDF %s", filename)
    extraction = extract_pdf_text(file_stream)

    LOGGER.info("Lawyer pipeline step 2/5: retrieving legal dictionary context")
    relevant_sections = find_relevant_sections(extraction.text)
    matched_laws = [entry["law"] for entry in relevant_sections]

    LOGGER.info("Lawyer pipeline step 3/5: producing source-grounded strategy facts")
    try:
        raw_outputs, failed_chunks = analyse_with_lawyer_llm(extraction.text, relevant_sections)
        if failed_chunks:
            failed_text = "\n\n".join(chunk["text"] for chunk in failed_chunks)
            errors = list(dict.fromkeys(chunk["error"] for chunk in failed_chunks))
            affected = ", ".join(str(chunk["chunk_index"]) for chunk in failed_chunks)
            recovery = extractive_lawyer_analysis(
                failed_text,
                (f"affected chunk(s) {affected}; " + "; ".join(errors))[:1500],
            )
            raw_outputs.append(json.dumps(recovery, ensure_ascii=False))
        if not raw_outputs:
            analysis = extractive_lawyer_analysis(extraction.text, "The LLM returned no completed chunks.")
        else:
            analysis = parse_and_merge_lawyer(raw_outputs)
            if not analysis.get("strategy_facts"):
                analysis = extractive_lawyer_analysis(extraction.text, "The LLM returned no strategy facts.")
    except (LlmLawyerError, ValueError, json.JSONDecodeError) as exc:
        LOGGER.exception("Lawyer LLM analysis could not be completed")
        analysis = extractive_lawyer_analysis(extraction.text, str(exc))

    LOGGER.info("Lawyer pipeline step 4/5: validating strategy output")
    document = extraction.to_dict()
    document.pop("text", None)
    document["filename"] = filename
    result = format_lawyer_for_frontend(analysis, document, matched_laws)

    LOGGER.info("Lawyer pipeline step 5/5: completed with %d strategy fact(s)", result["fact_count"])
    return result
