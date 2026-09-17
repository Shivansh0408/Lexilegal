"""Evidence-grounded Act-name discovery through an Ollama-compatible LLM."""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import requests


LOGGER = logging.getLogger("lexbrief.llama_processor")


class LlamaProcessingError(RuntimeError):
    """Raised when Act discovery cannot be completed."""


# Seeds are candidates only; every one must resolve to an official source.
KNOWN_ACTS_BY_LAW_TYPE: dict[str, list[str]] = {
    "business law": [
        "The Indian Contract Act, 1872",
        "The Negotiable Instruments Act, 1881",
        "The Sale of Goods Act, 1930",
        "The Indian Partnership Act, 1932",
        "The Companies Act, 2013",
        "The Limited Liability Partnership Act, 2008",
        "The Competition Act, 2002",
        "The Insolvency and Bankruptcy Code, 2016",
        "The Arbitration and Conciliation Act, 1996",
        "The Specific Relief Act, 1963",
        "The Securities and Exchange Board of India Act, 1992",
        "The Securities Contracts (Regulation) Act, 1956",
        "The Foreign Exchange Management Act, 1999",
        "The Micro, Small and Medium Enterprises Development Act, 2006",
        "The Consumer Protection Act, 2019",
        "The Central Goods and Services Tax Act, 2017",
        "The Information Technology Act, 2000",
        "The Trade Marks Act, 1999",
        "The Copyright Act, 1957",
    ],
    "corporate law": [
        "The Companies Act, 2013",
        "The Limited Liability Partnership Act, 2008",
        "The Competition Act, 2002",
        "The Insolvency and Bankruptcy Code, 2016",
        "The Securities and Exchange Board of India Act, 1992",
        "The Securities Contracts (Regulation) Act, 1956",
    ],
    "consumer law": [
        "The Consumer Protection Act, 2019",
        "The Legal Metrology Act, 2009",
        "The Food Safety and Standards Act, 2006",
    ],
    "labour law": [
        "The Code on Wages, 2019",
        "The Industrial Relations Code, 2020",
        "The Code on Social Security, 2020",
        "The Occupational Safety, Health and Working Conditions Code, 2020",
    ],
    "labor law": [
        "The Code on Wages, 2019",
        "The Industrial Relations Code, 2020",
        "The Code on Social Security, 2020",
        "The Occupational Safety, Health and Working Conditions Code, 2020",
    ],
}


class LlamaProcessor:
    def __init__(
        self,
        api_url: str = "http://localhost:11434/api/generate",
        model: str = "llama3.2:3b",
        timeout: int = 300,
        allow_keyword_fallback: bool = True,
        max_text_chars: int = 28_000,
        max_retries: int = 4,
        retry_base_delay: float = 5.0,
        large_act_threshold: int = 160,
        large_act_batch_chars: int = 7_000,
        large_act_max_sections: int = 8,
        large_act_batch_pause: float = 1.0,
    ) -> None:
        self.api_url = api_url
        self.model = model
        self.timeout = timeout
        self.allow_keyword_fallback = allow_keyword_fallback
        self.max_text_chars = max_text_chars
        self.max_retries = max(1, max_retries)
        self.retry_base_delay = max(0.0, retry_base_delay)
        self.large_act_threshold = max(1, large_act_threshold)
        self.large_act_batch_chars = max(4_000, large_act_batch_chars)
        self.large_act_max_sections = max(1, large_act_max_sections)
        self.large_act_batch_pause = max(0.0, large_act_batch_pause)

    def discover_indian_acts(self, law_type: str, evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        evidence_text = json.dumps(evidence, ensure_ascii=False)[: self.max_text_chars]
        try:
            response = requests.post(self.api_url, json={"model": self.model, "prompt": self._build_discovery_prompt(law_type, evidence_text), "stream": False, "format": "json"}, timeout=self.timeout)
            response.raise_for_status()
            parsed = self._parse_response(response.json())
            for item in parsed.get("acts", []):
                if not isinstance(item, dict):
                    continue
                name = self._normalize_act_name(str(item.get("act_name", "")))
                if self._looks_like_indian_act(name):
                    candidates.append({"act_name": name, "year": self._extract_year(name), "reason": str(item.get("reason", "Discovered from web evidence")), "discovery_method": "llama_from_search_evidence"})
        except (requests.RequestException, ValueError, KeyError, json.JSONDecodeError) as error:
            if not self.allow_keyword_fallback:
                raise LlamaProcessingError(f"Llama Act discovery failed: {error}") from error

        for name in self._extract_act_names(evidence_text):
            candidates.append({"act_name": name, "year": self._extract_year(name), "reason": "Act name appears in collected search evidence", "discovery_method": "evidence_regex"})

        normalized_type = re.sub(r"\s+", " ", law_type.strip().lower())
        seed_names = KNOWN_ACTS_BY_LAW_TYPE.get(normalized_type, [])
        if not seed_names and "business" in normalized_type:
            seed_names = KNOWN_ACTS_BY_LAW_TYPE["business law"]
        for name in seed_names:
            candidates.append({"act_name": name, "year": self._extract_year(name), "reason": f"Foundational Indian {law_type} candidate; official text must be validated", "discovery_method": "verified_seed_candidate"})

        deduplicated: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            name = self._normalize_act_name(candidate["act_name"])
            if not self._looks_like_indian_act(name):
                continue
            key = re.sub(r"\W+", "", name.lower().removeprefix("the"))
            existing = deduplicated.get(key)
            if existing is None or existing["discovery_method"] == "verified_seed_candidate":
                candidate["act_name"] = name
                deduplicated[key] = candidate
        priority = {"verified_seed_candidate": 0, "llama_from_search_evidence": 1, "evidence_regex": 2}
        return sorted(deduplicated.values(), key=lambda item: (priority.get(item["discovery_method"], 9), item.get("year") or "9999", item["act_name"]))

    def select_relevant_catalog_acts(
        self, law_type: str, catalog: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Select relevant entries from the fully crawled Drishti catalogue.

        Llama performs semantic selection. Deterministic category/name matches
        are unioned with its answer so obvious Acts are not lost if a small
        local model omits one.
        """
        deterministic = self._deterministic_catalog_matches(law_type, catalog)
        normalized_type = re.sub(r"\s+", " ", law_type.strip().lower())
        category_driven = any(
            term in normalized_type
            for term in ("criminal", "civil", "family", "constitution", "business", "corporate", "commercial")
        )
        if deterministic and category_driven:
            return [catalog[index] for index in sorted(deterministic)]

        indexed = [
            {
                "id": index,
                "title": item.get("title", ""),
                "category": item.get("category", ""),
            }
            for index, item in enumerate(catalog)
        ]
        selected_ids: set[int] = set()
        prompt = f"""You are selecting Indian Bare Acts for a legal research dataset.

Requested law type: {law_type}

From the catalogue below, return every Act substantively related to the
requested law type. Do not invent entries. Return only JSON in this form:
{{"selected_ids": [0, 1], "reasons": {{"0": "short reason"}}}}

CATALOGUE:
{json.dumps(indexed, ensure_ascii=False)}
"""
        try:
            response = self._request_json(prompt)
            for value in response.get("selected_ids", []):
                if isinstance(value, int) and 0 <= value < len(catalog):
                    selected_ids.add(value)
        except (requests.RequestException, ValueError, KeyError, json.JSONDecodeError):
            if not self.allow_keyword_fallback:
                raise LlamaProcessingError("Llama could not select relevant Drishti catalogue Acts")

        selected_ids.update(deterministic)
        if not selected_ids:
            raise LlamaProcessingError(
                f"No Drishti Bare Acts were matched to requested law type {law_type!r}"
            )
        return [catalog[index] for index in sorted(selected_ids)]

    def generate_section_index_record(
        self,
        law_type: str,
        act_name: str,
        sections: dict[str, dict[str, str]],
        source_metadata: dict[str, Any],
        batch_chars: int = 18_000,
        checkpoint_path: Path | None = None,
    ) -> dict[str, Any]:
        """Summarize every parsed statutory section and build an Act index."""
        if not sections:
            raise LlamaProcessingError("No parsed sections were supplied to the Llama indexer")
        safe_limit = max(4_000, min(batch_chars, self.max_text_chars - 5_000))
        large_act_mode = len(sections) > self.large_act_threshold
        max_sections_per_batch: int | None = None
        if large_act_mode:
            safe_limit = min(safe_limit, self.large_act_batch_chars)
            max_sections_per_batch = self.large_act_max_sections
            LOGGER.info(
                "Large Act mode enabled: %s sections; batch limit %s sections/%s characters",
                len(sections),
                max_sections_per_batch,
                safe_limit,
            )
        completed = self._load_section_checkpoint(checkpoint_path, act_name)
        if completed:
            LOGGER.info(
                "Resuming %s from checkpoint: %s/%s sections already indexed",
                act_name,
                len(completed),
                len(sections),
            )
        ordered_items = list(sections.items())
        pending: list[tuple[str, dict[str, str]]] = []
        pending_chars = 0

        def flush_pending() -> None:
            nonlocal pending, pending_chars, completed
            if not pending:
                return
            LOGGER.info(
                "Llama indexing section batch (%s sections; %s/%s already complete)",
                len(pending),
                len(completed),
                len(ordered_items),
            )
            results = self._summarize_section_batch(law_type, act_name, pending)
            completed.update(results)
            self._write_section_checkpoint(checkpoint_path, act_name, completed)
            pending, pending_chars = [], 0
            if large_act_mode and self.large_act_batch_pause:
                time.sleep(self.large_act_batch_pause)

        for number, section in ordered_items:
            if number in completed and str(completed[number].get("what", "")).strip():
                continue
            section_size = len(section.get("text", ""))
            if section_size > safe_limit:
                flush_pending()
                LOGGER.info(
                    "Llama indexing large section %s in multiple parts (%s characters)",
                    number,
                    section_size,
                )
                completed[number] = self._summarize_large_section(
                    law_type, act_name, number, section, safe_limit
                )
                self._write_section_checkpoint(checkpoint_path, act_name, completed)
                continue
            if pending and (
                pending_chars + section_size > safe_limit
                or (max_sections_per_batch is not None and len(pending) >= max_sections_per_batch)
            ):
                flush_pending()
            pending.append((number, section))
            pending_chars += section_size
        flush_pending()

        missing = [number for number, _ in ordered_items if number not in completed]
        if missing:
            raise LlamaProcessingError("Llama did not index sections: " + ", ".join(missing[:20]))

        indexed_sections: dict[str, dict[str, str]] = {}
        processed_characters = 0
        synthesis_input: list[dict[str, Any]] = []
        for number, original in ordered_items:
            summary = completed[number]
            what = str(summary.get("what", "")).strip()
            indexed_sections[number] = {
                "section": number,
                "title": original.get("title", f"Section {number}"),
                "what": what,
                "chapter": original.get("chapter", ""),
            }
            processed_characters += len(original.get("text", ""))
            synthesis_input.append(
                {
                    "section": number,
                    "title": original.get("title", ""),
                    "what": what,
                    "section_references": [number],
                    "key_points": [],
                }
            )

        synthesis = self._synthesize_chunk_results(law_type, act_name, synthesis_input)
        return {
            "schema_version": "4.0",
            "record_type": "indian_bare_act_section_index",
            "country": "India",
            "law_type": law_type,
            "law": synthesis.get("law") or act_name,
            "what": synthesis["what"],
            "section_count": len(indexed_sections),
            "sections": indexed_sections,
            "key_points": synthesis.get("key_points", []),
            "section_references": synthesis.get("section_references", []),
            "pdf_extraction": source_metadata.get("extraction", {}),
            "llm_processing": {
                "model": self.model,
                "mode": "numbered_section_indexing",
                "large_act_mode": large_act_mode,
                "section_count": len(indexed_sections),
                "processed_section_characters": processed_characters,
                "every_parsed_section_indexed": len(indexed_sections) == len(sections),
            },
            "source": {
                "name": source_metadata.get("source_name"),
                "url": source_metadata.get("resolved_source_url") or source_metadata.get("source_url"),
                "catalog_url": source_metadata.get("category_url"),
                "category": source_metadata.get("category"),
                "listed_date": source_metadata.get("listed_date"),
                "format": "pdf",
                "content_sha256": source_metadata.get("content_sha256"),
                "collected_at": source_metadata.get("collected_at"),
            },
            "notice": "Each 'what' value is an LLM explanation of the corresponding extracted statutory section. Verify exact wording and current legal status against the downloaded PDF.",
        }

    def _summarize_section_batch(
        self,
        law_type: str,
        act_name: str,
        items: list[tuple[str, dict[str, str]]],
    ) -> dict[str, dict[str, str]]:
        payload = [
            {
                "section": number,
                "title": item.get("title", ""),
                "text": item.get("text", ""),
            }
            for number, item in items
        ]
        prompt = f"""Index the supplied numbered sections of this Indian Bare Act.
For every input section return one result explaining what that section states.
Preserve duties, rights, prohibitions, powers, procedures, penalties,
exceptions and definitions. Do not merge section numbers and do not invent law.

Law: {act_name}
Requested law type: {law_type}

Return only valid JSON in this exact shape:
{{"sections":[{{"section":"1","what":"what section 1 states"}}]}}

INPUT SECTIONS:
{json.dumps(payload, ensure_ascii=False)}
"""
        try:
            raw = self._request_json(prompt)
        except (requests.RequestException, ValueError, KeyError, json.JSONDecodeError) as error:
            raise LlamaProcessingError(f"Llama section batch failed: {error}") from error
        results = self._coerce_section_results(raw)
        output: dict[str, dict[str, str]] = {}
        for number, item in items:
            what = results.get(number, "").strip()
            if not what:
                what = self._retry_single_section(law_type, act_name, number, item)
            output[number] = {"what": what}
        return output

    def _summarize_large_section(
        self,
        law_type: str,
        act_name: str,
        number: str,
        section: dict[str, str],
        limit: int,
    ) -> dict[str, str]:
        parts = self._split_all_text(section.get("text", ""), limit)
        part_summaries: list[str] = []
        for index, part in enumerate(parts, start=1):
            part_item = {
                "title": f"{section.get('title', '')} (part {index}/{len(parts)})",
                "text": part,
            }
            part_summaries.append(self._retry_single_section(law_type, act_name, number, part_item))
        if len(part_summaries) == 1:
            return {"what": part_summaries[0]}
        prompt = f"""Combine these partial explanations into one complete explanation of
Section {number} of {act_name}. Do not omit distinct rules or invent content.
Return only JSON: {{"section":"{number}","what":"combined explanation"}}

PART EXPLANATIONS:
{json.dumps(part_summaries, ensure_ascii=False)}
"""
        result = self._request_json(prompt)
        what = self._extract_what(result)
        if not what:
            what = " ".join(part_summaries)
        return {"what": what}

    def _retry_single_section(
        self,
        law_type: str,
        act_name: str,
        number: str,
        section: dict[str, str],
    ) -> str:
        prompt = f"""Explain only what this statutory section states in clear language.
Keep all material rules, exceptions, procedures and penalties. Do not provide advice.
Return only JSON: {{"section":"{number}","what":"complete explanation"}}

Law: {act_name}
Law type: {law_type}
Section: {number}
Title: {section.get('title', '')}
Text: {section.get('text', '')}
"""
        try:
            result = self._request_json(prompt)
        except (requests.RequestException, ValueError, KeyError, json.JSONDecodeError) as error:
            raise LlamaProcessingError(f"Llama failed on section {number}: {error}") from error
        what = self._extract_what(result)
        if not what:
            keys = ", ".join(map(str, result.keys())) or "none"
            raise LlamaProcessingError(
                f"Llama returned no section explanation for section {number}; returned keys: {keys}"
            )
        return what

    @classmethod
    def _coerce_section_results(cls, result: dict[str, Any]) -> dict[str, str]:
        output: dict[str, str] = {}
        container = result.get("sections", result.get("section_summaries", result))
        if isinstance(container, list):
            for item in container:
                if not isinstance(item, dict):
                    continue
                number = str(item.get("section") or item.get("section_number") or item.get("number") or "").strip()
                what = cls._extract_what(item)
                if number and what:
                    output[number] = what
        elif isinstance(container, dict):
            if any(key in container for key in ("section", "section_number", "number")):
                number = str(container.get("section") or container.get("section_number") or container.get("number") or "").strip()
                what = cls._extract_what(container)
                if number and what:
                    output[number] = what
            else:
                for number, value in container.items():
                    if isinstance(value, dict):
                        what = cls._extract_what(value)
                    else:
                        what = str(value).strip() if isinstance(value, str) else ""
                    if what:
                        output[str(number)] = what
        return output

    @staticmethod
    def _extract_what(result: dict[str, Any]) -> str:
        for key in ("what", "summary", "explanation", "meaning", "description", "statement"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    @staticmethod
    def _load_section_checkpoint(path: Path | None, act_name: str) -> dict[str, dict[str, str]]:
        if path is None or not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if payload.get("law") != act_name or not isinstance(payload.get("sections"), dict):
            return {}
        return payload["sections"]

    @staticmethod
    def _write_section_checkpoint(
        path: Path | None, act_name: str, sections: dict[str, dict[str, str]]
    ) -> None:
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps({"law": act_name, "sections": sections}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(path)

    def generate_law_key_value_record(
        self,
        law_type: str,
        act_name: str,
        full_pdf_text: str,
        source_metadata: dict[str, Any],
        chunk_chars: int = 24_000,
    ) -> dict[str, Any]:
        """Send the complete extracted PDF text to Llama without truncation.

        Large Acts are processed in lossless sequential chunks. Every chunk is
        summarized first, and those summaries are hierarchically consolidated
        so the final ``law``/``what`` pair is grounded in the entire PDF.
        """
        if not full_pdf_text.strip():
            raise LlamaProcessingError("The extracted PDF text is empty")
        safe_chunk_size = max(4_000, min(chunk_chars, self.max_text_chars - 4_000))
        chunks = self._split_all_text(full_pdf_text, safe_chunk_size)
        chunk_results: list[dict[str, Any]] = []
        processed_characters = 0

        for index, chunk in enumerate(chunks, start=1):
            prompt = f"""You are converting an Indian Bare Act into clean legal research data.
Use only the supplied PDF text chunk. Preserve material duties, rights,
prohibitions, powers, procedures, penalties, exceptions and definitions.
Do not provide legal advice and do not invent missing provisions.

Law: {act_name}
Requested law type: {law_type}
Chunk: {index} of {len(chunks)}

Return only JSON:
{{
  "law": "exact Act title if this chunk shows it, otherwise empty string",
  "what": "clear explanation of what this chunk of the law states",
  "key_points": ["specific provision or rule"],
  "section_references": ["section/chapter references found in this chunk"]
}}

FULL CHUNK TEXT:
{chunk}
"""
            try:
                result = self._request_json(prompt)
            except (requests.RequestException, ValueError, KeyError, json.JSONDecodeError) as error:
                raise LlamaProcessingError(
                    f"Llama failed while processing PDF chunk {index}/{len(chunks)}: {error}"
                ) from error
            what = str(result.get("what", "")).strip()
            if not what:
                raise LlamaProcessingError(f"Llama returned no 'what' value for chunk {index}")
            chunk_results.append(
                {
                    "chunk": index,
                    "law": str(result.get("law", "")).strip(),
                    "what": what,
                    "key_points": self._string_list(result.get("key_points")),
                    "section_references": self._string_list(result.get("section_references")),
                }
            )
            processed_characters += len(chunk)

        synthesis = self._synthesize_chunk_results(law_type, act_name, chunk_results)
        return {
            "schema_version": "3.0",
            "record_type": "indian_bare_act_llm_dictionary",
            "country": "India",
            "law_type": law_type,
            "law": synthesis.get("law") or act_name,
            "what": synthesis["what"],
            "key_points": synthesis["key_points"],
            "section_references": synthesis["section_references"],
            "chunk_analysis": chunk_results,
            "pdf_extraction": source_metadata.get("extraction", {}),
            "llm_processing": {
                "model": self.model,
                "input_characters": len(full_pdf_text),
                "processed_characters": processed_characters,
                "chunk_count": len(chunks),
                "every_extracted_character_sent": processed_characters == len(full_pdf_text),
            },
            "source": {
                "name": source_metadata.get("source_name"),
                "url": source_metadata.get("resolved_source_url")
                or source_metadata.get("source_url"),
                "catalog_url": source_metadata.get("category_url"),
                "category": source_metadata.get("category"),
                "listed_date": source_metadata.get("listed_date"),
                "format": "pdf",
                "content_sha256": source_metadata.get("content_sha256"),
                "collected_at": source_metadata.get("collected_at"),
            },
            "notice": "LLM-generated research summary grounded in the fully extracted PDF. Verify current amendments, commencement, repeal status and exact wording against the downloaded source before legal use.",
        }

    def _synthesize_chunk_results(
        self, law_type: str, act_name: str, chunk_results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        working = chunk_results
        while len(json.dumps(working, ensure_ascii=False)) > self.max_text_chars - 5_000:
            reduced: list[dict[str, Any]] = []
            group: list[dict[str, Any]] = []
            group_size = 0
            for item in working:
                item_size = len(json.dumps(item, ensure_ascii=False))
                if group and group_size + item_size > self.max_text_chars - 7_000:
                    reduced.append(self._reduce_summary_group(law_type, act_name, group))
                    group, group_size = [], 0
                group.append(item)
                group_size += item_size
            if group:
                reduced.append(self._reduce_summary_group(law_type, act_name, group))
            if len(reduced) >= len(working):
                raise LlamaProcessingError("Chunk summaries could not be reduced within the Llama context limit")
            working = reduced

        prompt = f"""Create the final legal dictionary entry for this Indian Bare Act.
The supplied material represents all PDF chunks. Consolidate it without
inventing rules. Return only JSON with exactly these keys:
{{"law": "exact official Act title and year found in the material",
  "what": "comprehensive plain-language statement of what the law states",
  "key_points": ["important rule"],
  "section_references": ["important section/chapter reference"]}}

Law: {act_name}
Requested law type: {law_type}
ALL CHUNK SUMMARIES:
{json.dumps(working, ensure_ascii=False)}
"""
        try:
            result = self._request_json(prompt)
        except (requests.RequestException, ValueError, KeyError, json.JSONDecodeError) as error:
            raise LlamaProcessingError(f"Final Llama synthesis failed: {error}") from error
        what = str(result.get("what", "")).strip()
        if not what:
            raise LlamaProcessingError("Final Llama synthesis returned no 'what' value")
        return {
            "law": str(result.get("law", "")).strip() or act_name,
            "what": what,
            "key_points": self._string_list(result.get("key_points")),
            "section_references": self._string_list(result.get("section_references")),
        }

    def _reduce_summary_group(
        self, law_type: str, act_name: str, group: list[dict[str, Any]]
    ) -> dict[str, Any]:
        prompt = f"""Compress these partial summaries of {act_name} for {law_type}.
Retain distinct legal rules, exceptions, penalties and section references.
Return only JSON: {{"law":"exact Act title if present","what":"combined summary","key_points":[],"section_references":[]}}

SUMMARIES:
{json.dumps(group, ensure_ascii=False)}
"""
        try:
            result = self._request_json(prompt)
        except (requests.RequestException, ValueError, KeyError, json.JSONDecodeError) as error:
            raise LlamaProcessingError(f"Intermediate Llama synthesis failed: {error}") from error
        return {
            "law": str(result.get("law", "")).strip() or act_name,
            "what": str(result.get("what", "")).strip(),
            "key_points": self._string_list(result.get("key_points")),
            "section_references": self._string_list(result.get("section_references")),
        }

    def _request_json(self, prompt: str) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            retry_prompt = prompt
            if attempt > 1:
                retry_prompt += (
                    "\n\nIMPORTANT RETRY: Return one valid JSON object only. "
                    "Do not use Markdown, code fences, commentary, or plain text outside JSON."
                )
            try:
                response = requests.post(
                    self.api_url,
                    json={
                        "model": self.model,
                        "prompt": retry_prompt,
                        "stream": False,
                        "format": "json",
                        "options": {"temperature": 0},
                    },
                    timeout=(10, self.timeout),
                )
                response.raise_for_status()
                return self._parse_response(response.json())
            except (requests.RequestException, ValueError, KeyError, json.JSONDecodeError) as error:
                last_error = error
                if attempt >= self.max_retries:
                    break
                delay = self.retry_base_delay * (2 ** (attempt - 1))
                LOGGER.warning(
                    "Llama request failed (%s/%s): %s. Retrying in %.1f seconds",
                    attempt,
                    self.max_retries,
                    error,
                    delay,
                )
                self.wait_until_ready(max_attempts=3, delay=min(max(delay, 1.0), 10.0))
                if delay:
                    time.sleep(delay)
        raise LlamaProcessingError(
            f"Llama request failed after {self.max_retries} attempts: {last_error}"
        ) from last_error

    def wait_until_ready(self, max_attempts: int = 6, delay: float = 5.0) -> bool:
        """Wait for the local Ollama service before resuming checkpointed work."""
        parsed = urlparse(self.api_url)
        tags_url = urlunparse((parsed.scheme, parsed.netloc, "/api/tags", "", "", ""))
        for attempt in range(1, max(1, max_attempts) + 1):
            try:
                response = requests.get(tags_url, timeout=(5, 15))
                response.raise_for_status()
                LOGGER.info("Ollama is ready")
                return True
            except requests.RequestException as error:
                if attempt >= max_attempts:
                    LOGGER.warning("Ollama readiness check failed: %s", error)
                    return False
                LOGGER.warning(
                    "Ollama is not ready (%s/%s); waiting %.1f seconds",
                    attempt,
                    max_attempts,
                    delay,
                )
                time.sleep(max(0.0, delay))
        return False

    @staticmethod
    def _split_all_text(text: str, limit: int) -> list[str]:
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + limit, len(text))
            if end < len(text):
                boundary = text.rfind("\n", start + limit // 2, end)
                if boundary > start:
                    end = boundary + 1
            chunks.append(text[start:end])
            start = end
        return chunks

    @staticmethod
    def _string_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    @staticmethod
    def _deterministic_catalog_matches(
        law_type: str, catalog: list[dict[str, Any]]
    ) -> set[int]:
        normalized = re.sub(r"\s+", " ", law_type.lower()).strip()
        all_words = {word for word in re.findall(r"[a-z]+", normalized) if len(word) > 2}
        business_terms = {
            "business", "corporate", "commercial", "company", "companies",
            "contract", "partnership", "goods", "trademark", "trade marks",
            "copyright", "arbitration", "tax", "consumer", "insolvency",
            "competition", "negotiable", "property", "trust", "stamp",
        }
        selected: set[int] = set()
        for index, item in enumerate(catalog):
            title = str(item.get("title", "")).lower()
            category = str(item.get("category", "")).lower()
            haystack = f"{title} {category}"
            if "criminal" in normalized:
                if "criminal" in category:
                    selected.add(index)
                continue
            if "civil" in normalized:
                if "civil" in category:
                    selected.add(index)
                continue
            if "family" in normalized:
                if "family" in category:
                    selected.add(index)
                continue
            if "constitution" in normalized:
                if "constitution" in category:
                    selected.add(index)
                continue
            if any(term in normalized for term in ("business", "corporate", "commercial")):
                if any(term in haystack for term in business_terms):
                    selected.add(index)
            elif all_words and any(word in haystack for word in all_words):
                selected.add(index)
        return selected

    @staticmethod
    def _build_discovery_prompt(law_type: str, evidence_text: str) -> str:
        return f"""You are an Indian legislation data librarian. From ONLY the supplied search evidence, identify Acts or Codes enacted in India that are substantively related to the requested law type. Do not include bills, rules, regulations, foreign laws, cases, articles, or an Act merely mentioned incidentally. Do not invent names or years.

Requested law type: {law_type}
Jurisdiction: India only

Return only a JSON object with one key, acts. acts must be an array of objects with:
- act_name: exact official-style Act or Code name including year
- reason: one short explanation of relevance to {law_type}

SEARCH EVIDENCE:
{evidence_text}
"""

    @staticmethod
    def _parse_response(payload: dict[str, Any]) -> dict[str, Any]:
        content = payload.get("response")
        if content is None and isinstance(payload.get("message"), dict):
            content = payload["message"].get("content")
        if isinstance(content, dict):
            return content
        if not isinstance(content, str):
            raise ValueError("Llama response contained no generated content")
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise ValueError("Llama response did not contain a JSON object")
        parsed = json.loads(match.group(0))
        if not isinstance(parsed, dict):
            raise ValueError("Llama JSON output must be an object")
        return parsed

    @classmethod
    def _extract_act_names(cls, text: str) -> list[str]:
        pattern = re.compile(r"\b(?:The\s+)?(?:Indian\s+)?[A-Z][A-Za-z0-9&'’().,/\- ]{2,110}?\s+(?:Act|Code)\s*,?\s+(?:of\s+)?(?:18|19|20)\d{2}\b")
        return [cls._normalize_act_name(match.group(0)) for match in pattern.finditer(text) if cls._looks_like_indian_act(cls._normalize_act_name(match.group(0)))]

    @staticmethod
    def _normalize_act_name(name: str) -> str:
        name = re.sub(r"\s+", " ", name).strip(" -–—:;.")
        name = re.sub(r"^.*?\b(?:include|includes|including|such as|namely|comprise|comprises)\b\s*", "", name, flags=re.IGNORECASE)
        return re.sub(r"\s*,?\s+(?=(?:18|19|20)\d{2}$)", ", ", name)

    @staticmethod
    def _extract_year(name: str) -> str:
        match = re.search(r"\b((?:18|19|20)\d{2})\b", name)
        return match.group(1) if match else ""

    @staticmethod
    def _looks_like_indian_act(name: str) -> bool:
        lowered = name.lower()
        if not (8 <= len(name) <= 150):
            return False
        if not re.search(r"\b(?:act|code)\b", lowered) or not re.search(r"\b(?:18|19|20)\d{2}\b", name):
            return False
        if any(term in lowered for term in ("united states", "united kingdom", "california", "australia")):
            return False
        if name.count("(") != name.count(")"):
            return False
        if re.match(r"^[^()]*\)", name):
            # A ")" before any "(" means the head of the name (e.g. "Wildlife" in
            # "Wildlife (Protection) Act, 1972") was cut off during extraction.
            return False
        if len(re.findall(r"\b(?:act|code)\b[^,;]{0,25}\b(?:18|19|20)\d{2}\b", lowered)) > 1:
            # More than one Act/Code+year pair means several Acts were merged
            # into a single name, e.g. "...IPC), the Code of Criminal Procedure,
            # 1973 (CrPC), and the Indian Evidence Act, 1872".
            return False
        return True
