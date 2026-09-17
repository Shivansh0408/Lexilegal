"""Store one India-only section dictionary per Act and maintain an index."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "unnamed"


def canonical_law_key(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value.lower().replace("&", " and ")).strip()
    normalized = re.sub(r"^the\s+", "", normalized)
    aliases = {
        "it act": "information technology act 2000",
        "ni act": "negotiable instruments act 1881",
        "ipc": "indian penal code 1860",
        "crpc": "code of criminal procedure 1973",
    }
    normalized = aliases.get(normalized, normalized)
    return re.sub(r"[^a-z0-9]+", "", normalized)


class IndianActDictionaryBuilder:
    def __init__(self, dictionary_root: Path, law_type: str) -> None:
        self.root = Path(dictionary_root)
        self.law_type = law_type
        self.root.mkdir(parents=True, exist_ok=True)
        self.failures: list[dict[str, Any]] = []

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)

    def store(self, record: dict[str, Any]) -> Path:
        target = self.root / f"{slugify(record['law'])}.json"
        self._write_json(target, record)
        return target

    def record_failure(self, candidate: dict[str, Any], error: str) -> None:
        self.failures.append({"act_name": candidate.get("act_name"), "year": candidate.get("year"), "error": error})

    def find_existing_complete(
        self, act_name: str, source_url: str = ""
    ) -> tuple[Path, dict[str, Any]] | None:
        """Find a fully indexed Act by canonical title or exact source URL."""
        wanted_key = canonical_law_key(act_name)
        for path in sorted(self.root.glob("*.json")):
            if path.name in {"index.json", "discovery_manifest.json"}:
                continue
            try:
                item = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            stored_url = str(item.get("source", {}).get("url") or "")
            title_matches = canonical_law_key(str(item.get("law") or "")) == wanted_key
            url_matches = bool(source_url and stored_url == source_url)
            sections = item.get("sections")
            section_count = int(item.get("section_count") or 0)
            complete = (
                item.get("record_type") == "indian_bare_act_section_index"
                and isinstance(sections, dict)
                and section_count > 0
                and len(sections) == section_count
                and all(str(section.get("what", "")).strip() for section in sections.values())
                and item.get("llm_processing", {}).get("every_parsed_section_indexed") is True
            )
            if complete and (title_matches or url_matches):
                return path, item
        return None

    def write_discovery(self, candidates: list[dict[str, Any]], evidence: list[dict[str, Any]]) -> Path:
        target = self.root / "discovery_manifest.json"
        self._write_json(target, {
            "country": "India",
            "law_type": self.law_type,
            "catalog_source_url": "https://www.drishtijudiciary.com/downloads/bare-acts/general-bare-acts?page=1",
            "catalog_item_count": len(evidence),
            "candidate_count": len(candidates),
            "candidates": candidates,
            "catalog_items": evidence,
            # Retained for compatibility with readers of the previous schema.
            "search_evidence": evidence,
        })
        return target

    def write_index(self) -> Path:
        records_by_key: dict[str, dict[str, Any]] = {}
        for path in sorted(self.root.glob("*.json")):
            if path.name in {"index.json", "discovery_manifest.json"}:
                continue
            try:
                item = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if item.get("record_type") not in {
                "indian_act_section_dictionary",
                "indian_bare_act_llm_dictionary",
                "indian_bare_act_section_index",
            }:
                continue
            record = {
                "law": item.get("law"),
                "year": item.get("year"),
                "law_type": item.get("law_type"),
                "section_count": item.get("section_count"),
                "what": item.get("what"),
                "key_point_count": len(item.get("key_points") or []),
                "pdf_page_count": item.get("pdf_extraction", {}).get("page_count"),
                "llm_chunk_count": item.get("llm_processing", {}).get("chunk_count"),
                "source_url": item.get("source", {}).get("url"),
                "content_sha256": item.get("source", {}).get("content_sha256"),
                "path": path.name,
            }
            key = canonical_law_key(str(item.get("law") or path.stem))
            existing = records_by_key.get(key)
            if existing is None or int(record.get("section_count") or 0) > int(existing.get("section_count") or 0):
                records_by_key[key] = record
        records = sorted(records_by_key.values(), key=lambda item: str(item.get("law") or "").lower())
        target = self.root / "index.json"
        self._write_json(target, {
            "schema_version": "2.0",
            "record_type": "indian_law_type_index",
            "country": "India",
            "law_type": self.law_type,
            "total_acts": len(records),
            "total_sections": sum(int(item.get("section_count") or 0) for item in records),
            "acts": records,
            "failed_candidates": self.failures,
            "notice": "Research dataset only. Verify current text, amendments, commencement and repeal status with the cited official source.",
        })
        return target
