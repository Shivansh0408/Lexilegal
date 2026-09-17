"""Strict download and complete text extraction for Drishti Bare Act PDFs."""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from pypdf import PdfReader

try:
    from .drishti_crawler import DrishtiBareAct, USER_AGENT
    from .section_parser import SectionParsingError, parse_ordered_bare_act_sections
except ImportError:
    from drishti_crawler import DrishtiBareAct, USER_AGENT
    from section_parser import SectionParsingError, parse_ordered_bare_act_sections


ALLOWED_PDF_HOST = "vault.drishtijudiciary.com"


@dataclass(frozen=True)
class ExtractedBareAct:
    source: DrishtiBareAct
    pdf_path: Path
    text_path: Path
    text: str
    sections: dict[str, dict[str, str]]
    metadata: dict[str, Any]


class PdfProcessingError(RuntimeError):
    """Raised when a PDF cannot be downloaded or completely text-extracted."""


class DrishtiPdfProcessor:
    def __init__(self, timeout: int = 60, max_download_bytes: int = 100 * 1024 * 1024) -> None:
        self.timeout = timeout
        self.max_download_bytes = max_download_bytes
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": USER_AGENT, "Accept": "application/pdf,*/*;q=0.5"}
        )

    def download_and_extract(self, source: DrishtiBareAct, destination: Path) -> ExtractedBareAct:
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        cached = self._load_cached_extraction(source, destination)
        if cached is not None:
            return cached
        pdf_path, download_metadata = self._download(source.pdf_url, destination)
        text, page_records = self._extract_every_page(pdf_path)
        text_path = destination / "extracted_text.txt"
        self._atomic_write_text(text_path, text)
        try:
            sections = parse_ordered_bare_act_sections(text)
        except SectionParsingError as error:
            raise PdfProcessingError(f"Numbered sections could not be parsed from the PDF: {error}") from error
        self._atomic_write_json(destination / "parsed_sections.json", sections)

        report = {
            "page_count": len(page_records),
            "extracted_page_count": sum(bool(item["text"].strip()) for item in page_records),
            "blank_pages": [item["page"] for item in page_records if not item["text"].strip()],
            "total_characters": len(text),
            "full_text_extraction": all(bool(item["text"].strip()) for item in page_records),
            "parsed_section_count": len(sections),
            "pages": [{"page": item["page"], "characters": len(item["text"])} for item in page_records],
        }
        if report["blank_pages"]:
            raise PdfProcessingError(
                "PDF extraction was incomplete; no machine-readable text was found on pages "
                + ", ".join(map(str, report["blank_pages"][:25]))
            )
        if len(text) < max(500, len(page_records) * 40):
            raise PdfProcessingError("PDF produced too little text to be considered fully extracted")

        metadata = {
            **download_metadata,
            "catalog_title": source.title,
            "category": source.category,
            "category_url": source.category_url,
            "listed_date": source.listed_date,
            "extraction": report,
        }
        self._atomic_write_json(destination / "extraction_report.json", metadata)
        return ExtractedBareAct(source, pdf_path, text_path, text, sections, metadata)

    @staticmethod
    def _load_cached_extraction(
        source: DrishtiBareAct, destination: Path
    ) -> ExtractedBareAct | None:
        pdf_path = destination / "bare_act.pdf"
        text_path = destination / "extracted_text.txt"
        sections_path = destination / "parsed_sections.json"
        report_path = destination / "extraction_report.json"
        if not all(path.exists() for path in (pdf_path, text_path, sections_path, report_path)):
            return None
        try:
            metadata = json.loads(report_path.read_text(encoding="utf-8"))
            sections = json.loads(sections_path.read_text(encoding="utf-8"))
            text = text_path.read_text(encoding="utf-8")
            with pdf_path.open("rb") as handle:
                signature = handle.read(5)
        except (OSError, json.JSONDecodeError):
            return None
        if (
            signature != b"%PDF-"
            or metadata.get("source_url") != source.pdf_url
            or not metadata.get("extraction", {}).get("full_text_extraction")
            or not isinstance(sections, dict)
            or len(sections) < 2
            or not text.strip()
        ):
            return None
        return ExtractedBareAct(source, pdf_path, text_path, text, sections, metadata)

    def _download(self, url: str, destination: Path) -> tuple[Path, dict[str, Any]]:
        if not self._is_allowed_pdf_url(url):
            raise PdfProcessingError(f"Rejected PDF URL outside {ALLOWED_PDF_HOST}: {url}")
        target = destination / "bare_act.pdf"
        partial = destination / "bare_act.pdf.part"
        digest = hashlib.sha256()
        size = 0
        first_bytes = b""
        try:
            with self.session.get(url, timeout=self.timeout, stream=True, allow_redirects=True) as response:
                response.raise_for_status()
                if not self._is_allowed_pdf_url(response.url):
                    raise PdfProcessingError("PDF redirected outside the Drishti Judiciary vault")
                with partial.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=65_536):
                        if not chunk:
                            continue
                        if not first_bytes:
                            first_bytes = chunk[:5]
                        size += len(chunk)
                        if size > self.max_download_bytes:
                            raise PdfProcessingError(
                                f"PDF exceeds {self.max_download_bytes // (1024 * 1024)} MB"
                            )
                        digest.update(chunk)
                        handle.write(chunk)
                content_type = response.headers.get("Content-Type", "").lower()
                resolved_url = response.url
        except requests.RequestException as error:
            partial.unlink(missing_ok=True)
            raise PdfProcessingError(f"Could not download Drishti PDF: {error}") from error
        except (OSError, PdfProcessingError):
            partial.unlink(missing_ok=True)
            raise
        if first_bytes != b"%PDF-":
            partial.unlink(missing_ok=True)
            raise PdfProcessingError(f"Download was not a valid PDF response ({content_type or 'unknown type'})")
        partial.replace(target)
        return target, {
            "source_name": "Drishti Judiciary Bare Acts",
            "source_url": url,
            "resolved_source_url": resolved_url,
            "content_type": content_type,
            "content_sha256": digest.hexdigest(),
            "downloaded_bytes": size,
            "collected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    @staticmethod
    def _extract_every_page(path: Path) -> tuple[str, list[dict[str, Any]]]:
        try:
            reader = PdfReader(str(path), strict=False)
        except Exception as error:
            raise PdfProcessingError(f"PDF could not be opened: {error}") from error
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as error:
                raise PdfProcessingError(f"Encrypted PDF could not be read: {error}") from error
        page_records: list[dict[str, Any]] = []
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                standard = page.extract_text() or ""
                layout = ""
                # Layout mode can emit "Rotated text discovered" warnings and
                # intentionally omit rotated fragments. Use it only as a
                # fallback when normal extraction produced almost no text.
                if len(standard.strip()) < 40:
                    try:
                        layout = page.extract_text(extraction_mode="layout") or ""
                    except (TypeError, ValueError):
                        layout = ""
            except Exception as error:
                raise PdfProcessingError(f"Text extraction failed on page {page_number}: {error}") from error
            selected = layout if len(layout.strip()) > len(standard.strip()) else standard
            selected = DrishtiPdfProcessor._normalize_page_text(selected)
            page_records.append({"page": page_number, "text": selected})
        text = "\n\n".join(
            f"===== PDF PAGE {item['page']} =====\n{item['text']}" for item in page_records
        ).strip()
        return text, page_records

    @staticmethod
    def _normalize_page_text(text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00ad", "")
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{4,}", "\n\n\n", text)
        return text.strip()

    @staticmethod
    def _is_allowed_pdf_url(url: str) -> bool:
        parsed = urlparse(url)
        return parsed.scheme == "https" and (parsed.hostname or "").lower() == ALLOWED_PDF_HOST

    @staticmethod
    def _atomic_write_text(path: Path, value: str) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(value, encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def _atomic_write_json(path: Path, value: Any) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)
