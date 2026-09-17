"""PDF text extraction for uploaded court files.

This module deliberately has no dependency on the collector pipeline.  It uses
``pypdf`` (the maintained successor to PyPDF2) and returns page-aware text so
the analysis layer can retain useful source references.
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, asdict
from typing import BinaryIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError

LOGGER = logging.getLogger("lexbrief.pdf")


class PdfProcessingError(ValueError):
    """Raised when an uploaded PDF cannot provide analysable text."""


@dataclass(frozen=True)
class PdfExtraction:
    text: str
    page_count: int
    pages_with_text: int
    character_count: int
    metadata: dict[str, str]

    def to_dict(self) -> dict:
        return asdict(self)


def _normalise_text(value: str) -> str:
    value = value.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[\t\f\v ]+", " ", value)
    value = re.sub(r" *\n *", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def extract_pdf_text(source: BinaryIO | bytes) -> PdfExtraction:
    """Extract all available text from a PDF while retaining page markers."""

    LOGGER.info("PDF step 1/3: opening uploaded PDF")
    stream = io.BytesIO(source) if isinstance(source, bytes) else source
    try:
        reader = PdfReader(stream, strict=False)
    except (PdfReadError, OSError, ValueError) as exc:
        raise PdfProcessingError("The uploaded file is not a readable PDF.") from exc

    if reader.is_encrypted:
        LOGGER.info("PDF is encrypted; attempting empty-password decryption")
        try:
            if reader.decrypt("") == 0:
                raise PdfProcessingError("The PDF is password protected and cannot be analysed.")
        except PdfProcessingError:
            raise
        except Exception as exc:  # pypdf can surface provider-specific errors
            raise PdfProcessingError("The PDF is password protected and cannot be analysed.") from exc

    page_count = len(reader.pages)
    if page_count == 0:
        raise PdfProcessingError("The uploaded PDF contains no pages.")

    LOGGER.info("PDF step 2/3: extracting text from %d page(s)", page_count)
    page_blocks: list[str] = []
    pages_with_text = 0
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            page_text = _normalise_text(page.extract_text() or "")
        except Exception as exc:
            LOGGER.warning("Could not extract page %d: %s", page_number, exc)
            page_text = ""
        if page_text:
            pages_with_text += 1
            page_blocks.append(f"[Page {page_number}]\n{page_text}")

    text = "\n\n".join(page_blocks).strip()
    if not text:
        raise PdfProcessingError(
            "No selectable text was found. This appears to be a scanned/image-only PDF; run OCR first."
        )

    raw_metadata = reader.metadata or {}
    metadata = {
        str(key).lstrip("/"): str(value)
        for key, value in raw_metadata.items()
        if value is not None
    }
    LOGGER.info(
        "PDF step 3/3: extracted %d characters from %d/%d page(s)",
        len(text),
        pages_with_text,
        page_count,
    )
    return PdfExtraction(
        text=text,
        page_count=page_count,
        pages_with_text=pages_with_text,
        character_count=len(text),
        metadata=metadata,
    )

