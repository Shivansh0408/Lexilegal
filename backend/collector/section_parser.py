"""Extract section-keyed dictionaries from official Indian Act PDF/HTML text."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from pypdf import PdfReader


class SectionParsingError(RuntimeError):
    """Raised when an official source cannot produce a useful section map."""


SECTION_RE = re.compile(
    r"(?m)^[ \t]*(?:SECTION[ \t]+)?(?P<number>\d{1,4}[A-Z]{0,3}(?:-[A-Z]{1,3})?)[ \t]*[.\-–—][ \t]*(?P<heading>[^\n]{2,240})"
)
BARE_SECTION_LINE_RE = re.compile(
    r"(?m)^[ \t]*(?:\d+\[)?(?P<number>\d{1,4}[A-Z]{0,3})\.[ \t]+(?P<line>[^\n]{2,500})"
)
CHAPTER_RE = re.compile(r"(?im)^[ \t]*(?P<chapter>(?:CHAPTER|PART)[ \t]+[IVXLCDM0-9A-Z-]+(?:[^\n]{0,160})?)$")


def extract_source_text(path: Path, source_format: str) -> str:
    if source_format == "pdf":
        try:
            reader = PdfReader(str(path))
            pages = [(page.extract_text() or "") for page in reader.pages]
        except Exception as error:
            raise SectionParsingError(f"PDF text extraction failed: {error}") from error
        text = "\n".join(pages)
    elif source_format == "html":
        try:
            soup = BeautifulSoup(path.read_bytes(), "html.parser")
        except OSError as error:
            raise SectionParsingError(f"HTML source could not be read: {error}") from error
        for node in soup(["script", "style", "noscript", "svg", "nav", "footer", "header", "aside", "form"]):
            node.decompose()
        text = (soup.find("main") or soup.find("article") or soup.body or soup).get_text("\n", strip=True)
    else:
        raise SectionParsingError(f"Unsupported official source format: {source_format}")
    text = _normalize_text(text)
    if len(text) < 500:
        raise SectionParsingError("Official source contained too little machine-readable text; it may be a scanned PDF")
    return text


def parse_sections(text: str) -> dict[str, dict[str, str]]:
    matches = list(SECTION_RE.finditer(text))
    if not matches:
        raise SectionParsingError("No section headings were detected in the official source")

    chapters = [(match.start(), _clean_inline(match.group("chapter"))) for match in CHAPTER_RE.finditer(text)]
    candidates: dict[str, list[dict[str, str]]] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        number = match.group("number").upper().replace("-", "")
        raw_heading = _clean_inline(match.group("heading"))
        title, opening_text = _split_heading_and_body(raw_heading)
        body = _clean_body(" ".join(part for part in (opening_text, text[match.end():end]) if part))
        if len(body) < 8:
            continue
        chapter = ""
        for chapter_position, chapter_name in chapters:
            if chapter_position > match.start():
                break
            chapter = chapter_name
        candidates.setdefault(number, []).append({
            "section": number,
            "title": title or f"Section {number}",
            "what": body,
            "chapter": chapter,
        })

    # India Code PDFs normally contain an arrangement of sections followed by
    # the Act body. Keeping the longest occurrence removes the short TOC copy.
    sections = {
        number: max(items, key=lambda item: len(item["what"]))
        for number, items in candidates.items()
    }
    if len(sections) < 2:
        raise SectionParsingError("Fewer than two complete sections could be extracted")
    return dict(sorted(sections.items(), key=lambda item: _natural_section_key(item[0])))


def parse_ordered_bare_act_sections(text: str) -> dict[str, dict[str, str]]:
    """Parse the main Act body while rejecting TOC and schedule footnote numbers.

    Drishti PDFs commonly contain an ``ARRANGEMENT OF SECTIONS`` followed by a
    second occurrence of section 1 where the enacted text begins.  After that
    point real section numbers move forward; footnotes and schedules restart at
    small numbers and are therefore excluded from section boundaries.
    """
    normalized = _normalize_text(text)
    matches = list(BARE_SECTION_LINE_RE.finditer(normalized))
    if not matches:
        raise SectionParsingError("No numbered section lines were detected in the Bare Act PDF")

    arrangement = normalized.upper().find("ARRANGEMENT OF SECTIONS")
    one_matches = [match for match in matches if match.group("number").upper() == "1"]
    body_start = 0
    if arrangement >= 0 and len(one_matches) >= 2:
        body_start = one_matches[1].start()
    elif one_matches:
        body_start = one_matches[0].start()

    schedule_match = re.search(
        r"(?m)^[ \t]*(?:THE[ \t]+)?(?:FIRST[ \t]+)?SCHEDULE(?:[ \t]+I)?[ \t]*$",
        normalized[body_start:],
    )
    body_end = body_start + schedule_match.start() if schedule_match else len(normalized)
    body_matches = [match for match in matches if body_start <= match.start() < body_end]

    toc_titles: dict[str, str] = {}
    if arrangement >= 0 and body_start > arrangement:
        for toc_match in BARE_SECTION_LINE_RE.finditer(normalized[arrangement:body_start]):
            toc_titles.setdefault(
                toc_match.group("number").upper(),
                _clean_inline(toc_match.group("line")),
            )
    accepted: list[re.Match[str]] = []
    last_number = 0
    for match in body_matches:
        number = match.group("number").upper()
        base_match = re.match(r"(\d+)", number)
        if not base_match:
            continue
        base_number = int(base_match.group(1))
        line = _clean_inline(match.group("line"))
        expected_title = toc_titles.get(number)
        if expected_title and not _heading_matches_toc(line, expected_title):
            continue
        if not accepted:
            if base_number != 1:
                continue
        elif base_number < last_number:
            # Schedules and footnotes restart numbering; they are not new Act sections.
            continue
        elif base_number > last_number + 25:
            # Prevent page numbers or table values from becoming boundaries.
            continue
        if re.match(r"(?i)^(?:ins\.|subs\.|omitted|effective date|w\.e\.f\.)", line):
            continue
        accepted.append(match)
        last_number = max(last_number, base_number)

    if len(accepted) < 2:
        raise SectionParsingError("Fewer than two ordered Act sections could be extracted")

    sections: dict[str, dict[str, str]] = {}
    for index, match in enumerate(accepted):
        number = match.group("number").upper()
        end = accepted[index + 1].start() if index + 1 < len(accepted) else body_end
        first_line = _clean_inline(match.group("line"))
        title, opening = _split_statutory_heading(first_line)
        body = _clean_body(" ".join(part for part in (opening, normalized[match.end():end]) if part))
        if not body:
            continue
        existing = sections.get(number)
        candidate = {
            "section": number,
            "title": title or f"Section {number}",
            "text": body,
            "chapter": _nearest_chapter(normalized, match.start()),
        }
        if existing is None or len(candidate["text"]) > len(existing["text"]):
            sections[number] = candidate

    if len(sections) < 2:
        raise SectionParsingError("Fewer than two complete ordered sections could be extracted")
    return dict(sorted(sections.items(), key=lambda item: _natural_section_key(item[0])))


def _split_statutory_heading(value: str) -> tuple[str, str]:
    for pattern in (r"\s*\.\s*[—–-]\s*", r"\s*[—–]\s*", r"\.(?=\s*\(1\))"):
        parts = re.split(pattern, value, maxsplit=1)
        if len(parts) == 2:
            return parts[0].rstrip(". "), parts[1].strip()
    return value.rstrip(". "), ""


def _nearest_chapter(text: str, position: int) -> str:
    chapter = ""
    for match in CHAPTER_RE.finditer(text, 0, position):
        chapter = _clean_inline(match.group("chapter"))
    return chapter


def _heading_matches_toc(body_line: str, toc_line: str) -> bool:
    ignored = {"the", "a", "an", "of", "to", "and", "or", "for", "in", "by", "with"}
    body_words = {
        word for word in re.findall(r"[a-z]{3,}", body_line.lower()) if word not in ignored
    }
    toc_words = {
        word for word in re.findall(r"[a-z]{3,}", toc_line.lower()) if word not in ignored
    }
    if not toc_words:
        return True
    required = 1 if len(toc_words) <= 3 else 2
    return len(body_words & toc_words) >= required


def build_act_record(source_path: Path, source_format: str, law_type: str, candidate: dict[str, Any], source_metadata: dict[str, Any]) -> dict[str, Any]:
    text = extract_source_text(source_path, source_format)
    sections = parse_sections(text)
    act_name = str(candidate["act_name"])
    year_match = re.search(r"\b((?:18|19|20)\d{2})\b", act_name)
    citation_match = re.search(r"(?i)\bACT\s+(?:NO\.?\s*)?(\d+)\s+OF\s+((?:18|19|20)\d{2})\b", text[:30_000])
    return {
        "schema_version": "2.0",
        "record_type": "indian_act_section_dictionary",
        "country": "India",
        "law_type": law_type,
        "law": act_name,
        "year": candidate.get("year") or (year_match.group(1) if year_match else ""),
        "act_number": citation_match.group(1) if citation_match else "",
        "official_citation": f"Act No. {citation_match.group(1)} of {citation_match.group(2)}" if citation_match else "",
        "discovery": {
            "method": candidate.get("discovery_method"),
            "reason": candidate.get("reason"),
        },
        "section_count": len(sections),
        "sections": sections,
        "source": {
            "name": source_metadata.get("source_name"),
            "url": source_metadata.get("resolved_source_url") or source_metadata.get("source_url"),
            "format": source_format,
            "title": source_metadata.get("source_title"),
            "collected_at": source_metadata.get("collected_at"),
            "content_sha256": source_metadata.get("content_sha256"),
        },
        "notice": "The 'what' value preserves machine-extracted statutory text. Verify formatting, amendments, commencement and current legal status against the official source before legal use.",
    }


def _split_heading_and_body(value: str) -> tuple[str, str]:
    # Full Act text commonly starts after an em dash; TOC headings do not.
    parts = re.split(r"\s*[—–]\s*", value, maxsplit=1)
    if len(parts) == 2:
        return parts[0].rstrip(". "), parts[1]
    return value.rstrip(". "), ""


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00ad", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _clean_body(text: str) -> str:
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" -–—\n")


def _clean_inline(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _natural_section_key(number: str) -> tuple[int, str]:
    match = re.match(r"(\d+)(.*)", number)
    return (int(match.group(1)), match.group(2)) if match else (10**9, number)
