"""Drishti Judiciary Bare Act PDF-to-LLM dictionary pipeline.

Run from the project root:
    python backend/collector/collector.py

The command asks for a field/type of law, crawls the Drishti Judiciary Bare
Acts catalogue with Beautiful Soup, downloads matching PDFs, completely
extracts their text, and sends every extracted character to the configured
Llama endpoint to generate ``law``/``what`` dictionary records.
"""

from __future__ import annotations

import argparse
import difflib
import logging
import os
import re
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

try:
    from .dictionary_builder import IndianActDictionaryBuilder, slugify
    from .drishti_crawler import DrishtiBareAct, DrishtiBareActCrawler, DrishtiCrawlError
    from .llama_processor import LlamaProcessingError, LlamaProcessor
    from .pdf_py import DrishtiPdfProcessor, PdfProcessingError
except ImportError:  # Allows: python backend/collector/collector.py
    from dictionary_builder import IndianActDictionaryBuilder, slugify
    from drishti_crawler import DrishtiBareAct, DrishtiBareActCrawler, DrishtiCrawlError
    from llama_processor import LlamaProcessingError, LlamaProcessor
    from pdf_py import DrishtiPdfProcessor, PdfProcessingError


LOGGER = logging.getLogger("lexbrief.india_collector")
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "downloads" / "india"
DICTIONARY_DIR = BASE_DIR / "dictionary" / "india"


def _searchable_act_name(value: str) -> str:
    """Normalize an Act name for safe, deterministic catalogue searching."""
    value = value.lower().replace("&", " and ")
    value = re.sub(r"\bthe\b", " ", value)
    return re.sub(r"[^a-z0-9]+", "", value)


def find_specific_catalog_act(
    act_name: str, catalog: list[dict[str, str]]
) -> dict[str, str]:
    """Resolve one user-supplied Act name without silently choosing a wrong Act."""
    query = act_name.strip()
    if not query:
        raise ValueError("Specific Act name cannot be empty")

    query_key = DrishtiBareActCrawler.canonical_act_key(query)
    exact_matches = [
        item
        for item in catalog
        if DrishtiBareActCrawler.canonical_act_key(item["title"]) == query_key
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]

    searchable_query = _searchable_act_name(query)
    partial_matches = [
        item
        for item in catalog
        if searchable_query
        and (
            searchable_query in _searchable_act_name(item["title"])
            or _searchable_act_name(item["title"]) in searchable_query
        )
    ]
    if len(partial_matches) == 1:
        return partial_matches[0]

    if len(exact_matches) > 1 or len(partial_matches) > 1:
        matches = exact_matches or partial_matches
        choices = "; ".join(item["title"] for item in matches[:10])
        raise ValueError(
            f'Specific Act name "{query}" is ambiguous. Enter one complete Act name. Matches: {choices}'
        )

    title_by_key = {
        _searchable_act_name(item["title"]): item["title"] for item in catalog
    }
    close_keys = difflib.get_close_matches(
        searchable_query, title_by_key.keys(), n=5, cutoff=0.45
    )
    suggestions = [title_by_key[key] for key in close_keys]
    suffix = f" Closest catalogue names: {'; '.join(suggestions)}" if suggestions else ""
    raise ValueError(
        f'No Drishti Bare Act matched "{query}". Check the name and try again.{suffix}'
    )


def _build_processor(args: argparse.Namespace) -> LlamaProcessor:
    return LlamaProcessor(
        api_url=args.llama_url or os.getenv("LLAMA_API_URL", "http://localhost:11434/api/generate"),
        model=args.model or os.getenv("LLAMA_MODEL", "llama3.2:3b"),
        timeout=args.llama_timeout,
        allow_keyword_fallback=True,
        max_retries=args.llama_max_retries,
        retry_base_delay=args.retry_base_delay,
        large_act_threshold=args.large_act_threshold,
        large_act_batch_chars=args.large_act_batch_chars,
        large_act_max_sections=args.large_act_max_sections,
        large_act_batch_pause=args.large_act_batch_pause,
    )


def run_india_pipeline(args: argparse.Namespace) -> int:
    """Crawl Drishti, process matching PDFs and store LLM dictionaries."""
    load_dotenv(BASE_DIR / ".env")
    law_type = args.law_type.strip()
    specific_act_name = (args.act_name or "").strip()
    if not specific_act_name and not law_type:
        raise ValueError("Law type cannot be empty")

    crawler = DrishtiBareActCrawler(
        timeout=args.timeout,
        delay=args.delay,
        max_pages=args.max_crawl_pages,
    )
    pdf_processor = DrishtiPdfProcessor(
        timeout=args.timeout,
        max_download_bytes=args.max_download_mb * 1024 * 1024,
    )
    processor = _build_processor(args)
    LOGGER.info("Stage 1/3: crawling the Drishti Judiciary Bare Acts catalogue")
    categories, catalog_items = crawler.crawl_catalog()
    catalog = [item.to_dict() for item in catalog_items]
    LOGGER.info("Crawled %s categories and %s unique PDFs", len(categories), len(catalog))

    if specific_act_name:
        LOGGER.info("Stage 2/3: searching for the specific Act: %s", specific_act_name)
        matched_act = find_specific_catalog_act(specific_act_name, catalog)
        law_type = matched_act["category"] or "Specific Act"
        candidates = [{**matched_act, "act_name": matched_act["title"]}]
        LOGGER.info("Matched specific Act: %s", matched_act["title"])
    else:
        LOGGER.info("Stage 2/3: selecting Acts related to %s", law_type)
        selected = processor.select_relevant_catalog_acts(law_type, catalog)
        candidates = [{**item, "act_name": item["title"]} for item in selected]
        if args.max_acts > 0:
            candidates = candidates[: args.max_acts]

    raw_root = RAW_DIR / slugify(law_type)
    dictionary_root = DICTIONARY_DIR / slugify(law_type)
    raw_root.mkdir(parents=True, exist_ok=True)
    builder = IndianActDictionaryBuilder(dictionary_root, law_type)
    builder.write_discovery(candidates, catalog)

    LOGGER.info("Selected %s matching Bare Acts", len(candidates))
    LOGGER.info("Stage 3/3: downloading PDFs, fully extracting text and running Llama")
    succeeded = 0
    skipped = 0
    halted = False
    for position, candidate in enumerate(candidates, start=1):
        act_name = candidate["act_name"]
        LOGGER.info("[%s/%s] %s", position, len(candidates), act_name)
        existing = builder.find_existing_complete(act_name, candidate.get("pdf_url", ""))
        if existing is not None:
            LOGGER.info("Skipping already indexed unique Act: %s (%s)", act_name, existing[0])
            skipped += 1
            continue
        try:
            source = DrishtiBareAct(
                title=candidate["title"],
                pdf_url=candidate["pdf_url"],
                category=candidate["category"],
                category_url=candidate["category_url"],
                listed_date=candidate.get("listed_date", ""),
            )
            extracted = pdf_processor.download_and_extract(
                source, raw_root / slugify(act_name)
            )
            LOGGER.info(
                "Extracted %s/%s PDF pages and parsed %s numbered sections",
                extracted.metadata["extraction"].get("extracted_page_count"),
                extracted.metadata["extraction"].get("page_count"),
                len(extracted.sections),
            )
            act_output_dir = raw_root / slugify(act_name)
            is_large_act = len(extracted.sections) > args.large_act_threshold
            if is_large_act:
                LOGGER.info(
                    "Large Act lock enabled for %s (%s sections). The next Act will not start until this Act is fully indexed.",
                    act_name,
                    len(extracted.sections),
                )

            record = None
            last_llama_error: LlamaProcessingError | None = None
            for act_attempt in range(1, args.act_max_retries + 1):
                if act_attempt > 1:
                    LOGGER.warning(
                        "Resuming %s from its section checkpoint (Act recovery attempt %s/%s)",
                        act_name,
                        act_attempt,
                        args.act_max_retries,
                    )
                processor.wait_until_ready(max_attempts=6, delay=args.retry_base_delay)
                try:
                    record = processor.generate_section_index_record(
                        law_type=law_type,
                        act_name=act_name,
                        sections=extracted.sections,
                        source_metadata=extracted.metadata,
                        batch_chars=args.llm_chunk_chars,
                        checkpoint_path=act_output_dir / "llm_section_checkpoint.json",
                    )
                    break
                except LlamaProcessingError as error:
                    last_llama_error = error
                    if act_attempt >= args.act_max_retries:
                        break
                    recovery_delay = args.retry_base_delay * act_attempt
                    LOGGER.warning(
                        "Act indexing paused after a Llama error: %s. Waiting %.1f seconds before checkpoint resume.",
                        error,
                        recovery_delay,
                    )
                    time.sleep(recovery_delay)

            if record is None:
                raise last_llama_error or LlamaProcessingError("Act indexing did not complete")
            target = builder.store(record)
            LOGGER.info(
                "Saved %s after processing %s PDF pages and indexing %s numbered sections",
                target,
                record["pdf_extraction"].get("page_count"),
                record["section_count"],
            )
            succeeded += 1
            if is_large_act:
                LOGGER.info(
                    "Large Act fully indexed. Cooling down Ollama for %.1f seconds before the next Act.",
                    args.between_act_cooldown,
                )
                time.sleep(max(0.0, args.between_act_cooldown))
                processor.wait_until_ready(max_attempts=6, delay=args.retry_base_delay)
        except (PdfProcessingError, LlamaProcessingError, OSError, ValueError) as error:
            LOGGER.error("Could not collect %s: %s", act_name, error)
            builder.record_failure(candidate, str(error))
            LOGGER.error(
                "Pipeline stopped before the next Act so %s cannot be silently left partially indexed. Rerun the same command to resume its checkpoint.",
                act_name,
            )
            halted = True
            break

    index_path = builder.write_index()
    LOGGER.info(
        "Complete: %s newly saved, %s existing skipped, %s selected; index: %s",
        succeeded,
        skipped,
        len(candidates),
        index_path,
    )
    return 1 if halted or (succeeded + skipped == 0) else 0


def _prompt(label: str, default: str) -> str:
    response = input(f"{label} [{default}]: ").strip()
    return response or default


def apply_interactive_inputs(args: argparse.Namespace) -> argparse.Namespace:
    print("\nLexBrief Drishti Bare Act PDF Collector")
    print("Press Enter to accept the default shown in brackets.\n")
    print("1. Existing pipeline: collect every Act related to a law type")
    print("2. Specific Act search: collect and index only one typed Act")
    mode = _prompt("Choose collection mode (1 or 2)", "1")
    if mode == "1":
        args.act_name = None
        args.law_type = _prompt("Law type", "Business Law")
        raw_max = _prompt("Maximum Acts to collect (0 = every discovered Act)", str(args.max_acts))
        try:
            args.max_acts = max(0, int(raw_max))
        except ValueError as error:
            raise ValueError("Maximum Acts must be a whole number") from error
        print(f"\nCrawling Drishti and creating an India-only {args.law_type} library...\n")
    elif mode == "2":
        args.act_name = input("Enter the specific Act name: ").strip()
        if not args.act_name:
            raise ValueError("Specific Act name cannot be empty")
        args.max_acts = 1
        print(f'\nCrawling Drishti and indexing only "{args.act_name}"...\n')
    else:
        raise ValueError("Collection mode must be 1 or 2")
    return args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Crawl Drishti Bare Acts, fully extract matching PDFs, and create LLM law/what dictionaries."
    )
    parser.add_argument("--law-type", "--field", dest="law_type", default="Business Law", help="Law type, for example Business Law")
    parser.add_argument("--act-name", help="Collect and index only the single matching Bare Act")
    parser.add_argument("--max-acts", type=int, default=0, help="Maximum Acts to collect; default 0 collects every discovered Act")
    parser.add_argument("--search-results", type=int, default=30, help="Legacy option retained for compatibility")
    parser.add_argument("--model", help="Ollama model name")
    parser.add_argument("--llama-url", help="Ollama-compatible /api/generate endpoint")
    parser.add_argument("--timeout", type=int, default=60, help="Web request timeout in seconds")
    parser.add_argument("--llama-timeout", type=int, default=300, help="Llama read timeout per request in seconds")
    parser.add_argument("--llama-max-retries", type=int, default=4, help="Retries for timeout, connection, HTTP, or invalid JSON responses")
    parser.add_argument("--act-max-retries", type=int, default=3, help="Checkpoint resume attempts before stopping the pipeline")
    parser.add_argument("--retry-base-delay", type=float, default=5.0, help="Base retry delay in seconds")
    parser.add_argument("--delay", type=float, default=0.5, help="Polite pause between requests")
    parser.add_argument("--max-download-mb", type=int, default=100, help="Maximum size of one Bare Act PDF")
    parser.add_argument("--max-crawl-pages", type=int, default=20, help="Safety limit for category pagination")
    parser.add_argument("--llm-chunk-chars", type=int, default=24000, help="Maximum section text characters sent in one Llama batch")
    parser.add_argument("--large-act-threshold", type=int, default=160, help="Section count that enables protected large-Act mode")
    parser.add_argument("--large-act-batch-chars", type=int, default=7000, help="Maximum characters per Llama batch in large-Act mode")
    parser.add_argument("--large-act-max-sections", type=int, default=8, help="Maximum sections per Llama batch in large-Act mode")
    parser.add_argument("--large-act-batch-pause", type=float, default=1.0, help="Pause between large-Act batches")
    parser.add_argument("--between-act-cooldown", type=float, default=5.0, help="Ollama cooldown after a large Act completes")
    parser.add_argument("--ignore-robots", action="store_true", help="Legacy option retained for compatibility")
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        args = build_parser().parse_args()
        if len(sys.argv) == 1:
            args = apply_interactive_inputs(args)
        if (
            args.max_acts < 0
            or args.search_results < 1
            or args.max_download_mb < 1
            or args.max_crawl_pages < 1
            or args.llm_chunk_chars < 4000
            or args.llama_max_retries < 1
            or args.act_max_retries < 1
            or args.retry_base_delay < 0
            or args.large_act_threshold < 1
            or args.large_act_batch_chars < 4000
            or args.large_act_max_sections < 1
            or args.large_act_batch_pause < 0
            or args.between_act_cooldown < 0
        ):
            raise ValueError("Numeric limits must be positive; only --max-acts may be 0")
        return run_india_pipeline(args)
    except KeyboardInterrupt:
        LOGGER.info("Collection cancelled by user")
        return 130
    except (ValueError, DrishtiCrawlError, PdfProcessingError, LlamaProcessingError, requests.RequestException) as error:
        LOGGER.error("Collection stopped: %s", error)
        return 2


if __name__ == "__main__":
    sys.exit(main())
