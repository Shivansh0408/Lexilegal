"""Search and official-source download support for Indian Acts."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import logging
import re
import socket
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS


LOGGER = logging.getLogger("lexbrief.india_web_collector")
GOOGLE_SEARCH_URL = "https://www.googleapis.com/customsearch/v1"
USER_AGENT = "LexBrief-India-Law-Collector/3.0 (+educational legal research)"
# India Code relaunched at indiacode.gov.in on 13 Aug 2026 with a new UUID-based
# URL scheme (/act/<uuid>); the legacy indiacode.nic.in DSpace handle/bitstream
# IDs are no longer stable, so both domains are searched and the actual PDF
# assets the new portal links to are hosted on the s3waas.gov.in CDN.
OFFICIAL_DOMAINS = ("indiacode.gov.in", "cdnbbsr.s3waas.gov.in", "indiacode.nic.in", "legislative.gov.in", "gov.in", "nic.in")


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    search_provider: str
    query: str


@dataclass(frozen=True)
class DownloadedAct:
    path: Path
    source_format: str
    metadata: dict[str, Any]


class WebCollectionError(RuntimeError):
    """Raised when search or official-source collection cannot continue."""


class IndianLawCollector:
    def __init__(self, raw_root: Path, google_api_key: str = "", google_cx: str = "", timeout: int = 60, delay: float = 0.5, respect_robots: bool = True, max_download_bytes: int = 50 * 1024 * 1024) -> None:
        self.raw_root = Path(raw_root)
        self.google_api_key = google_api_key
        self.google_cx = google_cx
        self.timeout = timeout
        self.delay = delay
        self.respect_robots = respect_robots
        self.max_download_bytes = max_download_bytes
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "en-IN,en;q=0.9"})

    @property
    def search_provider(self) -> str:
        return "google" if self.google_api_key and self.google_cx else "ddgs"

    def collect_discovery_evidence(self, law_type: str, limit: int) -> list[dict[str, Any]]:
        queries = [
            f'Indian "{law_type}" major Acts list',
            f'India {law_type} Acts Codes legislation',
            f'site:indiacode.gov.in India "{law_type}" Act',
            f'site:indiacode.nic.in India "{law_type}" Act',
            f'site:legislative.gov.in "{law_type}" Acts India',
        ]
        results = self._search_queries(queries, limit)
        if not results:
            LOGGER.warning("No web discovery results; known candidates may still be validated individually")
        evidence: list[dict[str, Any]] = []
        for result in results:
            item = asdict(result)
            # Search snippets are sufficient for Act-name discovery. A small page
            # excerpt gives Llama context without treating commentary as law text.
            if self._is_safe_public_url(result.url) and self._is_html_candidate(result.url):
                item["page_excerpt"] = self._fetch_page_excerpt(result.url, 6_000)
            evidence.append(item)
            time.sleep(self.delay)
        return evidence

    def locate_official_act_sources(self, act_name: str, limit: int = 5) -> list[SearchResult]:
        """Return every plausible official source for an Act, ranked best-first.

        Search engines still index a mix of live and stale India Code links:
        the portal moved from indiacode.nic.in to indiacode.gov.in in August
        2026, individual handle/bitstream IDs are not stable across that move,
        and the new portal's actual PDF assets live on a separate CDN
        (cdnbbsr.s3waas.gov.in). Callers must try more than one candidate,
        since the top-ranked link can still 404.
        """
        queries = [
            f'site:cdnbbsr.s3waas.gov.in "{act_name}" filetype:pdf',
            f'site:indiacode.gov.in "{act_name}"',
            f'site:indiacode.nic.in "{act_name}"',
            f'site:legislative.gov.in "{act_name}" filetype:pdf',
        ]
        results = self._search_queries(queries, 20)
        scored = sorted(results, key=lambda item: self._official_source_score(item, act_name), reverse=True)
        sources: list[SearchResult] = []
        seen_urls: set[str] = set()
        for result in scored:
            if not self._is_official_url(result.url) or not self._source_matches_act(result, act_name):
                continue
            candidate = result if self._is_pdf_candidate(result.url) else self._find_official_pdf_link(result, act_name)
            if candidate is None or candidate.url in seen_urls:
                continue
            seen_urls.add(candidate.url)
            sources.append(candidate)
            if len(sources) >= limit:
                break
        return sources

    def download_act(self, source: SearchResult, destination: Path) -> DownloadedAct:
        destination.mkdir(parents=True, exist_ok=True)
        if not self._is_safe_public_url(source.url) or not self._is_official_url(source.url):
            raise WebCollectionError(f"Rejected non-official or unsafe Act URL: {source.url}")
        if self.respect_robots and not self._robots_allowed(source.url):
            raise WebCollectionError(f"robots.txt does not permit collection: {source.url}")

        try:
            with self.session.get(source.url, timeout=self.timeout, stream=True, allow_redirects=True) as response:
                response.raise_for_status()
                if not self._is_safe_public_url(response.url) or not self._is_official_url(response.url):
                    raise WebCollectionError("Official URL redirected outside an allowed Indian government domain")
                content_type = response.headers.get("Content-Type", "").lower()
                is_pdf = "application/pdf" in content_type or response.url.lower().split("?")[0].endswith(".pdf")
                suffix = ".pdf" if is_pdf else ".html"
                target = destination / f"official_source{suffix}"
                digest = hashlib.sha256()
                size = 0
                with target.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=65_536):
                        if not chunk:
                            continue
                        size += len(chunk)
                        if size > self.max_download_bytes:
                            raise WebCollectionError(f"Official source exceeds {self.max_download_bytes // (1024 * 1024)} MB")
                        digest.update(chunk)
                        handle.write(chunk)
        except requests.RequestException as error:
            raise WebCollectionError(f"Could not download official Act source: {error}") from error

        metadata = {
            "source_name": "India Code" if "indiacode.nic.in" in urlparse(source.url).netloc.lower() else "Legislative Department, Government of India",
            "source_url": source.url,
            "resolved_source_url": response.url,
            "source_title": source.title,
            "search_query": source.query,
            "search_provider": source.search_provider,
            "content_type": content_type,
            "content_sha256": digest.hexdigest(),
            "downloaded_bytes": size,
            "collected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self._write_json(destination / "source_metadata.json", metadata)
        return DownloadedAct(target, "pdf" if is_pdf else "html", metadata)

    def _search_queries(self, queries: list[str], limit: int) -> list[SearchResult]:
        if limit < 1:
            return []
        collected: list[SearchResult] = []
        seen: set[str] = set()
        for query in queries:
            remaining = limit - len(collected)
            if remaining <= 0:
                break
            batch = self._google_search(query, min(remaining, 10)) if self.search_provider == "google" else self._ddgs_search(query, remaining)
            for result in batch:
                normalized = self._normalize_url(result.url)
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                collected.append(SearchResult(result.title, normalized, result.snippet, result.search_provider, result.query))
                if len(collected) >= limit:
                    break
        return collected

    def _google_search(self, query: str, limit: int) -> list[SearchResult]:
        response = self.session.get(GOOGLE_SEARCH_URL, params={"key": self.google_api_key, "cx": self.google_cx, "q": query, "num": min(limit, 10)}, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        if payload.get("error"):
            raise WebCollectionError(payload["error"].get("message", "Google search failed"))
        return [SearchResult(str(item.get("title", "")), str(item.get("link", "")), str(item.get("snippet", "")), "Google Programmable Search", query) for item in payload.get("items", []) if item.get("link")]

    def _ddgs_search(self, query: str, limit: int) -> list[SearchResult]:
        try:
            raw_results = DDGS(timeout=self.timeout).text(query=query, region="in-en", safesearch="moderate", max_results=limit, backend="auto")
        except Exception as error:
            LOGGER.warning("Search failed for %r: %s", query, error)
            return []
        results: list[SearchResult] = []
        for item in raw_results or []:
            url = str(item.get("href") or item.get("url") or item.get("link") or "").strip()
            if url:
                results.append(SearchResult(str(item.get("title") or ""), url, str(item.get("body") or item.get("snippet") or item.get("description") or ""), "DDGS key-free metasearch", query))
        return results[:limit]

    def _find_official_pdf_link(self, page: SearchResult, act_name: str) -> SearchResult | None:
        if not self._is_safe_public_url(page.url):
            return None
        try:
            response = self.session.get(page.url, timeout=self.timeout, allow_redirects=True)
            response.raise_for_status()
        except requests.RequestException:
            return None
        if not self._is_safe_public_url(response.url) or not self._is_official_url(response.url):
            return None
        if "html" not in response.headers.get("Content-Type", "").lower():
            return None
        soup = BeautifulSoup(response.content, "html.parser")
        links: list[tuple[int, str, str]] = []
        terms = {term for term in re.findall(r"[a-z0-9]+", act_name.lower()) if len(term) > 3 and term not in {"the", "act", "code"}}
        for anchor in soup.find_all("a", href=True):
            url = urljoin(response.url, anchor["href"])
            if not self._is_official_url(url) or not self._is_pdf_candidate(url):
                continue
            label = anchor.get_text(" ", strip=True)
            haystack = f"{label} {url}".lower()
            score = sum(term in haystack for term in terms)
            links.append((score, url, label))
        if not links:
            return None
        _, url, label = max(links)
        return SearchResult(label or act_name, url, page.snippet, page.search_provider, page.query)

    def _fetch_page_excerpt(self, url: str, max_chars: int) -> str:
        if self.respect_robots and not self._robots_allowed(url):
            return ""
        try:
            response = self.session.get(url, timeout=self.timeout, allow_redirects=True)
            response.raise_for_status()
            if "html" not in response.headers.get("Content-Type", "").lower():
                return ""
        except requests.RequestException:
            return ""
        if not self._is_safe_public_url(response.url):
            return ""
        soup = BeautifulSoup(response.content, "html.parser")
        for node in soup(["script", "style", "noscript", "svg", "nav", "footer", "header", "aside", "form"]):
            node.decompose()
        return self._clean_text((soup.find("main") or soup.find("article") or soup.body or soup).get_text("\n", strip=True))[:max_chars]

    def _official_source_score(self, result: SearchResult, act_name: str) -> int:
        if not self._is_official_url(result.url):
            return -100
        domain = urlparse(result.url).netloc.lower()
        if "cdnbbsr.s3waas.gov.in" in domain:
            score = 35  # direct official PDF asset host used by the current India Code portal
        elif "indiacode.gov.in" in domain:
            score = 30  # current India Code portal (relaunched August 2026)
        elif "indiacode.nic.in" in domain:
            score = 20  # legacy portal; still partly live but individual links may 404
        elif "legislative.gov.in" in domain:
            score = 20
        else:
            score = 5
        score += 15 if self._is_pdf_candidate(result.url) else 0
        expected = {term for term in re.findall(r"[a-z0-9]+", act_name.lower()) if len(term) > 3}
        haystack = f"{result.title} {result.snippet} {result.url}".lower()
        score += sum(3 for term in expected if term in haystack)
        year = re.search(r"\b(?:18|19|20)\d{2}\b", act_name)
        if year and year.group(0) in haystack:
            score += 10
        return score

    @staticmethod
    def _source_matches_act(result: SearchResult, act_name: str) -> bool:
        ignored = {"the", "act", "code", "india", "indian"}
        expected = {term for term in re.findall(r"[a-z0-9]+", act_name.lower()) if len(term) > 3 and term not in ignored}
        haystack = f"{result.title} {result.snippet} {result.url}".lower()
        words_matched = sum(term in haystack for term in expected)
        year_match = re.search(r"\b(?:18|19|20)\d{2}\b", act_name)
        year_ok = not year_match or year_match.group(0) in haystack
        required_words = 1 if len(expected) <= 2 else 2
        return year_ok and words_matched >= required_words

    @staticmethod
    def _is_official_url(url: str) -> bool:
        host = (urlparse(url).hostname or "").lower()
        return any(host == domain or host.endswith("." + domain) for domain in OFFICIAL_DOMAINS)

    @staticmethod
    def _is_pdf_candidate(url: str) -> bool:
        lowered = url.lower()
        return lowered.split("?")[0].endswith(".pdf") or any(marker in lowered for marker in ("/bitstream/", "view-casepdf", "showfile?"))

    @staticmethod
    def _is_html_candidate(url: str) -> bool:
        return not IndianLawCollector._is_pdf_candidate(url)

    def _robots_allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        parser = RobotFileParser()
        parser.set_url(f"{parsed.scheme}://{parsed.netloc}/robots.txt")
        try:
            response = self.session.get(parser.url, timeout=min(self.timeout, 10))
            if response.status_code == 404:
                return True
            response.raise_for_status()
            parser.parse(response.text.splitlines())
            return parser.can_fetch(USER_AGENT, url)
        except (requests.RequestException, UnicodeError):
            return True

    @staticmethod
    def _is_safe_public_url(url: str) -> bool:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.hostname.lower() in {"localhost", "localhost.localdomain"}:
            return False
        try:
            for info in socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM):
                address = ipaddress.ip_address(info[4][0])
                if address.is_private or address.is_loopback or address.is_link_local or address.is_reserved:
                    return False
        except (socket.gaierror, ValueError):
            return False
        return True

    @staticmethod
    def _normalize_url(url: str) -> str:
        parsed = urlparse(url.strip())
        return parsed._replace(fragment="").geturl() if parsed.scheme in {"http", "https"} and parsed.netloc else ""

    @staticmethod
    def _clean_text(text: str) -> str:
        return "\n".join(line for line in (re.sub(r"\s+", " ", line).strip() for line in text.splitlines()) if line)

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)
