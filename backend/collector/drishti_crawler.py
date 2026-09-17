"""Beautiful Soup crawler for the Drishti Judiciary Bare Acts catalogue."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup


LOGGER = logging.getLogger("lexbrief.drishti_crawler")
CATALOG_URL = "https://www.drishtijudiciary.com/downloads/bare-acts/general-bare-acts?page=1"
ALLOWED_PAGE_HOSTS = {"drishtijudiciary.com", "www.drishtijudiciary.com"}
ALLOWED_PDF_HOSTS = {"vault.drishtijudiciary.com"}
USER_AGENT = "LexBrief-Drishti-Bare-Act-Collector/1.0 (+educational legal research)"
LOCATION_RE = re.compile(r"location\.href\s*=\s*(['\"])(?P<url>.+?)\1", re.IGNORECASE)
DATE_RE = re.compile(r"\b\d{1,2}-[A-Za-z]{3}-\d{4}\b")


@dataclass(frozen=True)
class DrishtiCategory:
    title: str
    url: str
    listed_date: str = ""


@dataclass(frozen=True)
class DrishtiBareAct:
    title: str
    pdf_url: str
    category: str
    category_url: str
    listed_date: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DrishtiCrawlError(RuntimeError):
    """Raised when the Drishti catalogue cannot be crawled safely."""


class DrishtiBareActCrawler:
    def __init__(self, timeout: int = 60, delay: float = 0.5, max_pages: int = 20) -> None:
        self.timeout = timeout
        self.delay = max(0.0, delay)
        self.max_pages = max(1, max_pages)
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": USER_AGENT, "Accept-Language": "en-IN,en;q=0.9"}
        )

    def crawl_catalog(self) -> tuple[list[DrishtiCategory], list[DrishtiBareAct]]:
        """Crawl category pages and return every unique PDF exposed by the site."""
        categories = self._crawl_categories()
        if not categories:
            raise DrishtiCrawlError("No Bare Act categories were found on the Drishti catalogue page")

        acts_by_identity: dict[str, DrishtiBareAct] = {}
        for position, category in enumerate(categories, start=1):
            LOGGER.info("Crawling Drishti category %s/%s: %s", position, len(categories), category.title)
            for act in self._crawl_category_acts(category):
                identity = self.canonical_act_key(act.title)
                existing = acts_by_identity.get(identity)
                if existing is None:
                    acts_by_identity[identity] = act
                elif existing.pdf_url == act.pdf_url:
                    merged_categories = " | ".join(
                        dict.fromkeys(existing.category.split(" | ") + act.category.split(" | "))
                    )
                    preferred = act if len(act.title) > len(existing.title) else existing
                    acts_by_identity[identity] = DrishtiBareAct(
                        title=preferred.title,
                        pdf_url=preferred.pdf_url,
                        category=merged_categories,
                        category_url=preferred.category_url,
                        listed_date=preferred.listed_date,
                    )
                else:
                    LOGGER.info(
                        "Skipping duplicate Act listing %r (%s)", act.title, act.pdf_url
                    )
            time.sleep(self.delay)
        acts = sorted(acts_by_identity.values(), key=lambda item: (item.category.lower(), item.title.lower()))
        if not acts:
            raise DrishtiCrawlError("The Drishti categories did not expose any downloadable PDF files")
        return categories, acts

    def _crawl_categories(self) -> list[DrishtiCategory]:
        categories: dict[str, DrishtiCategory] = {}
        seen_fingerprints: set[tuple[str, ...]] = set()
        for page_number in range(1, self.max_pages + 1):
            page_url = self._with_page(CATALOG_URL, page_number)
            soup = self._get_soup(page_url)
            page_items: list[DrishtiCategory] = []
            for node in soup.select("[onclick]"):
                target = self._onclick_url(node.get("onclick", ""))
                if not target or not self._is_category_url(target):
                    continue
                title, listed_date = self._title_and_date(node.get_text(" ", strip=True))
                page_items.append(DrishtiCategory(title=title, url=target, listed_date=listed_date))
            fingerprint = tuple(sorted(item.url for item in page_items))
            if not fingerprint or fingerprint in seen_fingerprints:
                break
            seen_fingerprints.add(fingerprint)
            for item in page_items:
                categories.setdefault(item.url, item)
            time.sleep(self.delay)
        return sorted(categories.values(), key=lambda item: item.title.lower())

    def _crawl_category_acts(self, category: DrishtiCategory) -> list[DrishtiBareAct]:
        acts: dict[str, DrishtiBareAct] = {}
        seen_fingerprints: set[tuple[str, ...]] = set()
        for page_number in range(1, self.max_pages + 1):
            page_url = self._with_page(category.url, page_number)
            soup = self._get_soup(page_url)
            page_items: list[DrishtiBareAct] = []
            for node in soup.select("[onclick]"):
                target = self._onclick_url(node.get("onclick", ""))
                if not target or not self._is_pdf_url(target):
                    continue
                title, listed_date = self._title_and_date(node.get_text(" ", strip=True))
                page_items.append(
                    DrishtiBareAct(
                        title=title,
                        pdf_url=target,
                        category=category.title,
                        category_url=category.url,
                        listed_date=listed_date,
                    )
                )
            fingerprint = tuple(sorted(item.pdf_url for item in page_items))
            if not fingerprint or fingerprint in seen_fingerprints:
                break
            seen_fingerprints.add(fingerprint)
            for item in page_items:
                acts.setdefault(self.canonical_act_key(item.title), item)
            time.sleep(self.delay)
        return list(acts.values())

    def _get_soup(self, url: str) -> BeautifulSoup:
        if not self._is_page_url(url):
            raise DrishtiCrawlError(f"Rejected page outside Drishti Judiciary: {url}")
        try:
            response = self.session.get(url, timeout=self.timeout, allow_redirects=True)
            response.raise_for_status()
        except requests.RequestException as error:
            raise DrishtiCrawlError(f"Could not load Drishti page {url}: {error}") from error
        if not self._is_page_url(response.url):
            raise DrishtiCrawlError("Drishti page redirected outside the expected website")
        if "html" not in response.headers.get("Content-Type", "").lower():
            raise DrishtiCrawlError(f"Expected HTML catalogue page but received {response.headers.get('Content-Type')}")
        return BeautifulSoup(response.content, "html.parser")

    @staticmethod
    def _onclick_url(value: str) -> str:
        match = LOCATION_RE.search(value or "")
        return match.group("url").strip() if match else ""

    @staticmethod
    def _title_and_date(value: str) -> tuple[str, str]:
        value = re.sub(r"\s+", " ", value).strip()
        date_match = DATE_RE.search(value)
        listed_date = date_match.group(0) if date_match else ""
        title = DATE_RE.sub("", value).strip(" -")
        return title or "Untitled Bare Act", listed_date

    @staticmethod
    def _with_page(url: str, page_number: int) -> str:
        parsed = urlparse(url)
        query = dict(parse_qsl(parsed.query, keep_blank_values=True))
        query["page"] = str(page_number)
        return urlunparse(parsed._replace(query=urlencode(query)))

    @staticmethod
    def _is_page_url(url: str) -> bool:
        parsed = urlparse(url)
        return parsed.scheme == "https" and (parsed.hostname or "").lower() in ALLOWED_PAGE_HOSTS

    @staticmethod
    def _is_category_url(url: str) -> bool:
        parsed = urlparse(url)
        return (
            DrishtiBareActCrawler._is_page_url(url)
            and "/downloads/general-bare-acts/" in parsed.path
            and not parsed.path.lower().endswith(".pdf")
        )

    @staticmethod
    def _is_pdf_url(url: str) -> bool:
        parsed = urlparse(url)
        return (
            parsed.scheme == "https"
            and (parsed.hostname or "").lower() in ALLOWED_PDF_HOSTS
            and parsed.path.lower().endswith(".pdf")
        )

    @staticmethod
    def canonical_act_key(title: str) -> str:
        """Return a stable identity so duplicate catalogue cards are collected once."""
        normalized = title.lower().replace("&", " and ")
        normalized = re.sub(r"\bthe\b", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        aliases = {
            "it act": "information technology act 2000",
            "ni act": "negotiable instruments act 1881",
            "ipc": "indian penal code 1860",
            "crpc": "code of criminal procedure 1973",
        }
        normalized = aliases.get(normalized, normalized)
        return re.sub(r"[^a-z0-9]+", "", normalized)
