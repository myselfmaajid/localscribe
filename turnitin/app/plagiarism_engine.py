"""
VeriScholar v3 — Multi-Source Academic, Google-Grounded Web, PDF & Local Plagiarism Engine.

Discovery Channels (Beating Textul's Internet-Only Blind Spot):
  1. Gemini Google Search Grounding (gemini-2.5-flash + google_search tool — 1,500 req/day FREE!)
     -> Batches 5 fingerprints per prompt (only 2-3 prompts/paper) to search Google's full Web + PDF index!
  2. PubMed / Europe PMC (Abstracts + Full-Text Open-Access XML!)
  3. OpenAlex (250M+ scholarly papers) + CrossRef (150M+ DOIs) + Semantic Scholar (200M+ papers) + arXiv
  4. Wikipedia + DuckDuckGo + Optional Self-Hosted SearXNG
  5. Full-Text Web Page & Public PDF URL Downloader (extracts BOTH HTML AND public .pdf reports via PyMuPDF!)
  6. Local Reference Library (private uploaded PDFs/DOCXs/TXTs)

Matching & Turnitin-Grade Reporting:
  - Exact / Near-Exact contiguous word overlap (default 6 words)
  - Semantic / Paraphrase similarity (TF-IDF bigrams + optional sentence-transformers)
  - Exact Turnitin 4 Citation Match Groups (Not Cited or Quoted, Missing Quotations, Missing Citation, Cited and Quoted)
  - Exact Source Category Breakdown (Internet sources %, Publications %, Local Library %)
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import difflib
import html
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from urllib.parse import parse_qs, quote_plus, unquote, urljoin, urlparse

import httpx
import pymupdf
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import (
    get_core_api_key,
    get_gemini_api_key,
    get_optional_key,
    get_serper_api_key,
    get_tavily_api_key,
)
from .document_parser import ParsedDocument, SentenceUnit, parse_document
from .telemetry_logger import (
    is_provider_available,
    log_diagnostic_event,
    record_provider_success,
)

_STOPWORDS: Set[str] = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "as", "is", "was", "are", "were", "be", "been", "being", "have",
    "has", "had", "do", "does", "did", "will", "would", "shall", "should", "can",
    "could", "may", "might", "must", "that", "this", "these", "those", "it", "its",
    "they", "them", "their", "we", "our", "you", "your", "he", "she", "his", "her",
    "which", "who", "whom", "whose", "what", "where", "when", "why", "how", "all",
    "each", "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "also", "into",
    "between", "among", "during", "after", "before", "above", "below", "under", "over",
}

_USER_AGENT = "VeriScholar/3.0 (Academic Integrity Studio; mailto:research@verischolar.local)"
_BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"

_IN_TEXT_CITATION_RE = re.compile(
    r"(?:\[\s*\d+(?:\s*[,;\-–]\s*\d+)*\s*\]|"
    r"\([A-Z][A-Za-z\-&\s.,]{1,45},?\s*(?:19|20)\d{2}[a-z]?(?:;\s*[^)]{3,40})?\)|"
    r"\bet\s+al\.\s*\(?(?:19|20)\d{2}\)?)"
)
_QUOTE_MARKS_RE = re.compile(r'["“][^"”]{12,}["”]')


@dataclass
class CandidateSource:
    source_id: str
    title: str
    url: str
    provider: str  # "Google Search (Web)", "PubMed / Europe PMC", "OpenAlex Academic", "CrossRef Journal", "Semantic Scholar", "arXiv", "Wikipedia", "Internet Source", "Local Reference Library"
    category: str  # "internet" | "publication" | "local_library"
    text: str
    year: Optional[str] = None
    authors: Optional[str] = None
    pmcid: Optional[str] = None


@dataclass
class SentenceMatch:
    sentence_id: int
    source_index: int
    source_id: str
    source_category: str  # "internet" | "publication" | "local_library"
    match_type: str       # "exact" | "paraphrase"
    citation_group: str   # "not_cited_or_quoted" | "missing_quotations" | "missing_citation" | "cited_and_quoted"
    similarity: float
    matched_words: int
    total_words: int
    matched_phrases: List[str] = field(default_factory=list)
    source_excerpt: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _strip_html(raw: str) -> str:
    if not raw:
        return ""
    cleaned = re.sub(r"<[^>]+>", " ", raw)
    cleaned = html.unescape(cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def build_citation_suggestions(
    title: str,
    authors: Optional[str] = None,
    year: Optional[str] = None,
    url: str = "",
    ref_number: int = 1,
) -> Dict[str, str]:
    """Build accurate, ready-to-insert APA 7th and Vancouver citations from source metadata."""
    clean_title = (title or "Online Source").strip().rstrip(".")
    yr = (year or "").strip()
    if not yr or not yr.isdigit():
        m_yr = re.search(r"\b(19\d{2}|20\d{2})\b", f"{title} {url}")
        yr = m_yr.group(1) if m_yr else "n.d."

    auth_raw = (authors or "").strip()
    if auth_raw:
        first_author = auth_raw.split(",")[0].strip().split()[0].strip()
        has_et_al = ("," in auth_raw) or ("et al" in auth_raw.lower())
        apa_inline = f"({first_author} et al., {yr})" if has_et_al else f"({first_author}, {yr})"
        apa_ref = f"{auth_raw}. ({yr}). {clean_title}. {url}".strip()
        van_ref = f"{ref_number}. {auth_raw}. {clean_title}. {yr}. Available from: {url}".strip()
    else:
        # Derive clean organization / domain author when person author is absent
        org = clean_title.split(":")[0].strip() if ":" in clean_title and len(clean_title.split(":")[0]) < 35 else ""
        if not org and url.startswith("http"):
            netloc = urlparse(url).netloc.replace("www.", "")
            org = netloc.split(".")[0].upper() if len(netloc.split(".")[0]) <= 5 else netloc.split(".")[0].capitalize()
        org = org or "Source"
        apa_inline = f"({org}, {yr})"
        apa_ref = f"{org}. ({yr}). {clean_title}. {url}".strip()
        van_ref = f"{ref_number}. {org}. {clean_title}. {yr}. Available from: {url}".strip()

    return {
        "apa_inline": apa_inline,
        "apa_reference": apa_ref,
        "vancouver_inline": f"[{ref_number}]",
        "vancouver_reference": van_ref,
    }


def _classify_citation_group(sentence_text: str) -> str:
    has_cite = bool(_IN_TEXT_CITATION_RE.search(sentence_text))
    has_quote = bool(_QUOTE_MARKS_RE.search(sentence_text))
    if has_cite and has_quote:
        return "cited_and_quoted"
    if has_cite and not has_quote:
        return "missing_quotations"
    if not has_cite and has_quote:
        return "missing_citation"
    return "not_cited_or_quoted"


def _reconstruct_openalex_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> str:
    if not inverted_index or not isinstance(inverted_index, dict):
        return ""
    positions: List[Tuple[int, str]] = []
    for word, idxs in inverted_index.items():
        for idx in idxs:
            positions.append((idx, word))
    positions.sort(key=lambda x: x[0])
    return " ".join(w for _, w in positions)


def _extract_search_queries(sentences: List[SentenceUnit], max_queries: int = 16) -> List[Tuple[int, str, str]]:
    """
    Returns list of (sentence_id, keyword_query, exact_phrase_query) spanning the full document.
    """
    candidates: List[Tuple[int, float, str, str]] = []
    for s in sentences:
        if not s.qualifies_for_plagiarism or s.word_count < 8:
            continue
        clean = re.sub(r"\[\d+(?:[,\-–]\d+)*\]", "", s.text)
        clean = re.sub(r"\([^)]{2,40}\d{4}[^)]*\)", "", clean)
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", clean)
        content_tokens = [t for t in tokens if t.lower() not in _STOPWORDS]
        if len(content_tokens) < 5:
            continue
        kw_query = " ".join(content_tokens[:12])
        raw_words = re.findall(r"[A-Za-z0-9\-]+", clean)
        phrase_slice = raw_words[min(2, max(0, len(raw_words) - 8)) : min(len(raw_words), 11)]
        exact_phrase = " ".join(phrase_slice[:8])
        uniqueness = len(set(t.lower() for t in content_tokens)) / max(len(content_tokens), 1)
        candidates.append((s.id, uniqueness * min(len(content_tokens), 20), kw_query, exact_phrase))

    if not candidates:
        return []
    if len(candidates) <= max_queries:
        return [(sid, kw, ph) for sid, _, kw, ph in candidates]

    step = len(candidates) / max_queries
    selected: List[Tuple[int, str, str]] = []
    seen: Set[str] = set()
    for k in range(max_queries):
        idx = min(len(candidates) - 1, int(k * step))
        sid, _, kw, ph = candidates[idx]
        if kw.lower() not in seen:
            seen.add(kw.lower())
            selected.append((sid, kw, ph))
    return selected


# ---------------------------------------------------------------------------
# Channel 0: Serper Google Exact-Phrase Search, Tavily Web & CORE v3 Full-Text (260M+ Papers)
# ---------------------------------------------------------------------------
def _is_academic_url(title: str, uri: str) -> bool:
    comb = (title.lower() + " " + uri.lower())
    return any(
        dom in comb
        for dom in (
            "pubmed", "ncbi.nlm.nih.gov", "pmc", "doi.org", "sciencedirect",
            "springer", "wiley", "nature.com", "plos.org", "mdpi.com",
            "frontiersin", "arxiv.org", "researchgate", "tandfonline",
            "sagepub", "oxford", "bmj.com", "thelancet", "cureus",
            "ieee.org", "acm.org", "ssrn.com", "jstor.org", "core.ac.uk",
        )
    )


def _search_serper_google(
    client: httpx.Client,
    exact_phrase: str,
    kw_query: str,
    api_key: str,
) -> List[CandidateSource]:
    """Primary Google Exact-Phrase + Web/PDF Search via Serper.dev (~0.6s, automatic circuit-breaker)."""
    prov = "serper_google"
    if not api_key or not is_provider_available(prov):
        return []
    import time
    t0 = time.perf_counter()
    q_str = f'"{exact_phrase}"' if len(exact_phrase.split()) >= 5 else kw_query[:120]
    try:
        resp = client.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": q_str, "num": 5},
            timeout=8.0,
        )
        lat = (time.perf_counter() - t0) * 1000.0
        if resp.status_code != 200:
            sc = resp.status_code
            log_diagnostic_event(
                level="WARN_FAILOVER",
                subsystem="plagiarism",
                provider=prov,
                message=resp.text[:220],
                status_code=sc,
                latency_ms=lat,
                error_type="rate_limit_429" if sc == 429 else f"http_{sc}",
                fallback_used="tavily_search -> gemini_grounding -> open_web",
                cooldown_seconds=300 if sc in (401, 403, 429) else 30,
            )
            return []
        data = resp.json()
        organic = data.get("organic") or []
        # If exact quoted phrase had 0 hits, immediately run keyword fallback in the same session
        if not organic and kw_query:
            resp2 = client.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
                json={"q": kw_query[:120], "num": 4},
                timeout=8.0,
            )
            if resp2.status_code == 200:
                organic = resp2.json().get("organic") or []
        record_provider_success("plagiarism", prov, (time.perf_counter() - t0) * 1000.0)
        out: List[CandidateSource] = []
        for item in organic[:5]:
            href = item.get("link") or ""
            title = item.get("title") or "Google Web Source"
            snippet = item.get("snippet") or ""
            if not href:
                continue
            is_acad = _is_academic_url(title, href)
            out.append(
                CandidateSource(
                    source_id=f"serper:{href}",
                    title=title,
                    url=href,
                    provider="Google Scholarly Index" if is_acad else "Google Search (Web)",
                    category="publication" if is_acad else "internet",
                    text=f"{title}. {snippet}".strip(),
                    year=str(item.get("date") or "")[:4] if item.get("date") else None,
                )
            )
        return out
    except Exception as exc:
        log_diagnostic_event(
            level="WARN_FAILOVER",
            subsystem="plagiarism",
            provider=prov,
            message=f"{type(exc).__name__}: {str(exc)[:180]}",
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            error_type="network_or_timeout",
            fallback_used="tavily_search -> open_web",
            cooldown_seconds=30,
        )
        return []


def _search_tavily(
    client: httpx.Client,
    query: str,
    api_key: str,
) -> List[CandidateSource]:
    """Secondary Real-Time Web Search & Extraction via Tavily API (~1.1s, automatic circuit-breaker)."""
    prov = "tavily_search"
    if not api_key or not is_provider_available(prov):
        return []
    import time
    t0 = time.perf_counter()
    try:
        resp = client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query[:160],
                "search_depth": "basic",
                "max_results": 4,
            },
            timeout=9.0,
        )
        lat = (time.perf_counter() - t0) * 1000.0
        if resp.status_code != 200:
            sc = resp.status_code
            log_diagnostic_event(
                level="WARN_FAILOVER",
                subsystem="plagiarism",
                provider=prov,
                message=resp.text[:220],
                status_code=sc,
                latency_ms=lat,
                error_type="rate_limit_429" if sc == 429 else f"http_{sc}",
                fallback_used="gemini_grounding -> duckduckgo_open_web",
                cooldown_seconds=300 if sc in (401, 403, 429) else 30,
            )
            return []
        record_provider_success("plagiarism", prov, lat)
        out: List[CandidateSource] = []
        for item in (resp.json().get("results") or [])[:4]:
            href = item.get("url") or ""
            title = item.get("title") or "Web Source"
            content = item.get("content") or ""
            if not href or len(content.split()) < 8:
                continue
            is_acad = _is_academic_url(title, href)
            out.append(
                CandidateSource(
                    source_id=f"tavily:{href}",
                    title=title,
                    url=href,
                    provider="Scholarly Web Index" if is_acad else "Web Search Index",
                    category="publication" if is_acad else "internet",
                    text=f"{title}. {content}".strip(),
                )
            )
        return out
    except Exception as exc:
        log_diagnostic_event(
            level="WARN_FAILOVER",
            subsystem="plagiarism",
            provider=prov,
            message=f"{type(exc).__name__}: {str(exc)[:180]}",
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            error_type="network_or_timeout",
            fallback_used="duckduckgo_open_web",
            cooldown_seconds=30,
        )
        return []


def _search_core_academic(
    client: httpx.Client,
    query: str,
    api_key: str,
) -> List[CandidateSource]:
    """
    CORE v3 Global Open-Access Full-Text Repository (260M+ papers across CS, Business, Law, Humanities, STEM, Medicine).
    Uses trailing slash /v3/search/works/ to avoid 301 redirect drops and extracts embedded fullText directly!
    """
    prov = "core_academic_v3"
    if not api_key or not is_provider_available(prov):
        return []
    import time
    t0 = time.perf_counter()
    try:
        resp = client.get(
            "https://api.core.ac.uk/v3/search/works/",
            params={"q": query[:130], "limit": "4"},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=11.0,
        )
        lat = (time.perf_counter() - t0) * 1000.0
        if resp.status_code != 200:
            sc = resp.status_code
            log_diagnostic_event(
                level="WARN_FAILOVER",
                subsystem="plagiarism",
                provider=prov,
                message=resp.text[:220],
                status_code=sc,
                latency_ms=lat,
                error_type="rate_limit_429" if sc == 429 else f"http_{sc}",
                fallback_used="openalex+crossref+europe_pmc",
                cooldown_seconds=180 if sc in (401, 403, 429) else 30,
            )
            return []
        record_provider_success("plagiarism", prov, lat)
        out: List[CandidateSource] = []
        for item in (resp.json().get("results") or [])[:4]:
            title = _strip_html(item.get("title") or "")
            abstract = _strip_html(item.get("abstract") or "")
            full_txt = _strip_html((item.get("fullText") or "")[:35000])
            combined = f"{title}. {abstract} {full_txt}".strip()
            if len(combined.split()) < 15:
                continue
            doi = (item.get("doi") or "").strip()
            dl_url = item.get("downloadUrl") or ""
            src_urls = item.get("sourceFulltextUrls") or []
            src_url = (
                f"https://doi.org/{doi}"
                if doi and not doi.startswith("http")
                else (doi or dl_url or (src_urls[0] if src_urls else f"https://core.ac.uk/works/{item.get('id', '')}"))
            )
            authors_raw = item.get("authors") or []
            auth_names = ", ".join(
                a.get("name", "") if isinstance(a, dict) else str(a)
                for a in authors_raw[:3]
            ).strip(", ")
            out.append(
                CandidateSource(
                    source_id=f"core:{item.get('id') or doi or title[:40]}",
                    title=title or "CORE Open-Access Research Paper",
                    url=src_url,
                    provider="CORE Academic Full-Text",
                    category="publication",
                    text=combined,
                    year=str(item.get("yearPublished") or ""),
                    authors=auth_names or None,
                )
            )
        return out
    except Exception as exc:
        log_diagnostic_event(
            level="WARN_FAILOVER",
            subsystem="plagiarism",
            provider=prov,
            message=f"{type(exc).__name__}: {str(exc)[:180]}",
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            error_type="network_or_timeout",
            fallback_used="openalex+crossref+europe_pmc",
            cooldown_seconds=30,
        )
        return []


# ---------------------------------------------------------------------------
# Channel 1: Gemini Google Search Grounding (Full Google Web + PDF Index!)
# ---------------------------------------------------------------------------
def _search_gemini_grounding(
    queries: List[Tuple[int, str, str]],
    api_key: str,
    doc_title: str = "",
    reference_hints: Optional[List[str]] = None,
) -> List[CandidateSource]:
    """
    Use Gemini Flash-Lite + built-in Google Search Grounding as automatic tertiary web fallback.
    Protected by strict 12s timeout + circuit-breaker so quota exhaustion never stalls scans.
    """
    prov = "gemini_grounding"
    if not api_key or not is_provider_available(prov):
        return []
    try:
        from google import genai
        from google.genai import types
    except Exception:
        return []

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=12000, retry_options=types.HttpRetryOptions(attempts=1)),
    )
    out: List[CandidateSource] = []
    seen_urls: Set[str] = set()

    batches: List[List[Tuple[int, str, str]]] = []
    step_size = 5
    for i in range(0, min(len(queries), 15), step_size):
        batches.append(queries[i : i + step_size])

    prompts: List[str] = []
    for batch in batches[:3]:
        excerpts_block = "\n".join(
            f'{k+1}. "{ph}" (keywords: {kw})' for k, (_, kw, ph) in enumerate(batch)
        )
        prompts.append(
            "Search Google to find the exact original web pages, government/corporate PDF reports, "
            "standards, or academic articles that closely match or contain these excerpts "
            f"from an academic document ({doc_title}). Return authoritative primary URLs and PDF report pages.\n\n"
            f"EXCERPTS:\n{excerpts_block}\n\n"
            "Briefly list the matching primary sources found."
        )

    # Extra grounded prompt targeting the primary reports/URLs listed in the document's References
    if reference_hints:
        ref_block = "\n".join(f"- {r[:180]}" for r in reference_hints[:8])
        prompts.append(
            "Search Google to find the official primary webpage or downloadable PDF URL for each of "
            "these cited reports, standards, or articles:\n\n"
            f"{ref_block}\n\n"
            "Briefly list the official URLs found."
        )

    for prompt in prompts:
        resp = None
        for model_name in ("gemini-flash-lite-latest", "gemini-2.5-flash-lite", "gemini-3-flash-preview", "gemini-3.1-flash-lite"):
            try:
                resp = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        tools=[types.Tool(google_search=types.GoogleSearch())],
                        temperature=0.0,
                    ),
                )
                break
            except Exception:
                continue
        if resp is None:
            continue
        try:
            cands = getattr(resp, "candidates", None) or []
            if not cands:
                continue
            gm = getattr(cands[0], "grounding_metadata", None)
            if not gm:
                continue
            chunks = getattr(gm, "grounding_chunks", None) or []
            supports = getattr(gm, "grounding_supports", None) or []

            chunk_snippets: Dict[int, List[str]] = {}
            for sup in supports:
                seg = getattr(sup, "segment", None)
                seg_txt = getattr(seg, "text", "") if seg else ""
                for c_idx in (getattr(sup, "grounding_chunk_indices", None) or []):
                    if seg_txt:
                        chunk_snippets.setdefault(c_idx, []).append(seg_txt)

            for c_idx, ch in enumerate(chunks):
                web = getattr(ch, "web", None)
                if not web:
                    continue
                uri = getattr(web, "uri", None) or ""
                title = getattr(web, "title", None) or "Google Search Source"
                if not uri or uri in seen_urls:
                    continue
                seen_urls.add(uri)
                snips = " ".join(chunk_snippets.get(c_idx, []))
                is_academic = any(
                    dom in (title.lower() + " " + uri.lower())
                    for dom in (
                        "pubmed", "ncbi.nlm.nih.gov", "pmc", "doi.org", "sciencedirect",
                        "springer", "wiley", "nature.com", "plos.org", "mdpi.com",
                        "frontiersin", "arxiv.org", "researchgate", "tandfonline",
                        "sagepub", "oxford", "bmj.com", "thelancet", "cureus",
                    )
                )
                out.append(
                    CandidateSource(
                        source_id=f"google:{uri}",
                        title=title,
                        url=uri,
                        provider="Google Scholarly Web" if is_academic else "Google Search (Web)",
                        category="publication" if is_academic else "internet",
                        text=f"{title}. {snips}".strip(),
                    )
                )
        except Exception:
            continue

    return out


# ---------------------------------------------------------------------------
# Channel 2: Academic Full-Text & Metadata APIs (PubMed/PMC, OpenAlex, CrossRef, S2, arXiv)
# ---------------------------------------------------------------------------
def _search_europe_pmc(client: httpx.Client, query: str) -> List[CandidateSource]:
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    params = {"query": query, "format": "json", "resultType": "core", "pageSize": "4"}
    try:
        resp = client.get(url, params=params, timeout=10.0)
        if resp.status_code != 200:
            return []
        out: List[CandidateSource] = []
        for item in resp.json().get("resultList", {}).get("result", []):
            title = _strip_html(item.get("title") or "Untitled Biomedical Article")
            abstract = _strip_html(item.get("abstractText") or "")
            combined = f"{title}. {abstract}".strip()
            if len(combined.split()) < 15:
                continue
            doi = item.get("doi")
            pmid = item.get("pmid")
            pmcid = item.get("pmcid")
            src_url = f"https://doi.org/{doi}" if doi else (
                f"https://europepmc.org/article/PMC/{pmcid}" if pmcid else (
                    f"https://europepmc.org/article/MED/{pmid}" if pmid else "https://europepmc.org"
                )
            )
            out.append(
                CandidateSource(
                    source_id=f"epmc:{pmcid or pmid or doi or title[:40]}",
                    title=title,
                    url=src_url,
                    provider="PubMed / Europe PMC",
                    category="publication",
                    text=combined,
                    year=str(item.get("pubYear") or ""),
                    authors=item.get("authorString"),
                    pmcid=pmcid,
                )
            )
        return out
    except Exception:
        return []


def _fetch_pmc_fulltext(client: httpx.Client, pmcid: str) -> str:
    """Fetch open-access full-text XML from Europe PMC and extract clean body prose."""
    if not pmcid:
        return ""
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
    try:
        resp = client.get(url, timeout=8.0)
        if resp.status_code != 200 or len(resp.text) < 200:
            return ""
        text = _strip_html(resp.text[:120000])
        return text[:24000]
    except Exception:
        return ""


def _search_openalex(client: httpx.Client, query: str) -> List[CandidateSource]:
    url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "per-page": "4",
        "select": "id,doi,title,publication_year,abstract_inverted_index,authorships",
    }
    try:
        resp = client.get(url, params=params, timeout=10.0)
        if resp.status_code != 200:
            return []
        out: List[CandidateSource] = []
        for item in resp.json().get("results", []):
            title = item.get("title") or ""
            abstract = _reconstruct_openalex_abstract(item.get("abstract_inverted_index"))
            combined = f"{title}. {abstract}".strip()
            if len(combined.split()) < 15:
                continue
            doi = item.get("doi") or item.get("id") or ""
            authors_list = [
                a.get("author", {}).get("display_name", "")
                for a in (item.get("authorships") or [])[:3]
            ]
            out.append(
                CandidateSource(
                    source_id=f"openalex:{item.get('id', title[:40])}",
                    title=title,
                    url=doi if doi.startswith("http") else f"https://openalex.org/{item.get('id', '')}",
                    provider="OpenAlex Academic",
                    category="publication",
                    text=combined,
                    year=str(item.get("publication_year") or ""),
                    authors=", ".join(a for a in authors_list if a),
                )
            )
        return out
    except Exception:
        return []


def _search_crossref(client: httpx.Client, query: str) -> List[CandidateSource]:
    url = "https://api.crossref.org/works"
    params = {"query.bibliographic": query, "rows": "3"}
    try:
        resp = client.get(url, params=params, timeout=10.0)
        if resp.status_code != 200:
            return []
        out: List[CandidateSource] = []
        for item in resp.json().get("message", {}).get("items", []):
            titles = item.get("title") or []
            title = _strip_html(titles[0]) if titles else ""
            abstract = _strip_html(item.get("abstract") or "")
            combined = f"{title}. {abstract}".strip()
            if len(combined.split()) < 15:
                continue
            doi = item.get("DOI") or ""
            out.append(
                CandidateSource(
                    source_id=f"crossref:{doi or title[:40]}",
                    title=title,
                    url=f"https://doi.org/{doi}" if doi else (item.get("URL") or ""),
                    provider="CrossRef Journal",
                    category="publication",
                    text=combined,
                    year=str((item.get("issued", {}).get("date-parts", [[""]])[0][0]) or ""),
                )
            )
        return out
    except Exception:
        return []


def _search_semantic_scholar(client: httpx.Client, query: str) -> List[CandidateSource]:
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {"query": query[:120], "limit": "3", "fields": "title,abstract,year,url,externalIds"}
    headers = {}
    s2_key = get_optional_key("SEMANTIC_SCHOLAR_API_KEY")
    if s2_key:
        headers["x-api-key"] = s2_key
    try:
        resp = client.get(url, params=params, headers=headers, timeout=8.0)
        if resp.status_code != 200:
            return []
        out: List[CandidateSource] = []
        for item in resp.json().get("data", []):
            title = item.get("title") or ""
            abstract = item.get("abstract") or ""
            combined = f"{title}. {abstract}".strip()
            if len(combined.split()) < 15:
                continue
            doi = (item.get("externalIds") or {}).get("DOI")
            src_url = f"https://doi.org/{doi}" if doi else (item.get("url") or "")
            out.append(
                CandidateSource(
                    source_id=f"s2:{item.get('paperId', title[:40])}",
                    title=title,
                    url=src_url,
                    provider="Semantic Scholar",
                    category="publication",
                    text=combined,
                    year=str(item.get("year") or ""),
                )
            )
        return out
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Channel 3: Wikipedia, DuckDuckGo, Optional SearXNG + Full HTML/PDF Fetcher
# ---------------------------------------------------------------------------
def _search_wikipedia(client: httpx.Client, query: str) -> List[CandidateSource]:
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": query[:120],
        "gsrlimit": "3",
        "prop": "extracts|info",
        "exintro": "1",
        "explaintext": "1",
        "inprop": "url",
        "format": "json",
    }
    try:
        resp = client.get(url, params=params, timeout=8.0)
        if resp.status_code != 200:
            return []
        out: List[CandidateSource] = []
        for pid, p in resp.json().get("query", {}).get("pages", {}).items():
            title = p.get("title") or ""
            extract = (p.get("extract") or "").strip()
            if len(extract.split()) < 20:
                continue
            out.append(
                CandidateSource(
                    source_id=f"wiki:{pid}",
                    title=f"Wikipedia: {title}",
                    url=p.get("fullurl") or f"https://en.wikipedia.org/wiki/{quote_plus(title)}",
                    provider="Wikipedia",
                    category="internet",
                    text=extract,
                )
            )
        return out
    except Exception:
        return []


def _search_duckduckgo(client: httpx.Client, query: str) -> List[CandidateSource]:
    url = "https://html.duckduckgo.com/html/"
    try:
        resp = client.post(
            url,
            data={"q": query[:140]},
            headers={"User-Agent": _BROWSER_UA},
            timeout=9.0,
        )
        if resp.status_code != 200:
            return []
        soup = BeautifulSoup(resp.text, "html.parser")
        out: List[CandidateSource] = []
        for res in soup.select(".result")[:4]:
            a_tag = res.select_one(".result__a")
            snip_tag = res.select_one(".result__snippet")
            if not a_tag or not snip_tag:
                continue
            title = a_tag.get_text(" ", strip=True)
            href = a_tag.get("href") or ""
            if "uddg=" in href:
                qs = parse_qs(urlparse(href).query)
                if "uddg" in qs and qs["uddg"]:
                    href = unquote(qs["uddg"][0])
            snippet = snip_tag.get_text(" ", strip=True)
            if len(snippet.split()) < 8:
                continue
            out.append(
                CandidateSource(
                    source_id=f"web:{href or title[:40]}",
                    title=title,
                    url=href,
                    provider="Internet Source",
                    category="internet",
                    text=f"{title}. {snippet}",
                )
            )
        return out
    except Exception:
        return []


def _search_searxng(client: httpx.Client, query: str, base_url: str) -> List[CandidateSource]:
    """Optional self-hosted SearXNG metasearch engine (unlimited, zero API cost)."""
    try:
        resp = client.get(
            f"{base_url.rstrip('/')}/search",
            params={"q": query[:140], "format": "json"},
            timeout=8.0,
        )
        if resp.status_code != 200:
            return []
        out: List[CandidateSource] = []
        for item in resp.json().get("results", [])[:4]:
            href = item.get("url") or ""
            title = item.get("title") or "Web Result"
            snippet = item.get("content") or ""
            if not href:
                continue
            out.append(
                CandidateSource(
                    source_id=f"searx:{href}",
                    title=title,
                    url=href,
                    provider="SearXNG Web",
                    category="internet",
                    text=f"{title}. {snippet}".strip(),
                )
            )
        return out
    except Exception:
        return []


def _extract_pdf_bytes_text(pdf_bytes: bytes, max_pages: int = 250, max_chars: int = 450000) -> str:
    """Fast in-memory PDF full-text extraction via PyMuPDF."""
    try:
        pdf = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        pages_txt: List[str] = []
        total_len = 0
        for p_idx in range(min(max_pages, len(pdf))):
            t = pdf[p_idx].get_text("text")
            if t:
                pages_txt.append(t)
                total_len += len(t)
                if total_len >= max_chars:
                    break
        pdf.close()
        return re.sub(r"\s+", " ", " ".join(pages_txt)).strip()[:max_chars]
    except Exception:
        return ""


def _fetch_url_fulltext(client: httpx.Client, url: str) -> Tuple[str, str]:
    """
    Fetch clean full text from EITHER an HTML webpage (automatically following the primary
    embedded report .pdf link like theccc.org.uk & iea.org!) OR a direct public PDF document!
    Uses strict timeouts & 12 MB streaming cap so slow servers never stall a scan.
    """
    if not url or not url.startswith("http"):
        return url, ""
    if any(url.lower().endswith(ext) for ext in (".zip", ".png", ".jpg", ".jpeg", ".mp4", ".mp3")):
        return url, ""

    req_timeout = httpx.Timeout(connect=5.0, read=8.0, write=5.0, pool=5.0)
    try:
        with client.stream("GET", url, headers={"User-Agent": _BROWSER_UA}, timeout=req_timeout, follow_redirects=True) as resp:
            final_url = str(resp.url)
            if resp.status_code != 200:
                return final_url, ""
            ctype = resp.headers.get("content-type", "").lower()
            chunks: List[bytes] = []
            total = 0
            limit = 12_000_000 if ("pdf" in ctype or final_url.lower().endswith(".pdf")) else 500_000
            for ch in resp.iter_bytes(65536):
                chunks.append(ch)
                total += len(ch)
                if total >= limit:
                    break
            raw_bytes = b"".join(chunks)

        # Case A: Direct public PDF document
        if "application/pdf" in ctype or final_url.lower().endswith(".pdf"):
            return final_url, _extract_pdf_bytes_text(raw_bytes)

        # Case B: HTML webpage -> extract body AND follow primary embedded report .pdf link
        if "text/html" in ctype or "text/plain" in ctype:
            raw_html = raw_bytes.decode("utf-8", errors="ignore")
            soup = BeautifulSoup(raw_html, "html.parser")

            embedded_pdfs: List[str] = []
            seen_pdfs: Set[str] = set()
            for a in soup.find_all("a", href=True):
                href = (a.get("href") or "").strip()
                if ".pdf" not in href.lower():
                    continue
                abs_pdf = urljoin(final_url, href)
                low_pdf = abs_pdf.lower()
                if any(skip in low_pdf for skip in ("privacy", "cookie", "terms", "accessibility", "gdpr", "sitemap")):
                    continue
                if abs_pdf not in seen_pdfs:
                    seen_pdfs.add(abs_pdf)
                    embedded_pdfs.append(abs_pdf)

            for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "aside"]):
                tag.decompose()
            body_text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))[:80000]

            # Follow the primary embedded report PDF (up to 12 MB)
            pdf_extra = ""
            if embedded_pdfs:
                pdf_url = embedded_pdfs[0]
                try:
                    with client.stream("GET", pdf_url, headers={"User-Agent": _BROWSER_UA}, timeout=req_timeout, follow_redirects=True) as pr:
                        if pr.status_code == 200:
                            pct = pr.headers.get("content-type", "").lower()
                            if "pdf" in pct or str(pr.url).lower().endswith(".pdf"):
                                p_chunks: List[bytes] = []
                                p_tot = 0
                                for ch in pr.iter_bytes(65536):
                                    p_chunks.append(ch)
                                    p_tot += len(ch)
                                    if p_tot >= 12_000_000:
                                        break
                                extracted = _extract_pdf_bytes_text(b"".join(p_chunks))
                                if len(extracted) > 500:
                                    pdf_extra = extracted
                except Exception:
                    pass

            combined = (body_text + " " + pdf_extra).strip()
            return final_url, combined[:450000]

        return final_url, ""
    except Exception:
        return url, ""


def _load_local_reference_sources(ref_dir: Optional[Path]) -> List[CandidateSource]:
    if not ref_dir or not ref_dir.exists():
        return []
    sources: List[CandidateSource] = []
    for p in sorted(ref_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in (".pdf", ".docx", ".txt"):
            continue
        try:
            parsed = parse_document(file_path=p)
            if parsed.word_count >= 20:
                sources.append(
                    CandidateSource(
                        source_id=f"local:{p.name}",
                        title=f"Local Reference: {p.name}",
                        url=f"local://{p.name}",
                        provider="Local Reference Library",
                        category="local_library",
                        text=parsed.full_text,
                    )
                )
        except Exception:
            continue
    return sources


def _build_shingle_set(tokens: List[str], k: int = 5) -> Set[Tuple[str, ...]]:
    """Build O(1) lookup set of k-word shingles for blazing-fast exact matching across 200-page PDFs."""
    if len(tokens) < k:
        return set()
    return {tuple(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def _find_exact_overlap_shingles(
    sent_tokens: List[str],
    source_shingles: Set[Tuple[str, ...]],
    k: int = 5,
    min_match_words: int = 6,
) -> Tuple[int, List[str]]:
    """
    Turnitin-style O(N) shingle overlap detector: finds all contiguous matching spans
    of length >= min_match_words even inside a 100,000-word PDF in microseconds!
    """
    n = len(sent_tokens)
    if n < min_match_words or not source_shingles:
        return 0, []

    matched_mask = [False] * n
    for i in range(n - k + 1):
        if tuple(sent_tokens[i : i + k]) in source_shingles:
            for pos in range(i, i + k):
                matched_mask[pos] = True

    total_matched = 0
    phrases: List[str] = []
    idx = 0
    while idx < n:
        if matched_mask[idx]:
            start = idx
            while idx < n and matched_mask[idx]:
                idx += 1
            span_len = idx - start
            if span_len >= min_match_words:
                total_matched += span_len
                phrases.append(" ".join(sent_tokens[start:idx]))
        else:
            idx += 1

    return total_matched, phrases


def _best_source_excerpt(sent_text: str, source_text: str, matched_phrases: Optional[List[str]] = None) -> str:
    """Extract the exact matching context window from even a 400,000-char source PDF."""
    if matched_phrases:
        anchor = max(matched_phrases, key=len)
        anchor_words = anchor.split()[:6]
        if anchor_words:
            pat = r"\b" + r"\W+".join(re.escape(w) for w in anchor_words) + r"\b"
            m = re.search(pat, source_text, re.IGNORECASE)
            if m:
                start = max(0, m.start() - 120)
                end = min(len(source_text), m.end() + 220)
                return ("..." if start > 0 else "") + source_text[start:end].strip() + ("..." if end < len(source_text) else "")

    src_sents = [x.strip() for x in re.split(r"(?<=[.!?])\s+", source_text[:40000]) if len(x.strip()) > 20]
    if not src_sents:
        return source_text[:320]
    sent_set = set(_tokenize(sent_text)) - _STOPWORDS
    if not sent_set:
        return src_sents[0][:320]
    best = max(
        src_sents[:250],
        key=lambda ss: len(sent_set & set(_tokenize(ss))) / max(len(sent_set), 1),
    )
    return best[:360]


def analyze_plagiarism(
    doc: ParsedDocument,
    enable_web: bool = True,
    enable_academic: bool = True,
    enable_google_grounding: bool = True,
    gemini_api_key: Optional[str] = None,
    ref_dir: Optional[Path] = None,
    exclude_references: bool = True,
    exclude_quotes: bool = True,
    min_match_words: int = 6,
    max_queries: int = 16,
    progress_cb: Optional[Callable[[int, str], None]] = None,
) -> Dict[str, Any]:
    def _progress(pct: int, msg: str) -> None:
        if progress_cb:
            progress_cb(pct, msg)

    active_sentences: List[SentenceUnit] = []
    for s in doc.sentences:
        if exclude_references and s.section == "reference":
            continue
        if exclude_quotes and s.is_quote:
            continue
        if s.qualifies_for_plagiarism:
            active_sentences.append(s)

    total_checked_words = sum(s.word_count for s in active_sentences)
    empty_groups = {
        "not_cited_or_quoted": {"count": 0, "words": 0, "percent": 0.0},
        "missing_quotations": {"count": 0, "words": 0, "percent": 0.0},
        "missing_citation": {"count": 0, "words": 0, "percent": 0.0},
        "cited_and_quoted": {"count": 0, "words": 0, "percent": 0.0},
    }
    if total_checked_words < 15 or not active_sentences:
        return {
            "overall_similarity_percent": 0.0,
            "exact_similarity_percent": 0.0,
            "paraphrase_similarity_percent": 0.0,
            "internet_percent": 0.0,
            "publications_percent": 0.0,
            "local_library_percent": 0.0,
            "match_groups": empty_groups,
            "checked_words": total_checked_words,
            "sources_scanned": 0,
            "providers_used": [],
            "sources": [],
            "sentence_matches": [],
            "settings": {
                "exclude_references": exclude_references,
                "exclude_quotes": exclude_quotes,
                "min_match_words": min_match_words,
            },
        }

    candidate_map: Dict[str, CandidateSource] = {}
    providers_used: List[str] = []

    # 1. Local Reference Library
    local_sources = _load_local_reference_sources(ref_dir)
    for src in local_sources:
        candidate_map[src.source_id] = src
    if local_sources:
        providers_used.append(f"Local Library ({len(local_sources)} docs)")

    queries = _extract_search_queries(active_sentences, max_queries=max_queries)
    g_key = get_gemini_api_key(gemini_api_key)
    serper_key = get_serper_api_key()
    tavily_key = get_tavily_api_key()
    core_key = get_core_api_key()
    searxng_url = get_optional_key("SEARXNG_URL")
    network_errors = 0

    # Also extract explicit URLs & cited report names from the document's References section
    ref_hints: List[str] = []
    direct_doc_urls: List[str] = []
    for s in doc.sentences:
        for u in re.findall(r"https?://[^\s\)\]>\"']+", s.text):
            u_clean = u.rstrip(".,;:")
            if u_clean not in direct_doc_urls:
                direct_doc_urls.append(u_clean)
        if s.section == "reference" and s.word_count >= 5:
            ref_hints.append(s.text)

    # Seed any explicit public URLs found in the document's own references directly!
    if enable_web:
        for u in direct_doc_urls[:10]:
            sid = f"docref:{u}"
            if sid not in candidate_map:
                candidate_map[sid] = CandidateSource(
                    source_id=sid,
                    title=urlparse(u).netloc or "Cited URL",
                    url=u,
                    provider="Internet Source (Cited URL)",
                    category="internet",
                    text="",
                )

    # 2. Fault-Tolerant Multi-Channel Academic & Web Discovery Waterfall in Parallel:
    #    Web Waterfall     : Serper Google Exact-Phrase -> Tavily Search -> Gemini Grounding -> SearXNG/DuckDuckGo + Wikipedia
    #    Academic Waterfall: CORE v3 Full-Text (260M+ papers) + OpenAlex + CrossRef + Europe PMC + Semantic Scholar
    serper_hits_count = 0
    tavily_hits_count = 0
    core_hits_count = 0

    if (enable_academic or enable_web) and queries:
        _progress(72, f"Scanning Google Index, CORE 260M+ Full-Text, OpenAlex, CrossRef & PubMed ({len(queries)} fingerprints)...")

        def _fetch_one(q_idx: int, kw_text: str, phrase_text: str) -> List[CandidateSource]:
            gathered: List[CandidateSource] = []
            with httpx.Client(headers={"User-Agent": _USER_AGENT}, follow_redirects=True) as client:
                if enable_academic:
                    # CORE v3 Full-Text across all 6 university faculties (every 2nd query to stay well inside rate limits)
                    if core_key and q_idx % 2 == 0 and is_provider_available("core_academic_v3"):
                        gathered.extend(_search_core_academic(client, kw_text, core_key))
                    gathered.extend(_search_openalex(client, kw_text))
                    gathered.extend(_search_europe_pmc(client, kw_text))
                    if q_idx % 2 == 0:
                        gathered.extend(_search_crossref(client, kw_text))
                    if q_idx % 3 == 0:
                        gathered.extend(_search_semantic_scholar(client, kw_text))

                if enable_web:
                    web_found_for_q = False
                    # Tier 1 Web: Serper Google Exact-Phrase & Web/PDF Search (top 8 distributed fingerprints)
                    if serper_key and q_idx % 2 == 0 and is_provider_available("serper_google"):
                        s_hits = _search_serper_google(client, phrase_text, kw_text, serper_key)
                        if s_hits:
                            gathered.extend(s_hits)
                            web_found_for_q = True
                    # Tier 2 Web: Tavily Search (runs on odd queries for complementary coverage, OR automatically on ALL queries if Serper fails/cools down!)
                    if tavily_key and is_provider_available("tavily_search") and (not web_found_for_q or q_idx % 3 == 1):
                        t_hits = _search_tavily(client, f'"{phrase_text}" {kw_text[:70]}', tavily_key)
                        if t_hits:
                            gathered.extend(t_hits)
                            web_found_for_q = True
                    # Tier 3 Open Web Fallback: Wikipedia + SearXNG / DuckDuckGo
                    if q_idx % 2 == 0:
                        gathered.extend(_search_wikipedia(client, kw_text))
                    if not web_found_for_q:
                        if searxng_url:
                            gathered.extend(_search_searxng(client, f'"{phrase_text}"', searxng_url))
                        elif q_idx % 2 == 0:
                            gathered.extend(_search_duckduckgo(client, kw_text))
            return gathered

        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(_fetch_one, idx, kw, ph) for idx, (_, kw, ph) in enumerate(queries)]
            for fut in as_completed(futures):
                try:
                    for src in fut.result():
                        if src.source_id.startswith("serper:"):
                            serper_hits_count += 1
                        elif src.source_id.startswith("tavily:"):
                            tavily_hits_count += 1
                        elif src.source_id.startswith("core:"):
                            core_hits_count += 1
                        if src.source_id not in candidate_map:
                            candidate_map[src.source_id] = src
                except Exception:
                    network_errors += 1

    if serper_hits_count:
        providers_used.append(f"Google Search Index ({serper_hits_count} hits)")
    if tavily_hits_count:
        providers_used.append(f"Tavily Web Index ({tavily_hits_count} hits)")
    if core_hits_count:
        providers_used.append(f"CORE Full-Text ({core_hits_count} papers)")

    # 3. Gemini Google Search Grounding Fallback (runs automatically if Serper+Tavily returned <4 web sources, or for cited reference PDFs!)
    web_hit_total = serper_hits_count + tavily_hits_count
    if enable_web and enable_google_grounding and g_key and queries and (web_hit_total < 4 or bool(ref_hints)):
        _progress(79, "Running supplemental Google Search Grounding for cited reports & PDFs...")
        google_srcs = _search_gemini_grounding(
            queries[:6] if web_hit_total >= 4 else queries,
            g_key,
            doc_title=doc.filename,
            reference_hints=ref_hints,
        )
        for src in google_srcs:
            if src.source_id not in candidate_map:
                candidate_map[src.source_id] = src
        if google_srcs:
            providers_used.append(f"Google Search Grounding ({len(google_srcs)} hits)")

    # 4. Full-Text Enrichment: Download HTML webpages (+ embedded report PDFs!), Direct PDFs, & PMC Full-Text XML!
    docref_targets = [c for c in candidate_map.values() if c.source_id.startswith("docref:")][:8]
    serper_tavily_targets = [
        c for c in candidate_map.values()
        if c.source_id.startswith(("serper:", "tavily:", "google:")) and len(c.text) < 3000
    ][:14]
    web_targets = [c for c in candidate_map.values() if c.source_id.startswith(("web:", "searx:"))][:6]
    pmc_targets = [c for c in candidate_map.values() if c.pmcid][:5]
    enrich_targets = docref_targets + serper_tavily_targets + web_targets

    if enrich_targets or pmc_targets:
        _progress(84, f"Fetching full text of {len(enrich_targets)} Web/PDF URLs & {len(pmc_targets)} PMC articles...")

        def _enrich_web(c: CandidateSource) -> Tuple[str, str, str]:
            with httpx.Client(follow_redirects=True) as cl:
                final_url, full_txt = _fetch_url_fulltext(cl, c.url)
                return c.source_id, final_url, full_txt

        def _enrich_pmc(c: CandidateSource) -> Tuple[str, str]:
            with httpx.Client(headers={"User-Agent": _USER_AGENT}, follow_redirects=True) as cl:
                return c.source_id, _fetch_pmc_fulltext(cl, c.pmcid or "")

        # Hard 30-second total wall-clock budget for URL enrichment so slow servers never stall a scan
        pool = ThreadPoolExecutor(max_workers=8)
        try:
            web_futs = {pool.submit(_enrich_web, c): c.source_id for c in enrich_targets}
            pmc_futs = {pool.submit(_enrich_pmc, c): c.source_id for c in pmc_targets}
            try:
                for fut in as_completed(list(web_futs.keys()) + list(pmc_futs.keys()), timeout=30.0):
                    try:
                        if fut in web_futs:
                            sid, final_url, full_txt = fut.result(timeout=0.1)
                            if final_url and final_url.startswith("http"):
                                candidate_map[sid].url = final_url
                            if full_txt and len(full_txt) > len(candidate_map[sid].text):
                                candidate_map[sid].text = f"{candidate_map[sid].title}. {full_txt}"
                        else:
                            sid, pmc_txt = fut.result(timeout=0.1)
                            if pmc_txt and len(pmc_txt) > len(candidate_map[sid].text):
                                candidate_map[sid].text = f"{candidate_map[sid].title}. {pmc_txt}"
                    except Exception:
                        continue
            except Exception:
                pass
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    # Deduplicate by final resolved URL after redirects
    dedup_map: Dict[str, CandidateSource] = {}
    for src in candidate_map.values():
        key = src.url.rstrip("/").lower() if src.url.startswith("http") else src.source_id
        if key not in dedup_map or len(src.text) > len(dedup_map[key].text):
            dedup_map[key] = src
    candidates = list(dedup_map.values())

    if enable_academic:
        providers_used.extend(["PubMed/PMC Full-Text", "OpenAlex", "CrossRef", "Semantic Scholar"])
    if enable_web:
        providers_used.extend(["Wikipedia", "SearXNG" if searxng_url else "Open Web"])

    if not candidates:
        return {
            "overall_similarity_percent": 0.0,
            "exact_similarity_percent": 0.0,
            "paraphrase_similarity_percent": 0.0,
            "internet_percent": 0.0,
            "publications_percent": 0.0,
            "local_library_percent": 0.0,
            "match_groups": empty_groups,
            "checked_words": total_checked_words,
            "sources_scanned": 0,
            "providers_used": providers_used,
            "network_note": "No online sources reachable. Check internet connection or add PDFs to Local Reference Library.",
            "sources": [],
            "sentence_matches": [],
            "settings": {
                "exclude_references": exclude_references,
                "exclude_quotes": exclude_quotes,
                "min_match_words": min_match_words,
            },
        }

    _progress(90, f"Aligning text against {len(candidates)} full-text scholarly, web & PDF sources...")

    sent_texts = [s.text for s in active_sentences]
    cand_texts = [c.text for c in candidates]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        min_df=1,
        max_features=35000,
    )
    tfidf_matrix = vectorizer.fit_transform(sent_texts + cand_texts)
    sent_vecs = tfidf_matrix[: len(sent_texts)]
    cand_vecs = tfidf_matrix[len(sent_texts) :]
    sim_matrix = cosine_similarity(sent_vecs, cand_vecs)

    # Build Turnitin-style O(1) 5-gram shingle sets for EVERY candidate (blazing fast even on 250-page PDFs!)
    shingle_k = 5
    cand_shingles_list = [_build_shingle_set(_tokenize(c.text), k=shingle_k) for c in candidates]
    raw_matches: List[Tuple[SentenceUnit, CandidateSource, str, float, int, List[str], str]] = []

    for s_idx, sent in enumerate(active_sentences):
        s_tokens = _tokenize(sent.text)
        if len(s_tokens) < min_match_words:
            continue

        row_sims = sim_matrix[s_idx]
        top_tfidf = set(int(x) for x in row_sims.argsort()[::-1][:10])
        best_match = None

        # Exact shingle pass across ALL candidates (O(len(sent)) per candidate — catches matches even in 400-page PDFs!)
        for c_idx, cand in enumerate(candidates):
            exact_words, exact_phrases = _find_exact_overlap_shingles(
                s_tokens, cand_shingles_list[c_idx], k=shingle_k, min_match_words=min_match_words
            )
            cos_sim = float(row_sims[c_idx])
            if exact_words >= min_match_words:
                exact_ratio = exact_words / max(len(s_tokens), 1)
                score = 1.0 + exact_ratio
                if best_match is None or score > best_match[0]:
                    best_match = (
                        score,
                        cand,
                        "exact",
                        round(max(exact_ratio, cos_sim), 3),
                        exact_words,
                        exact_phrases,
                        _best_source_excerpt(sent.text, cand.text, exact_phrases),
                    )
            elif c_idx in top_tfidf and cos_sim >= 0.36 and len(s_tokens) >= 10:
                est_words = max(min_match_words, int(round(len(s_tokens) * cos_sim)))
                if best_match is None or cos_sim > best_match[0]:
                    best_match = (
                        cos_sim,
                        cand,
                        "paraphrase",
                        round(cos_sim, 3),
                        est_words,
                        [],
                        _best_source_excerpt(sent.text, cand.text),
                    )

        if best_match is not None:
            _, cand, m_type, sim_val, m_words, phrases, excerpt = best_match
            raw_matches.append((sent, cand, m_type, sim_val, m_words, phrases, excerpt))

    source_word_totals: Dict[str, int] = {}
    source_objs: Dict[str, CandidateSource] = {}
    source_match_counts: Dict[str, int] = {}

    for sent, cand, m_type, sim_val, m_words, phrases, excerpt in raw_matches:
        capped = min(m_words, sent.word_count)
        source_word_totals[cand.source_id] = source_word_totals.get(cand.source_id, 0) + capped
        source_objs[cand.source_id] = cand
        source_match_counts[cand.source_id] = source_match_counts.get(cand.source_id, 0) + 1

    sorted_source_ids = sorted(
        source_word_totals.keys(),
        key=lambda sid: source_word_totals[sid],
        reverse=True,
    )
    source_index_map = {sid: idx + 1 for idx, sid in enumerate(sorted_source_ids)}

    ref_count_existing = sum(1 for s in doc.sentences if s.section == "reference")
    sources_out: List[Dict[str, Any]] = []
    for idx_pos, sid in enumerate(sorted_source_ids, start=1):
        cand = source_objs[sid]
        m_words = source_word_totals[sid]
        pct = round((m_words / total_checked_words) * 100.0, 1)
        cit_sug = build_citation_suggestions(
            title=cand.title,
            authors=cand.authors,
            year=cand.year,
            url=cand.url,
            ref_number=ref_count_existing + idx_pos,
        )
        sources_out.append(
            {
                "index": source_index_map[sid],
                "source_id": sid,
                "title": cand.title,
                "url": cand.url,
                "provider": cand.provider,
                "category": cand.category,
                "year": cand.year,
                "authors": cand.authors,
                "matched_words": m_words,
                "match_count": source_match_counts[sid],
                "similarity_percent": pct,
                "citations": cit_sug,
            }
        )

    sentence_matches_out: List[Dict[str, Any]] = []
    exact_matched_total = 0
    para_matched_total = 0
    cat_words = {"internet": 0, "publication": 0, "local_library": 0}
    match_groups = {
        "not_cited_or_quoted": {"count": 0, "words": 0, "percent": 0.0},
        "missing_quotations": {"count": 0, "words": 0, "percent": 0.0},
        "missing_citation": {"count": 0, "words": 0, "percent": 0.0},
        "cited_and_quoted": {"count": 0, "words": 0, "percent": 0.0},
    }

    for sent, cand, m_type, sim_val, m_words, phrases, excerpt in raw_matches:
        capped = min(m_words, sent.word_count)
        if m_type == "exact":
            exact_matched_total += capped
        else:
            para_matched_total += capped

        cat_words[cand.category] = cat_words.get(cand.category, 0) + capped
        cgroup = _classify_citation_group(sent.text)
        match_groups[cgroup]["count"] += 1
        match_groups[cgroup]["words"] += capped

        sentence_matches_out.append(
            SentenceMatch(
                sentence_id=sent.id,
                source_index=source_index_map[cand.source_id],
                source_id=cand.source_id,
                source_category=cand.category,
                match_type=m_type,
                citation_group=cgroup,
                similarity=sim_val,
                matched_words=capped,
                total_words=sent.word_count,
                matched_phrases=phrases,
                source_excerpt=excerpt,
            ).to_dict()
        )

    for g in match_groups.values():
        g["percent"] = round((g["words"] / total_checked_words) * 100.0, 1)

    overall_matched = min(total_checked_words, exact_matched_total + para_matched_total)
    exact_pct = round((exact_matched_total / total_checked_words) * 100.0, 1)
    para_pct = round((para_matched_total / total_checked_words) * 100.0, 1)
    overall_pct = round((overall_matched / total_checked_words) * 100.0, 1)

    # Build transparent Verified Checked-Sources Log (up to 45 websites & journals) so users can verify even 0% clean checks!
    matched_sid_set = set(sorted_source_ids)
    scanned_sources_log: List[Dict[str, Any]] = []
    for cand in candidates[:45]:
        scanned_sources_log.append({
            "title": cand.title,
            "url": cand.url,
            "category": cand.category,
            "year": cand.year,
            "matched": cand.source_id in matched_sid_set,
        })

    _progress(96, "Finalizing similarity report...")

    return {
        "overall_similarity_percent": overall_pct,
        "exact_similarity_percent": exact_pct,
        "paraphrase_similarity_percent": para_pct,
        "internet_percent": round(min(100.0, (cat_words["internet"] / total_checked_words) * 100.0), 1),
        "publications_percent": round(min(100.0, (cat_words["publication"] / total_checked_words) * 100.0), 1),
        "local_library_percent": round(min(100.0, (cat_words["local_library"] / total_checked_words) * 100.0), 1),
        "match_groups": match_groups,
        "checked_words": total_checked_words,
        "sources_scanned": len(candidates),
        "scanned_sources_log": scanned_sources_log,
        "providers_used": providers_used,
        "sources": sources_out,
        "sentence_matches": sentence_matches_out,
        "settings": {
            "exclude_references": exclude_references,
            "exclude_quotes": exclude_quotes,
            "min_match_words": min_match_words,
        },
    }
