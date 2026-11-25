"""
Document Parser & Academic Section Extractor.
Supports PDF (PyMuPDF), DOCX (python-docx), and plain text.
Classifies blocks into: cover, heading, prose, list_item, table, reference.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


@dataclass
class SentenceUnit:
    id: int
    text: str
    page: int
    section: str  # cover, heading, prose, list_item, table, reference
    paragraph_id: int
    word_count: int
    char_start: int
    char_end: int
    is_quote: bool = False
    qualifies_for_ai: bool = True
    qualifies_for_plagiarism: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ParsedDocument:
    filename: str
    file_size: int
    page_count: int
    full_text: str
    word_count: int
    char_count: int
    qualifying_ai_words: int
    qualifying_plagiarism_words: int
    sentences: List[SentenceUnit] = field(default_factory=list)
    paragraphs: List[Dict[str, Any]] = field(default_factory=list)
    references_start_index: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "file_size": self.file_size,
            "page_count": self.page_count,
            "word_count": self.word_count,
            "char_count": self.char_count,
            "qualifying_ai_words": self.qualifying_ai_words,
            "qualifying_plagiarism_words": self.qualifying_plagiarism_words,
            "references_start_index": self.references_start_index,
            "metadata": self.metadata,
            "paragraphs": self.paragraphs,
            "sentences": [s.to_dict() for s in self.sentences],
        }


# Regex patterns for academic structure detection
_REF_HEADING_RE = re.compile(
    r"^\s*(?:\d+\.?\s*)?(?:references|bibliography|works\s+cited|literature\s+cited)\s*:?\s*$",
    re.IGNORECASE,
)
_NUMBERED_HEADING_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?|[A-Z]\.)\s+[A-Z][A-Za-z0-9\s,&/\-()]{2,90}$"
)
_ALL_CAPS_HEADING_RE = re.compile(r"^[A-Z0-9\s.,:;?&/\-()]{4,220}$")
_BULLET_RE = re.compile(r"^\s*(?:[\u2022\u25cf\u25cb\u25aa\uf0b7•\-–—*]|\d+[\.\)]|[a-zA-Z][\.\)])\s+")
_COVER_KEYWORDS_RE = re.compile(
    r"\b(?:synopsis|dissertation|thesis|submitted\s+to|name\s+of\s+candidate|"
    r"name\s+of\s+supervisor|rtmc\s+no|roll\s+no|registration\s+no|specialty\s*/\s*faculty|"
    r"duration\s+of\s+study|study\s+setting|college\s+of\s+physicians|university\s+of)\b",
    re.IGNORECASE,
)
_FORMULA_RE = re.compile(r"[=×÷±≤≥∑√]|Z[₁1]|P[₁1]\s*\(")
_CITATION_REF_LINE_RE = re.compile(
    r"^\s*(?:\[\d+\]|\d+\.)\s+[A-Z][a-zA-Z\-]+[\s,]+[A-Z]{1,3}\b"
)


def _clean_text(text: str) -> str:
    """Clean PDF extraction artifacts, soft hyphens, replacement chars, and broken line wraps."""
    if not text:
        return ""
    # Replace PDF soft hyphen / replacement char between letters (e.g., gonadotropin\ufffdreleasing)
    text = re.sub(r"([a-zA-Z])\ufffd([a-zA-Z])", r"\1-\2", text)
    text = text.replace("\ufffd", "-")
    text = text.replace("\u00ad", "")  # soft hyphen
    text = text.replace("\uf0b7", "•")  # Wingdings bullet
    text = text.replace("\xa0", " ")
    # Fix hyphenated line breaks: "word-\nnext" -> "word-next" or "wordnext"
    text = re.sub(r"([a-zA-Z])-\s*\n\s*([a-z])", r"\1-\2", text)
    # Collapse multiple spaces
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _split_sentences(text: str) -> List[str]:
    """
    Academic-aware sentence splitter that preserves abbreviations
    (Dr., Prof., et al., vs., e.g., i.e., Fig., Eq., No., Jan., mg/dL., etc.)
    and decimal numbers (10.7%, 23.0 kg/m²).
    """
    if not text or not text.strip():
        return []

    # Protect common academic abbreviations from triggering sentence splits
    abbrevs = [
        "Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "St.", "vs.", "etc.", "e.g.", "i.e.",
        "al.", "Fig.", "Figs.", "Eq.", "Eqs.", "No.", "Vol.", "pp.", "Ref.", "Refs.",
        "approx.", "min.", "max.", "Jan.", "Feb.", "Mar.", "Apr.", "Aug.", "Sept.",
        "Oct.", "Nov.", "Dec.", "Inc.", "Ltd.", "Dept.", "Univ.", "Hosp.", "Assoc.",
        "Clin.", "Med.", "Surg.", "Endocrinol.", "Gynaecol.", "Obstet.", "Reprod.",
    ]
    protected = text
    placeholders: List[Tuple[str, str]] = []
    for idx, abbr in enumerate(abbrevs):
        token = f"__ABBR{idx}__"
        if abbr in protected:
            protected = protected.replace(abbr, token)
            placeholders.append((token, abbr))

    # Protect numbered list prefixes at start like "1. " or "14. "
    protected = re.sub(r"^(\s*\d+)\.\s+", r"\1__DOT__ ", protected)

    # Split on sentence-ending punctuation followed by whitespace and uppercase/quote/bracket
    raw_parts = re.split(
        r"(?<=[.!?])(?:\s*\[\d+(?:[,\-–]\d+)*\])?\s+(?=[A-Z\"'\(\[\d])",
        protected,
    )

    sentences: List[str] = []
    for part in raw_parts:
        s = part.replace("__DOT__", ".")
        for token, abbr in placeholders:
            s = s.replace(token, abbr)
        s = s.strip()
        if s:
            sentences.append(s)

    return sentences


def _count_words(text: str) -> int:
    """Standard MS Word / WordCounter whitespace word count (exact match: 2,555 on CPSP synopsis)."""
    if not text:
        return 0
    return len([tok for tok in text.split() if tok != "|"])


def _is_quoted_sentence(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) >= 10:
        if (stripped.startswith('"') and stripped.endswith('"')) or (
            stripped.startswith("“") and stripped.endswith("”")
        ):
            return True
    # More than 70% of characters inside quotation marks
    quoted_chars = sum(len(m.group(0)) for m in re.finditer(r'["“][^"”]{10,}["”]', stripped))
    return len(stripped) > 0 and (quoted_chars / len(stripped)) > 0.70


def _extract_pdf_blocks(file_path: Path) -> Tuple[int, List[Tuple[int, str, bool]]]:
    """
    Extract structured text blocks from PDF using PyMuPDF.
    Returns (page_count, list of (page_num, paragraph_text, is_table_hint)).
    """
    import pymupdf

    doc = pymupdf.open(str(file_path))
    page_count = len(doc)
    blocks_out: List[Tuple[int, str, bool]] = []

    for page_idx, page in enumerate(doc, start=1):
        # Detect table bounding boxes if available to tag table content
        table_rects = []
        try:
            tabs = page.find_tables()
            if tabs and tabs.tables:
                table_rects = [pymupdf.Rect(t.bbox) for t in tabs.tables]
        except Exception:
            table_rects = []

        raw_blocks = page.get_text("blocks")
        # Sort top-to-bottom, then left-to-right
        raw_blocks.sort(key=lambda b: (round(b[1], 1), round(b[0], 1)))

        for b in raw_blocks:
            if b[6] != 0:  # skip image blocks
                continue
            rect = pymupdf.Rect(b[0], b[1], b[2], b[3])
            text = _clean_text(b[4])
            if not text:
                continue
            # Skip standalone page numbers (e.g., "1", "2", "Page 1 of 9")
            if re.match(r"^(?:page\s+\d+(?:\s+of\s+\d+)?|\d{1,3})$", text, re.IGNORECASE):
                continue

            is_table = any(rect.intersects(tr) for tr in table_rects)

            # Merge wrapped lines within a paragraph block unless they are distinct bullet items
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            current_buf: List[str] = []
            for ln in lines:
                if _BULLET_RE.match(ln) or _REF_HEADING_RE.match(ln) or _NUMBERED_HEADING_RE.match(ln):
                    if current_buf:
                        blocks_out.append((page_idx, _clean_text(" ".join(current_buf)), is_table))
                        current_buf = []
                    current_buf.append(ln)
                else:
                    current_buf.append(ln)
            if current_buf:
                blocks_out.append((page_idx, _clean_text(" ".join(current_buf)), is_table))

    doc.close()
    return page_count, blocks_out


def _extract_docx_blocks(file_path: Path) -> Tuple[int, List[Tuple[int, str, bool]]]:
    """Extract paragraphs and tables from DOCX."""
    import docx

    doc = docx.Document(str(file_path))
    blocks_out: List[Tuple[int, str, bool]] = []
    words_so_far = 0

    for para in doc.paragraphs:
        text = _clean_text(para.text)
        if not text:
            continue
        page_est = max(1, (words_so_far // 300) + 1)
        blocks_out.append((page_est, text, False))
        words_so_far += _count_words(text)

    for table in doc.tables:
        for row in table.rows:
            seen_cells = set()
            cell_texts = []
            for cell in row.cells:
                ct = _clean_text(cell.text)
                if ct and ct not in seen_cells:
                    seen_cells.add(ct)
                    cell_texts.append(ct)
            row_text = " | ".join(cell_texts)
            if row_text:
                page_est = max(1, (words_so_far // 300) + 1)
                blocks_out.append((page_est, row_text, True))
                words_so_far += _count_words(row_text)

    page_count = max(1, (words_so_far // 300) + 1)
    return page_count, blocks_out


def _extract_txt_blocks(text: str) -> Tuple[int, List[Tuple[int, str, bool]]]:
    """Split raw text into paragraphs."""
    raw_paras = re.split(r"\n\s*\n+", text.replace("\r\n", "\n"))
    blocks_out: List[Tuple[int, str, bool]] = []
    words_so_far = 0
    for rp in raw_paras:
        cleaned = _clean_text(" ".join(ln.strip() for ln in rp.splitlines() if ln.strip()))
        if not cleaned:
            continue
        page_est = max(1, (words_so_far // 300) + 1)
        blocks_out.append((page_est, cleaned, False))
        words_so_far += _count_words(cleaned)
    page_count = max(1, (words_so_far // 300) + 1)
    return page_count, blocks_out


def parse_document(
    file_path: Optional[Path] = None,
    raw_text: Optional[str] = None,
    filename: str = "document.txt",
) -> ParsedDocument:
    """Parse a PDF, DOCX, or raw text into structured academic sections and sentences."""
    file_size = 0
    if file_path is not None:
        file_path = Path(file_path)
        filename = file_path.name
        file_size = file_path.stat().st_size if file_path.exists() else 0
        ext = file_path.suffix.lower()
        if ext == ".pdf":
            page_count, raw_blocks = _extract_pdf_blocks(file_path)
        elif ext == ".docx":
            page_count, raw_blocks = _extract_docx_blocks(file_path)
        else:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            page_count, raw_blocks = _extract_txt_blocks(content)
    else:
        content = raw_text or ""
        file_size = len(content.encode("utf-8"))
        page_count, raw_blocks = _extract_txt_blocks(content)

    # Detect if page 1 is a Cover / Title Page
    page1_texts = [t for p, t, _ in raw_blocks if p == 1]
    page1_joined = " ".join(page1_texts)
    page1_is_cover = (
        page_count >= 2
        and len(page1_joined.split()) < 180
        and bool(_COVER_KEYWORDS_RE.search(page1_joined))
    )

    in_references = False
    references_start_idx: Optional[int] = None
    sentences: List[SentenceUnit] = []
    paragraphs: List[Dict[str, Any]] = []
    full_text_parts: List[str] = []
    cursor = 0
    sent_id = 0

    for para_idx, (page_num, block_text, is_table) in enumerate(raw_blocks):
        wc = _count_words(block_text)

        # Classify section
        if _REF_HEADING_RE.match(block_text):
            in_references = True
            section = "heading"
        elif in_references:
            section = "reference"
        elif page1_is_cover and page_num == 1:
            section = "cover"
        elif is_table:
            section = "table"
        elif wc <= 28 and (_NUMBERED_HEADING_RE.match(block_text) or _ALL_CAPS_HEADING_RE.match(block_text)):
            section = "heading"
        elif _BULLET_RE.match(block_text) and wc < 32:
            section = "list_item"
        elif _CITATION_REF_LINE_RE.match(block_text) and page_num >= max(2, page_count - 2):
            in_references = True
            section = "reference"
        else:
            section = "prose"

        para_start = cursor
        if full_text_parts:
            full_text_parts.append("\n\n")
            cursor += 2

        para_sent_ids: List[int] = []

        if section in ("heading", "cover", "reference"):
            unit_texts = [block_text]
        else:
            unit_texts = _split_sentences(block_text) or [block_text]

        for u_idx, u_text in enumerate(unit_texts):
            if u_idx > 0:
                full_text_parts.append(" ")
                cursor += 1
            s_start = cursor
            full_text_parts.append(u_text)
            cursor += len(u_text)
            s_end = cursor

            u_wc = _count_words(u_text)
            is_q = _is_quoted_sentence(u_text)
            is_formula = bool(_FORMULA_RE.search(u_text)) and u_wc < 25

            # Turnitin-like qualification rules:
            # AI detection applies to qualifying prose (including prose inside Word table templates >= 18 words)
            qualifies_ai = (
                (section == "prose" and u_wc >= 8 and not is_formula)
                or (section == "table" and u_wc >= 18 and not is_formula and ("." in u_text or "," in u_text))
            )
            # Plagiarism applies to prose, list items, and substantial table text (excluding cover & references)
            qualifies_plag = (
                section in ("prose", "list_item", "table")
                and u_wc >= 6
                and not is_formula
            )

            if section == "reference" and references_start_idx is None:
                references_start_idx = sent_id

            sentences.append(
                SentenceUnit(
                    id=sent_id,
                    text=u_text,
                    page=page_num,
                    section=section,
                    paragraph_id=para_idx,
                    word_count=u_wc,
                    char_start=s_start,
                    char_end=s_end,
                    is_quote=is_q,
                    qualifies_for_ai=qualifies_ai,
                    qualifies_for_plagiarism=qualifies_plag,
                )
            )
            para_sent_ids.append(sent_id)
            sent_id += 1

        paragraphs.append(
            {
                "id": para_idx,
                "page": page_num,
                "section": section,
                "text": block_text,
                "word_count": wc,
                "sentence_ids": para_sent_ids,
                "char_start": para_start,
                "char_end": cursor,
            }
        )

    full_text = "".join(full_text_parts)
    total_words = sum(s.word_count for s in sentences)
    qual_ai_words = sum(s.word_count for s in sentences if s.qualifies_for_ai)
    qual_plag_words = sum(s.word_count for s in sentences if s.qualifies_for_plagiarism)

    return ParsedDocument(
        filename=filename,
        file_size=file_size,
        page_count=page_count,
        full_text=full_text,
        word_count=total_words,
        char_count=len(full_text),
        qualifying_ai_words=qual_ai_words,
        qualifying_plagiarism_words=qual_plag_words,
        sentences=sentences,
        paragraphs=paragraphs,
        references_start_index=references_start_idx,
        metadata={
            "page1_is_cover": page1_is_cover,
            "has_references": references_start_idx is not None,
            "total_sentences": len(sentences),
            "qualifying_ai_sentences": sum(1 for s in sentences if s.qualifies_for_ai),
        },
    )
