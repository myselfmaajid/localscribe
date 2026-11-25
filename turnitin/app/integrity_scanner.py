"""
Document Integrity & Anti-Cheat Forensics Scanner (Turnitin Integrity Flags Equivalent+).

Detects deliberate manipulation techniques used to bypass AI & Plagiarism checkers:
  1. Hidden / White-on-White Text (PDF & DOCX)
  2. Micro-Font Stuffing (< 3.5 pt invisible text in PDF & DOCX)
  3. Invisible / Zero-Width Unicode Character Stuffing (U+200B, U+200C, U+200D, U+2060, U+FEFF)
  4. Cyrillic / Greek Homoglyph Character Substitution inside English words (e.g. Cyrillic 'а' vs Latin 'a')
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

# Zero-width / invisible characters commonly used to break n-gram plagiarism or AI tokenizers
_ZERO_WIDTH_CHARS = {
    "\u200b": "ZERO WIDTH SPACE",
    "\u200c": "ZERO WIDTH NON-JOINER",
    "\u200d": "ZERO WIDTH JOINER",
    "\u2060": "WORD JOINER",
    "\ufeff": "ZERO WIDTH NO-BREAK SPACE (BOM)",
    "\u180e": "MONGOLIAN VOWEL SEPARATOR",
}

# Cyrillic & Greek lookalike characters commonly swapped into English words to bypass detectors
_CONFUSABLE_MAP = {
    "\u0430": "a (Cyrillic)", "\u0435": "e (Cyrillic)", "\u043e": "o (Cyrillic)",
    "\u0440": "p (Cyrillic)", "\u0441": "c (Cyrillic)", "\u0443": "y (Cyrillic)",
    "\u0445": "x (Cyrillic)", "\u0456": "i (Cyrillic)", "\u0458": "j (Cyrillic)",
    "\u0410": "A (Cyrillic)", "\u0412": "B (Cyrillic)", "\u0415": "E (Cyrillic)",
    "\u041a": "K (Cyrillic)", "\u041c": "M (Cyrillic)", "\u041d": "H (Cyrillic)",
    "\u041e": "O (Cyrillic)", "\u0420": "P (Cyrillic)", "\u0421": "C (Cyrillic)",
    "\u0422": "T (Cyrillic)", "\u0425": "X (Cyrillic)",
    "\u03bf": "o (Greek)",    "\u03b1": "a (Greek)",    "\u03b5": "e (Greek)",
    "\u03c1": "p (Greek)",    "\u0391": "A (Greek)",    "\u0392": "B (Greek)",
    "\u0395": "E (Greek)",    "\u039f": "O (Greek)",
}


def _scan_text_anomalies(raw_text: str) -> List[Dict[str, Any]]:
    flags: List[Dict[str, Any]] = []
    if not raw_text:
        return flags

    # 1. Zero-width invisible characters (ignore leading BOM at index 0)
    body = raw_text[1:] if raw_text.startswith("\ufeff") else raw_text
    zw_count = sum(body.count(ch) for ch in _ZERO_WIDTH_CHARS)
    if zw_count >= 5:
        flags.append(
            {
                "type": "zero_width_characters",
                "severity": "high" if zw_count >= 25 else "medium",
                "title": "Hidden Zero-Width Unicode Characters Detected",
                "count": zw_count,
                "description": (
                    f"Found {zw_count} invisible zero-width characters embedded in the text. "
                    "These characters are invisible to readers but break word matching and AI tokenizers."
                ),
            }
        )

    # 2. Mixed-script homoglyph substitution (Cyrillic/Greek letters inside Latin words)
    homoglyph_words: List[str] = []
    homoglyph_chars = 0
    for word in re.findall(r"\S+", body):
        has_latin = bool(re.search(r"[A-Za-z]", word))
        confusables = [ch for ch in word if ch in _CONFUSABLE_MAP]
        if has_latin and confusables:
            homoglyph_chars += len(confusables)
            if len(homoglyph_words) < 6:
                homoglyph_words.append(word)

    if homoglyph_chars >= 3:
        flags.append(
            {
                "type": "homoglyph_substitution",
                "severity": "high" if homoglyph_chars >= 10 else "medium",
                "title": "Replaced Characters (Cyrillic/Greek Homoglyphs)",
                "count": homoglyph_chars,
                "samples": homoglyph_words,
                "description": (
                    f"Detected {homoglyph_chars} non-Latin lookalike characters mixed inside English words "
                    f"(e.g., {', '.join(homoglyph_words[:4])}). This is a known plagiarism/AI-detector bypass technique."
                ),
            }
        )
    return flags


def _scan_pdf_visual_anomalies(file_path: Path) -> List[Dict[str, Any]]:
    import pymupdf

    flags: List[Dict[str, Any]] = []
    white_chars = 0
    white_samples: List[str] = []
    micro_chars = 0
    micro_samples: List[str] = []
    affected_pages = set()

    try:
        doc = pymupdf.open(str(file_path))
        for page_idx, page in enumerate(doc, start=1):
            pdict = page.get_text("dict")
            for block in pdict.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        txt = (span.get("text") or "").strip()
                        if len(txt) < 2:
                            continue
                        color_int = span.get("color", 0)
                        r = ((color_int >> 16) & 255) / 255.0
                        g = ((color_int >> 8) & 255) / 255.0
                        b = (color_int & 255) / 255.0
                        size = float(span.get("size", 11.0))

                        # Near-white text (often placed on white background to inject hidden words)
                        if r > 0.94 and g > 0.94 and b > 0.94 and len(txt) >= 3:
                            white_chars += len(txt)
                            affected_pages.add(page_idx)
                            if len(white_samples) < 4:
                                white_samples.append(f"p.{page_idx}: '{txt[:40]}'")

                        # Micro-font text (< 3.5 pt)
                        if 0.1 < size < 3.5 and len(re.findall(r"[A-Za-z]", txt)) >= 3:
                            micro_chars += len(txt)
                            affected_pages.add(page_idx)
                            if len(micro_samples) < 4:
                                micro_samples.append(f"p.{page_idx} ({size:.1f}pt): '{txt[:40]}'")
        doc.close()
    except Exception:
        return flags

    if white_chars >= 20:
        flags.append(
            {
                "type": "hidden_white_text",
                "severity": "high" if white_chars >= 80 else "medium",
                "title": "Hidden White-Colored Text Detected",
                "count": white_chars,
                "pages": sorted(affected_pages)[:10],
                "samples": white_samples,
                "description": (
                    f"Detected {white_chars} characters rendered in near-white ink on pages "
                    f"{sorted(affected_pages)[:6]}. Hidden text can distort word counts and AI/similarity scores."
                ),
            }
        )

    if micro_chars >= 20:
        flags.append(
            {
                "type": "micro_font_text",
                "severity": "medium",
                "title": "Micro-Font Hidden Text (< 3.5pt)",
                "count": micro_chars,
                "pages": sorted(affected_pages)[:10],
                "samples": micro_samples,
                "description": (
                    f"Detected {micro_chars} characters in ultra-small font size (< 3.5pt) "
                    "which is unreadable to humans but parsed by automated checkers."
                ),
            }
        )
    return flags


def _scan_docx_visual_anomalies(file_path: Path) -> List[Dict[str, Any]]:
    import docx

    flags: List[Dict[str, Any]] = []
    white_chars = 0
    white_samples: List[str] = []
    micro_chars = 0
    hidden_prop_chars = 0

    try:
        d = docx.Document(str(file_path))
        # Check body paragraphs for white text (table headers often legitimately use white text on dark shading)
        para_entries = [(p, False) for p in d.paragraphs]
        for tbl in d.tables:
            for row in tbl.rows:
                for cell in row.cells:
                    # Detect dark cell shading via XML <w:shd w:fill="...">
                    tc_xml = cell._tc.xml if hasattr(cell, "_tc") else ""
                    m_fill = re.search(r'w:fill="([0-9A-Fa-f]{6})"', tc_xml)
                    has_dark_bg = False
                    if m_fill:
                        hx = m_fill.group(1).upper()
                        if hx not in ("FFFFFF", "FEFEFE", "F8F8F8", "F5F5F5"):
                            has_dark_bg = True
                    for p in cell.paragraphs:
                        para_entries.append((p, has_dark_bg))

        for para, has_dark_bg in para_entries:
            for run in para.runs:
                txt = (run.text or "").strip()
                if len(txt) < 2:
                    continue
                if run.font.hidden:
                    hidden_prop_chars += len(txt)
                if not has_dark_bg and run.font.color and run.font.color.rgb:
                    rgb_hex = str(run.font.color.rgb).upper()
                    if len(rgb_hex) == 6:
                        r = int(rgb_hex[0:2], 16)
                        g = int(rgb_hex[2:4], 16)
                        b = int(rgb_hex[4:6], 16)
                        if r > 240 and g > 240 and b > 240:
                            white_chars += len(txt)
                            if len(white_samples) < 4:
                                white_samples.append(f"'{txt[:40]}'")
                if run.font.size is not None:
                    pt = run.font.size.pt
                    if pt < 3.5 and len(re.findall(r"[A-Za-z]", txt)) >= 3:
                        micro_chars += len(txt)
    except Exception:
        return flags

    if white_chars >= 20:
        flags.append(
            {
                "type": "hidden_white_text",
                "severity": "high" if white_chars >= 80 else "medium",
                "title": "Hidden White-Colored Text Detected (DOCX)",
                "count": white_chars,
                "samples": white_samples,
                "description": f"Found {white_chars} characters formatted in near-white font color inside the Word document.",
            }
        )
    if hidden_prop_chars >= 15:
        flags.append(
            {
                "type": "docx_hidden_property",
                "severity": "high",
                "title": "Hidden Font Attribute Text (DOCX)",
                "count": hidden_prop_chars,
                "description": f"Found {hidden_prop_chars} characters marked with Word's 'Hidden' text attribute.",
            }
        )
    if micro_chars >= 20:
        flags.append(
            {
                "type": "micro_font_text",
                "severity": "medium",
                "title": "Micro-Font Hidden Text (< 3.5pt)",
                "count": micro_chars,
                "description": f"Found {micro_chars} characters formatted below 3.5pt font size.",
            }
        )
    return flags


def scan_document_integrity(
    file_path: Optional[Path] = None,
    full_text: str = "",
) -> Dict[str, Any]:
    """Run all Document Integrity anti-cheat checks and return Turnitin-style Integrity Flags."""
    flags: List[Dict[str, Any]] = []
    flags.extend(_scan_text_anomalies(full_text))

    if file_path is not None and file_path.exists():
        ext = file_path.suffix.lower()
        if ext == ".pdf":
            flags.extend(_scan_pdf_visual_anomalies(file_path))
        elif ext == ".docx":
            flags.extend(_scan_docx_visual_anomalies(file_path))

    high_count = sum(1 for f in flags if f.get("severity") == "high")
    return {
        "flag_count": len(flags),
        "status": "flagged" if flags else "clean",
        "risk_level": "high" if high_count > 0 else ("medium" if flags else "clean"),
        "flags": flags,
    }
