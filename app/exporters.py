"""Export utilities for SRT, VTT, Short-Line Reels/Shorts captions, TXT, DOCX, and JSON."""

from __future__ import annotations

import io
import json
import re
from typing import Any
from docx import Document
from docx.shared import Pt, RGBColor


def format_srt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis == 1000:
        secs += 1
        millis = 0
        if secs == 60:
            minutes += 1
            secs = 0
            if minutes == 60:
                hours += 1
                minutes = 0
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_vtt_time(seconds: float) -> str:
    return format_srt_time(seconds).replace(",", ".")


def format_short_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def build_short_caption_segments(
    segments: list[dict[str, Any]],
    max_words: int = 3,
    max_chars: int = 28,
) -> list[dict[str, Any]]:
    """
    Build tight, punchy captions ideal for YouTube Shorts, Instagram Reels, and TikTok.
    Uses word-level timestamps when available; falls back to proportional splitting.
    """
    max_words = max(1, min(30, int(max_words)))
    max_chars = max(8, min(140, int(max_chars)))

    cues: list[dict[str, Any]] = []

    for seg in segments:
        words = seg.get("words") or []
        if words:
            chunk_words: list[dict[str, Any]] = []
            chunk_len = 0

            for w in words:
                w_text = str(w.get("word", "")).strip()
                if not w_text:
                    continue
                projected_len = chunk_len + (1 if chunk_words else 0) + len(w_text)
                if chunk_words and (len(chunk_words) >= max_words or projected_len > max_chars):
                    cues.append({
                        "start": float(chunk_words[0]["start"]),
                        "end": float(chunk_words[-1]["end"]),
                        "text": " ".join(str(x["word"]).strip() for x in chunk_words),
                    })
                    chunk_words = []
                    chunk_len = 0

                chunk_words.append(w)
                chunk_len += (1 if len(chunk_words) > 1 else 0) + len(w_text)

                # Break cleanly on sentence-ending punctuation
                if w_text.endswith((".", "!", "?", "।", "۔")):
                    cues.append({
                        "start": float(chunk_words[0]["start"]),
                        "end": float(chunk_words[-1]["end"]),
                        "text": " ".join(str(x["word"]).strip() for x in chunk_words),
                    })
                    chunk_words = []
                    chunk_len = 0

            if chunk_words:
                cues.append({
                    "start": float(chunk_words[0]["start"]),
                    "end": float(chunk_words[-1]["end"]),
                    "text": " ".join(str(x["word"]).strip() for x in chunk_words),
                })
        else:
            # Fallback when word-level timestamps are absent (e.g., Gemini mode or edited text)
            text = str(seg.get("text", "")).strip()
            if not text:
                continue
            tokens = text.split()
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start + 1.0))
            total_dur = max(0.2, end - start)

            groups: list[list[str]] = []
            cur: list[str] = []
            cur_len = 0
            for tok in tokens:
                proj = cur_len + (1 if cur else 0) + len(tok)
                if cur and (len(cur) >= max_words or proj > max_chars):
                    groups.append(cur)
                    cur = []
                    cur_len = 0
                cur.append(tok)
                cur_len += (1 if len(cur) > 1 else 0) + len(tok)
            if cur:
                groups.append(cur)

            total_tokens = max(1, len(tokens))
            cursor = start
            for g in groups:
                g_dur = total_dur * (len(g) / total_tokens)
                cues.append({
                    "start": cursor,
                    "end": min(end, cursor + g_dur),
                    "text": " ".join(g),
                })
                cursor += g_dur

    return cues


_ABBREVS = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "vs.",
    "etc.", "e.g.", "i.e.", "u.s.", "u.k.", "a.m.", "p.m.", "no.",
}


def _is_sentence_end(token: str) -> bool:
    t = token.strip()
    if not t:
        return False
    stripped = t.rstrip("\"'”’)]}")
    if not stripped:
        return False
    if stripped[-1] not in ".!?।۔":
        return False
    if stripped.lower() in _ABBREVS:
        return False
    # Ignore single-letter initials like "J." or "A."
    if len(stripped) == 2 and stripped[0].isalpha() and stripped[1] == ".":
        return False
    return True


def build_sentence_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Re-segment transcript into 100% complete, meaningful sentences using exact
    word-level timestamps (never cutting mid-sentence). Ideal for AI video/image
    clip syncing and clean TurboScribe-style SRT/VTT subtitles.
    """
    all_words: list[dict[str, Any]] = []
    has_any_words = any(seg.get("words") for seg in segments)

    if has_any_words:
        for seg in segments:
            words = seg.get("words") or []
            if words:
                for w in words:
                    wt = str(w.get("word", "")).strip()
                    if wt:
                        all_words.append({
                            "start": round(float(w["start"]), 3),
                            "end": round(float(w["end"]), 3),
                            "word": wt,
                        })
            else:
                text = str(seg.get("text", "")).strip()
                if not text:
                    continue
                tokens = text.split()
                st = float(seg.get("start", 0.0))
                en = float(seg.get("end", st + 1.0))
                step = max(0.05, (en - st) / max(1, len(tokens)))
                for idx, tok in enumerate(tokens):
                    all_words.append({
                        "start": round(st + idx * step, 3),
                        "end": round(min(en, st + (idx + 1) * step), 3),
                        "word": tok,
                    })

        sentences: list[dict[str, Any]] = []
        buf: list[dict[str, Any]] = []
        for w in all_words:
            buf.append(w)
            if _is_sentence_end(w["word"]):
                st = buf[0]["start"]
                en = buf[-1]["end"]
                sentences.append({
                    "id": len(sentences) + 1,
                    "start": st,
                    "end": en,
                    "duration": round(max(0.0, en - st), 2),
                    "text": " ".join(x["word"] for x in buf),
                    "words": list(buf),
                })
                buf = []
        if buf:
            st = buf[0]["start"]
            en = buf[-1]["end"]
            sentences.append({
                "id": len(sentences) + 1,
                "start": st,
                "end": en,
                "duration": round(max(0.0, en - st), 2),
                "text": " ".join(x["word"] for x in buf),
                "words": list(buf),
            })
        return sentences

    # Fallback when word-level timestamps are absent (e.g., Gemini mode or edited segments)
    sentence_split_re = re.compile(r"(?<=[.!?।۔])\s+")
    sentences = []
    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", seg_start + 1.0))
        parts = [p.strip() for p in sentence_split_re.split(text) if p.strip()]
        total_chars = max(1, sum(len(p) for p in parts))
        cursor = seg_start
        dur = max(0.0, seg_end - seg_start)
        for p in parts:
            p_dur = dur * (len(p) / total_chars)
            if sentences and not _is_sentence_end(sentences[-1]["text"].split()[-1]):
                sentences[-1]["text"] = (sentences[-1]["text"] + " " + p).strip()
                sentences[-1]["end"] = round(cursor + p_dur, 3)
                sentences[-1]["duration"] = round(max(0.0, sentences[-1]["end"] - sentences[-1]["start"]), 2)
            else:
                st = round(cursor, 3)
                en = round(cursor + p_dur, 3)
                sentences.append({
                    "id": len(sentences) + 1,
                    "start": st,
                    "end": en,
                    "duration": round(max(0.0, en - st), 2),
                    "text": p,
                    "words": [],
                })
            cursor += p_dur
    return sentences


def export_srt(
    segments: list[dict[str, Any]],
    short_mode: bool = False,
    max_words: int = 3,
    max_chars: int = 28,
) -> str:
    if short_mode:
        items = build_short_caption_segments(segments, max_words, max_chars)
    else:
        # Complete, meaningful sentences with exact word-level start & end timestamps
        items = build_sentence_segments(segments)
    lines: list[str] = []
    idx = 1
    for seg in items:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        start = format_srt_time(seg.get("start", 0.0))
        end = format_srt_time(seg.get("end", 0.0))
        lines.append(f"{idx}\n{start} --> {end}\n{text}\n")
        idx += 1
    return "\n".join(lines).strip() + "\n"


def export_vtt(
    segments: list[dict[str, Any]],
    short_mode: bool = False,
    max_words: int = 3,
    max_chars: int = 28,
) -> str:
    if short_mode:
        items = build_short_caption_segments(segments, max_words, max_chars)
    else:
        items = build_sentence_segments(segments)
    lines: list[str] = ["WEBVTT\n"]
    for seg in items:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        start = format_vtt_time(seg.get("start", 0.0))
        end = format_vtt_time(seg.get("end", 0.0))
        lines.append(f"{start} --> {end}\n{text}\n")
    return "\n".join(lines).strip() + "\n"


def build_turboscribe_paragraphs(
    segments: list[dict[str, Any]],
    sentences_per_paragraph: int = 3,
    max_chars_per_paragraph: int = 280,
) -> list[dict[str, Any]]:
    """
    Group exact word-synced complete sentences into clean, TurboScribe-style
    2-3 sentence paragraphs with exact start & end timestamps.
    """
    sentences = build_sentence_segments(segments)
    paragraphs: list[dict[str, Any]] = []
    cur_sents: list[dict[str, Any]] = []
    cur_len = 0

    for sent in sentences:
        s_text = sent["text"]
        projected = cur_len + (1 if cur_sents else 0) + len(s_text)
        if cur_sents and (
            len(cur_sents) >= sentences_per_paragraph
            or (len(cur_sents) >= 2 and projected > max_chars_per_paragraph)
        ):
            st = cur_sents[0]["start"]
            en = cur_sents[-1]["end"]
            paragraphs.append({
                "id": len(paragraphs) + 1,
                "start": st,
                "end": en,
                "duration": round(max(0.0, en - st), 2),
                "text": " ".join(x["text"] for x in cur_sents),
            })
            cur_sents = []
            cur_len = 0

        cur_sents.append(sent)
        cur_len += (1 if len(cur_sents) > 1 else 0) + len(s_text)

    if cur_sents:
        st = cur_sents[0]["start"]
        en = cur_sents[-1]["end"]
        paragraphs.append({
            "id": len(paragraphs) + 1,
            "start": st,
            "end": en,
            "duration": round(max(0.0, en - st), 2),
            "text": " ".join(x["text"] for x in cur_sents),
        })

    return paragraphs


def export_txt(segments: list[dict[str, Any]], include_timestamps: bool = False) -> str:
    if not include_timestamps:
        paragraphs = build_turboscribe_paragraphs(segments)
        if not paragraphs:
            return ""
        return "\n\n".join(p["text"] for p in paragraphs) + "\n"

    # Timestamped TXT: one complete sentence per line with exact start -> end & duration
    sentences = build_sentence_segments(segments)
    if not sentences:
        return ""
    lines = [
        f"[{format_short_time(s['start'])} --> {format_short_time(s['end'])}] ({s['duration']:.1f}s) {s['text']}"
        for s in sentences
    ]
    return "\n".join(lines) + "\n"


def export_docx(job: dict[str, Any], include_timestamps: bool = True) -> bytes:
    doc = Document()
    title = doc.add_heading(job.get("filename", "Transcript"), level=1)
    title.runs[0].font.size = Pt(18)

    meta_parts = []
    if job.get("language"):
        meta_parts.append(f"Language: {job['language']}")
    if job.get("duration"):
        meta_parts.append(f"Duration: {format_short_time(job['duration'])}")
    if job.get("engine_label"):
        meta_parts.append(f"Engine: {job['engine_label']}")
    if meta_parts:
        p = doc.add_paragraph("  |  ".join(meta_parts))
        for run in p.runs:
            run.font.size = Pt(9.5)
            run.font.color.rgb = RGBColor(100, 116, 139)

    doc.add_paragraph("")

    paragraphs = build_turboscribe_paragraphs(job.get("segments", []))
    for para in paragraphs:
        text = para["text"]
        if not text:
            continue
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(8)
        if include_timestamps:
            ts_run = p.add_run(f"[{format_short_time(para['start'])}] ")
            ts_run.bold = True
            ts_run.font.size = Pt(10)
            ts_run.font.color.rgb = RGBColor(79, 70, 229)
        body_run = p.add_run(text)
        body_run.font.size = Pt(11)

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def export_json(job: dict[str, Any]) -> str:
    payload = {
        "id": job.get("id"),
        "filename": job.get("filename"),
        "backend": job.get("backend"),
        "mode": job.get("mode"),
        "engine_label": job.get("engine_label"),
        "language": job.get("language"),
        "duration": job.get("duration"),
        "created_at": job.get("created_at"),
        "full_text": " ".join(str(s.get("text", "")).strip() for s in job.get("segments", [])),
        "segments": job.get("segments", []),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
