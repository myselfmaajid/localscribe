"""
VeriScholar v5 — Professional Academic Integrity & Before/After Revision PDF Generator.

Features:
  - Separate Plagiarism-Only, AI-Only, and Full Combined PDF modes
  - Green highlighting ([✓ Updated]) for humanized / rewritten / newly-cited passages
  - Distinct highlighting for Uncited Matches ([1 · Uncited]) vs Properly Cited Matches ([1 · Cited ✓])
  - Smart Highlight Guide (hides unused badges — e.g., never shows [1] Copied on a 0.0% Similarity report!)
  - Dedicated Before & After Revision Summary + Side-by-Side Comparison Table when edits exist!
  - Zero internal model/library names exposed
"""
from __future__ import annotations

import html as html_lib
import io
from datetime import datetime
from typing import Any, Dict, List

import pymupdf

_SOURCE_BG_COLORS = [
    ("#ffe4e6", "#9f1239", "#e11d48"),  # rose
    ("#ede9fe", "#5b21b6", "#7c3aed"),  # violet
    ("#e0f2fe", "#075985", "#0284c7"),  # sky
    ("#fef3c7", "#92400e", "#d97706"),  # amber
    ("#fce7f3", "#9d174d", "#db2777"),  # pink
    ("#e0e7ff", "#3730a3", "#4f46e5"),  # indigo
    ("#ffedd5", "#9a3412", "#ea580c"),  # orange
    ("#dbeafe", "#1e40af", "#2563eb"),  # blue
]


def _palette_for_source(index: int) -> tuple[str, str, str]:
    return _SOURCE_BG_COLORS[(max(1, index) - 1) % len(_SOURCE_BG_COLORS)]


def render_report_html(job: Dict[str, Any], mode: str = "both") -> str:
    mode = mode.lower().strip()
    if mode not in ("plagiarism", "ai", "both"):
        mode = "both"

    show_plag = mode in ("plagiarism", "both")
    show_ai = mode in ("ai", "both")

    doc = job.get("document") or {}
    ai = job.get("ai_result") or {}
    plag = job.get("plagiarism_result") or {}
    integ = job.get("integrity_result") or {"flag_count": 0, "flags": []}
    baseline = job.get("baseline") or {}
    revisions: Dict[str, Any] = job.get("revisions") or {}

    filename = html_lib.escape(str(doc.get("filename") or job.get("filename") or "Document"))
    job_id = html_lib.escape(str(job.get("id") or "VS-REPORT"))
    submitted_at = html_lib.escape(str(job.get("created_at") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    downloaded_at = html_lib.escape(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    report_subtitle = (
        "Originality & Similarity Verification Report"
        if mode == "plagiarism"
        else ("AI Writing Assessment Report" if mode == "ai" else "Combined Originality & AI Writing Assessment Report")
    )

    sim_pct = float(plag.get("overall_similarity_percent", 0.0))
    exact_pct = float(plag.get("exact_similarity_percent", 0.0))
    para_pct = float(plag.get("paraphrase_similarity_percent", 0.0))
    internet_pct = float(plag.get("internet_percent", 0.0))
    pub_pct = float(plag.get("publications_percent", 0.0))
    local_pct = float(plag.get("local_library_percent", 0.0))

    ai_pct = float(ai.get("overall_ai_percent", 0.0))
    ai_gen_pct = float(ai.get("ai_generated_percent", 0.0))
    ai_para_pct = float(ai.get("ai_paraphrased_percent", 0.0))
    ai_low_pct = float(ai.get("ai_low_signal_percent", 0.0))
    human_pct = float(max(0.0, 100.0 - ai_pct))

    metrics = ai.get("document_metrics") or {}
    burst_cv = metrics.get("burstiness_cv", 0.4)
    ppl_val = metrics.get("avg_perplexity")
    rhythm_label = "Natural Variation" if (burst_cv or 0.4) >= 0.32 else "Uniform / Robotic"
    phrasing_label = "Natural Academic Phrasing" if (ppl_val is None or ppl_val >= 42.0) else "Predictable Phrasing"

    mg = plag.get("match_groups") or {}
    uncited_cnt = (mg.get("not_cited_or_quoted", {}).get("count", 0)) + (mg.get("missing_citation", {}).get("count", 0))
    cited_cnt = (mg.get("missing_quotations", {}).get("count", 0)) + (mg.get("cited_and_quoted", {}).get("count", 0))

    # Before -> After header badges if revisions exist
    init_sim = baseline.get("initial_similarity_percent")
    init_ai = baseline.get("initial_ai_percent")
    sim_before_html = (
        f'<div style="font-size:9pt;color:#059669;font-weight:bold;margin-bottom:4px;">Before Revision: {float(init_sim):.1f}% &rarr; After Revision: {sim_pct:.1f}%</div>'
        if (revisions and init_sim is not None and float(init_sim) != sim_pct)
        else ""
    )
    ai_before_html = (
        f'<div style="font-size:9pt;color:#059669;font-weight:bold;margin-bottom:4px;">Before Humanizer: {float(init_ai):.1f}% &rarr; After Humanizer: {ai_pct:.1f}%</div>'
        if (revisions and init_ai is not None and float(init_ai) != ai_pct)
        else ""
    )

    cards_td: List[str] = []
    if show_plag:
        cards_td.append(
            f"""
            <td class="score-card">
              <div style="font-size:9pt;color:#64748b;font-weight:bold;">OVERALL SIMILARITY</div>
              <div class="score-big-sim">{sim_pct:.1f}%</div>
              {sim_before_html}
              <div style="font-size:9pt;color:#334155;">
                Web Sources: <strong>{internet_pct:.1f}%</strong> &nbsp;|&nbsp; Academic Journals: <strong>{pub_pct:.1f}%</strong> &nbsp;|&nbsp; Reference Files: <strong>{local_pct:.1f}%</strong><br/>
                Copied Word-for-Word: <strong>{exact_pct:.1f}%</strong> &nbsp;|&nbsp; Reworded: <strong>{para_pct:.1f}%</strong><br/>
                Uncited Matches: <strong>{uncited_cnt}</strong> &nbsp;|&nbsp; Properly Cited: <strong>{cited_cnt}</strong>
              </div>
            </td>
            """
        )
    if show_ai:
        cards_td.append(
            f"""
            <td class="score-card">
              <div style="font-size:9pt;color:#64748b;font-weight:bold;">AI WRITING SCORE</div>
              <div class="score-big-ai">{ai_pct:.1f}%</div>
              {ai_before_html}
              <div style="font-size:9pt;color:#334155;">
                Likely AI-Generated: <strong>{ai_gen_pct:.1f}%</strong> &nbsp;|&nbsp; AI-Assisted / Edited: <strong>{ai_para_pct:.1f}%</strong><br/>
                Mild AI Pattern: <strong>{ai_low_pct:.1f}%</strong> &nbsp;|&nbsp; Human-Written: <strong>{human_pct:.1f}%</strong><br/>
                Sentence Rhythm: <strong>{rhythm_label}</strong> &nbsp;|&nbsp; Word Choice: <strong>{phrasing_label}</strong>
              </div>
            </td>
            """
        )

    # Matched Sources table (plus Verified Scanned Sources proof when 0% matched!)
    sources_section_html = ""
    if show_plag:
        sources = plag.get("sources") or []
        scanned_log = plag.get("scanned_sources_log") or []
        if sources:
            sources_rows = []
            for src in sources[:30]:
                idx = src["index"]
                _, _, badge_col = _palette_for_source(idx)
                title = html_lib.escape(str(src.get("title") or "Matched Source"))
                cat_label = (
                    "Academic Journal / Paper"
                    if src.get("category") == "publication"
                    else ("Uploaded Reference" if src.get("category") == "local_library" else "Website / Online Document")
                )
                url = html_lib.escape(str(src.get("url") or ""))
                pct = float(src.get("similarity_percent", 0.0))
                words = int(src.get("matched_words", 0))
                sources_rows.append(
                    f"""
                    <tr>
                      <td style="width:36px;text-align:center;">
                        <span style="background-color:{badge_col};color:#ffffff;font-weight:bold;padding:2px 6px;font-size:9pt;">[{idx}]</span>
                      </td>
                      <td>
                        <div style="font-weight:bold;color:#0f172a;font-size:10pt;">{title}</div>
                        <div style="color:#475569;font-size:8.5pt;">{cat_label}{(' &bull; ' + url[:95]) if url else ''}</div>
                      </td>
                      <td style="width:68px;text-align:right;font-weight:bold;color:{badge_col};font-size:10pt;">{pct:.1f}%</td>
                      <td style="width:65px;text-align:right;color:#475569;font-size:9pt;">{words} words</td>
                    </tr>
                    """
                )
            sources_section_html = f"""
              <div class="section-title">Matched Sources ({len(sources)})</div>
              <table class="sources-table">
                <thead>
                  <tr><th>#</th><th>Matched Source &amp; Verification Link</th><th style="text-align:right;">Match</th><th style="text-align:right;">Words</th></tr>
                </thead>
                <tbody>{''.join(sources_rows)}</tbody>
              </table>
            """
        else:
            proof_items = "".join(
                f'<li style="margin-bottom:3px;"><strong>{html_lib.escape(s.get("title","")[:70])}</strong> '
                f'<span style="color:#64748b;">({html_lib.escape(s.get("url","")[:75])})</span> &mdash; <span style="color:#059669;font-weight:bold;">0% Match Verified</span></li>'
                for s in scanned_log[:8]
            )
            proof_html = f'<ul style="margin:6px 0 0 18px;font-size:8.5pt;color:#334155;">{proof_items}</ul>' if proof_items else ""
            sources_section_html = f"""
              <div class="section-title">Matched Sources (0)</div>
              <div style="padding:9px 12px;background-color:#ecfdf5;color:#065f46;font-size:9.5pt;">
                <strong>0.0% Similarity Verified:</strong> Cross-checked against {plag.get("sources_scanned", 0)} live websites and academic journal papers with no matching passages found.
                {proof_html}
              </div>
            """

    # Before & After Revision Comparison Table (when edits exist relevant to this report mode!)
    revision_section_html = ""
    relevant_revs = []
    for rev in sorted(revisions.values(), key=lambda x: x.get("sentence_id", 0)):
        ctype = rev.get("change_type", "humanized")
        if mode == "plagiarism" and ctype == "humanized_ai" and float(rev.get("before_plag_percent", 0)) == 0:
            continue
        if mode == "ai" and ctype in ("rewritten_plagiarism", "citation_added") and float(rev.get("before_ai_percent", 0)) < 30:
            continue
        relevant_revs.append(rev)

    if relevant_revs:
        rev_rows = []
        for r in relevant_revs[:35]:
            b_tag = (
                f'Similarity: {r.get("before_plag_percent", 0):.0f}%'
                if mode == "plagiarism"
                else (
                    f'AI: {r.get("before_ai_percent", 0):.0f}%'
                    if mode == "ai"
                    else f'AI: {r.get("before_ai_percent", 0):.0f}% | Sim: {r.get("before_plag_percent", 0):.0f}%'
                )
            )
            a_tag = (
                f'Similarity: {r.get("after_plag_percent", 0):.0f}%'
                if mode == "plagiarism"
                else (
                    f'AI: {r.get("after_ai_percent", 0):.0f}%'
                    if mode == "ai"
                    else f'AI: {r.get("after_ai_percent", 0):.0f}% | Sim: {r.get("after_plag_percent", 0):.0f}%'
                )
            )
            rev_rows.append(
                f"""
                <div class="rev-card">
                  <div style="font-size:8.5pt;color:#475569;font-weight:bold;margin-bottom:4px;">Page {r.get("page", 1)}</div>
                  <div style="margin-bottom:5px;font-size:9pt;color:#881337;">
                    <span style="background-color:#e11d48;color:#ffffff;font-weight:bold;font-size:7.8pt;">&nbsp;BEFORE ({b_tag})&nbsp;</span>
                    &nbsp;<span style="background-color:#ffe4e6;color:#881337;">{html_lib.escape(r.get("original_text", ""))}</span>
                  </div>
                  <div style="font-size:9pt;color:#14532d;">
                    <span style="background-color:#16a34a;color:#ffffff;font-weight:bold;font-size:7.8pt;">&nbsp;UPDATED ({a_tag})&nbsp;</span>
                    &nbsp;<span style="background-color:#dcfce7;color:#14532d;">{html_lib.escape(r.get("updated_text", ""))}</span>
                  </div>
                </div>
                """
            )
        revision_section_html = f"""
          <div class="section-title">Before &amp; After Revision Summary ({len(relevant_revs)} Updated Passages)</div>
          {''.join(rev_rows)}
        """

    integ_html = (
        "".join(
            f'<div style="padding:7px 10px;background-color:#fef2f2;color:#991b1b;margin-bottom:6px;font-size:9.5pt;">'
            f'<strong>Document Alert:</strong> {html_lib.escape(f.get("description",""))}</div>'
            for f in (integ.get("flags") or [])
        )
        if integ.get("flag_count", 0) > 0
        else '<div style="padding:7px 10px;background-color:#ecfdf5;color:#065f46;font-size:9.5pt;"><strong>Document Formatting Verified:</strong> Authentic text formatting with no hidden characters or manipulation tricks.</div>'
    )

    ai_by_sid = {s["sentence_id"]: s for s in (ai.get("sentences") or [])}
    plag_by_sid = {m["sentence_id"]: m for m in (plag.get("sentence_matches") or [])}
    sent_by_id = {s["id"]: s for s in (doc.get("sentences") or [])}

    used_uncited = False
    used_cited = False
    used_ai_gen = False
    used_ai_para = False
    used_ai_low = False
    used_green = False

    paragraphs_html: List[str] = []
    for para in (doc.get("paragraphs") or []):
        sec = para.get("section", "prose")
        sids = para.get("sentence_ids") or []
        spans: List[str] = []

        for sid in sids:
            s_obj = sent_by_id.get(sid)
            if not s_obj:
                continue
            txt = html_lib.escape(s_obj["text"])
            is_revised = str(sid) in revisions
            p_match = plag_by_sid.get(sid) if show_plag else None
            a_res = ai_by_sid.get(sid) if show_ai else None

            if is_revised and (not p_match or p_match.get("citation_group") in ("missing_quotations", "cited_and_quoted")):
                used_green = True
                spans.append(
                    f'<span style="background-color:#16a34a;color:#ffffff;font-weight:bold;font-size:8pt;">&nbsp;UPDATED&nbsp;</span>'
                    f'<span style="background-color:#dcfce7;color:#14532d;">{txt}</span>'
                )
            elif p_match:
                is_cited = p_match.get("citation_group") in ("missing_quotations", "cited_and_quoted")
                if is_cited:
                    used_cited = True
                    spans.append(
                        f'<span style="background-color:#0d9488;color:#ffffff;font-weight:bold;font-size:8pt;">&nbsp;[{p_match["source_index"]} &bull; Cited]&nbsp;</span>'
                        f'<span style="background-color:#ccfbf1;color:#115e59;">{txt}</span>'
                    )
                else:
                    used_uncited = True
                    bg_col, fg_col, badge_col = _palette_for_source(p_match["source_index"])
                    spans.append(
                        f'<span style="background-color:{badge_col};color:#ffffff;font-weight:bold;font-size:8pt;">&nbsp;[{p_match["source_index"]} &bull; Uncited]&nbsp;</span>'
                        f'<span style="background-color:{bg_col};color:{fg_col};">{txt}</span>'
                    )
            elif a_res and a_res.get("label") == "ai_generated":
                used_ai_gen = True
                spans.append(f'<span style="background-color:#cffafe;color:#0e7490;">{txt}</span>')
            elif a_res and a_res.get("label") == "ai_paraphrased":
                used_ai_para = True
                spans.append(f'<span style="background-color:#ede9fe;color:#5b21b6;">{txt}</span>')
            elif a_res and a_res.get("label") == "low_ai_signal":
                used_ai_low = True
                spans.append(f'<span style="background-color:#fef9c3;color:#854d0e;">{txt}</span>')
            else:
                spans.append(txt)

        inner = " ".join(spans)
        if sec == "heading":
            paragraphs_html.append(f'<h3 class="doc-heading">{inner}</h3>')
        elif sec == "cover":
            paragraphs_html.append(f'<p class="doc-cover">{inner}</p>')
        elif sec == "reference":
            paragraphs_html.append(f'<p class="doc-ref">{inner}</p>')
        else:
            paragraphs_html.append(f'<p class="doc-para">{inner}</p>')

    # Build Smart Highlight Guide ONLY showing badges actually present in this document!
    legend_items: List[str] = []
    if used_green:
        legend_items.append('<span style="background-color:#dcfce7;color:#14532d;padding:1px 5px;">[UPDATED] Humanized / Revised Passage</span>')
    if used_uncited:
        legend_items.append('<span style="background-color:#ffe4e6;color:#9f1239;padding:1px 5px;">[1 &bull; Uncited] Matched Source (Needs Citation)</span>')
    if used_cited:
        legend_items.append('<span style="background-color:#ccfbf1;color:#115e59;padding:1px 5px;">[1 &bull; Cited] Properly Cited Match</span>')
    if used_ai_gen:
        legend_items.append('<span style="background-color:#cffafe;color:#0e7490;padding:1px 5px;">Likely AI-Generated</span>')
    if used_ai_para:
        legend_items.append('<span style="background-color:#ede9fe;color:#5b21b6;padding:1px 5px;">AI-Assisted / Edited</span>')
    if used_ai_low:
        legend_items.append('<span style="background-color:#fef9c3;color:#854d0e;padding:1px 5px;">Mild AI Pattern</span>')

    legend_html = (
        f'<div class="legend"><strong>Highlight Guide:</strong> &nbsp; {" &nbsp;&nbsp; ".join(legend_items)}</div>'
        if legend_items
        else '<div class="legend" style="color:#065f46;background-color:#ecfdf5;"><strong>Clean Submission Text:</strong> No flagged passages present in this report view.</div>'
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>VeriScholar Report — {filename}</title>
<style>
  body {{
    font-family: Helvetica, Arial, sans-serif;
    color: #0f172a;
    font-size: 10.5pt;
    line-height: 1.55;
    margin: 0;
    padding: 0;
  }}
  .header-bar {{
    border-bottom: 2px solid #0284c7;
    padding-bottom: 8px;
    margin-bottom: 14px;
  }}
  .brand {{ font-size: 18pt; font-weight: bold; color: #0284c7; }}
  .sub-brand {{ font-size: 9.5pt; color: #475569; }}
  .doc-title {{ font-size: 14pt; font-weight: bold; margin: 10px 0 6px 0; color: #0f172a; }}
  .meta-table {{
    width: 100%;
    border-collapse: collapse;
    margin: 10px 0 16px 0;
    border: 1px solid #cbd5e1;
  }}
  .meta-table td {{ padding: 6px 10px; font-size: 9pt; border-bottom: 1px solid #e2e8f0; }}
  .score-grid {{ width: 100%; border-collapse: separate; border-spacing: 8px 0; margin: 12px -8px 16px -8px; }}
  .score-card {{ border: 1.5px solid #cbd5e1; padding: 12px; vertical-align: top; }}
  .rev-card {{ border-bottom: 1px solid #e2e8f0; padding: 8px 0; margin-bottom: 6px; }}
  .score-big-sim {{ font-size: 25pt; font-weight: bold; color: #e11d48; margin: 2px 0; }}
  .score-big-ai {{ font-size: 25pt; font-weight: bold; color: #0284c7; margin: 2px 0; }}
  .section-title {{
    font-size: 12pt;
    font-weight: bold;
    color: #0f172a;
    border-bottom: 1px solid #cbd5e1;
    padding-bottom: 4px;
    margin: 18px 0 10px 0;
  }}
  .sources-table {{ width: 100%; border-collapse: collapse; margin-bottom: 14px; }}
  .sources-table th {{ text-align: left; font-size: 9pt; color: #475569; border-bottom: 1.5px solid #cbd5e1; padding: 6px 4px; }}
  .sources-table td {{ padding: 6px 4px; border-bottom: 1px solid #e2e8f0; vertical-align: top; }}
  .legend {{ background-color: #f8fafc; border: 1px solid #e2e8f0; padding: 8px 10px; font-size: 9pt; margin-bottom: 14px; }}
  .doc-heading {{ font-size: 11.5pt; font-weight: bold; margin: 14px 0 6px 0; color: #1e293b; }}
  .doc-para {{ margin: 0 0 9px 0; text-align: justify; font-size: 10pt; }}
  .doc-cover {{ margin: 0 0 5px 0; color: #334155; font-size: 9.5pt; }}
  .doc-ref {{ margin: 0 0 5px 0; color: #64748b; font-size: 8.8pt; }}
  .disclaimer {{ margin-top: 20px; padding: 10px; background-color: #f1f5f9; font-size: 8.5pt; color: #475569; }}
</style>
</head>
<body>
  <div class="header-bar">
    <div class="brand">VeriScholar</div>
    <div class="sub-brand">{report_subtitle}</div>
  </div>

  <div class="doc-title">{filename}</div>

  <table class="meta-table">
    <tr>
      <td><strong>Report ID:</strong> {job_id}</td>
      <td><strong>Submission Date:</strong> {submitted_at}</td>
    </tr>
    <tr>
      <td><strong>Download Date:</strong> {downloaded_at}</td>
      <td><strong>Document Stats:</strong> {doc.get("page_count", 1)} Pages &bull; {doc.get("word_count", 0):,} Words &bull; {doc.get("char_count", 0):,} Characters</td>
    </tr>
  </table>

  <table class="score-grid">
    <tr>{''.join(cards_td)}</tr>
  </table>

  {integ_html}

  {sources_section_html}

  {revision_section_html}

  <div class="section-title">Submission Text</div>
  {legend_html}

  {''.join(paragraphs_html)}

  <div class="disclaimer">
    <strong>Verification Notice:</strong> Generated by VeriScholar Academic Integrity Studio on {downloaded_at}. Highlighted passages indicate source overlap, automated writing patterns, or author-approved revisions.
  </div>
</body>
</html>"""


def generate_pdf_bytes(job: Dict[str, Any], mode: str = "both") -> bytes:
    html_str = render_report_html(job, mode=mode)
    story = pymupdf.Story(html=html_str)
    mediabox = pymupdf.paper_rect("a4")
    where = mediabox + (42, 46, -42, -46)

    buf = io.BytesIO()
    dw = pymupdf.DocumentWriter(buf)
    more = 1
    page_guard = 0
    while more and page_guard < 160:
        dev = dw.begin_page(mediabox)
        more, _ = story.place(where)
        story.draw(dev)
        dw.end_page()
        page_guard += 1
    dw.close()

    pdf_doc = pymupdf.open(stream=buf.getvalue(), filetype="pdf")
    total_pages = len(pdf_doc)
    job_id = str(job.get("id") or "VS-REPORT")
    mode_tag = "Similarity Report" if mode == "plagiarism" else ("AI Writing Report" if mode == "ai" else "Full Integrity Report")
    dl_date = datetime.now().strftime("%Y-%m-%d %H:%M")

    for idx, page in enumerate(pdf_doc, start=1):
        footer_text = f"VeriScholar {mode_tag}  |  Downloaded: {dl_date}  |  Page {idx} of {total_pages}  |  ID: {job_id}"
        page.insert_text(
            pymupdf.Point(42, mediabox.height - 22),
            footer_text,
            fontsize=8,
            color=(0.42, 0.47, 0.55),
        )
    final_bytes = pdf_doc.tobytes(garbage=4, deflate=True)
    pdf_doc.close()
    return final_bytes
