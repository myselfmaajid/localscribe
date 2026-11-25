"""
VeriScholar v4 FastAPI Backend Server.
Includes Plagiarism, AI Detection, Separate Report Exports (Similarity / AI / Combined),
Closed-Loop Academic Humanizer & Originality Rewriter, and Word (.docx) Export.
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .ai_detector import analyze_ai_content
from .config import (
    get_core_api_key,
    get_gemini_api_key,
    get_groq_api_key,
    get_mistral_api_key,
    get_optional_key,
    get_serper_api_key,
    get_tavily_api_key,
)
from .document_parser import parse_document
from .humanizer import (
    apply_sentence_edits_and_rescore,
    export_job_docx_bytes,
    humanize_entire_job,
    humanize_single_sentence,
    insert_citation_into_job,
    validate_job_authenticity,
)
from .integrity_scanner import scan_document_integrity
from .plagiarism_engine import analyze_plagiarism
from .report_generator import generate_pdf_bytes, render_report_html
from .telemetry_logger import get_diagnostics_summary, log_diagnostic_event

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
JOBS_DIR = DATA_DIR / "jobs"
REFS_DIR = DATA_DIR / "references"

for d in (UPLOADS_DIR, JOBS_DIR, REFS_DIR):
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="VeriScholar — Academic Originality & AI Integrity Studio", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_JOBS_LOCK = threading.Lock()
_JOBS: Dict[str, Dict[str, Any]] = {}


def _save_job(job: Dict[str, Any]) -> None:
    job_id = job["id"]
    with _JOBS_LOCK:
        _JOBS[job_id] = job
    path = JOBS_DIR / f"{job_id}.json"
    tmp = JOBS_DIR / f"{job_id}.json.tmp"
    tmp.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _load_existing_jobs() -> None:
    for p in sorted(JOBS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if data.get("status") in ("queued", "running"):
                data["status"] = "failed"
                data["error"] = "Interrupted by server restart"
            _JOBS[data["id"]] = data
        except Exception:
            continue


_load_existing_jobs()


def _job_summary(job: Dict[str, Any]) -> Dict[str, Any]:
    doc = job.get("document") or {}
    ai = job.get("ai_result") or {}
    plag = job.get("plagiarism_result") or {}
    integ = job.get("integrity_result") or {}
    return {
        "id": job["id"],
        "filename": job.get("filename"),
        "status": job.get("status"),
        "progress": job.get("progress", 0),
        "stage": job.get("stage", ""),
        "created_at": job.get("created_at"),
        "word_count": doc.get("word_count", 0),
        "char_count": doc.get("char_count", 0),
        "page_count": doc.get("page_count", 0),
        "overall_ai_percent": ai.get("overall_ai_percent"),
        "turnitin_display": ai.get("turnitin_display"),
        "overall_similarity_percent": plag.get("overall_similarity_percent"),
        "integrity_flags": integ.get("flag_count", 0),
        "error": job.get("error"),
    }


def _run_analysis_worker(
    job_id: str,
    file_path: Optional[Path],
    raw_text: Optional[str],
    filename: str,
    ai_mode: str,
    enable_academic: bool,
    enable_web: bool,
    enable_google_grounding: bool,
    enable_gemini_critic: bool,
    exclude_references: bool,
    exclude_quotes: bool,
    min_match_words: int,
    gemini_api_key: Optional[str],
) -> None:
    def _update(pct: int, stage: str) -> None:
        with _JOBS_LOCK:
            job = _JOBS.get(job_id)
            if not job:
                return
            job["progress"] = pct
            job["stage"] = stage
            job["status"] = "running"

    try:
        _update(5, "Reading document structure, word counts & sections...")
        parsed = parse_document(file_path=file_path, raw_text=raw_text, filename=filename)

        _update(10, "Verifying document formatting authenticity...")
        integrity_result = scan_document_integrity(file_path=file_path, full_text=parsed.full_text)

        _update(16, "Analyzing writing patterns & AI authorship probability...")
        ai_result = analyze_ai_content(
            parsed,
            mode=ai_mode,
            gemini_api_key=gemini_api_key,
            enable_gemini_critic=enable_gemini_critic,
            progress_cb=lambda p, _: _update(p, "Evaluating sentence-by-sentence AI writing patterns..."),
        )

        _update(70, "Checking web pages, online PDFs, and academic journals for similarity...")
        plag_result = analyze_plagiarism(
            parsed,
            enable_web=enable_web,
            enable_academic=enable_academic,
            enable_google_grounding=enable_google_grounding,
            gemini_api_key=gemini_api_key,
            ref_dir=REFS_DIR,
            exclude_references=exclude_references,
            exclude_quotes=exclude_quotes,
            min_match_words=min_match_words,
            progress_cb=lambda p, _: _update(p, "Cross-checking matched web & journal sources..."),
        )

        with _JOBS_LOCK:
            job = _JOBS[job_id]
            job["status"] = "completed"
            job["progress"] = 100
            job["stage"] = "Complete"
            job["document"] = parsed.to_dict()
            job["integrity_result"] = integrity_result
            job["ai_result"] = ai_result
            job["plagiarism_result"] = plag_result
            job["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _save_job(job)

    except Exception as exc:
        log_diagnostic_event(
            level="ERROR",
            subsystem="server",
            provider="analysis_worker",
            message=f"{type(exc).__name__}: {str(exc)[:300]}",
            job_id=job_id,
            error_type=type(exc).__name__,
        )
        with _JOBS_LOCK:
            job = _JOBS.get(job_id)
            if job:
                job["status"] = "failed"
                job["stage"] = "Failed"
                job["error"] = str(exc)
                _save_job(job)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    ref_files = [p.name for p in REFS_DIR.iterdir() if p.is_file()]
    channels = {
        "primary_humanizer_groq": bool(get_groq_api_key()),
        "secondary_humanizer_mistral": bool(get_mistral_api_key()),
        "fallback_humanizer_gemini": bool(get_gemini_api_key()),
        "serper_web_search": bool(get_serper_api_key()),
        "tavily_web_search": bool(get_tavily_api_key()),
        "core_academic_fulltext": bool(get_core_api_key()),
        "offline_base_engine": True,
    }
    return {
        "status": "ok",
        "version": "7.0.0",
        "local_references_count": len(ref_files),
        "jobs_count": len(_JOBS),
        "cloud_ready": any(v for k, v in channels.items() if k != "offline_base_engine"),
        "active_channels": sum(1 for v in channels.values() if v),
    }


@app.get("/api/diagnostics")
async def diagnostics(limit: int = Query(default=60, ge=5, le=250)) -> Dict[str, Any]:
    """Structured diagnostic database summary, provider circuit-breaker states & secret-scrubbed logs."""
    summary = get_diagnostics_summary(limit=limit)
    summary["configured_waterfall"] = {
        "humanizer_chain": [
            {"tier": "1. Primary High-Speed Cloud", "configured": bool(get_groq_api_key())},
            {"tier": "2. Secondary Cross-Tokenizer Cloud", "configured": bool(get_mistral_api_key())},
            {"tier": "3. Tertiary Cloud Backup", "configured": bool(get_gemini_api_key())},
            {"tier": "4. Default Offline Deterministic Base", "configured": True},
        ],
        "plagiarism_chain": [
            {"tier": "1. Google Exact-Phrase Index (Serper)", "configured": bool(get_serper_api_key())},
            {"tier": "2. Real-Time Web Search (Tavily)", "configured": bool(get_tavily_api_key())},
            {"tier": "3. CORE 260M+ Open-Access Full-Text", "configured": bool(get_core_api_key())},
            {"tier": "4. OpenAlex + CrossRef + PubMed/PMC + S2", "configured": True},
            {"tier": "5. Default Offline Reference + Shingle Engine", "configured": True},
        ],
    }
    return summary


@app.post("/api/analyze")
async def create_analysis(
    file: Optional[UploadFile] = File(default=None),
    text: Optional[str] = Form(default=None),
    title: Optional[str] = Form(default=None),
    ai_mode: str = Form(default="deep"),
    enable_academic: bool = Form(default=True),
    enable_web: bool = Form(default=True),
    enable_google_grounding: bool = Form(default=True),
    enable_gemini_critic: bool = Form(default=False),
    exclude_references: bool = Form(default=True),
    exclude_quotes: bool = Form(default=True),
    min_match_words: int = Form(default=6),
    gemini_api_key: Optional[str] = Form(default=None),
) -> Dict[str, Any]:
    if file is None and (not text or not text.strip()):
        raise HTTPException(status_code=400, detail="Provide either a file (.pdf/.docx/.txt) or pasted text.")

    job_id = f"VS-{uuid.uuid4().hex[:9].upper()}"
    saved_path: Optional[Path] = None
    filename = (title or "Pasted_Document.txt").strip()

    if file is not None and file.filename:
        filename = Path(file.filename).name
        ext = Path(filename).suffix.lower()
        if ext not in (".pdf", ".docx", ".txt"):
            raise HTTPException(status_code=400, detail="Unsupported file type. Upload PDF, DOCX, or TXT.")
        saved_path = UPLOADS_DIR / f"{job_id}_{filename}"
        content = await file.read()
        saved_path.write_bytes(content)

    job: Dict[str, Any] = {
        "id": job_id,
        "filename": filename,
        "status": "queued",
        "progress": 1,
        "stage": "Starting scan...",
        "ai_mode": ai_mode,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "document": None,
        "integrity_result": None,
        "ai_result": None,
        "plagiarism_result": None,
        "error": None,
    }
    _save_job(job)

    t = threading.Thread(
        target=_run_analysis_worker,
        args=(
            job_id,
            saved_path,
            text,
            filename,
            ai_mode,
            enable_academic,
            enable_web,
            enable_google_grounding,
            enable_gemini_critic,
            exclude_references,
            exclude_quotes,
            max(4, min(20, min_match_words)),
            gemini_api_key,
        ),
        daemon=True,
    )
    t.start()
    return _job_summary(job)


@app.get("/api/jobs")
async def list_jobs() -> List[Dict[str, Any]]:
    with _JOBS_LOCK:
        items = list(_JOBS.values())
    items.sort(key=lambda j: j.get("created_at") or "", reverse=True)
    return [_job_summary(j) for j in items]


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> Dict[str, Any]:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str) -> Dict[str, str]:
    with _JOBS_LOCK:
        _JOBS.pop(job_id, None)
    p = JOBS_DIR / f"{job_id}.json"
    if p.exists():
        p.unlink()
    return {"deleted": job_id}


@app.get("/api/jobs/{job_id}/report.pdf")
async def download_pdf(
    job_id: str,
    mode: str = Query(default="both", description="plagiarism | ai | both"),
) -> Response:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if not job or job.get("status") != "completed":
        raise HTTPException(status_code=404, detail="Completed report not found")
    pdf_bytes = generate_pdf_bytes(job, mode=mode)
    safe_name = Path(job.get("filename") or job_id).stem
    suffix = "Similarity_Report" if mode == "plagiarism" else ("AI_Report" if mode == "ai" else "Full_Report")
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="VeriScholar_{suffix}_{safe_name}.pdf"'},
    )


@app.get("/api/jobs/{job_id}/report.html")
async def download_html(
    job_id: str,
    mode: str = Query(default="both"),
) -> HTMLResponse:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if not job or job.get("status") != "completed":
        raise HTTPException(status_code=404, detail="Completed report not found")
    return HTMLResponse(content=render_report_html(job, mode=mode))


@app.get("/api/jobs/{job_id}/export_docx")
async def download_docx(
    job_id: str,
    tracked: bool = Query(default=False, description="Highlight updated passages in green"),
) -> Response:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if not job or job.get("status") != "completed":
        raise HTTPException(status_code=404, detail="Completed job not found")
    docx_bytes = export_job_docx_bytes(job, highlight_changes=tracked, uploads_dir=UPLOADS_DIR)
    safe_name = Path(job.get("filename") or job_id).stem
    tag = "Tracked_Green_Changes" if tracked else "Final_Clean_Submission"
    return Response(
        content=docx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}_{tag}.docx"'},
    )


@app.post("/api/jobs/{job_id}/insert_citation")
async def api_insert_citation(
    job_id: str,
    payload: Dict[str, Any] = Body(...),
) -> Dict[str, Any]:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if not job or job.get("status") != "completed":
        raise HTTPException(status_code=404, detail="Completed job not found")
    sid = int(payload.get("sentence_id", -1))
    style = str(payload.get("style") or "auto")
    updated = insert_citation_into_job(job, sentence_id=sid, style=style)
    _save_job(updated)
    return updated


@app.post("/api/jobs/{job_id}/humanize_sentence")
async def api_humanize_sentence(
    job_id: str,
    payload: Dict[str, Any] = Body(...),
) -> Dict[str, Any]:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if not job or job.get("status") != "completed":
        raise HTTPException(status_code=404, detail="Completed job not found")

    sid = int(payload.get("sentence_id", -1))
    sents = (job.get("document") or {}).get("sentences") or []
    s_obj = next((s for s in sents if s["id"] == sid), None)
    if not s_obj:
        raise HTTPException(status_code=404, detail="Sentence not found")

    plag_matches = (job.get("plagiarism_result") or {}).get("sentence_matches") or []
    pm = next((m for m in plag_matches if m["sentence_id"] == sid), None)
    ctx_before = " ".join(s["text"] for s in sents if sid - 3 <= s["id"] < sid and s.get("qualifies_for_ai"))
    ctx_after = " ".join(s["text"] for s in sents if sid < s["id"] <= sid + 3 and s.get("qualifies_for_ai"))
    reason = "both" if pm else "ai"
    res = humanize_single_sentence(
        sentence_text=s_obj["text"],
        reason=reason,
        matched_source_excerpt=pm.get("source_excerpt", "") if pm else "",
        context_before=ctx_before,
        context_after=ctx_after,
    )
    return res


@app.post("/api/jobs/{job_id}/apply_edit")
async def api_apply_edit(
    job_id: str,
    payload: Dict[str, Any] = Body(...),
) -> Dict[str, Any]:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if not job or job.get("status") != "completed":
        raise HTTPException(status_code=404, detail="Completed job not found")

    sid = int(payload.get("sentence_id", -1))
    new_text = str(payload.get("new_text") or "").strip()
    if not new_text:
        raise HTTPException(status_code=400, detail="New text cannot be empty")

    updated = apply_sentence_edits_and_rescore(job, {sid: new_text})
    _save_job(updated)
    return updated


@app.post("/api/jobs/{job_id}/humanize_all")
async def api_humanize_all(
    job_id: str,
    payload: Dict[str, Any] = Body(default={}),
) -> Dict[str, Any]:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if not job or job.get("status") != "completed":
        raise HTTPException(status_code=404, detail="Completed job not found")

    target = str(payload.get("target") or "both")
    updated, count = humanize_entire_job(job, target=target)
    _save_job(updated)
    return {"updated_sentences": count, "job": updated}


@app.post("/api/jobs/{job_id}/validate")
async def api_validate_job(job_id: str) -> Dict[str, Any]:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if not job or job.get("status") != "completed":
        raise HTTPException(status_code=404, detail="Completed job not found")
    return validate_job_authenticity(job)


@app.get("/api/references")
async def list_references() -> List[Dict[str, Any]]:
    refs = []
    for p in sorted(REFS_DIR.iterdir()):
        if p.is_file():
            refs.append({"name": p.name, "size": p.stat().st_size})
    return refs


@app.post("/api/references")
async def upload_reference(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    name = Path(file.filename).name
    if Path(name).suffix.lower() not in (".pdf", ".docx", ".txt"):
        raise HTTPException(status_code=400, detail="Only PDF, DOCX, and TXT files are supported.")
    dest = REFS_DIR / name
    dest.write_bytes(await file.read())
    return {"name": name, "size": dest.stat().st_size}


@app.delete("/api/references/{name}")
async def delete_reference(name: str) -> Dict[str, str]:
    safe = Path(name).name
    dest = REFS_DIR / safe
    if dest.exists() and dest.is_file():
        dest.unlink()
    return {"deleted": safe}
