"""FastAPI server for LocalScribe — Unlimited local TurboScribe alternative."""

from __future__ import annotations

import json
import mimetypes
import os
import re
import shutil
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

from app.exporters import (
    build_short_caption_segments,
    export_docx,
    export_json,
    export_srt,
    export_txt,
    export_vtt,
)
from app.transcriber import (
    WHISPER_MODES,
    detect_hardware,
    transcribe_with_gemini,
    transcribe_with_whisper,
    translate_segments_with_gemini,
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
JOBS_DIR = DATA_DIR / "jobs"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="LocalScribe", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_JOBS_LOCK = threading.Lock()
_JOBS: dict[str, dict[str, Any]] = {}


def _safe_stem(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"[^a-zA-Z0-9_\-\. ]+", "_", stem).strip()
    return stem or "transcript"


def _save_job(job: dict[str, Any]) -> None:
    path = JOBS_DIR / f"{job['id']}.json"
    tmp = JOBS_DIR / f"{job['id']}.json.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(job, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def _load_existing_jobs() -> None:
    for p in JOBS_DIR.glob("*.json"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                job = json.load(f)
            if job.get("status") in ("queued", "processing"):
                job["status"] = "failed"
                job["error"] = "Interrupted by server restart. Please re-transcribe."
                _save_job(job)
            _JOBS[job["id"]] = job
        except Exception:
            continue


_load_existing_jobs()


def _update_job(job_id: str, **updates: Any) -> None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return
        job.update(updates)
        _save_job(job)


def _run_transcription_job(
    job_id: str,
    audio_path: str,
    backend: str,
    mode: str,
    language: str,
    task: str,
    target_language: str,
    initial_prompt: str,
    vad_filter: bool,
    gemini_api_key: str,
    gemini_model: str,
) -> None:
    def progress_cb(msg: str, pct: int) -> None:
        _update_job(job_id, status="processing", progress=pct, status_message=msg)

    try:
        progress_cb("Starting transcription...", 2)
        if backend == "gemini":
            key = gemini_api_key.strip() or os.environ.get("GEMINI_API_KEY", "").strip()
            result = transcribe_with_gemini(
                audio_path=audio_path,
                api_key=key,
                gemini_model=gemini_model or "gemini-2.5-flash",
                language=language,
                task=task,
                target_language=target_language,
                initial_prompt=initial_prompt,
                progress_cb=progress_cb,
            )
        else:
            result = transcribe_with_whisper(
                audio_path=audio_path,
                mode=mode,
                language=language,
                task=task,
                initial_prompt=initial_prompt,
                vad_filter=vad_filter,
                progress_cb=progress_cb,
            )

        _update_job(
            job_id,
            status="completed",
            progress=100,
            status_message="Completed",
            segments=result["segments"],
            language=result["language"],
            language_probability=result.get("language_probability", 1.0),
            duration=result.get("duration", 0.0),
            elapsed_seconds=result.get("elapsed_seconds", 0.0),
            engine_label=result.get("engine_label", backend),
            error=None,
        )
    except Exception as exc:
        _update_job(
            job_id,
            status="failed",
            status_message=f"Error: {exc}",
            error=str(exc),
        )


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "hardware": detect_hardware(),
        "whisper_modes": WHISPER_MODES,
        "has_env_gemini_key": bool(os.environ.get("GEMINI_API_KEY", "").strip()),
    }


@app.post("/api/transcribe")
async def create_transcription(
    file: UploadFile = File(...),
    backend: str = Form("whisper"),
    mode: str = Form("whale"),
    language: str = Form("auto"),
    task: str = Form("transcribe"),
    target_language: str = Form("English"),
    initial_prompt: str = Form(""),
    vad_filter: bool = Form(True),
    gemini_api_key: str = Form(""),
    gemini_model: str = Form("gemini-2.5-flash"),
) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    job_id = uuid.uuid4().hex[:12]
    ext = Path(file.filename).suffix.lower() or ".mp3"
    saved_audio = UPLOADS_DIR / f"{job_id}{ext}"

    with open(saved_audio, "wb") as out:
        shutil.copyfileobj(file.file, out)

    job = {
        "id": job_id,
        "filename": file.filename,
        "audio_file": saved_audio.name,
        "backend": backend,
        "mode": mode if backend == "whisper" else gemini_model,
        "engine_label": WHISPER_MODES.get(mode, {}).get("label", mode) if backend == "whisper" else f"Gemini ({gemini_model})",
        "language": language,
        "task": task,
        "status": "queued",
        "progress": 0,
        "status_message": "Queued...",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp": time.time(),
        "duration": 0.0,
        "elapsed_seconds": 0.0,
        "segments": [],
        "error": None,
    }

    with _JOBS_LOCK:
        _JOBS[job_id] = job
        _save_job(job)

    worker = threading.Thread(
        target=_run_transcription_job,
        args=(
            job_id,
            str(saved_audio),
            backend,
            mode,
            language,
            task,
            target_language,
            initial_prompt,
            vad_filter,
            gemini_api_key,
            gemini_model,
        ),
        daemon=True,
    )
    worker.start()

    return {"job_id": job_id, "job": job}


@app.get("/api/jobs")
def list_jobs() -> list[dict[str, Any]]:
    with _JOBS_LOCK:
        jobs = sorted(_JOBS.values(), key=lambda j: j.get("timestamp", 0), reverse=True)
        return [
            {
                "id": j["id"],
                "filename": j["filename"],
                "backend": j.get("backend"),
                "engine_label": j.get("engine_label"),
                "language": j.get("language"),
                "status": j.get("status"),
                "progress": j.get("progress", 0),
                "status_message": j.get("status_message", ""),
                "duration": j.get("duration", 0.0),
                "elapsed_seconds": j.get("elapsed_seconds", 0.0),
                "created_at": j.get("created_at"),
                "segment_count": len(j.get("segments", [])),
                "error": j.get("error"),
            }
            for j in jobs
        ]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job


class SegmentsUpdate(BaseModel):
    segments: list[dict[str, Any]]


@app.put("/api/jobs/{job_id}/segments")
def update_segments(job_id: str, payload: SegmentsUpdate) -> dict[str, Any]:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        job["segments"] = payload.segments
        _save_job(job)
        return {"ok": True, "segment_count": len(payload.segments)}


class TranslateRequest(BaseModel):
    target_language: str
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"


@app.post("/api/jobs/{job_id}/translate")
def translate_job(job_id: str, payload: TranslateRequest) -> dict[str, Any]:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        segments = list(job.get("segments", []))

    if not segments:
        raise HTTPException(status_code=400, detail="No transcript segments to translate")

    key = payload.gemini_api_key.strip() or os.environ.get("GEMINI_API_KEY", "").strip()
    try:
        translated = translate_segments_with_gemini(
            segments=segments,
            target_language=payload.target_language,
            api_key=key,
            gemini_model=payload.gemini_model,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    _update_job(job_id, segments=translated, language=f"translated:{payload.target_language}")
    return {"ok": True, "segments": translated}


@app.get("/api/jobs/{job_id}/preview_short")
def preview_short_captions(
    job_id: str,
    max_words: int = Query(3, ge=1, le=12),
    max_chars: int = Query(28, ge=8, le=80),
) -> dict[str, Any]:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        segments = job.get("segments", [])
    cues = build_short_caption_segments(segments, max_words=max_words, max_chars=max_chars)
    return {"cues": cues}


@app.get("/api/jobs/{job_id}/export/{fmt}")
def export_job(
    job_id: str,
    fmt: str,
    max_words: int = Query(3, ge=1, le=12),
    max_chars: int = Query(28, ge=8, le=80),
) -> Response:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        job_copy = dict(job)

    segments = job_copy.get("segments", [])
    stem = _safe_stem(job_copy.get("filename", "transcript"))

    if fmt == "srt":
        content = export_srt(segments, short_mode=False)
        return Response(
            content=content.encode("utf-8"),
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{stem}.srt"'},
        )
    if fmt == "srt_short":
        content = export_srt(segments, short_mode=True, max_words=max_words, max_chars=max_chars)
        return Response(
            content=content.encode("utf-8"),
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{stem}_shorts.srt"'},
        )
    if fmt == "vtt":
        content = export_vtt(segments, short_mode=False)
        return Response(
            content=content.encode("utf-8"),
            media_type="text/vtt; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{stem}.vtt"'},
        )
    if fmt == "vtt_short":
        content = export_vtt(segments, short_mode=True, max_words=max_words, max_chars=max_chars)
        return Response(
            content=content.encode("utf-8"),
            media_type="text/vtt; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{stem}_shorts.vtt"'},
        )
    if fmt == "txt":
        content = export_txt(segments, include_timestamps=False)
        return Response(
            content=content.encode("utf-8"),
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{stem}.txt"'},
        )
    if fmt == "txt_timestamps":
        content = export_txt(segments, include_timestamps=True)
        return Response(
            content=content.encode("utf-8"),
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{stem}_timestamps.txt"'},
        )
    if fmt == "docx":
        data = export_docx(job_copy, include_timestamps=True)
        return Response(
            content=data,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f'attachment; filename="{stem}.docx"'},
        )
    if fmt == "json":
        content = export_json(job_copy)
        return Response(
            content=content.encode("utf-8"),
            media_type="application/json; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{stem}.json"'},
        )

    raise HTTPException(status_code=400, detail=f"Unsupported export format: {fmt}")


@app.get("/api/jobs/{job_id}/audio")
def get_job_audio(job_id: str) -> FileResponse:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        audio_file = job.get("audio_file")

    if not audio_file:
        raise HTTPException(status_code=404, detail="Audio not found")

    path = UPLOADS_DIR / audio_file
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audio file missing on disk")

    media_type, _ = mimetypes.guess_type(str(path))
    return FileResponse(str(path), media_type=media_type or "audio/mpeg")


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict[str, Any]:
    with _JOBS_LOCK:
        job = _JOBS.pop(job_id, None)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

    job_file = JOBS_DIR / f"{job_id}.json"
    if job_file.exists():
        job_file.unlink(missing_ok=True)
    if job.get("audio_file"):
        audio_path = UPLOADS_DIR / job["audio_file"]
        audio_path.unlink(missing_ok=True)

    return {"ok": True}
