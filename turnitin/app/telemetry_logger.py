"""
VeriScholar v7.0 — Structured Diagnostic Database, Human-Readable Error Logger & API Circuit-Breaker.

Guarantees:
  1. Every API error, quota exhaustion (429), tier restriction (403), timeout, or fallback switch is recorded in:
     - SQLite DB : turnitin/app/data/logs/verischolar_diagnostics.db (tables: system_events, provider_health)
     - JSONL File: turnitin/app/data/logs/errors_and_failovers.jsonl
     - Text Log  : turnitin/app/data/logs/diagnostics_readable.log
  2. Automatic Secret Scrubbing: API keys / Bearer tokens are masked before touching disk.
  3. Smart Circuit-Breaker: Temporarily cools down quota-exhausted (429) or unauthorized (401/403) providers
     so VeriScholar fails over in 0ms without stalling user requests, and ALWAYS falls back to the 100% offline
     local base architecture when all external APIs are unavailable.
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

LOGS_DIR = Path(__file__).resolve().parent / "data" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = LOGS_DIR / "verischolar_diagnostics.db"
JSONL_PATH = LOGS_DIR / "errors_and_failovers.jsonl"
READABLE_LOG_PATH = LOGS_DIR / "diagnostics_readable.log"

_LOG_LOCK = threading.Lock()
_COOLDOWN_UNTIL: Dict[str, float] = {}

_SECRET_SCRUB_RE = re.compile(
    r"(?:AIza[0-9A-Za-z\-_]{20,}|gsk_[0-9A-Za-z]{20,}|tvly-[0-9A-Za-z\-_]{15,}|sk-[0-9A-Za-z\-_]{15,}|Bearer\s+[0-9A-Za-z\-_.]{12,})"
)


def _scrub(text: str) -> str:
    if not text:
        return ""
    return _SECRET_SCRUB_RE.sub("[REDACTED_KEY]", str(text))


def _init_db() -> None:
    with _LOG_LOCK:
        conn = sqlite3.connect(str(DB_PATH), timeout=10.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS system_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    level TEXT NOT NULL,
                    subsystem TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    job_id TEXT,
                    status_code INTEGER,
                    latency_ms REAL,
                    error_type TEXT,
                    fallback_used TEXT,
                    message TEXT
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS provider_health (
                    provider TEXT PRIMARY KEY,
                    subsystem TEXT NOT NULL,
                    success_count INTEGER DEFAULT 0,
                    error_count INTEGER DEFAULT 0,
                    last_status TEXT,
                    last_latency_ms REAL,
                    last_error TEXT,
                    cooldown_until_epoch REAL DEFAULT 0,
                    updated_at TEXT
                );
                """
            )
            conn.commit()
        finally:
            conn.close()


_init_db()


def is_provider_available(provider: str) -> bool:
    """Return False if the circuit-breaker placed this provider on temporary quota/auth cooldown."""
    until = _COOLDOWN_UNTIL.get(provider, 0.0)
    return time.time() >= until


def record_provider_success(
    subsystem: str,
    provider: str,
    latency_ms: float,
    job_id: Optional[str] = None,
) -> None:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _COOLDOWN_UNTIL.pop(provider, None)
    with _LOG_LOCK:
        try:
            conn = sqlite3.connect(str(DB_PATH), timeout=5.0)
            try:
                conn.execute(
                    """
                    INSERT INTO provider_health (provider, subsystem, success_count, error_count, last_status, last_latency_ms, last_error, cooldown_until_epoch, updated_at)
                    VALUES (?, ?, 1, 0, 'OK', ?, '', 0, ?)
                    ON CONFLICT(provider) DO UPDATE SET
                        success_count = success_count + 1,
                        last_status = 'OK',
                        last_latency_ms = excluded.last_latency_ms,
                        cooldown_until_epoch = 0,
                        updated_at = excluded.updated_at;
                    """,
                    (provider, subsystem, round(latency_ms, 1), now_str),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception:
            pass


def log_diagnostic_event(
    level: str,  # "ERROR" | "WARN_FAILOVER" | "INFO"
    subsystem: str,  # "humanizer" | "plagiarism" | "ai_detector" | "parser" | "server"
    provider: str,
    message: str,
    job_id: Optional[str] = None,
    status_code: Optional[int] = None,
    latency_ms: Optional[float] = None,
    error_type: str = "runtime_error",
    fallback_used: str = "",
    cooldown_seconds: int = 0,
) -> None:
    """Write a structured, secret-scrubbed event to SQLite, JSONL, and readable log file + set circuit-breaker."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    clean_msg = _scrub(message)[:500]
    cd_epoch = 0.0
    if cooldown_seconds > 0:
        cd_epoch = time.time() + cooldown_seconds
        _COOLDOWN_UNTIL[provider] = cd_epoch

    record = {
        "timestamp": now_str,
        "level": level,
        "subsystem": subsystem,
        "provider": provider,
        "job_id": job_id or "-",
        "status_code": status_code,
        "latency_ms": round(latency_ms, 1) if latency_ms is not None else None,
        "error_type": error_type,
        "fallback_used": fallback_used,
        "cooldown_seconds": cooldown_seconds,
        "message": clean_msg,
    }

    readable_line = (
        f"[{now_str}] [{level}] [{subsystem}] provider={provider}"
        f"{f' status={status_code}' if status_code else ''}"
        f"{f' ({round(latency_ms)}ms)' if latency_ms is not None else ''}"
        f" type={error_type}"
        f"{f' -> FALLBACK={fallback_used}' if fallback_used else ''}"
        f" | job={job_id or '-'} | {clean_msg}\n"
    )

    with _LOG_LOCK:
        try:
            with JSONL_PATH.open("a", encoding="utf-8") as jf:
                jf.write(json.dumps(record, ensure_ascii=False) + "\n")
            with READABLE_LOG_PATH.open("a", encoding="utf-8") as rf:
                rf.write(readable_line)
        except Exception:
            pass

        try:
            conn = sqlite3.connect(str(DB_PATH), timeout=5.0)
            try:
                conn.execute(
                    """
                    INSERT INTO system_events (timestamp, level, subsystem, provider, job_id, status_code, latency_ms, error_type, fallback_used, message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        now_str,
                        level,
                        subsystem,
                        provider,
                        job_id or "",
                        status_code,
                        round(latency_ms, 1) if latency_ms is not None else None,
                        error_type,
                        fallback_used,
                        clean_msg,
                    ),
                )
                if level in ("ERROR", "WARN_FAILOVER"):
                    conn.execute(
                        """
                        INSERT INTO provider_health (provider, subsystem, success_count, error_count, last_status, last_latency_ms, last_error, cooldown_until_epoch, updated_at)
                        VALUES (?, ?, 0, 1, ?, ?, ?, ?, ?)
                        ON CONFLICT(provider) DO UPDATE SET
                            error_count = error_count + 1,
                            last_status = excluded.last_status,
                            last_latency_ms = COALESCE(excluded.last_latency_ms, provider_health.last_latency_ms),
                            last_error = excluded.last_error,
                            cooldown_until_epoch = MAX(provider_health.cooldown_until_epoch, excluded.cooldown_until_epoch),
                            updated_at = excluded.updated_at;
                        """,
                        (
                            provider,
                            subsystem,
                            f"FAIL ({error_type})",
                            round(latency_ms, 1) if latency_ms is not None else None,
                            clean_msg[:240],
                            cd_epoch,
                            now_str,
                        ),
                    )
                conn.commit()
            finally:
                conn.close()
        except Exception:
            pass


def get_diagnostics_summary(limit: int = 80) -> Dict[str, Any]:
    """Return structured provider health + recent events for easy debugging."""
    now_ep = time.time()
    providers: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    with _LOG_LOCK:
        conn = sqlite3.connect(str(DB_PATH), timeout=5.0)
        conn.row_factory = sqlite3.Row
        try:
            for row in conn.execute("SELECT * FROM provider_health ORDER BY subsystem, provider"):
                d = dict(row)
                rem_cd = max(0, int(round((d.get("cooldown_until_epoch") or 0) - now_ep)))
                d["cooldown_remaining_sec"] = rem_cd
                d["circuit_state"] = "COOLDOWN_SKIP" if rem_cd > 0 else "ACTIVE"
                providers.append(d)
            for row in conn.execute("SELECT * FROM system_events ORDER BY id DESC LIMIT ?", (limit,)):
                events.append(dict(row))
        finally:
            conn.close()
    return {
        "database_path": str(DB_PATH),
        "jsonl_log_path": str(JSONL_PATH),
        "readable_log_path": str(READABLE_LOG_PATH),
        "provider_health": providers,
        "recent_events": events,
    }
