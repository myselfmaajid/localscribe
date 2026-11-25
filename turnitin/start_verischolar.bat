@echo off
title VeriScholar - Academic Plagiarism & AI Detection Studio
cd /d "%~dp0"
echo ============================================================
echo   Starting VeriScholar (Turnitin Alternative)
echo   URL: http://127.0.0.1:8010
echo ============================================================
start "" "http://127.0.0.1:8010"
python -m uvicorn app.server:app --host 127.0.0.1 --port 8010
pause
