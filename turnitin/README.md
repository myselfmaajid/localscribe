# VeriScholar — Self-Hosted Academic Plagiarism & AI Detection Studio

> A free, unlimited, privacy-first Turnitin & Textul alternative designed for medical synopses, theses, dissertations, and research papers.

## Features

1. **Multi-Engine Academic-Calibrated AI Detection**
   - **Pretrained Neural Classifier (`RoBERTa`)** using Turnitin-style overlapping sliding prose windows.
   - **Zero-Shot Causal-LM Forensics (`DistilGPT-2`)** measuring real token-level **Perplexity**, **Burstiness**, and **GLTR Top-100 Token Rank Concentration**.
   - **Academic-Aware Stylometry** with a clinical/scientific false-positive guard (automatically excludes cover pages, headings, tables, formulas, and bibliographies).
   - **Optional Gemini 2.5 Flash Forensic Critic** when you supply a Gemini API key.
   - Displays both exact percentages and the **Turnitin-equivalent band** (`*%` for 1%–19% low-confidence detection).

2. **Multi-Source Plagiarism & Paraphrase Detection**
   - **PubMed / Europe PMC** (biomedical & clinical literature)
   - **OpenAlex** (250M+ scholarly works)
   - **CrossRef** (150M+ academic DOIs)
   - **Wikipedia & Open Web (DuckDuckGo)**
   - **Local Reference Library** — drop your own PDFs/DOCXs to check against previous drafts or source papers.
   - Detects both **Exact Verbatim Overlap** and **Paraphrased Semantic Similarity**, with Turnitin-style filters (*Exclude References*, *Exclude Quotes*, *Minimum Match Words*).

3. **Interactive Turnitin-Style Workspace & PDF Reports**
   - Click any highlighted sentence to inspect its AI forensics or side-by-side matched source excerpt.
   - One-click export to **Turnitin-Style PDF Report**, **HTML**, and **JSON**.

## Quick Start

Double-click `start_verischolar.bat`, or run from terminal inside `turnitin/`:

```bash
python -m uvicorn app.server:app --host 127.0.0.1 --port 8010
```

Then open **http://127.0.0.1:8010** in your browser.
