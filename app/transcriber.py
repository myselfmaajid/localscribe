"""Dual-backend transcription engine: Local Whisper (faster-whisper) + Gemini AI."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Any, Callable

# Note: faster-whisper uses PyAV (av), which bundles FFmpeg natively without external binaries.


WHISPER_MODES: dict[str, dict[str, str]] = {
    "cheetah": {
        "model_id": "base",
        "label": "Cheetah (Whisper Base)",
        "desc": "Fastest on CPU (~8-15x realtime). Best for quick drafts.",
        "download_size": "~145 MB",
    },
    "dolphin": {
        "model_id": "small",
        "label": "Dolphin (Whisper Small)",
        "desc": "Great balance of speed & accuracy on CPU.",
        "download_size": "~485 MB",
    },
    "whale": {
        "model_id": "deepdml/faster-whisper-large-v3-turbo-ct2",
        "fallback_id": "large-v3-turbo",
        "label": "Whale Turbo (Whisper Large-v3-Turbo)",
        "desc": "Near-flagship 99.8% accuracy, ~4-6x faster than Large-v3. Recommended!",
        "download_size": "~1.6 GB",
    },
    "whale-max": {
        "model_id": "large-v3",
        "label": "Whale Max (Whisper Large-v3)",
        "desc": "Maximum accuracy (TurboScribe's Whale model). Best for difficult audio; slower on CPU.",
        "download_size": "~3.1 GB",
    },
}

_MODEL_CACHE: dict[str, Any] = {}
_MODEL_LOCK = threading.Lock()


_HW_CACHE: dict[str, Any] | None = None


def detect_hardware() -> dict[str, Any]:
    global _HW_CACHE
    if _HW_CACHE is not None:
        return {**_HW_CACHE, "cached_models": list(_MODEL_CACHE.keys())}

    import importlib.util

    cuda = False
    gpu_name = None
    try:
        import ctranslate2
        cuda = ctranslate2.get_cuda_device_count() > 0
        if cuda:
            gpu_name = "NVIDIA CUDA GPU"
    except Exception:
        pass

    has_faster_whisper = importlib.util.find_spec("faster_whisper") is not None
    has_gemini = importlib.util.find_spec("google.genai") is not None

    _HW_CACHE = {
        "cuda": cuda,
        "gpu_name": gpu_name,
        "cpu_threads": os.cpu_count() or 4,
        "device": "cuda" if cuda else "cpu",
        "compute_type": "float16" if cuda else "int8",
        "has_faster_whisper": has_faster_whisper,
        "has_gemini": has_gemini,
    }
    return {**_HW_CACHE, "cached_models": list(_MODEL_CACHE.keys())}


def _get_whisper_model(mode: str, status_cb: Callable[[str, int], None] | None = None):
    from faster_whisper import WhisperModel

    mode_info = WHISPER_MODES.get(mode, WHISPER_MODES["whale"])
    model_id = mode_info["model_id"]
    hw = detect_hardware()
    cache_key = f"{model_id}:{hw['device']}:{hw['compute_type']}"

    with _MODEL_LOCK:
        if cache_key in _MODEL_CACHE:
            return _MODEL_CACHE[cache_key]

        if status_cb:
            status_cb(
                f"Loading {mode_info['label']} (first run downloads {mode_info['download_size']})...",
                5,
            )

        try:
            model = WhisperModel(
                model_id,
                device=hw["device"],
                compute_type=hw["compute_type"],
                cpu_threads=hw["cpu_threads"],
            )
        except Exception:
            fallback = mode_info.get("fallback_id")
            if fallback:
                model = WhisperModel(
                    fallback,
                    device=hw["device"],
                    compute_type=hw["compute_type"],
                    cpu_threads=hw["cpu_threads"],
                )
            else:
                raise

        _MODEL_CACHE[cache_key] = model
        return model


def decode_audio_to_float32(audio_path: str, sample_rate: int = 16000):
    """
    Decode any audio/video file to 16kHz mono float32 numpy array using PyAV.
    Avoids faster-whisper's `metadata_errors="ignore"` kwarg which was removed in PyAV 19+.
    """
    import av
    import numpy as np

    resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)
    chunks: list[np.ndarray] = []

    with av.open(audio_path, mode="r") as container:
        audio_streams = [s for s in container.streams if s.type == "audio"]
        if not audio_streams:
            raise ValueError("No audio stream found in the uploaded file.")
        stream = audio_streams[0]
        for frame in container.decode(stream):
            resampled_frames = resampler.resample(frame)
            if resampled_frames is None:
                continue
            if not isinstance(resampled_frames, list):
                resampled_frames = [resampled_frames]
            for rf in resampled_frames:
                arr = rf.to_ndarray()
                chunks.append(arr.reshape(-1))

        # Flush resampler
        flushed = resampler.resample(None)
        if flushed:
            if not isinstance(flushed, list):
                flushed = [flushed]
            for rf in flushed:
                chunks.append(rf.to_ndarray().reshape(-1))

    if not chunks:
        return np.zeros(0, dtype=np.float32)

    audio_s16 = np.concatenate(chunks, axis=0)
    return audio_s16.astype(np.float32) / 32768.0


def transcribe_with_whisper(
    audio_path: str,
    mode: str = "whale",
    language: str | None = None,
    task: str = "transcribe",
    initial_prompt: str | None = None,
    vad_filter: bool = True,
    progress_cb: Callable[[str, int], None] | None = None,
) -> dict[str, Any]:
    """Transcribe audio locally using faster-whisper with word-level timestamps."""
    t0 = time.time()
    mode_info = WHISPER_MODES.get(mode, WHISPER_MODES["whale"])
    model = _get_whisper_model(mode, progress_cb)

    if progress_cb:
        progress_cb("Decoding audio...", 10)

    audio_array = decode_audio_to_float32(audio_path, sample_rate=16000)

    if progress_cb:
        progress_cb("Transcribing audio locally...", 12)

    lang_arg = None if (not language or language == "auto") else language
    task_arg = "translate" if task == "translate" else "transcribe"

    hw = detect_hardware()
    if hw["cuda"]:
        beam_size = 5
    else:
        beam_size = {"cheetah": 1, "dolphin": 1, "whale": 2, "whale-max": 5}.get(mode, 2)

    kwargs: dict[str, Any] = {
        "language": lang_arg,
        "task": task_arg,
        "beam_size": beam_size,
        "word_timestamps": True,
        "vad_filter": vad_filter,
        "condition_on_previous_text": True,
    }
    if vad_filter:
        kwargs["vad_parameters"] = {"min_silence_duration_ms": 400}
    if initial_prompt and initial_prompt.strip():
        # Strip ElevenLabs stage tags like [thoughtful], [short pause], [whisper]
        cleaned_prompt = re.sub(r"\[[^\]]+\]", " ", initial_prompt)
        cleaned_prompt = re.sub(r"\s+", " ", cleaned_prompt).strip()
        if cleaned_prompt:
            kwargs["initial_prompt"] = cleaned_prompt

    segments_gen, info = model.transcribe(audio_array, **kwargs)
    total_duration = float(getattr(info, "duration", 0.0) or 0.0)
    detected_lang = getattr(info, "language", lang_arg or "unknown")
    lang_prob = float(getattr(info, "language_probability", 1.0) or 1.0)

    segments: list[dict[str, Any]] = []
    for seg in segments_gen:
        words_list = []
        if seg.words:
            for w in seg.words:
                words_list.append({
                    "start": round(float(w.start), 3),
                    "end": round(float(w.end), 3),
                    "word": w.word.strip(),
                    "prob": round(float(getattr(w, "probability", 1.0)), 3),
                })
        segments.append({
            "id": len(segments) + 1,
            "start": round(float(seg.start), 3),
            "end": round(float(seg.end), 3),
            "text": seg.text.strip(),
            "words": words_list,
        })
        if progress_cb and total_duration > 0:
            pct = min(96, 12 + int((float(seg.end) / total_duration) * 84))
            progress_cb(f"Transcribing... ({len(segments)} segments)", pct)

    elapsed = round(time.time() - t0, 2)
    return {
        "segments": segments,
        "language": detected_lang,
        "language_probability": round(lang_prob, 3),
        "duration": round(total_duration, 2),
        "elapsed_seconds": elapsed,
        "engine_label": mode_info["label"],
    }


def _parse_timestamp_to_seconds(ts: Any) -> float:
    if isinstance(ts, (int, float)):
        return max(0.0, float(ts))
    s = str(ts).strip().replace(",", ".")
    parts = s.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(s)
    except Exception:
        return 0.0


def _extract_json_block(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def transcribe_with_gemini(
    audio_path: str,
    api_key: str,
    gemini_model: str = "gemini-2.5-flash",
    language: str | None = None,
    task: str = "transcribe",
    target_language: str | None = None,
    initial_prompt: str | None = None,
    progress_cb: Callable[[str, int], None] | None = None,
) -> dict[str, Any]:
    """Transcribe (and optionally translate) audio using Google Gemini API."""
    from google import genai
    from google.genai import types

    if not api_key or not api_key.strip():
        raise ValueError("Gemini API key is required for Gemini Cloud mode.")

    t0 = time.time()
    client = genai.Client(api_key=api_key.strip())

    if progress_cb:
        progress_cb("Uploading audio to Gemini...", 15)

    uploaded = client.files.upload(file=audio_path)

    try:
        # Wait for file to become ACTIVE if needed
        while getattr(uploaded.state, "name", "") == "PROCESSING":
            time.sleep(1)
            uploaded = client.files.get(name=uploaded.name)

        if progress_cb:
            progress_cb(f"Transcribing with {gemini_model}...", 45)

        lang_hint = f"The spoken language is {language}." if (language and language != "auto") else "Auto-detect the spoken language."
        if task == "translate":
            dest = target_language or "English"
            task_instruction = f"Transcribe and translate the speech accurately into {dest}."
        else:
            task_instruction = "Transcribe the speech verbatim with 100% accuracy in the original spoken language."

        vocab_hint = ""
        if initial_prompt and initial_prompt.strip():
            vocab_hint = f"\nReference script / spelling hints (use for exact spelling of names and terms):\n{initial_prompt.strip()}\n"

        prompt = (
            f"You are a professional subtitle transcriber.\n"
            f"{lang_hint}\n"
            f"{task_instruction}\n"
            f"{vocab_hint}\n"
            "Return ONLY valid JSON with this exact schema:\n"
            '{\n'
            '  "language": "detected ISO language code like en, ur, hi, es",\n'
            '  "segments": [\n'
            '    {"start": 0.0, "end": 3.45, "text": "Exact spoken sentence or subtitle phrase."}\n'
            '  ]\n'
            '}\n'
            "Rules:\n"
            "- Split into natural subtitle segments of 1 to 5 seconds each.\n"
            "- start and end MUST be accurate numbers in seconds (float).\n"
            "- Do not skip any spoken words and do not hallucinate text.\n"
            "- Output raw JSON only, no markdown fences or extra commentary."
        )

        response = client.models.generate_content(
            model=gemini_model,
            contents=[uploaded, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
            ),
        )

        if progress_cb:
            progress_cb("Parsing Gemini timestamps...", 88)

        data = _extract_json_block(response.text)
        raw_segments = data.get("segments", []) if isinstance(data, dict) else data
        detected_lang = data.get("language", language or "auto") if isinstance(data, dict) else (language or "auto")

        segments: list[dict[str, Any]] = []
        max_end = 0.0
        for item in raw_segments:
            st = round(_parse_timestamp_to_seconds(item.get("start", 0.0)), 3)
            en = round(_parse_timestamp_to_seconds(item.get("end", st + 1.0)), 3)
            if en < st:
                en = round(st + 0.5, 3)
            txt = str(item.get("text", "")).strip()
            if not txt:
                continue
            max_end = max(max_end, en)
            segments.append({
                "id": len(segments) + 1,
                "start": st,
                "end": en,
                "text": txt,
                "words": [],
            })

        elapsed = round(time.time() - t0, 2)
        return {
            "segments": segments,
            "language": detected_lang,
            "language_probability": 1.0,
            "duration": round(max_end, 2),
            "elapsed_seconds": elapsed,
            "engine_label": f"Gemini Cloud ({gemini_model})",
        }
    finally:
        try:
            client.files.delete(name=uploaded.name)
        except Exception:
            pass


def translate_segments_with_gemini(
    segments: list[dict[str, Any]],
    target_language: str,
    api_key: str,
    gemini_model: str = "gemini-2.5-flash",
) -> list[dict[str, Any]]:
    """Translate existing subtitle segments into any language while preserving exact timestamps."""
    from google import genai
    from google.genai import types

    if not api_key or not api_key.strip():
        raise ValueError("Gemini API key is required for translation.")

    client = genai.Client(api_key=api_key.strip())
    compact = [{"id": s.get("id", i + 1), "text": s.get("text", "")} for i, s in enumerate(segments)]

    prompt = (
        f"Translate each subtitle entry's 'text' naturally and accurately into {target_language}.\n"
        "Preserve the exact 'id' values and return ONLY a JSON array of objects: "
        '[{"id": 1, "text": "..."}].\n'
        f"Input:\n{json.dumps(compact, ensure_ascii=False)}"
    )

    response = client.models.generate_content(
        model=gemini_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1,
        ),
    )

    translated_list = _extract_json_block(response.text)
    by_id = {int(item["id"]): str(item["text"]).strip() for item in translated_list}

    out: list[dict[str, Any]] = []
    for i, s in enumerate(segments):
        sid = int(s.get("id", i + 1))
        new_seg = dict(s)
        if sid in by_id:
            new_seg["text"] = by_id[sid]
            # Clear word-level timings because word count changes after translation
            new_seg["words"] = []
        out.append(new_seg)
    return out
