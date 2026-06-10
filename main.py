import os
import uuid
import tempfile
import subprocess
import threading
import re
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
import whisper
from deep_translator import GoogleTranslator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Subtitle Generator")

jobs: dict = {}
job_files: dict = {}
loaded_models: dict = {}
model_lock = threading.Lock()

LANGUAGES = {
    "Arabic": "ar",
    "Bengali": "bn",
    "Chinese (Simplified)": "zh-CN",
    "Chinese (Traditional)": "zh-TW",
    "Czech": "cs",
    "Danish": "da",
    "Dutch": "nl",
    "English": "en",
    "Filipino": "tl",
    "Finnish": "fi",
    "French": "fr",
    "German": "de",
    "Greek": "el",
    "Hebrew": "iw",
    "Hindi": "hi",
    "Hungarian": "hu",
    "Indonesian": "id",
    "Italian": "it",
    "Japanese": "ja",
    "Korean": "ko",
    "Malay": "ms",
    "Norwegian": "no",
    "Persian": "fa",
    "Polish": "pl",
    "Portuguese": "pt",
    "Romanian": "ro",
    "Russian": "ru",
    "Spanish": "es",
    "Swedish": "sv",
    "Thai": "th",
    "Turkish": "tr",
    "Ukrainian": "uk",
    "Urdu": "ur",
    "Vietnamese": "vi",
}

WHISPER_LANGUAGES = {
    "Arabic": "ar", "Bengali": "bn", "Chinese (Simplified)": "zh",
    "Chinese (Traditional)": "zh", "Czech": "cs", "Danish": "da",
    "Dutch": "nl", "English": "en", "Filipino": "tl", "Finnish": "fi",
    "French": "fr", "German": "de", "Greek": "el", "Hebrew": "he",
    "Hindi": "hi", "Hungarian": "hu", "Indonesian": "id", "Italian": "it",
    "Japanese": "ja", "Korean": "ko", "Malay": "ms", "Norwegian": "no",
    "Persian": "fa", "Polish": "pl", "Portuguese": "pt", "Romanian": "ro",
    "Russian": "ru", "Spanish": "es", "Swedish": "sv", "Thai": "th",
    "Turkish": "tr", "Ukrainian": "uk", "Urdu": "ur", "Vietnamese": "vi",
}


def get_model(model_size: str):
    with model_lock:
        if model_size not in loaded_models:
            logger.info(f"Loading Whisper model: {model_size}")
            loaded_models[model_size] = whisper.load_model(model_size)
        return loaded_models[model_size]


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def segments_to_srt(segments) -> str:
    blocks = []
    idx = 1
    for seg in segments:
        text = seg["text"].strip()
        if text:
            start = format_timestamp(seg["start"])
            end = format_timestamp(seg["end"])
            blocks.append(f"{idx}\n{start} --> {end}\n{text}")
            idx += 1
    return "\n\n".join(blocks) + "\n" if blocks else ""


def translate_block(args):
    text, lang_code = args
    try:
        result = GoogleTranslator(source="auto", target=lang_code).translate(text)
        return result if result else text
    except Exception:
        return text


def translate_srt(srt_content: str, lang_code: str) -> str:
    raw_blocks = re.split(r"\n\n+", srt_content.strip())
    parsed = []
    for block in raw_blocks:
        lines = block.strip().split("\n")
        if len(lines) >= 3:
            parsed.append((lines[0], lines[1], "\n".join(lines[2:])))

    if not parsed:
        return srt_content

    texts = [p[2] for p in parsed]
    with ThreadPoolExecutor(max_workers=5) as ex:
        translated = list(ex.map(translate_block, [(t, lang_code) for t in texts]))

    result_blocks = [
        f"{idx}\n{ts}\n{tr}"
        for (idx, ts, _), tr in zip(parsed, translated)
    ]
    return "\n\n".join(result_blocks) + "\n"


def job_update(job_id: str, **kwargs):
    jobs[job_id].update(kwargs)


def process_job(job_id: str, video_path: str, output_dir: str,
                source_lang: str, target_langs: list, model_size: str):
    try:
        job_update(job_id, status="extracting_audio", progress=5)

        audio_path = os.path.join(output_dir, "audio.wav")
        proc = subprocess.run(
            ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
             "-ar", "16000", "-ac", "1", audio_path, "-y"],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Audio extraction failed:\n{proc.stderr[-800:]}")

        audio_size = os.path.getsize(audio_path)
        logger.info(f"Audio extracted: {audio_size:,} bytes")
        if audio_size < 4096:
            raise RuntimeError(
                f"Audio file is too small ({audio_size} bytes). "
                "The video may have no audio track, or the audio codec is unsupported."
            )

        job_update(job_id, status="loading_model", progress=15)
        model = get_model(model_size)

        job_update(job_id, status="transcribing", progress=25)
        whisper_lang = WHISPER_LANGUAGES.get(source_lang)
        kwargs = {"language": whisper_lang} if whisper_lang else {}
        # fp16=False is required on CPU/Apple Silicon — without it Whisper may return empty results
        result = model.transcribe(audio_path, fp16=False, **kwargs)

        detected_lang = result.get("language", "unknown")
        segments = result.get("segments", [])
        logger.info(f"Transcription: {len(segments)} segments, detected language={detected_lang}")

        if not segments or not any(s["text"].strip() for s in segments):
            raise RuntimeError(
                f"No speech detected (Whisper detected language: '{detected_lang}'). "
                "Suggestions: select the source language manually, or try a larger model."
            )

        original_srt = segments_to_srt(segments)
        job_update(job_id, status="saving", progress=65, detected_language=detected_lang)

        orig_path = os.path.join(output_dir, "original.srt")
        with open(orig_path, "w", encoding="utf-8") as f:
            f.write(original_srt)

        output_files = {"original": orig_path}

        for idx, lang_name in enumerate(target_langs):
            lang_code = LANGUAGES.get(lang_name, lang_name)
            progress = 65 + int((idx + 1) / max(len(target_langs), 1) * 32)
            job_update(job_id, status="translating", progress=progress, translating_to=lang_name)

            try:
                translated = translate_srt(original_srt, lang_code)
            except Exception as e:
                logger.error(f"Translation to {lang_name} failed: {e}")
                translated = original_srt

            t_path = os.path.join(output_dir, f"{lang_code}.srt")
            with open(t_path, "w", encoding="utf-8") as f:
                f.write(translated)
            output_files[lang_name] = t_path

        job_files[job_id] = output_files
        job_update(job_id, status="completed", progress=100, detected_language=detected_lang)

        for p in [video_path, audio_path]:
            try:
                os.unlink(p)
            except OSError:
                pass

    except Exception as e:
        logger.error(f"Job {job_id} error: {e}")
        job_update(job_id, status="error", error=str(e))
        try:
            os.unlink(video_path)
        except OSError:
            pass


@app.get("/")
async def root():
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/languages")
async def get_languages():
    return {"languages": list(LANGUAGES.keys())}


@app.post("/api/transcribe")
async def start_transcription(
    file: UploadFile = File(...),
    source_language: str = Form("Auto-detect"),
    target_languages: str = Form("English,Korean"),
    model_size: str = Form("medium"),
):
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    output_dir = tempfile.mkdtemp(prefix="subtitle_")
    video_path = os.path.join(output_dir, f"video{suffix}")

    with open(video_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)

    target_langs = [
        l.strip() for l in target_languages.split(",")
        if l.strip() and l.strip() in LANGUAGES
    ]

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "filename": file.filename or "video",
        "target_languages": target_langs,
    }

    t = threading.Thread(
        target=process_job,
        args=(job_id, video_path, output_dir, source_language, target_langs, model_size),
        daemon=True,
    )
    t.start()
    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]


@app.get("/api/download/{job_id}/{lang_key:path}")
async def download_subtitle(job_id: str, lang_key: str):
    if job_id not in job_files:
        raise HTTPException(404, "Job not found or not completed")
    files = job_files[job_id]
    if lang_key not in files:
        raise HTTPException(404, f"Language '{lang_key}' not found")
    file_path = files[lang_key]
    if not os.path.exists(file_path):
        raise HTTPException(404, "File missing from disk")
    fname = "subtitles_original.srt" if lang_key == "original" else f"subtitles_{LANGUAGES.get(lang_key, lang_key)}.srt"
    return FileResponse(file_path, media_type="text/plain; charset=utf-8", filename=fname)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8766))
    uvicorn.run(app, host="0.0.0.0", port=port)
