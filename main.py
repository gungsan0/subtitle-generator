import os
import uuid
import shutil
import subprocess
import threading
import re
import logging
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse
import whisper
from deep_translator import GoogleTranslator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Subtitle Generator")

_DEFAULT_TEMP = "/Volumes/Samsung T7/Agent/Subtitle Generator/Temp"
TEMP_BASE = Path(
    os.environ.get("SUBTITLE_TEMP_DIR")
    or (_DEFAULT_TEMP if os.path.isdir(os.path.dirname(_DEFAULT_TEMP)) else
        str(Path.home() / ".subtitle-generator" / "temp"))
)

jobs: dict = {}
job_files: dict = {}
loaded_models: dict = {}
model_lock = threading.Lock()


def make_job_dir(job_id: str) -> str:
    job_dir = TEMP_BASE / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    return str(job_dir)

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


def cleanup_after_download(job_id: str, lang_key: str, file_path: str):
    """Delete the served file, then remove the temp dir when all files are gone."""
    try:
        os.unlink(file_path)
    except OSError:
        pass
    if job_id not in job_files:
        return
    job_files[job_id].pop(lang_key, None)
    # _video counts as remaining — don't wipe the dir until video is also downloaded
    remaining = [k for k in job_files[job_id] if k != "_dir"]
    if not remaining:
        output_dir = job_files[job_id].get("_dir")
        if output_dir:
            shutil.rmtree(output_dir, ignore_errors=True)
        job_files.pop(job_id, None)
        jobs.pop(job_id, None)


def _extract_audio(input_path: str, output_dir: str) -> str:
    """Convert any video/audio file to 16kHz mono WAV. Returns wav path."""
    audio_path = os.path.join(output_dir, "audio.wav")
    proc = subprocess.run(
        ["ffmpeg", "-i", input_path, "-vn", "-acodec", "pcm_s16le",
         "-ar", "16000", "-ac", "1", audio_path, "-y"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Audio extraction failed:\n{proc.stderr[-800:]}")
    audio_size = os.path.getsize(audio_path)
    logger.info(f"Audio extracted: {audio_size:,} bytes")
    if audio_size < 4096:
        raise RuntimeError(
            f"Audio file too small ({audio_size} bytes). "
            "The video may have no audio track, or the codec is unsupported."
        )
    return audio_path


def _transcribe_and_translate(job_id: str, audio_path: str, output_dir: str,
                               source_lang: str, target_langs: list,
                               model_size: str, progress_base: int = 15):
    """Shared transcription + translation pipeline."""
    job_update(job_id, status="loading_model", progress=progress_base)
    model = get_model(model_size)

    job_update(job_id, status="transcribing", progress=progress_base + 12)
    whisper_lang = WHISPER_LANGUAGES.get(source_lang)
    kwargs = {"language": whisper_lang} if whisper_lang else {}
    result = model.transcribe(audio_path, fp16=False, **kwargs)

    detected_lang = result.get("language", "unknown")
    segments = result.get("segments", [])
    logger.info(f"Transcription: {len(segments)} segments, language={detected_lang}")

    if not segments or not any(s["text"].strip() for s in segments):
        raise RuntimeError(
            f"No speech detected (Whisper detected language: '{detected_lang}'). "
            "Try selecting the source language manually, or use a larger model."
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

    output_files["_dir"] = output_dir
    job_files[job_id] = output_files
    job_update(job_id, status="completed", progress=100, detected_language=detected_lang)

    try:
        os.unlink(audio_path)
    except OSError:
        pass


def process_job(job_id: str, video_path: str, output_dir: str,
                source_lang: str, target_langs: list, model_size: str):
    try:
        job_update(job_id, status="extracting_audio", progress=5)
        audio_path = _extract_audio(video_path, output_dir)
        try:
            os.unlink(video_path)
        except OSError:
            pass
        _transcribe_and_translate(job_id, audio_path, output_dir,
                                   source_lang, target_langs, model_size,
                                   progress_base=15)
    except Exception as e:
        logger.error(f"Job {job_id} error: {e}")
        job_update(job_id, status="error", error=str(e))


def process_youtube_job(job_id: str, url: str, output_dir: str,
                         source_lang: str, target_langs: list, model_size: str):
    try:
        job_update(job_id, status="downloading", progress=5)

        # Fetch title
        title_proc = subprocess.run(
            ["yt-dlp", "--print", "title", "--no-playlist", url],
            capture_output=True, text=True, timeout=30,
        )
        title = title_proc.stdout.strip().split("\n")[0] if title_proc.stdout.strip() else url
        jobs[job_id]["filename"] = title
        logger.info(f"YouTube title: {title}")

        job_update(job_id, status="downloading", progress=10)

        video_path = os.path.join(output_dir, "video.mp4")
        video_err = [None]

        def _dl_video():
            proc = subprocess.run(
                ["yt-dlp",
                 "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best",
                 "--merge-output-format", "mp4",
                 "-o", video_path,
                 "--no-playlist", "--no-progress", "--no-warnings", url],
                capture_output=True, text=True, timeout=7200,
            )
            if proc.returncode != 0:
                video_err[0] = proc.stderr[-400:]
                logger.warning(f"Video download failed for {job_id}: {video_err[0]}")

        # Start full video download in background so it runs in parallel with transcription
        video_thread = threading.Thread(target=_dl_video, daemon=True)
        video_thread.start()

        # Download audio track for Whisper
        audio_template = os.path.join(output_dir, "yt_audio.%(ext)s")
        dl_proc = subprocess.run(
            ["yt-dlp", "-f", "bestaudio/best", "-o", audio_template,
             "--no-playlist", "--no-progress", "--no-warnings", url],
            capture_output=True, text=True, timeout=600,
        )
        if dl_proc.returncode != 0:
            raise RuntimeError(f"YouTube audio download failed:\n{dl_proc.stderr[-600:]}")

        downloaded = sorted(f for f in os.listdir(output_dir) if f.startswith("yt_audio."))
        if not downloaded:
            raise RuntimeError("Audio file not found after download.")

        yt_path = os.path.join(output_dir, downloaded[0])
        logger.info(f"Audio downloaded: {yt_path} ({os.path.getsize(yt_path):,} bytes)")

        job_update(job_id, status="extracting_audio", progress=35)
        audio_path = _extract_audio(yt_path, output_dir)
        try:
            os.unlink(yt_path)
        except OSError:
            pass

        # Transcribe + translate (video download continues in background)
        _transcribe_and_translate(job_id, audio_path, output_dir,
                                   source_lang, target_langs, model_size,
                                   progress_base=42)

        # Wait for video download to finish (subtitles are already done)
        video_thread.join(timeout=7200)

        if not video_err[0] and os.path.exists(video_path):
            job_files[job_id]["_video"] = video_path
            job_update(job_id, video_ready=True)
        else:
            job_update(job_id, video_ready=False)

    except Exception as e:
        logger.error(f"YouTube job {job_id} error: {e}")
        job_update(job_id, status="error", error=str(e))


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
    job_id = str(uuid.uuid4())
    output_dir = make_job_dir(job_id)
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    video_path = os.path.join(output_dir, f"video{suffix}")

    with open(video_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)

    target_langs = [
        l.strip() for l in target_languages.split(",")
        if l.strip() and l.strip() in LANGUAGES
    ]
    jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "filename": file.filename or "video",
        "target_languages": target_langs,
        "source": "file",
    }

    threading.Thread(
        target=process_job,
        args=(job_id, video_path, output_dir, source_language, target_langs, model_size),
        daemon=True,
    ).start()
    return {"job_id": job_id}


@app.post("/api/transcribe-url")
async def transcribe_from_url(
    youtube_url: str = Form(...),
    source_language: str = Form("Auto-detect"),
    target_languages: str = Form("English,Korean"),
    model_size: str = Form("medium"),
):
    target_langs = [
        l.strip() for l in target_languages.split(",")
        if l.strip() and l.strip() in LANGUAGES
    ]

    job_id = str(uuid.uuid4())
    output_dir = make_job_dir(job_id)

    jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "filename": youtube_url,
        "target_languages": target_langs,
        "source": "youtube",
    }

    threading.Thread(
        target=process_youtube_job,
        args=(job_id, youtube_url, output_dir, source_language, target_langs, model_size),
        daemon=True,
    ).start()
    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]


@app.get("/api/download/{job_id}/{lang_key:path}")
async def download_subtitle(job_id: str, lang_key: str, background_tasks: BackgroundTasks):
    from fastapi.responses import Response as RawResponse
    if job_id not in job_files:
        raise HTTPException(404, "Job not found or not completed")
    files = job_files[job_id]
    if lang_key.startswith("_") or lang_key not in files:
        raise HTTPException(404, f"Language '{lang_key}' not found")
    file_path = files[lang_key]
    if not os.path.exists(file_path):
        raise HTTPException(404, "File missing from disk")
    fname = "subtitles_original.srt" if lang_key == "original" else f"subtitles_{LANGUAGES.get(lang_key, lang_key)}.srt"
    # Read into memory first so the background cleanup cannot race with streaming
    content = Path(file_path).read_bytes()
    background_tasks.add_task(cleanup_after_download, job_id, lang_key, file_path)
    return RawResponse(
        content=content,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@app.get("/api/download-video/{job_id}")
async def download_video(job_id: str, background_tasks: BackgroundTasks):
    if job_id not in job_files:
        raise HTTPException(404, "Job not found")
    video_path = job_files[job_id].get("_video")
    if not video_path:
        raise HTTPException(404, "Video not available for this job")
    if not os.path.exists(video_path):
        raise HTTPException(404, "Video file missing from disk")
    title = jobs.get(job_id, {}).get("filename", "video")
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', title).strip()[:80] or "video"
    fname = f"{safe}.mp4"
    background_tasks.add_task(cleanup_after_download, job_id, "_video", video_path)
    return FileResponse(video_path, media_type="video/mp4", filename=fname)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8766)
