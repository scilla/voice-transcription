from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel
from yt_dlp import DownloadError, YoutubeDL

import speech
from cill.shared import (
    classify_youtube_live_status,
    extract_youtube_video_id,
    is_youtube_url,
    sanitize_filename_component,
)
from cill.storage import (
    WEB_SUMMARY_FILENAME,
    WEB_TRANSCRIPT_FILENAME,
    create_storage_backend,
)


SUPPORTED_REQUEST_SUFFIXES = (".cill.app", ".localhost")
SUPPORTED_SOURCE_HOSTS = {"youtube.com", "www.youtube.com"}
UNSUPPORTED_STATUSES = {"unsupported_live", "unsupported_duration", "unsupported_size", "error"}
TERMINAL_STATUSES = {"complete", "cache_hit", *UNSUPPORTED_STATUSES}
ACTIVE_STATUSES = {"downloading", "transcribing", "summarizing"}
MAX_VIDEO_SECONDS = int(os.getenv("CILL_MAX_VIDEO_SECONDS", str(25 * 60)))
MAX_AUDIO_BYTES = int(os.getenv("CILL_MAX_AUDIO_BYTES", str(20 * 1024 * 1024)))
SUPPORTED_WEB_AUDIO_EXTENSIONS = {".m4a", ".mp3", ".webm", ".wav", ".ogg", ".aac", ".mpeg", ".mpga"}
YTDLP_WEB_FORMAT = (
    "bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio[ext=webm]/"
    "bestaudio[ext=ogg]/bestaudio[acodec!=none]/bestaudio"
)
FAVICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" role="img" aria-label="cill app icon">
  <defs>
    <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#111827" />
      <stop offset="100%" stop-color="#0f766e" />
    </linearGradient>
  </defs>
  <rect width="64" height="64" rx="16" fill="url(#bg)" />
  <path d="M22 20a12 12 0 1 0 0 24h6v-8h-6a4 4 0 1 1 0-8h14v-8H22Z" fill="#f8fafc" />
  <path d="M42 20h-8v24h8a12 12 0 1 0 0-24Zm0 16h-2v-8h2a4 4 0 1 1 0 8Z" fill="#fbbf24" />
</svg>
"""

app = FastAPI(title="cill.app transcript viewer")
storage = create_storage_backend()


class JobCreateRequest(BaseModel):
    source_url: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def strip_port(host: str) -> str:
    return host.split(":", 1)[0].lower()


def reconstruct_source_url(host: str, path: str, query: str) -> Optional[str]:
    hostname = strip_port(host)
    suffix = next((candidate for candidate in SUPPORTED_REQUEST_SUFFIXES if hostname.endswith(candidate)), None)
    if not suffix:
        return None

    original_host = hostname[: -len(suffix)] + ".com"
    if original_host not in SUPPORTED_SOURCE_HOSTS:
        raise ValueError("Only youtube.cill.app and www.youtube.cill.app are supported in v1.")

    if path == "/" and not query:
        return None

    source_url = f"https://{original_host}{path}"
    if query:
        source_url += f"?{query}"
    return source_url


def make_cache_key(video_id: str) -> str:
    return f"youtube:{video_id}:plain:auto"


def make_job_id(cache_key: str) -> str:
    return hashlib.sha1(cache_key.encode("utf-8")).hexdigest()[:16]


def create_job_state(source_url: str, metadata: dict[str, Any], status: str) -> dict[str, Any]:
    video_id = metadata.get("id") or "unknown"
    cache_key = make_cache_key(video_id)
    return {
        "job_id": make_job_id(cache_key),
        "cache_key": cache_key,
        "source_url": source_url,
        "video_id": video_id,
        "title": metadata.get("title") or "Unknown",
        "uploader": metadata.get("channel") or metadata.get("uploader") or "Unknown",
        "status": status,
        "error": None,
        "transcript_ready": False,
        "summary_ready": False,
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
    }


def create_error_state(source_url: str, error_message: str) -> dict[str, Any]:
    video_id = extract_youtube_video_id(source_url) or "unknown"
    cache_key = make_cache_key(video_id)
    return {
        "job_id": make_job_id(cache_key),
        "cache_key": cache_key,
        "source_url": source_url,
        "video_id": video_id,
        "title": "Unknown",
        "uploader": "Unknown",
        "status": "error",
        "error": error_message,
        "transcript_ready": False,
        "summary_ready": False,
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
    }


def update_state(state: dict[str, Any], *, status: Optional[str] = None, error: Optional[str] = None) -> dict[str, Any]:
    if status is not None:
        state["status"] = status
    state["error"] = error
    state["updated_at"] = utc_now_iso()
    return state


def parse_duration_seconds(metadata: dict[str, Any]) -> Optional[float]:
    value = metadata.get("duration")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_source_context(state: dict[str, Any], audio_path: str) -> dict[str, Any]:
    return {
        "source_type": "youtube",
        "selected_source_path": audio_path,
        "youtube_url": state["source_url"],
        "youtube_title": state["title"],
        "youtube_uploader": state["uploader"],
        "youtube_video_id": state["video_id"],
        "youtube_live_status": "not_live",
        "youtube_live_capture_outcome": None,
    }


def probe_youtube_metadata(source_url: str) -> dict[str, Any]:
    options = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "skip_download": True,
        "cachedir": False,
    }
    with YoutubeDL(options) as ydl:
        return ydl.extract_info(source_url, download=False)


def find_downloaded_file(download_dir: str, video_id: str) -> Optional[str]:
    marker = f"[{video_id}]"
    matches: list[tuple[float, str]] = []
    for candidate in Path(download_dir).glob("*"):
        if marker not in candidate.name or not candidate.is_file():
            continue
        matches.append((candidate.stat().st_mtime, str(candidate)))
    if not matches:
        return None
    matches.sort(key=lambda item: item[0], reverse=True)
    return matches[0][1]


def download_audio(source_url: str, state: dict[str, Any]) -> tuple[str, bool]:
    cached_audio = storage.find_cached_audio(state["video_id"])
    if cached_audio:
        return cached_audio, False

    temp_dir = tempfile.mkdtemp(prefix=f"cill_{state['video_id']}_")
    output_template = os.path.join(
        temp_dir,
        f"{sanitize_filename_component(state['title'])} [{state['video_id']}].%(ext)s",
    )
    options = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "cachedir": False,
        "format": YTDLP_WEB_FORMAT,
        "outtmpl": output_template,
        "overwrites": True,
    }

    try:
        with YoutubeDL(options) as ydl:
            info = ydl.extract_info(source_url, download=True)
    except DownloadError as exc:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(f"Unable to download YouTube audio: {exc}") from exc

    requested = info.get("requested_downloads", [])
    for item in requested:
        filepath = item.get("filepath")
        if filepath and os.path.exists(filepath):
            return filepath, True

    fallback = find_downloaded_file(temp_dir, state["video_id"])
    if fallback:
        return fallback, True

    shutil.rmtree(temp_dir, ignore_errors=True)
    raise RuntimeError("yt_dlp completed but no downloadable audio file was found")


def hydrate_state(state: dict[str, Any]) -> dict[str, Any]:
    video_id = state.get("video_id")
    transcript_text = storage.read_text(state["job_id"], WEB_TRANSCRIPT_FILENAME, video_id=video_id)
    summary_text = storage.read_text(state["job_id"], WEB_SUMMARY_FILENAME, video_id=video_id)

    hydrated = dict(state)
    hydrated["transcript"] = transcript_text
    hydrated["summary"] = summary_text
    hydrated["transcript_ready"] = bool(transcript_text)
    hydrated["summary_ready"] = bool(summary_text)

    if hydrated["status"] == "idle" and hydrated["transcript_ready"] and hydrated["summary_ready"]:
        hydrated["status"] = "cache_hit"
    return hydrated


def persist_state(state: dict[str, Any]) -> dict[str, Any]:
    storage.save_state(state["job_id"], state)
    return state


def process_job(state: dict[str, Any]) -> dict[str, Any]:
    hydrated = hydrate_state(state)
    transcript_text = hydrated.get("transcript")
    summary_text = hydrated.get("summary")

    if hydrated["status"] in ACTIVE_STATUSES:
        return hydrated
    if hydrated["status"] in UNSUPPORTED_STATUSES:
        return hydrated
    if transcript_text and summary_text:
        hydrated["status"] = "cache_hit"
        return persist_state(update_state(hydrated, status="cache_hit"))

    audio_path: Optional[str] = None
    cleanup_temp_dir = False

    try:
        if not transcript_text:
            persist_state(update_state(hydrated, status="downloading"))
            audio_path, cleanup_temp_dir = download_audio(hydrated["source_url"], hydrated)
            extension = Path(audio_path).suffix.lower()
            if extension not in SUPPORTED_WEB_AUDIO_EXTENSIONS:
                persist_state(update_state(hydrated, status="error", error=f"Unsupported audio format: {extension}"))
                return hydrate_state(hydrated)

            if os.path.getsize(audio_path) > MAX_AUDIO_BYTES:
                persist_state(
                    update_state(
                        hydrated,
                        status="unsupported_size",
                        error=(
                            f"Downloaded audio exceeds the v1 size cap of {MAX_AUDIO_BYTES // (1024 * 1024)} MB."
                        ),
                    )
                )
                return hydrate_state(hydrated)

            persist_state(update_state(hydrated, status="transcribing"))
            _, _, transcript_text = speech.transcribe_audio_file(audio_path, None, diarize=False)
            storage.write_text(hydrated["job_id"], WEB_TRANSCRIPT_FILENAME, transcript_text)
            hydrated["transcript_ready"] = True
            persist_state(update_state(hydrated, status="summarizing"))

        if not summary_text:
            if not audio_path:
                audio_path = storage.find_cached_audio(hydrated["video_id"]) or hydrated.get("selected_source_path") or ""
            persist_state(update_state(hydrated, status="summarizing"))
            source_context = build_source_context(hydrated, audio_path or hydrated["video_id"])
            summary_text = speech.summarize_transcript(transcript_text, source_context)
            storage.write_text(hydrated["job_id"], WEB_SUMMARY_FILENAME, summary_text)
            hydrated["summary_ready"] = True

        persist_state(update_state(hydrated, status="complete"))
        return hydrate_state(hydrated)
    except Exception as exc:  # pragma: no cover - network dependent
        persist_state(update_state(hydrated, status="error", error=str(exc)))
        return hydrate_state(hydrated)
    finally:
        if cleanup_temp_dir and audio_path:
            shutil.rmtree(str(Path(audio_path).parent), ignore_errors=True)


def create_or_reuse_job(source_url: str) -> dict[str, Any]:
    parsed = urlparse(source_url)
    if parsed.hostname not in SUPPORTED_SOURCE_HOSTS:
        raise HTTPException(status_code=400, detail="Only youtube.com URLs are supported in v1.")

    video_id = extract_youtube_video_id(source_url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Unable to extract a YouTube video ID from the URL.")

    cache_key = make_cache_key(video_id)
    job_id = make_job_id(cache_key)
    existing_state = storage.load_state(job_id)
    if existing_state:
        return hydrate_state(existing_state)

    cached_transcript = storage.read_text(job_id, WEB_TRANSCRIPT_FILENAME, video_id=video_id)
    cached_summary = storage.read_text(job_id, WEB_SUMMARY_FILENAME, video_id=video_id)
    if cached_transcript or cached_summary:
        cached_state = {
            "job_id": job_id,
            "cache_key": cache_key,
            "source_url": source_url,
            "video_id": video_id,
            "title": "Unknown",
            "uploader": "Unknown",
            "status": "cache_hit" if cached_transcript and cached_summary else "idle",
            "error": None,
            "transcript_ready": bool(cached_transcript),
            "summary_ready": bool(cached_summary),
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
        }
        return hydrate_state(persist_state(cached_state))

    try:
        metadata = probe_youtube_metadata(source_url)
    except Exception:
        fallback_state = create_error_state(
            source_url,
            (
                "The deployment could not access YouTube metadata. "
                "YouTube is blocking automated requests from the current runtime."
            ),
        )
        return hydrate_state(persist_state(fallback_state))

    live_status = classify_youtube_live_status(metadata)
    base_state = create_job_state(source_url, metadata, status="idle")

    if live_status != "not_live":
        base_state["status"] = "unsupported_live"
        base_state["error"] = "Live and archived live videos are not supported in the web app v1."
        return hydrate_state(persist_state(base_state))

    duration_seconds = parse_duration_seconds(metadata)
    if duration_seconds is None or duration_seconds > MAX_VIDEO_SECONDS:
        base_state["status"] = "unsupported_duration"
        base_state["error"] = (
            f"Video duration must be 25 minutes or less for the web app v1."
        )
        return hydrate_state(persist_state(base_state))

    hydrated = hydrate_state(base_state)
    if hydrated["transcript_ready"] and hydrated["summary_ready"]:
        hydrated["status"] = "cache_hit"
    return hydrate_state(persist_state(hydrated))


def render_instructions_page(message: Optional[str] = None) -> str:
    details = (
        f"<p class='notice'>{message}</p>" if message else ""
    )
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>cill.app</title>
    <link rel="icon" href="/favicon.svg" type="image/svg+xml" />
    <style>
      body {{ font-family: system-ui, sans-serif; margin: 3rem auto; max-width: 48rem; padding: 0 1rem; line-height: 1.5; }}
      code {{ background: #f3f4f6; padding: 0.15rem 0.35rem; border-radius: 0.35rem; }}
      .notice {{ background: #fff7ed; border: 1px solid #fdba74; padding: 0.75rem 1rem; border-radius: 0.5rem; }}
    </style>
  </head>
  <body>
    <h1>cill.app transcript viewer</h1>
    {details}
    <p>Replace <code>.com</code> with <code>.cill.app</code> on a YouTube URL.</p>
    <p>Example:</p>
    <pre>https://youtube.cill.app/watch?v=zFcyWFK1q8I</pre>
    <p>For local testing, use:</p>
    <pre>http://youtube.localhost:8000/watch?v=zFcyWFK1q8I</pre>
  </body>
</html>"""


def render_processing_page(source_url: str) -> str:
    serialized_url = json.dumps(source_url)
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>cill.app</title>
    <link rel="icon" href="/favicon.svg" type="image/svg+xml" />
    <style>
      body {{ font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 60rem; padding: 0 1rem 4rem; line-height: 1.5; }}
      h1, h2 {{ margin-bottom: 0.5rem; }}
      .meta {{ color: #4b5563; margin-bottom: 1rem; }}
      .status {{ background: #eff6ff; border: 1px solid #93c5fd; padding: 0.75rem 1rem; border-radius: 0.5rem; margin-bottom: 1rem; }}
      .error {{ background: #fef2f2; border-color: #fca5a5; }}
      pre {{ white-space: pre-wrap; background: #f8fafc; padding: 1rem; border-radius: 0.5rem; border: 1px solid #e5e7eb; }}
      section {{ margin-top: 1.5rem; }}
    </style>
  </head>
  <body>
    <h1>cill.app</h1>
    <p class="meta">Source URL: <a id="source-link" href={serialized_url}></a></p>
    <div id="status" class="status">Creating job…</div>
    <section id="transcript-section" hidden>
      <h2>Transcript</h2>
      <pre id="transcript"></pre>
    </section>
    <section id="summary-section" hidden>
      <h2>Summary</h2>
      <pre id="summary"></pre>
    </section>
    <script>
      const sourceUrl = {serialized_url};
      const sourceLink = document.getElementById('source-link');
      const statusEl = document.getElementById('status');
      const transcriptSection = document.getElementById('transcript-section');
      const summarySection = document.getElementById('summary-section');
      const transcriptEl = document.getElementById('transcript');
      const summaryEl = document.getElementById('summary');
      sourceLink.textContent = sourceUrl;

      const terminalStates = new Set(['complete', 'cache_hit', 'unsupported_live', 'unsupported_duration', 'unsupported_size', 'error']);

      function renderState(state) {{
        const title = state.title ? `${{state.title}}` : 'Untitled video';
        const uploader = state.uploader ? ` by ${{state.uploader}}` : '';
        const detail = state.error ? `: ${{state.error}}` : '';
        statusEl.textContent = `${{state.status}} — ${{title}}${{uploader}}${{detail}}`;
        statusEl.className = 'status' + (state.error ? ' error' : '');

        if (state.transcript) {{
          transcriptSection.hidden = false;
          transcriptEl.textContent = state.transcript;
        }}
        if (state.summary) {{
          summarySection.hidden = false;
          summaryEl.textContent = state.summary;
        }}
      }}

      async function fetchState(jobId) {{
        const response = await fetch(`/api/jobs/${{jobId}}`);
        if (!response.ok) {{
          throw new Error(`Unable to load job state (${{response.status}})`);
        }}
        return response.json();
      }}

      async function poll(jobId) {{
        while (true) {{
          const state = await fetchState(jobId);
          renderState(state);
          if (terminalStates.has(state.status) && (state.summary || state.error || state.status === 'cache_hit')) {{
            return;
          }}
          await new Promise((resolve) => setTimeout(resolve, 1500));
        }}
      }}

      async function start() {{
        const createResponse = await fetch('/api/jobs', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ source_url: sourceUrl }}),
        }});

        if (!createResponse.ok) {{
          const errorBody = await createResponse.text();
          statusEl.textContent = errorBody || 'Unable to create job.';
          statusEl.className = 'status error';
          return;
        }}

        const state = await createResponse.json();
        renderState(state);

        if (terminalStates.has(state.status)) {{
          return;
        }}

        fetch(`/api/jobs/${{state.job_id}}/run`, {{ method: 'POST' }}).catch((error) => {{
          statusEl.textContent = `error — ${{error.message}}`;
          statusEl.className = 'status error';
        }});

        await poll(state.job_id);
      }}

      start().catch((error) => {{
        statusEl.textContent = `error — ${{error.message}}`;
        statusEl.className = 'status error';
      }});
    </script>
  </body>
</html>"""


@app.post("/api/jobs")
def create_job(payload: JobCreateRequest) -> JSONResponse:
    if not is_youtube_url(payload.source_url):
        raise HTTPException(status_code=400, detail="Only YouTube URLs are supported.")

    try:
        state = create_or_reuse_job(payload.source_url)
    except Exception:
        state = create_error_state(
            payload.source_url,
            (
                "The deployment could not access YouTube metadata. "
                "YouTube is blocking automated requests from the current runtime."
            ),
        )
    return JSONResponse(hydrate_state(state))


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> JSONResponse:
    state = storage.load_state(job_id)
    if not state:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JSONResponse(hydrate_state(state))


@app.post("/api/jobs/{job_id}/run")
def run_job(job_id: str) -> JSONResponse:
    state = storage.load_state(job_id)
    if not state:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JSONResponse(process_job(state))


@app.get("/", response_class=HTMLResponse)
def root(request: Request) -> HTMLResponse:
    try:
        source_url = reconstruct_source_url(
            request.headers.get("host", ""),
            request.url.path,
            request.url.query,
        )
    except ValueError as exc:
        return HTMLResponse(render_instructions_page(str(exc)))

    if not source_url:
        return HTMLResponse(render_instructions_page())
    return HTMLResponse(render_processing_page(source_url))


@app.get("/favicon.ico")
def favicon() -> Response:
    return Response(content=FAVICON_SVG, media_type="image/svg+xml")


@app.get("/favicon.svg")
def favicon_svg() -> Response:
    return Response(content=FAVICON_SVG, media_type="image/svg+xml")


@app.get("/{full_path:path}", response_class=HTMLResponse)
def catch_all(full_path: str, request: Request) -> HTMLResponse:
    try:
        source_url = reconstruct_source_url(
            request.headers.get("host", ""),
            "/" + full_path,
            request.url.query,
        )
    except ValueError as exc:
        return HTMLResponse(render_instructions_page(str(exc)))

    if not source_url:
        return HTMLResponse(render_instructions_page())
    return HTMLResponse(render_processing_page(source_url))
