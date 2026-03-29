from __future__ import annotations

import copy
import concurrent.futures
import hashlib
import json
import os
import shutil
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel
from yt_dlp import DownloadError, YoutubeDL

import speech
from cill import rapidapi_youtube
from cill.shared import (
    classify_youtube_live_status,
    extract_youtube_video_id,
    is_youtube_url,
    sanitize_filename_component,
)
from cill.storage import (
    WEB_DIARIZED_SUMMARY_FILENAME,
    WEB_DIARIZED_TRANSCRIPT_FILENAME,
    WEB_RAPIDAPI_SUBTITLE_METADATA_FILENAME,
    WEB_RAPIDAPI_SUBTITLE_TEXT_FILENAME,
    WEB_SUMMARY_FILENAME,
    WEB_TRANSCRIPT_FILENAME,
    create_storage_backend,
)


SUPPORTED_REQUEST_SUFFIXES = (".cill.app", ".localhost")
SUPPORTED_SOURCE_HOSTS = {"youtube.com", "www.youtube.com"}
UNSUPPORTED_STATUSES = {"unsupported_live", "unsupported_duration", "unsupported_size", "error"}
TERMINAL_STATUSES = {"complete", "cache_hit", *UNSUPPORTED_STATUSES}
ACTIVE_STATUSES = {"downloading", "transcribing", "summarizing"}
QUEUED_STATUSES = {"queued"}
MAX_VIDEO_SECONDS = int(os.getenv("CILL_MAX_VIDEO_SECONDS", str(25 * 60)))
MAX_AUDIO_BYTES = int(os.getenv("CILL_MAX_AUDIO_BYTES", str(20 * 1024 * 1024)))
SUPPORTED_WEB_AUDIO_EXTENSIONS = {".m4a", ".mp3", ".webm", ".wav", ".ogg", ".aac", ".mpeg", ".mpga"}
YTDLP_WEB_FORMAT = (
    "bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio[ext=webm]/"
    "bestaudio[ext=ogg]/bestaudio[acodec!=none]/bestaudio"
)
VARIANT_DEFINITIONS = {
    "plain": {
        "label": "Plain transcript",
        "diarize": False,
        "transcript_filename": WEB_TRANSCRIPT_FILENAME,
        "summary_filename": WEB_SUMMARY_FILENAME,
    },
    "diarized": {
        "label": "Diarized transcript",
        "diarize": True,
        "transcript_filename": WEB_DIARIZED_TRANSCRIPT_FILENAME,
        "summary_filename": WEB_DIARIZED_SUMMARY_FILENAME,
    },
}
VARIANT_TERMINAL_STATUSES = {"complete", "cache_hit", "error"}
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


def build_variant_state(name: str) -> dict[str, Any]:
    config = VARIANT_DEFINITIONS[name]
    return {
        "name": name,
        "label": config["label"],
        "diarize": config["diarize"],
        "status": "idle",
        "error": None,
        "transcript_ready": False,
        "summary_ready": False,
    }


def ensure_variant_map(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    variants = state.setdefault("variants", {})
    for name in VARIANT_DEFINITIONS:
        merged = build_variant_state(name)
        merged.update(variants.get(name, {}))
        variants[name] = merged
    return variants


def ensure_rapidapi_state(state: dict[str, Any]) -> dict[str, Any]:
    rapidapi_state = {
        "configured": rapidapi_youtube.is_configured(),
        "subtitle_available": False,
        "subtitle_fetched": False,
        "subtitle_language": None,
        "subtitle_kind": None,
        "subtitle_word_count": 0,
        "subtitle_chars": 0,
        "subtitle_duration_seconds": None,
        "error": None,
    }
    rapidapi_state.update(state.get("rapidapi") or {})
    state["rapidapi"] = rapidapi_state
    return rapidapi_state


def variant_has_any_output(variant: dict[str, Any]) -> bool:
    return bool(variant.get("transcript_ready") or variant.get("summary_ready"))


def variant_is_terminal(variant: dict[str, Any]) -> bool:
    return variant.get("status") in VARIANT_TERMINAL_STATUSES


def any_variant_output(state: dict[str, Any]) -> bool:
    return any(variant_has_any_output(variant) for variant in ensure_variant_map(state).values())


def all_variants_ready(state: dict[str, Any]) -> bool:
    return all(variant.get("transcript_ready") and variant.get("summary_ready") for variant in ensure_variant_map(state).values())


def all_variants_terminal(state: dict[str, Any]) -> bool:
    return all(variant_is_terminal(variant) for variant in ensure_variant_map(state).values())


def derive_state_error(state: dict[str, Any]) -> Optional[str]:
    status = state.get("status")
    current_error = state.get("error")
    if status in {"unsupported_live", "unsupported_duration", "unsupported_size"}:
        return current_error

    variant_errors = [
        variant.get("error")
        for variant in ensure_variant_map(state).values()
        if variant.get("status") == "error" and variant.get("error")
    ]
    if all_variants_terminal(state) and not any_variant_output(state):
        return variant_errors[0] if variant_errors else current_error
    return current_error if status == "error" and not any_variant_output(state) else None


def derive_overall_status(state: dict[str, Any]) -> str:
    current_status = state.get("status") or "idle"
    if current_status in {"unsupported_live", "unsupported_duration", "unsupported_size"}:
        return current_status
    if current_status == "error" and not any_variant_output(state):
        return "error"

    variants = ensure_variant_map(state)
    if all(variant.get("status") == "cache_hit" for variant in variants.values()):
        return "cache_hit"
    if all_variants_terminal(state):
        return "complete" if any_variant_output(state) else "error"
    if any(variant.get("status") == "summarizing" for variant in variants.values()):
        return "summarizing"
    if any(variant.get("status") == "transcribing" for variant in variants.values()):
        return "transcribing"
    if current_status == "downloading":
        return "downloading"
    if any(variant.get("status") == "queued" for variant in variants.values()):
        return "queued"
    return current_status


def build_state_for_storage(state: dict[str, Any]) -> dict[str, Any]:
    stored = copy.deepcopy(state)
    stored.pop("transcript", None)
    stored.pop("summary", None)
    rapidapi_state = stored.get("rapidapi") or {}
    rapidapi_state.pop("subtitle_text", None)
    stored["rapidapi"] = rapidapi_state
    for variant in (stored.get("variants") or {}).values():
        variant.pop("transcript", None)
        variant.pop("summary", None)
    return stored


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
    state = {
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
    ensure_variant_map(state)
    ensure_rapidapi_state(state)
    return state


def create_error_state(source_url: str, error_message: str) -> dict[str, Any]:
    video_id = extract_youtube_video_id(source_url) or "unknown"
    cache_key = make_cache_key(video_id)
    state = {
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
    ensure_variant_map(state)
    ensure_rapidapi_state(state)
    return state


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


def maybe_fetch_rapidapi_subtitles(state: dict[str, Any], duration_seconds: Optional[float]) -> dict[str, Any]:
    rapidapi_state = ensure_rapidapi_state(state)
    if rapidapi_state.get("subtitle_fetched") or not rapidapi_youtube.is_configured():
        rapidapi_state["configured"] = rapidapi_youtube.is_configured()
        return state

    try:
        client = rapidapi_youtube.RapidAPIYoutubeClient.from_env()
        details = client.get_video_details(state["video_id"])
        track = rapidapi_youtube.choose_subtitle_track(
            rapidapi_youtube.extract_subtitle_tracks(details)
        )
        if not track:
            rapidapi_state.update(
                {
                    "configured": True,
                    "subtitle_available": False,
                    "subtitle_fetched": True,
                    "error": None,
                }
            )
            return persist_state(state)

        subtitle_text = rapidapi_youtube.normalize_subtitle_text(
            client.get_subtitle_text(track["url"], format="srt", fix_overlap=True)
        )
        subtitle_stats = rapidapi_youtube.build_subtitle_stats(subtitle_text, duration_seconds)
        subtitle_metadata = {
            "details": {
                "lengthSeconds": details.get("lengthSeconds"),
                "isLiveStream": details.get("isLiveStream"),
                "isLiveNow": details.get("isLiveNow"),
            },
            "selected_track": track,
            "stats": subtitle_stats,
            "fetched_at": utc_now_iso(),
        }
        storage.write_text(
            state["job_id"],
            WEB_RAPIDAPI_SUBTITLE_METADATA_FILENAME,
            json.dumps(subtitle_metadata, indent=2, sort_keys=True),
        )
        storage.write_text(
            state["job_id"],
            WEB_RAPIDAPI_SUBTITLE_TEXT_FILENAME,
            subtitle_text,
        )
        rapidapi_state.update(
            {
                "configured": True,
                "subtitle_available": True,
                "subtitle_fetched": True,
                "subtitle_language": track.get("code"),
                "subtitle_kind": track.get("kind"),
                "subtitle_word_count": subtitle_stats.get("word_count", 0),
                "subtitle_chars": subtitle_stats.get("char_count", 0),
                "subtitle_duration_seconds": subtitle_stats.get("duration_seconds"),
                "error": None,
            }
        )
        return persist_state(state)
    except Exception as exc:
        rapidapi_state.update(
            {
                "configured": True,
                "subtitle_available": False,
                "subtitle_fetched": True,
                "error": str(exc),
            }
        )
        return persist_state(state)


def queue_job_state(source_url: str, video_id: str, *, transcript_ready: bool = False, summary_ready: bool = False) -> dict[str, Any]:
    cache_key = make_cache_key(video_id)
    timestamp = utc_now_iso()
    state = {
        "job_id": make_job_id(cache_key),
        "cache_key": cache_key,
        "source_url": source_url,
        "video_id": video_id,
        "title": "Unknown",
        "uploader": "Unknown",
        "status": "queued",
        "error": None,
        "transcript_ready": transcript_ready,
        "summary_ready": summary_ready,
        "created_at": timestamp,
        "updated_at": timestamp,
    }
    variants = ensure_variant_map(state)
    variants["plain"]["transcript_ready"] = transcript_ready
    variants["plain"]["summary_ready"] = summary_ready
    if transcript_ready and summary_ready:
        variants["plain"]["status"] = "cache_hit"
    else:
        variants["plain"]["status"] = "queued"
    variants["diarized"]["status"] = "queued"
    ensure_rapidapi_state(state)
    return state


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


def get_variant_artifact_names(name: str) -> tuple[str, str]:
    config = VARIANT_DEFINITIONS[name]
    return config["transcript_filename"], config["summary_filename"]


def update_variant_state(
    state: dict[str, Any],
    variant_name: str,
    *,
    status: Optional[str] = None,
    error: Optional[str] = None,
    transcript_ready: Optional[bool] = None,
    summary_ready: Optional[bool] = None,
) -> dict[str, Any]:
    variant = ensure_variant_map(state)[variant_name]
    if status is not None:
        variant["status"] = status
    if error is not None or status == "error":
        variant["error"] = error
    if transcript_ready is not None:
        variant["transcript_ready"] = transcript_ready
    if summary_ready is not None:
        variant["summary_ready"] = summary_ready
    state["updated_at"] = utc_now_iso()
    return state


def variant_needs_work(variant: dict[str, Any]) -> bool:
    return not (variant.get("transcript_ready") and variant.get("summary_ready"))


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
    hydrated = dict(state)
    variants = ensure_variant_map(hydrated)
    rapidapi_state = ensure_rapidapi_state(hydrated)

    for name, config in VARIANT_DEFINITIONS.items():
        variant = variants[name]
        transcript_text = storage.read_text(
            hydrated["job_id"],
            config["transcript_filename"],
            video_id=video_id,
        )
        summary_text = storage.read_text(
            hydrated["job_id"],
            config["summary_filename"],
            video_id=video_id,
        )
        variant["transcript"] = transcript_text
        variant["summary"] = summary_text
        variant["transcript_ready"] = bool(transcript_text)
        variant["summary_ready"] = bool(summary_text)
        if variant["transcript_ready"] and variant["summary_ready"] and variant.get("status") in {"idle", "queued"}:
            variant["status"] = "cache_hit"
        elif variant["transcript_ready"] and not variant["summary_ready"] and variant.get("status") == "idle":
            variant["status"] = "queued"

    subtitle_metadata_raw = storage.read_text(
        hydrated["job_id"],
        WEB_RAPIDAPI_SUBTITLE_METADATA_FILENAME,
        video_id=video_id,
    )
    if subtitle_metadata_raw:
        try:
            subtitle_metadata = json.loads(subtitle_metadata_raw)
        except json.JSONDecodeError:
            subtitle_metadata = {}
        rapidapi_state.update(
            {
                "subtitle_available": bool(subtitle_metadata),
                "subtitle_fetched": bool(subtitle_metadata),
                "subtitle_language": subtitle_metadata.get("selected_track", {}).get("code"),
                "subtitle_kind": subtitle_metadata.get("selected_track", {}).get("kind"),
                "subtitle_word_count": subtitle_metadata.get("stats", {}).get("word_count", 0),
                "subtitle_chars": subtitle_metadata.get("stats", {}).get("char_count", 0),
                "subtitle_duration_seconds": subtitle_metadata.get("stats", {}).get("duration_seconds"),
                "error": subtitle_metadata.get("error"),
            }
        )

    hydrated["transcript"] = variants["plain"].get("transcript")
    hydrated["summary"] = variants["plain"].get("summary")
    hydrated["transcript_ready"] = variants["plain"]["transcript_ready"]
    hydrated["summary_ready"] = variants["plain"]["summary_ready"]
    hydrated["error"] = derive_state_error(hydrated)
    hydrated["status"] = derive_overall_status(hydrated)
    return hydrated


def persist_state(state: dict[str, Any]) -> dict[str, Any]:
    ensure_variant_map(state)
    ensure_rapidapi_state(state)
    state["transcript_ready"] = state["variants"]["plain"].get("transcript_ready", False)
    state["summary_ready"] = state["variants"]["plain"].get("summary_ready", False)
    state["error"] = derive_state_error(state)
    state["status"] = derive_overall_status(state)
    storage.save_state(state["job_id"], build_state_for_storage(state))
    return state


def prepare_job_for_processing(state: dict[str, Any]) -> dict[str, Any]:
    metadata = probe_youtube_metadata(state["source_url"])
    state["title"] = metadata.get("title") or state.get("title") or "Unknown"
    state["uploader"] = metadata.get("channel") or metadata.get("uploader") or state.get("uploader") or "Unknown"
    for variant_name in VARIANT_DEFINITIONS:
        variant = ensure_variant_map(state)[variant_name]
        if variant_needs_work(variant):
            variant["status"] = "queued"
            variant["error"] = None
    persist_state(update_state(state, status="queued", error=None))

    live_status = classify_youtube_live_status(metadata)
    if live_status != "not_live":
        persist_state(
            update_state(
                state,
                status="unsupported_live",
                error="Live and archived live videos are not supported in the web app v1.",
            )
        )
        return hydrate_state(state)

    duration_seconds = parse_duration_seconds(metadata)
    if duration_seconds is None or duration_seconds > MAX_VIDEO_SECONDS:
        persist_state(
            update_state(
                state,
                status="unsupported_duration",
                error="Video duration must be 25 minutes or less for the web app v1.",
            )
        )
        return hydrate_state(state)

    maybe_fetch_rapidapi_subtitles(state, duration_seconds)
    return hydrate_state(state)


def process_job(state: dict[str, Any]) -> dict[str, Any]:
    hydrated = hydrate_state(state)
    if hydrated["status"] in ACTIVE_STATUSES | UNSUPPORTED_STATUSES:
        return hydrated
    if all_variants_terminal(hydrated) and derive_overall_status(hydrated) in TERMINAL_STATUSES:
        return hydrate_state(persist_state(hydrated))

    audio_path: Optional[str] = None
    cleanup_temp_dir = False

    try:
        variants = ensure_variant_map(hydrated)
        needs_transcription = any(
            not variant.get("transcript_ready") for variant in variants.values()
        )
        variants_needing_work = [
            variant_name
            for variant_name, variant in variants.items()
            if variant_needs_work(variant)
        ]
        if not variants_needing_work:
            return hydrate_state(persist_state(hydrated))

        if needs_transcription:
            hydrated = prepare_job_for_processing(hydrated)
            if hydrated["status"] in TERMINAL_STATUSES:
                return hydrated

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
        else:
            audio_path = storage.find_cached_audio(hydrated["video_id"]) or ""

        state_lock = threading.Lock()

        def process_variant(variant_name: str) -> None:
            variant_config = VARIANT_DEFINITIONS[variant_name]
            transcript_filename, summary_filename = get_variant_artifact_names(variant_name)

            with state_lock:
                variant = hydrated["variants"][variant_name]
                transcript_text = variant.get("transcript")
                summary_text = variant.get("summary")
                if transcript_text and summary_text:
                    variant["status"] = "cache_hit"
                    variant["error"] = None
                    persist_state(hydrated)
                    return

            try:
                if not transcript_text:
                    with state_lock:
                        update_variant_state(
                            hydrated,
                            variant_name,
                            status="transcribing",
                            error=None,
                            transcript_ready=False,
                        )
                        persist_state(hydrated)
                    _, _, transcript_text = speech.transcribe_audio_file(
                        audio_path or hydrated["video_id"],
                        None,
                        diarize=variant_config["diarize"],
                    )
                    storage.write_text(hydrated["job_id"], transcript_filename, transcript_text)
                    with state_lock:
                        hydrated["variants"][variant_name]["transcript"] = transcript_text
                        update_variant_state(
                            hydrated,
                            variant_name,
                            status="summarizing" if not summary_text else "complete",
                            error=None,
                            transcript_ready=True,
                        )
                        persist_state(hydrated)

                if not summary_text:
                    with state_lock:
                        update_variant_state(
                            hydrated,
                            variant_name,
                            status="summarizing",
                            error=None,
                            summary_ready=False,
                        )
                        persist_state(hydrated)
                    source_context = build_source_context(hydrated, audio_path or hydrated["video_id"])
                    summary_text = speech.summarize_transcript(transcript_text or "", source_context)
                    storage.write_text(hydrated["job_id"], summary_filename, summary_text)
                    with state_lock:
                        hydrated["variants"][variant_name]["summary"] = summary_text
                        update_variant_state(
                            hydrated,
                            variant_name,
                            status="complete",
                            error=None,
                            transcript_ready=bool(transcript_text),
                            summary_ready=True,
                        )
                        persist_state(hydrated)
            except Exception as exc:
                with state_lock:
                    update_variant_state(hydrated, variant_name, status="error", error=str(exc))
                    persist_state(hydrated)

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(variants_needing_work)) as executor:
            futures = [executor.submit(process_variant, variant_name) for variant_name in variants_needing_work]
            for future in concurrent.futures.as_completed(futures):
                future.result()

        return hydrate_state(persist_state(update_state(hydrated, status=derive_overall_status(hydrated), error=derive_state_error(hydrated))))
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
        hydrated = hydrate_state(existing_state)
        if hydrated["status"] == "error" and not any_variant_output(hydrated):
            return hydrate_state(persist_state(update_state(existing_state, status="queued", error=None)))
        return hydrated

    cached_plain_transcript = storage.read_text(job_id, WEB_TRANSCRIPT_FILENAME, video_id=video_id)
    cached_plain_summary = storage.read_text(job_id, WEB_SUMMARY_FILENAME, video_id=video_id)
    cached_diarized_transcript = storage.read_text(job_id, WEB_DIARIZED_TRANSCRIPT_FILENAME, video_id=video_id)
    cached_diarized_summary = storage.read_text(job_id, WEB_DIARIZED_SUMMARY_FILENAME, video_id=video_id)
    if any(
        (
            cached_plain_transcript,
            cached_plain_summary,
            cached_diarized_transcript,
            cached_diarized_summary,
        )
    ):
        cached_state = queue_job_state(source_url, video_id)
        plain_variant = ensure_variant_map(cached_state)["plain"]
        diarized_variant = ensure_variant_map(cached_state)["diarized"]
        plain_variant["transcript_ready"] = bool(cached_plain_transcript)
        plain_variant["summary_ready"] = bool(cached_plain_summary)
        plain_variant["status"] = "cache_hit" if cached_plain_transcript and cached_plain_summary else "queued"
        diarized_variant["transcript_ready"] = bool(cached_diarized_transcript)
        diarized_variant["summary_ready"] = bool(cached_diarized_summary)
        diarized_variant["status"] = (
            "cache_hit" if cached_diarized_transcript and cached_diarized_summary else "queued"
        )
        if all_variants_ready(cached_state):
            cached_state["status"] = "cache_hit"
        return hydrate_state(persist_state(cached_state))

    return hydrate_state(persist_state(queue_job_state(source_url, video_id)))


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
      body {{ font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 84rem; padding: 0 1rem 4rem; line-height: 1.5; }}
      h1, h2 {{ margin-bottom: 0.5rem; }}
      .meta {{ color: #4b5563; margin-bottom: 1rem; }}
      .status {{ background: #eff6ff; border: 1px solid #93c5fd; padding: 0.75rem 1rem; border-radius: 0.5rem; margin-bottom: 1rem; }}
      .subtitle-meta {{ color: #475569; margin-bottom: 1rem; }}
      .error {{ background: #fef2f2; border-color: #fca5a5; }}
      pre {{ white-space: pre-wrap; background: #f8fafc; padding: 1rem; border-radius: 0.5rem; border: 1px solid #e5e7eb; }}
      section {{ margin-top: 1.5rem; }}
      .columns {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1.25rem; align-items: start; }}
      .column {{ border: 1px solid #e5e7eb; border-radius: 0.75rem; padding: 1rem; background: #ffffff; }}
      .column-status {{ color: #475569; font-size: 0.95rem; margin: 0 0 0.75rem; }}
      .column-status.error {{ color: #b91c1c; }}
      @media (max-width: 900px) {{
        .columns {{ grid-template-columns: 1fr; }}
      }}
    </style>
  </head>
  <body>
    <h1>cill.app</h1>
    <p class="meta">Source URL: <a id="source-link" href={serialized_url}></a></p>
    <div id="status" class="status">Creating job…</div>
    <p id="subtitle-meta" class="subtitle-meta" hidden></p>
    <div class="columns">
      <section class="column" id="plain-column">
        <h2>Plain</h2>
        <p id="plain-status" class="column-status">Waiting…</p>
        <section id="plain-transcript-section" hidden>
          <h3>Transcript</h3>
          <pre id="plain-transcript"></pre>
        </section>
        <section id="plain-summary-section" hidden>
          <h3>Summary</h3>
          <pre id="plain-summary"></pre>
        </section>
      </section>
      <section class="column" id="diarized-column">
        <h2>Diarized</h2>
        <p id="diarized-status" class="column-status">Waiting…</p>
        <section id="diarized-transcript-section" hidden>
          <h3>Transcript</h3>
          <pre id="diarized-transcript"></pre>
        </section>
        <section id="diarized-summary-section" hidden>
          <h3>Summary</h3>
          <pre id="diarized-summary"></pre>
        </section>
      </section>
    </div>
    <script>
      const sourceUrl = {serialized_url};
      const sourceLink = document.getElementById('source-link');
      const statusEl = document.getElementById('status');
      const subtitleMetaEl = document.getElementById('subtitle-meta');
      const variantElements = {{
        plain: {{
          status: document.getElementById('plain-status'),
          transcriptSection: document.getElementById('plain-transcript-section'),
          transcript: document.getElementById('plain-transcript'),
          summarySection: document.getElementById('plain-summary-section'),
          summary: document.getElementById('plain-summary'),
        }},
        diarized: {{
          status: document.getElementById('diarized-status'),
          transcriptSection: document.getElementById('diarized-transcript-section'),
          transcript: document.getElementById('diarized-transcript'),
          summarySection: document.getElementById('diarized-summary-section'),
          summary: document.getElementById('diarized-summary'),
        }},
      }};
      sourceLink.textContent = sourceUrl;

      const terminalStates = new Set(['complete', 'cache_hit', 'unsupported_live', 'unsupported_duration', 'unsupported_size', 'error']);

      function renderVariant(name, variant) {{
        const elements = variantElements[name];
        const detail = variant.error ? `: ${{variant.error}}` : '';
        elements.status.textContent = `${{variant.status || 'idle'}}${{detail}}`;
        elements.status.className = 'column-status' + (variant.error ? ' error' : '');

        if (variant.transcript) {{
          elements.transcriptSection.hidden = false;
          elements.transcript.textContent = variant.transcript;
        }}
        if (variant.summary) {{
          elements.summarySection.hidden = false;
          elements.summary.textContent = variant.summary;
        }}
      }}

      function renderRapidApiState(state) {{
        const rapidapi = state.rapidapi || {{}};
        if (!rapidapi.subtitle_fetched) {{
          subtitleMetaEl.hidden = true;
          return;
        }}
        if (rapidapi.subtitle_available) {{
          const details = [
            rapidapi.subtitle_language ? `language=${{rapidapi.subtitle_language}}` : null,
            rapidapi.subtitle_kind ? `kind=${{rapidapi.subtitle_kind}}` : null,
            rapidapi.subtitle_word_count ? `words=${{rapidapi.subtitle_word_count}}` : null,
          ].filter(Boolean).join(' · ');
          subtitleMetaEl.textContent = `RapidAPI subtitles available${{details ? ' (' + details + ')' : ''}}`;
          subtitleMetaEl.hidden = false;
          return;
        }}
        if (rapidapi.error) {{
          subtitleMetaEl.textContent = `RapidAPI subtitle lookup failed: ${{rapidapi.error}}`;
          subtitleMetaEl.hidden = false;
          return;
        }}
        subtitleMetaEl.textContent = 'RapidAPI subtitle lookup completed: no usable subtitles found.';
        subtitleMetaEl.hidden = false;
      }}

      function renderState(state) {{
        const title = state.title ? `${{state.title}}` : 'Untitled video';
        const uploader = state.uploader ? ` by ${{state.uploader}}` : '';
        const detail = state.error ? `: ${{state.error}}` : '';
        statusEl.textContent = `${{state.status}} — ${{title}}${{uploader}}${{detail}}`;
        statusEl.className = 'status' + (state.error ? ' error' : '');
        renderRapidApiState(state);
        renderVariant('plain', state.variants?.plain || {{}});
        renderVariant('diarized', state.variants?.diarized || {{}});
      }}

      async function fetchState(jobId) {{
        const response = await fetch(`/api/jobs/${{jobId}}`);
        if (!response.ok) {{
          throw new Error(`Unable to load job state (${{response.status}})`);
        }}
        return response.json();
      }}

      async function poll(jobId) {{
        let delayMs = 5000;
        while (true) {{
          const state = await fetchState(jobId);
          renderState(state);
          if (terminalStates.has(state.status)) {{
            return;
          }}
          await new Promise((resolve) => setTimeout(resolve, delayMs));
          delayMs = Math.min(60000, delayMs + 5000);
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
    hydrated = hydrate_state(state)
    if hydrated["status"] in TERMINAL_STATUSES or hydrated["status"] in ACTIVE_STATUSES:
        return JSONResponse(hydrate_state(hydrated))
    return JSONResponse(hydrate_state(persist_state(update_state(hydrated, status="queued", error=None))))


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
