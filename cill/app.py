from __future__ import annotations

import copy
import concurrent.futures
import hashlib
import json
import os
import requests
import shutil
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field

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
DOWNLOAD_CONNECT_TIMEOUT_SECONDS = float(os.getenv("CILL_AUDIO_DOWNLOAD_CONNECT_TIMEOUT_SECONDS", "10"))
DOWNLOAD_READ_TIMEOUT_SECONDS = float(os.getenv("CILL_AUDIO_DOWNLOAD_READ_TIMEOUT_SECONDS", "20"))
DOWNLOAD_DEADLINE_SECONDS = float(os.getenv("CILL_AUDIO_DOWNLOAD_DEADLINE_SECONDS", "180"))
STALE_ACTIVE_JOB_SECONDS = float(os.getenv("CILL_STALE_ACTIVE_JOB_SECONDS", "300"))
SUPPORTED_WEB_AUDIO_EXTENSIONS = {".m4a", ".mp3", ".webm", ".wav", ".ogg", ".aac", ".mpeg", ".mpga"}
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
TRANSCRIPT_SOURCE_LABELS = {
    "rapidapi_subtitles": "RapidAPI subtitles",
    "openai_audio_transcript": "OpenAI audio transcript",
    "legacy_cache": "legacy cache",
}
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


class AudioTooLargeError(RuntimeError):
    pass


class JobCreateRequest(BaseModel):
    source_url: str


class VariantRunRequest(BaseModel):
    transcript: bool = False
    summary: bool = False


class JobRunRequest(BaseModel):
    plain: VariantRunRequest = Field(default_factory=VariantRunRequest)
    diarized: VariantRunRequest = Field(default_factory=VariantRunRequest)


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
        "requested_transcript": False,
        "requested_summary": False,
        "transcript_ready": False,
        "summary_ready": False,
        "transcript_source": None,
        "summary_basis": None,
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
        "subtitle_usable": False,
        "subtitle_validation_reason": None,
        "error": None,
    }
    rapidapi_state.update(state.get("rapidapi") or {})
    state["rapidapi"] = rapidapi_state
    return rapidapi_state


def variant_has_any_output(variant: dict[str, Any]) -> bool:
    return bool(variant.get("transcript_ready") or variant.get("summary_ready"))


def variant_has_requested_work(variant: dict[str, Any]) -> bool:
    return bool(variant.get("requested_transcript") or variant.get("requested_summary"))


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
    if current_status == "complete" and any_variant_output(state):
        return "complete"
    if current_status == "cache_hit" and any_variant_output(state):
        return "cache_hit"
    if any_variant_output(state):
        if all(variant.get("transcript_ready") and variant.get("summary_ready") for variant in variants.values()):
            return "cache_hit"
        return "idle"
    return "idle"


def build_state_for_storage(state: dict[str, Any]) -> dict[str, Any]:
    stored = copy.deepcopy(state)
    stored.pop("transcript", None)
    stored.pop("summary", None)
    rapidapi_state = stored.get("rapidapi") or {}
    rapidapi_state.pop("subtitle_text", None)
    rapidapi_state.pop("subtitle_status_label", None)
    stored["rapidapi"] = rapidapi_state
    for variant in (stored.get("variants") or {}).values():
        variant.pop("transcript", None)
        variant.pop("summary", None)
        variant.pop("transcript_source_label", None)
        variant.pop("summary_basis_label", None)
        variant.pop("queue_actions", None)
        variant.pop("mode_hint", None)
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
        "duration_seconds": parse_duration_seconds(metadata),
        "live_status": classify_youtube_live_status(metadata),
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
        "duration_seconds": None,
        "live_status": None,
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
        value = metadata.get("lengthSeconds")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_state_timestamp(value: Any) -> Optional[datetime]:
    if not value or not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def state_has_active_work(state: dict[str, Any]) -> bool:
    if state.get("status") in ACTIVE_STATUSES:
        return True
    return any(
        variant.get("status") in ACTIVE_STATUSES
        for variant in ensure_variant_map(state).values()
    )


def is_stale_active_state(state: dict[str, Any]) -> bool:
    if not state_has_active_work(state):
        return False
    updated_at = parse_state_timestamp(state.get("updated_at")) or parse_state_timestamp(
        state.get("created_at")
    )
    if not updated_at:
        return False
    age_seconds = (datetime.now(timezone.utc) - updated_at).total_seconds()
    return age_seconds >= STALE_ACTIVE_JOB_SECONDS


def build_source_context(state: dict[str, Any], audio_path: str) -> dict[str, Any]:
    return {
        "source_type": "youtube",
        "selected_source_path": audio_path,
        "youtube_url": state["source_url"],
        "youtube_title": state["title"],
        "youtube_uploader": state["uploader"],
        "youtube_video_id": state["video_id"],
        "youtube_live_status": state.get("live_status") or "not_live",
        "youtube_live_capture_outcome": None,
    }


def probe_youtube_metadata(source_url: str) -> dict[str, Any]:
    video_id = extract_youtube_video_id(source_url)
    if not video_id:
        raise RuntimeError("Unable to extract a YouTube video ID from the URL.")
    client = rapidapi_youtube.RapidAPIYoutubeClient.from_env()
    details = client.get_video_details(video_id)
    return {
        "id": details.get("id") or video_id,
        "title": details.get("title"),
        "uploader": rapidapi_youtube.channel_name_from_details(details),
        "duration": details.get("lengthSeconds"),
        "live_status": rapidapi_youtube.live_status_from_details(details),
        "rapidapi_details": details,
    }


def maybe_fetch_rapidapi_subtitles(
    state: dict[str, Any],
    duration_seconds: Optional[float],
    *,
    details: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    rapidapi_state = ensure_rapidapi_state(state)
    if rapidapi_state.get("subtitle_fetched") or not rapidapi_youtube.is_configured():
        rapidapi_state["configured"] = rapidapi_youtube.is_configured()
        return state

    try:
        client = None
        if details is None:
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

        if client is None:
            client = rapidapi_youtube.RapidAPIYoutubeClient.from_env()
        subtitle_text = rapidapi_youtube.normalize_subtitle_text(
            client.get_subtitle_text(track["url"], format="srt", fix_overlap=True)
        )
        subtitle_stats = rapidapi_youtube.build_subtitle_stats(subtitle_text, duration_seconds)
        validation = rapidapi_youtube.validate_subtitle_stats(subtitle_stats)
        subtitle_metadata = {
            "details": {
                "lengthSeconds": details.get("lengthSeconds"),
                "isLiveStream": details.get("isLiveStream"),
                "isLiveNow": details.get("isLiveNow"),
            },
            "selected_track": track,
            "stats": subtitle_stats,
            "validation": validation,
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
                "subtitle_usable": bool(validation.get("usable")),
                "subtitle_validation_reason": validation.get("reason"),
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
                "subtitle_usable": False,
                "subtitle_validation_reason": "fetch_failed",
                "error": str(exc),
            }
        )
        return persist_state(state)


def get_subtitle_validation(state: dict[str, Any]) -> tuple[bool, Optional[str]]:
    rapidapi_state = ensure_rapidapi_state(state)
    return bool(rapidapi_state.get("subtitle_usable")), rapidapi_state.get("subtitle_validation_reason")


def load_subtitle_transcript(state: dict[str, Any]) -> Optional[str]:
    return storage.read_text(
        state["job_id"],
        WEB_RAPIDAPI_SUBTITLE_TEXT_FILENAME,
        video_id=state.get("video_id"),
    )


def variant_can_use_subtitles(state: dict[str, Any], variant_name: str, subtitle_text: Optional[str]) -> bool:
    subtitle_usable, _ = get_subtitle_validation(state)
    return variant_name == "plain" and subtitle_usable and bool(subtitle_text)


def format_source_label(source_key: Optional[str]) -> Optional[str]:
    if not source_key:
        return None
    return TRANSCRIPT_SOURCE_LABELS.get(source_key, source_key.replace("_", " "))


def subtitle_status_label(rapidapi_state: dict[str, Any]) -> str:
    if rapidapi_state.get("subtitle_available") and rapidapi_state.get("subtitle_usable"):
        return "Subtitles available and validated"
    if rapidapi_state.get("subtitle_available"):
        return "Subtitles available but unusable"
    if rapidapi_state.get("error"):
        return "Subtitle lookup failed"
    if rapidapi_state.get("subtitle_fetched"):
        return "No subtitles found"
    return "Subtitle lookup pending"


def get_variant_queue_actions(state: dict[str, Any], variant_name: str) -> dict[str, Any]:
    variant = ensure_variant_map(state)[variant_name]
    rapidapi_state = ensure_rapidapi_state(state)
    subtitle_usable = bool(rapidapi_state.get("subtitle_usable"))
    duration_seconds = state.get("duration_seconds")
    duration_exceeds_limit = duration_seconds is None or duration_seconds > MAX_VIDEO_SECONDS

    transcript_disabled_reason = None
    summary_disabled_reason = None

    if state.get("status") == "unsupported_live":
        transcript_disabled_reason = state.get("error")
        summary_disabled_reason = state.get("error")
    elif variant_name == "plain":
        if duration_exceeds_limit and not subtitle_usable:
            transcript_disabled_reason = "Audio processing is limited to 25 minutes and no usable subtitles were found."
            summary_disabled_reason = transcript_disabled_reason
    else:
        if duration_exceeds_limit:
            transcript_disabled_reason = "Diarized transcript requires audio processing and is limited to 25 minutes in the web app v1."
            summary_disabled_reason = transcript_disabled_reason

    if variant.get("transcript_ready"):
        transcript_disabled_reason = "Transcript already available."
    if variant.get("summary_ready"):
        summary_disabled_reason = "Summary already available."
    elif summary_disabled_reason is None and variant.get("transcript_ready"):
        summary_disabled_reason = None

    if variant_name == "plain":
        mode_hint = "Will use subtitles" if subtitle_usable and not variant.get("transcript_ready") else "Audio required"
    else:
        mode_hint = "Audio required"

    return {
        "transcript_only": {
            "enabled": transcript_disabled_reason is None,
            "disabled_reason": transcript_disabled_reason,
            "label": "Queue transcript only",
        },
        "transcript_and_summary": {
            "enabled": summary_disabled_reason is None,
            "disabled_reason": summary_disabled_reason,
            "label": "Queue transcript + summary" if not variant.get("transcript_ready") else "Queue summary",
        },
        "mode_hint": mode_hint,
    }


def idle_job_state(source_url: str, video_id: str) -> dict[str, Any]:
    cache_key = make_cache_key(video_id)
    timestamp = utc_now_iso()
    state = {
        "job_id": make_job_id(cache_key),
        "cache_key": cache_key,
        "source_url": source_url,
        "video_id": video_id,
        "title": "Unknown",
        "uploader": "Unknown",
        "duration_seconds": None,
        "live_status": None,
        "status": "idle",
        "error": None,
        "transcript_ready": False,
        "summary_ready": False,
        "created_at": timestamp,
        "updated_at": timestamp,
    }
    ensure_variant_map(state)
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
    requested_transcript: Optional[bool] = None,
    requested_summary: Optional[bool] = None,
    transcript_ready: Optional[bool] = None,
    summary_ready: Optional[bool] = None,
    transcript_source: Optional[str] = None,
    summary_basis: Optional[str] = None,
) -> dict[str, Any]:
    variant = ensure_variant_map(state)[variant_name]
    if status is not None:
        variant["status"] = status
    if error is not None or status == "error":
        variant["error"] = error
    if requested_transcript is not None:
        variant["requested_transcript"] = requested_transcript
    if requested_summary is not None:
        variant["requested_summary"] = requested_summary
    if transcript_ready is not None:
        variant["transcript_ready"] = transcript_ready
    if summary_ready is not None:
        variant["summary_ready"] = summary_ready
    if transcript_source is not None:
        variant["transcript_source"] = transcript_source
    if summary_basis is not None:
        variant["summary_basis"] = summary_basis
    state["updated_at"] = utc_now_iso()
    return state


def requeue_pending_work(state: dict[str, Any]) -> dict[str, Any]:
    hydrated = hydrate_state(state)
    queued_any = False
    for variant_name, variant in ensure_variant_map(hydrated).items():
        needs_transcript = bool(variant.get("requested_transcript") and not variant.get("transcript_ready"))
        needs_summary = bool(variant.get("requested_summary") and not variant.get("summary_ready"))
        if not needs_transcript and not needs_summary:
            continue
        queued_any = True
        update_variant_state(
            hydrated,
            variant_name,
            status="queued",
            error=None,
            requested_transcript=needs_transcript,
            requested_summary=needs_summary,
        )
    if queued_any:
        update_state(hydrated, status="queued", error=None)
    return hydrate_state(persist_state(hydrated))


def recover_stale_active_state(state: dict[str, Any]) -> dict[str, Any]:
    hydrated = hydrate_state(state)
    if not is_stale_active_state(hydrated):
        return hydrated
    return requeue_pending_work(hydrated)


def variant_needs_work(variant: dict[str, Any]) -> bool:
    if variant.get("status") in ACTIVE_STATUSES | QUEUED_STATUSES:
        return True
    if variant_is_terminal(variant):
        return False
    if variant.get("requested_summary") and not variant.get("summary_ready"):
        return True
    if variant.get("requested_transcript") and not variant.get("transcript_ready"):
        return True
    return False


def build_audio_download_selection(video_id: str) -> dict[str, Any]:
    client = rapidapi_youtube.RapidAPIYoutubeClient.from_env()
    details = client.get_video_details(video_id, subtitles=False)
    track = rapidapi_youtube.choose_audio_track(
        rapidapi_youtube.extract_audio_tracks(details)
    )
    if not track or not track.get("url"):
        raise RuntimeError("RapidAPI did not provide a downloadable audio stream for this video.")
    return track


def download_audio(source_url: str, state: dict[str, Any]) -> tuple[str, bool]:
    cached_audio = storage.find_cached_audio(state["video_id"])
    if cached_audio:
        return cached_audio, False

    temp_dir = tempfile.mkdtemp(prefix=f"cill_{state['video_id']}_")
    track = build_audio_download_selection(state["video_id"])
    track_size = track.get("size")
    if track_size and int(track_size) > MAX_AUDIO_BYTES:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise AudioTooLargeError(
            f"Downloaded audio exceeds the v1 size cap of {MAX_AUDIO_BYTES // (1024 * 1024)} MB."
        )

    extension = str(track.get("extension") or "bin").strip(".")
    target_path = Path(temp_dir) / f"{sanitize_filename_component(state['title'])} [{state['video_id']}].{extension}"
    deadline = time.monotonic() + DOWNLOAD_DEADLINE_SECONDS
    try:
        with requests.get(
            track["url"],
            stream=True,
            timeout=(DOWNLOAD_CONNECT_TIMEOUT_SECONDS, DOWNLOAD_READ_TIMEOUT_SECONDS),
        ) as response:
            response.raise_for_status()
            total_bytes = 0
            with open(target_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    if time.monotonic() > deadline:
                        raise RuntimeError(
                            f"Audio download timed out after {int(DOWNLOAD_DEADLINE_SECONDS)} seconds."
                        )
                    total_bytes += len(chunk)
                    if total_bytes > MAX_AUDIO_BYTES:
                        raise AudioTooLargeError(
                            f"Downloaded audio exceeds the v1 size cap of {MAX_AUDIO_BYTES // (1024 * 1024)} MB."
                        )
                    handle.write(chunk)
    except requests.exceptions.Timeout as exc:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError("Audio download timed out while waiting for the upstream media stream.") from exc
    except requests.exceptions.RequestException as exc:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(f"Audio download failed: {exc}") from exc
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    return str(target_path), True


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
        if variant["transcript_ready"] and not variant.get("transcript_source"):
            variant["transcript_source"] = "legacy_cache"
        if variant["summary_ready"] and not variant.get("summary_basis"):
            variant["summary_basis"] = "legacy_cache"
        if not variant_has_requested_work(variant) and variant.get("status") == "error" and not variant_has_any_output(variant):
            variant["status"] = "idle"
            variant["error"] = None
        if variant_has_requested_work(variant):
            continue
        if variant["transcript_ready"] and variant["summary_ready"]:
            if variant.get("status") not in {"complete", "error"}:
                variant["status"] = "cache_hit"
        elif variant_has_any_output(variant):
            if variant.get("status") not in {"complete", "error"}:
                variant["status"] = "idle"
        elif variant.get("status") not in {"error"}:
            variant["status"] = "idle"

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
        validation = subtitle_metadata.get("validation") or rapidapi_youtube.validate_subtitle_stats(
            subtitle_metadata.get("stats") or {}
        )
        rapidapi_state.update(
            {
                "subtitle_available": bool(subtitle_metadata),
                "subtitle_fetched": bool(subtitle_metadata),
                "subtitle_language": subtitle_metadata.get("selected_track", {}).get("code"),
                "subtitle_kind": subtitle_metadata.get("selected_track", {}).get("kind"),
                "subtitle_word_count": subtitle_metadata.get("stats", {}).get("word_count", 0),
                "subtitle_chars": subtitle_metadata.get("stats", {}).get("char_count", 0),
                "subtitle_duration_seconds": subtitle_metadata.get("stats", {}).get("duration_seconds"),
                "subtitle_usable": bool(validation.get("usable")),
                "subtitle_validation_reason": validation.get("reason"),
                "error": subtitle_metadata.get("error"),
            }
        )

    rapidapi_state["subtitle_status_label"] = subtitle_status_label(rapidapi_state)

    hydrated["transcript"] = variants["plain"].get("transcript")
    hydrated["summary"] = variants["plain"].get("summary")
    hydrated["transcript_ready"] = variants["plain"]["transcript_ready"]
    hydrated["summary_ready"] = variants["plain"]["summary_ready"]
    hydrated["error"] = derive_state_error(hydrated)
    hydrated["status"] = derive_overall_status(hydrated)
    variants = ensure_variant_map(hydrated)
    for name in list(variants.keys()):
        queue_actions = get_variant_queue_actions(hydrated, name)
        variant = ensure_variant_map(hydrated)[name]
        variant["transcript_source_label"] = format_source_label(variant.get("transcript_source"))
        variant["summary_basis_label"] = format_source_label(variant.get("summary_basis"))
        variant["queue_actions"] = {
            "transcript_only": queue_actions["transcript_only"],
            "transcript_and_summary": queue_actions["transcript_and_summary"],
        }
        variant["mode_hint"] = queue_actions["mode_hint"]
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


def handle_probe_failure(state: dict[str, Any], exc: Exception) -> dict[str, Any]:
    hydrated = hydrate_state(state)
    rapidapi_state = ensure_rapidapi_state(hydrated)
    rapidapi_state["configured"] = rapidapi_youtube.is_configured()
    rapidapi_state["error"] = str(exc)
    if any_variant_output(hydrated):
        return hydrate_state(persist_state(hydrated))
    return hydrate_state(
        persist_state(
            update_state(
                hydrated,
                status="error",
                error=f"The deployment could not access YouTube metadata from RapidAPI: {exc}",
            )
        )
    )


def probe_job_state(state: dict[str, Any]) -> dict[str, Any]:
    metadata = probe_youtube_metadata(state["source_url"])
    state["title"] = metadata.get("title") or state.get("title") or "Unknown"
    state["uploader"] = metadata.get("uploader") or state.get("uploader") or "Unknown"
    state["duration_seconds"] = parse_duration_seconds(metadata)
    state["live_status"] = metadata.get("live_status")
    persist_state(update_state(state, status=state.get("status") or "idle", error=None))

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

    duration_seconds = state["duration_seconds"]
    maybe_fetch_rapidapi_subtitles(
        state,
        duration_seconds,
        details=metadata.get("rapidapi_details"),
    )
    subtitle_usable, _ = get_subtitle_validation(state)
    duration_exceeds_limit = duration_seconds is None or duration_seconds > MAX_VIDEO_SECONDS
    if duration_exceeds_limit and not subtitle_usable:
        persist_state(
            update_state(
                state,
                status="unsupported_duration",
                error="Video duration must be 25 minutes or less for the web app v1.",
            )
        )
        return hydrate_state(state)
    return hydrate_state(state)


def process_job(state: dict[str, Any]) -> dict[str, Any]:
    hydrated = recover_stale_active_state(state)
    if hydrated["status"] in ACTIVE_STATUSES | UNSUPPORTED_STATUSES:
        return hydrated
    if not any(variant_needs_work(variant) for variant in ensure_variant_map(hydrated).values()):
        if hydrated["status"] == "queued":
            return hydrate_state(persist_state(update_state(hydrated, status="idle", error=None)))
    if all_variants_terminal(hydrated) and derive_overall_status(hydrated) in TERMINAL_STATUSES:
        return hydrate_state(persist_state(hydrated))

    audio_path: Optional[str] = None
    cleanup_temp_dir = False

    try:
        variants = ensure_variant_map(hydrated)
        needs_transcription = any(
            variant.get("requested_transcript") and not variant.get("transcript_ready")
            for variant in variants.values()
        )

        if needs_transcription:
            hydrated = probe_job_state(hydrated)
            if hydrated["status"] in TERMINAL_STATUSES:
                return hydrated

        variants = ensure_variant_map(hydrated)
        variants_needing_work = [
            variant_name
            for variant_name, variant in variants.items()
            if variant_needs_work(variant)
        ]
        if not variants_needing_work:
            return hydrate_state(persist_state(hydrated))

        subtitle_text = load_subtitle_transcript(hydrated)
        needs_audio_download = any(
            not variant_can_use_subtitles(hydrated, variant_name, subtitle_text)
            and not variants[variant_name].get("transcript_ready")
            for variant_name in variants_needing_work
        )

        if needs_audio_download:
            persist_state(update_state(hydrated, status="downloading"))
            try:
                audio_path, cleanup_temp_dir = download_audio(hydrated["source_url"], hydrated)
            except AudioTooLargeError as exc:
                persist_state(update_state(hydrated, status="unsupported_size", error=str(exc)))
                return hydrate_state(hydrated)
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
        elif not needs_transcription:
            audio_path = storage.find_cached_audio(hydrated["video_id"]) or ""

        state_lock = threading.Lock()

        def process_variant(variant_name: str) -> None:
            variant_config = VARIANT_DEFINITIONS[variant_name]
            transcript_filename, summary_filename = get_variant_artifact_names(variant_name)

            with state_lock:
                variant = hydrated["variants"][variant_name]
                transcript_text = variant.get("transcript")
                summary_text = variant.get("summary")
                if not variant_needs_work(variant):
                    variant["status"] = "cache_hit"
                    variant["error"] = None
                    persist_state(hydrated)
                    return

            try:
                if variant.get("requested_transcript") and not transcript_text:
                    if variant_can_use_subtitles(hydrated, variant_name, subtitle_text):
                        transcript_text = subtitle_text or ""
                        transcript_source = "rapidapi_subtitles"
                    else:
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
                        transcript_source = "openai_audio_transcript"
                    storage.write_text(hydrated["job_id"], transcript_filename, transcript_text)
                    with state_lock:
                        hydrated["variants"][variant_name]["transcript"] = transcript_text
                        update_variant_state(
                            hydrated,
                            variant_name,
                            status="summarizing" if variant.get("requested_summary") and not summary_text else "complete",
                            error=None,
                            requested_transcript=False,
                            transcript_ready=True,
                            transcript_source=transcript_source,
                        )
                        persist_state(hydrated)

                if variant.get("requested_summary") and not summary_text:
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
                            requested_summary=False,
                            transcript_ready=bool(transcript_text),
                            summary_ready=True,
                            summary_basis=hydrated["variants"][variant_name].get("transcript_source") or "legacy_cache",
                        )
                        persist_state(hydrated)
                elif transcript_text and not variant.get("requested_summary"):
                    with state_lock:
                        update_variant_state(
                            hydrated,
                            variant_name,
                            status="complete",
                            error=None,
                            requested_transcript=False,
                            requested_summary=False,
                            transcript_ready=bool(transcript_text),
                            summary_ready=bool(summary_text),
                        )
                        persist_state(hydrated)
            except Exception as exc:
                with state_lock:
                    update_variant_state(
                        hydrated,
                        variant_name,
                        status="error",
                        error=str(exc),
                        requested_transcript=False,
                        requested_summary=False,
                    )
                    persist_state(hydrated)

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(variants_needing_work)) as executor:
            futures = [executor.submit(process_variant, variant_name) for variant_name in variants_needing_work]
            for future in concurrent.futures.as_completed(futures):
                future.result()

        return hydrate_state(persist_state(update_state(hydrated, status="complete", error=derive_state_error(hydrated))))
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
        hydrated = recover_stale_active_state(existing_state)
        if hydrated["status"] in ACTIVE_STATUSES | QUEUED_STATUSES | UNSUPPORTED_STATUSES:
            return hydrated
        if (
            hydrated.get("title") in {None, "Unknown"}
            or hydrated.get("duration_seconds") is None
            or not ensure_rapidapi_state(hydrated).get("subtitle_fetched")
        ):
            try:
                return probe_job_state(hydrated)
            except Exception as exc:
                return handle_probe_failure(hydrated, exc)
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
        cached_state = idle_job_state(source_url, video_id)
        plain_variant = ensure_variant_map(cached_state)["plain"]
        diarized_variant = ensure_variant_map(cached_state)["diarized"]
        plain_variant["transcript_ready"] = bool(cached_plain_transcript)
        plain_variant["summary_ready"] = bool(cached_plain_summary)
        plain_variant["status"] = "cache_hit" if cached_plain_transcript and cached_plain_summary else "idle"
        diarized_variant["transcript_ready"] = bool(cached_diarized_transcript)
        diarized_variant["summary_ready"] = bool(cached_diarized_summary)
        diarized_variant["status"] = (
            "cache_hit" if cached_diarized_transcript and cached_diarized_summary else "idle"
        )
        if all_variants_ready(cached_state):
            cached_state["status"] = "cache_hit"
        else:
            cached_state["status"] = "idle"
        try:
            return probe_job_state(persist_state(cached_state))
        except Exception as exc:
            return handle_probe_failure(persist_state(cached_state), exc)

    fresh_state = persist_state(idle_job_state(source_url, video_id))
    try:
        return probe_job_state(fresh_state)
    except Exception as exc:
        return handle_probe_failure(fresh_state, exc)


def queue_requested_work(state: dict[str, Any], payload: JobRunRequest) -> dict[str, Any]:
    hydrated = hydrate_state(state)
    if hydrated["status"] in UNSUPPORTED_STATUSES:
        return hydrated

    request_map = {
        "plain": payload.plain,
        "diarized": payload.diarized,
    }

    queued_any = False
    for variant_name, request in request_map.items():
        variant = ensure_variant_map(hydrated)[variant_name]
        actions = variant.get("queue_actions") or get_variant_queue_actions(hydrated, variant_name)
        wants_transcript = request.transcript
        wants_summary = request.summary
        if not wants_transcript and not wants_summary:
            continue

        if wants_summary and not variant.get("transcript_ready"):
            wants_transcript = True

        transcript_allowed = actions["transcript_only"]["enabled"] or variant.get("transcript_ready")
        summary_allowed = actions["transcript_and_summary"]["enabled"] or variant.get("summary_ready")

        if wants_transcript and not transcript_allowed:
            update_variant_state(
                hydrated,
                variant_name,
                status="error",
                error=actions["transcript_only"]["disabled_reason"],
                requested_transcript=wants_transcript,
                requested_summary=wants_summary,
            )
            continue

        if wants_summary and not summary_allowed:
            update_variant_state(
                hydrated,
                variant_name,
                status="error",
                error=actions["transcript_and_summary"]["disabled_reason"],
                requested_transcript=wants_transcript,
                requested_summary=wants_summary,
            )
            continue

        request_transcript = wants_transcript and not variant.get("transcript_ready")
        request_summary = wants_summary and not variant.get("summary_ready")
        if not request_transcript and not request_summary:
            continue

        queued_any = True
        update_variant_state(
            hydrated,
            variant_name,
            status="queued",
            error=None,
            requested_transcript=request_transcript,
            requested_summary=request_summary,
        )

    if queued_any:
        update_state(hydrated, status="queued", error=None)
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
      :root {{
        color-scheme: dark;
        --bg: #12110f;
        --surface: #171614;
        --surface-2: #1c1b18;
        --surface-3: #22201c;
        --border: #34312c;
        --border-strong: #4b463d;
        --text: #f2ece3;
        --muted: #a7a091;
        --accent: #d1a247;
        --accent-dim: #8b6830;
        --danger: #db7c62;
        --success: #8ca27a;
        --shadow: 0 2px 8px rgba(0, 0, 0, 0.22);
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        background: var(--bg);
        color: var(--text);
        font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", sans-serif;
        line-height: 1.5;
      }}
      main {{
        max-width: 1320px;
        margin: 0 auto;
        padding: 24px 20px 40px;
      }}
      a {{ color: inherit; }}
      h1, h2, h3 {{ margin: 0; font-weight: 620; letter-spacing: -0.02em; }}
      p {{ margin: 0; }}
      .toolbar {{
        display: grid;
        grid-template-columns: minmax(0, 1.4fr) minmax(280px, 0.8fr);
        gap: 14px;
        margin-bottom: 16px;
      }}
      .panel {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 10px;
        box-shadow: var(--shadow);
      }}
      .panel-header {{
        padding: 16px 18px 14px;
        border-bottom: 1px solid var(--border);
      }}
      .panel-body {{
        padding: 16px 18px 18px;
      }}
      .title-block {{
        display: grid;
        gap: 10px;
      }}
      .title-row {{
        display: flex;
        justify-content: space-between;
        gap: 16px;
        align-items: start;
      }}
      .title-row h1 {{
        font-size: 1.55rem;
      }}
      .source-link {{
        display: block;
        color: var(--muted);
        text-decoration: none;
        overflow-wrap: anywhere;
      }}
      .facts {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px 14px;
      }}
      .fact {{
        display: grid;
        gap: 4px;
        padding: 10px 12px;
        background: var(--surface-2);
        border: 1px solid var(--border);
        border-radius: 8px;
      }}
      .fact-label {{
        color: var(--muted);
        font-size: 0.82rem;
      }}
      .status-panel {{
        display: grid;
        gap: 12px;
      }}
      .status-line {{
        padding: 14px 16px;
        background: var(--surface-2);
        border: 1px solid var(--border);
        border-radius: 8px;
        min-height: 56px;
      }}
      .status-line.error {{
        border-color: rgba(219, 124, 98, 0.55);
        color: #ffd2c6;
      }}
      .queue-panel {{
        margin-bottom: 16px;
      }}
      .queue-grid {{
        display: grid;
        gap: 12px;
      }}
      .queue-row {{
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto;
        gap: 16px;
        align-items: center;
        padding: 14px 16px;
        background: var(--surface-2);
        border: 1px solid var(--border);
        border-radius: 8px;
      }}
      .queue-copy {{
        display: grid;
        gap: 5px;
      }}
      .queue-title {{
        font-size: 1rem;
      }}
      .queue-meta {{
        color: var(--muted);
        font-size: 0.92rem;
      }}
      .queue-actions {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        justify-content: flex-end;
      }}
      button {{
        appearance: none;
        border: 1px solid var(--border-strong);
        background: var(--surface-3);
        color: var(--text);
        border-radius: 8px;
        padding: 9px 12px;
        font: inherit;
        cursor: pointer;
        transition: background 140ms ease, border-color 140ms ease, color 140ms ease;
      }}
      button:hover:enabled {{
        background: #2a2620;
        border-color: var(--accent-dim);
      }}
      button:disabled {{
        cursor: not-allowed;
        color: #847d71;
        background: #191815;
        border-color: #2a2722;
      }}
      .button-strong {{
        background: var(--accent);
        border-color: var(--accent);
        color: #17140f;
      }}
      .button-strong:hover:enabled {{
        background: #ddb05a;
        border-color: #ddb05a;
      }}
      .columns {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 16px;
        align-items: start;
      }}
      .column {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 10px;
        box-shadow: var(--shadow);
        overflow: hidden;
      }}
      .column-header {{
        display: flex;
        justify-content: space-between;
        gap: 16px;
        align-items: center;
        padding: 16px 18px 14px;
        border-bottom: 1px solid var(--border);
      }}
      .column-header h2 {{
        font-size: 1.05rem;
      }}
      .column-body {{
        padding: 16px 18px 18px;
        display: grid;
        gap: 16px;
      }}
      .column-status {{
        color: var(--muted);
        font-size: 0.95rem;
      }}
      .column-status.error {{
        color: #ffd2c6;
      }}
      .source-note {{
        padding: 10px 12px;
        background: var(--surface-2);
        border: 1px solid var(--border);
        border-radius: 8px;
        color: var(--muted);
        font-size: 0.92rem;
      }}
      .artifact {{
        display: grid;
        gap: 10px;
      }}
      .artifact-header {{
        display: flex;
        justify-content: space-between;
        gap: 12px;
        align-items: baseline;
      }}
      .artifact-header h3 {{
        font-size: 0.98rem;
      }}
      pre {{
        margin: 0;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
        background: #11100e;
        color: #f5efe6;
        padding: 14px;
        border-radius: 8px;
        border: 1px solid var(--border);
        min-height: 92px;
      }}
      .empty-state {{
        padding: 12px;
        border: 1px dashed var(--border);
        border-radius: 8px;
        color: var(--muted);
      }}
      .inline-note {{
        color: var(--muted);
        font-size: 0.9rem;
      }}
      @media (max-width: 900px) {{
        .toolbar,
        .columns {{ grid-template-columns: 1fr; }}
        .queue-row {{ grid-template-columns: 1fr; }}
        .queue-actions {{ justify-content: flex-start; }}
        .facts {{ grid-template-columns: 1fr; }}
      }}
    </style>
  </head>
  <body>
    <main>
      <section class="toolbar">
        <div class="panel">
          <div class="panel-header">
            <div class="title-block">
              <div class="title-row">
                <h1 id="page-title">cill.app</h1>
              </div>
              <a class="source-link" id="source-link" href={serialized_url}></a>
            </div>
          </div>
          <div class="panel-body">
            <div class="facts" id="facts">
              <div class="fact"><div class="fact-label">Uploader</div><div id="fact-uploader">Unknown</div></div>
              <div class="fact"><div class="fact-label">Duration</div><div id="fact-duration">Unknown</div></div>
              <div class="fact"><div class="fact-label">Live status</div><div id="fact-live">Unknown</div></div>
              <div class="fact"><div class="fact-label">Subtitles</div><div id="fact-subtitles">Pending</div></div>
            </div>
          </div>
        </div>
        <div class="panel">
          <div class="panel-header">
            <h2>Job state</h2>
          </div>
          <div class="panel-body status-panel">
            <div id="status" class="status-line">Loading job state…</div>
            <div id="subtitle-meta" class="inline-note"></div>
          </div>
        </div>
      </section>

      <section class="panel queue-panel">
        <div class="panel-header">
          <h2>Queue work</h2>
        </div>
        <div class="panel-body">
          <div class="queue-grid" id="queue-grid"></div>
        </div>
      </section>

      <section class="columns">
        <section class="column" id="plain-column">
          <div class="column-header">
            <h2>Plain</h2>
            <div id="plain-status" class="column-status">Idle</div>
          </div>
          <div class="column-body">
            <div id="plain-transcript-source" class="source-note" hidden></div>
            <section id="plain-transcript-section" class="artifact" hidden>
              <div class="artifact-header">
                <h3>Transcript</h3>
              </div>
              <pre id="plain-transcript"></pre>
            </section>
            <div id="plain-summary-basis" class="source-note" hidden></div>
            <section id="plain-summary-section" class="artifact" hidden>
              <div class="artifact-header">
                <h3>Summary</h3>
              </div>
              <pre id="plain-summary"></pre>
            </section>
            <div id="plain-empty" class="empty-state">Nothing queued yet.</div>
          </div>
        </section>
        <section class="column" id="diarized-column">
          <div class="column-header">
            <h2>Diarized</h2>
            <div id="diarized-status" class="column-status">Idle</div>
          </div>
          <div class="column-body">
            <div id="diarized-transcript-source" class="source-note" hidden></div>
            <section id="diarized-transcript-section" class="artifact" hidden>
              <div class="artifact-header">
                <h3>Transcript</h3>
              </div>
              <pre id="diarized-transcript"></pre>
            </section>
            <div id="diarized-summary-basis" class="source-note" hidden></div>
            <section id="diarized-summary-section" class="artifact" hidden>
              <div class="artifact-header">
                <h3>Summary</h3>
              </div>
              <pre id="diarized-summary"></pre>
            </section>
            <div id="diarized-empty" class="empty-state">Nothing queued yet.</div>
          </div>
        </section>
      </section>
    </main>
    <script>
      const sourceUrl = {serialized_url};
      const sourceLink = document.getElementById('source-link');
      const statusEl = document.getElementById('status');
      const subtitleMetaEl = document.getElementById('subtitle-meta');
      const queueGridEl = document.getElementById('queue-grid');
      const titleEl = document.getElementById('page-title');
      const uploaderEl = document.getElementById('fact-uploader');
      const durationEl = document.getElementById('fact-duration');
      const liveEl = document.getElementById('fact-live');
      const subtitlesEl = document.getElementById('fact-subtitles');
      const variantElements = {{
        plain: {{
          status: document.getElementById('plain-status'),
          transcriptSource: document.getElementById('plain-transcript-source'),
          transcriptSection: document.getElementById('plain-transcript-section'),
          transcript: document.getElementById('plain-transcript'),
          summaryBasis: document.getElementById('plain-summary-basis'),
          summarySection: document.getElementById('plain-summary-section'),
          summary: document.getElementById('plain-summary'),
          empty: document.getElementById('plain-empty'),
        }},
        diarized: {{
          status: document.getElementById('diarized-status'),
          transcriptSource: document.getElementById('diarized-transcript-source'),
          transcriptSection: document.getElementById('diarized-transcript-section'),
          transcript: document.getElementById('diarized-transcript'),
          summaryBasis: document.getElementById('diarized-summary-basis'),
          summarySection: document.getElementById('diarized-summary-section'),
          summary: document.getElementById('diarized-summary'),
          empty: document.getElementById('diarized-empty'),
        }},
      }};
      sourceLink.textContent = sourceUrl;

      const pollingStates = new Set(['queued', 'downloading', 'transcribing', 'summarizing']);
      let currentJobId = null;
      let pollSession = 0;

      function formatDuration(seconds) {{
        if (seconds === null || seconds === undefined) {{
          return 'Unknown';
        }}
        const total = Math.max(0, Math.round(seconds));
        const hours = Math.floor(total / 3600);
        const minutes = Math.floor((total % 3600) / 60);
        const secs = total % 60;
        if (hours > 0) {{
          return `${{hours}}h ${{String(minutes).padStart(2, '0')}}m`;
        }}
        return `${{minutes}}m ${{String(secs).padStart(2, '0')}}s`;
      }}

      function renderVariant(name, variant) {{
        const elements = variantElements[name];
        const detail = variant.error ? `: ${{variant.error}}` : '';
        elements.status.textContent = `${{variant.status || 'idle'}}${{detail}}`;
        elements.status.className = 'column-status' + (variant.error ? ' error' : '');

        if (variant.transcript_source_label) {{
          elements.transcriptSource.textContent = `Source: ${{variant.transcript_source_label}}`;
          elements.transcriptSource.hidden = false;
        }} else {{
          elements.transcriptSource.hidden = true;
        }}

        if (variant.summary_basis_label) {{
          elements.summaryBasis.textContent = `Based on: ${{variant.summary_basis_label}}`;
          elements.summaryBasis.hidden = false;
        }} else {{
          elements.summaryBasis.hidden = true;
        }}

        if (variant.transcript) {{
          elements.transcriptSection.hidden = false;
          elements.transcript.textContent = variant.transcript;
        }} else {{
          elements.transcriptSection.hidden = true;
        }}
        if (variant.summary) {{
          elements.summarySection.hidden = false;
          elements.summary.textContent = variant.summary;
        }} else {{
          elements.summarySection.hidden = true;
        }}

        elements.empty.hidden = Boolean(variant.transcript || variant.summary);
      }}

      function renderRapidApiState(state) {{
        const rapidapi = state.rapidapi || {{}};
        const details = [
          rapidapi.subtitle_language ? `language=${{rapidapi.subtitle_language}}` : null,
          rapidapi.subtitle_kind ? `kind=${{rapidapi.subtitle_kind}}` : null,
          rapidapi.subtitle_word_count ? `words=${{rapidapi.subtitle_word_count}}` : null,
        ].filter(Boolean).join(' · ');
        subtitleMetaEl.textContent = rapidapi.subtitle_status_label
          ? `${{rapidapi.subtitle_status_label}}${{details ? ' · ' + details : ''}}`
          : 'Subtitle lookup pending';
        subtitlesEl.textContent = rapidapi.subtitle_status_label || 'Pending';
        if (rapidapi.error) {{
          subtitleMetaEl.textContent += ` · ${{rapidapi.error}}`;
        }}
      }}

      function buildQueuePayload(variantName, includeSummary) {{
        return {{
          plain: {{ transcript: false, summary: false }},
          diarized: {{ transcript: false, summary: false }},
          [variantName]: {{
            transcript: true,
            summary: includeSummary,
          }},
        }};
      }}

      async function queueWork(variantName, includeSummary) {{
        if (!currentJobId) return;
        const response = await fetch(`/api/jobs/${{currentJobId}}/run`, {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify(buildQueuePayload(variantName, includeSummary)),
        }});
        if (!response.ok) {{
          const message = await response.text();
          throw new Error(message || 'Unable to queue work.');
        }}
        const state = await response.json();
        renderState(state);
        if (pollingStates.has(state.status)) {{
          pollSession += 1;
          poll(state.job_id, pollSession);
        }}
      }}

      function makeQueueButton(action, variantName, includeSummary, strong) {{
        const button = document.createElement('button');
        button.textContent = action.label;
        button.disabled = !action.enabled;
        if (strong) button.className = 'button-strong';
        if (action.disabled_reason) {{
          button.title = action.disabled_reason;
        }}
        if (action.enabled) {{
          button.addEventListener('click', () => {{
            queueWork(variantName, includeSummary).catch((error) => {{
              statusEl.textContent = `error — ${{error.message}}`;
              statusEl.className = 'status-line error';
            }});
          }});
        }}
        return button;
      }}

      function renderQueueRow(variantName, variant) {{
        const row = document.createElement('div');
        row.className = 'queue-row';

        const copy = document.createElement('div');
        copy.className = 'queue-copy';
        const title = document.createElement('div');
        title.className = 'queue-title';
        title.textContent = variant.label;
        const meta = document.createElement('div');
        const queueActions = variant.queue_actions || {{}};
        const transcriptAction = queueActions.transcript_only || {{}};
        const summaryAction = queueActions.transcript_and_summary || {{}};
        const queueNotes = [];
        if (!(variant.transcript_ready && variant.summary_ready) && variant.mode_hint) {{
          queueNotes.push(variant.mode_hint);
        }}
        if (variant.error) {{
          queueNotes.push(variant.error);
        }} else if (!transcriptAction.enabled && transcriptAction.disabled_reason && !variant.transcript_ready) {{
          queueNotes.push(transcriptAction.disabled_reason);
        }} else if (!summaryAction.enabled && summaryAction.disabled_reason && !variant.summary_ready) {{
          queueNotes.push(summaryAction.disabled_reason);
        }}
        meta.className = 'queue-meta';
        meta.textContent = queueNotes.filter(Boolean).join(' · ');
        copy.append(title, meta);

        const actions = document.createElement('div');
        actions.className = 'queue-actions';

        if (!variant.transcript_ready) {{
          actions.appendChild(makeQueueButton(transcriptAction, variantName, false, false));
          actions.appendChild(makeQueueButton(summaryAction, variantName, true, true));
        }} else if (!variant.summary_ready) {{
          actions.appendChild(makeQueueButton(summaryAction, variantName, true, true));
        }} else {{
          const done = document.createElement('div');
          done.className = 'inline-note';
          done.textContent = 'Already available.';
          actions.appendChild(done);
        }}

        row.append(copy, actions);
        return row;
      }}

      function renderState(state) {{
        const title = state.title ? `${{state.title}}` : 'Untitled video';
        const uploader = state.uploader ? ` by ${{state.uploader}}` : '';
        const detail = state.error ? `: ${{state.error}}` : '';
        statusEl.textContent = `${{state.status}} — ${{title}}${{uploader}}${{detail}}`;
        statusEl.className = 'status-line' + (state.error ? ' error' : '');
        titleEl.textContent = state.title || 'cill.app';
        uploaderEl.textContent = state.uploader || 'Unknown';
        durationEl.textContent = formatDuration(state.duration_seconds);
        liveEl.textContent = state.live_status || 'Unknown';
        renderRapidApiState(state);
        queueGridEl.replaceChildren(
          renderQueueRow('plain', state.variants?.plain || {{ label: 'Plain' }}),
          renderQueueRow('diarized', state.variants?.diarized || {{ label: 'Diarized' }}),
        );
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

      async function poll(jobId, sessionId) {{
        let delayMs = 5000;
        while (true) {{
          if (sessionId !== pollSession) {{
            return;
          }}
          const state = await fetchState(jobId);
          renderState(state);
          if (!pollingStates.has(state.status)) {{
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
        currentJobId = state.job_id;
        renderState(state);

        if (pollingStates.has(state.status)) {{
          pollSession += 1;
          await poll(state.job_id, pollSession);
        }}
      }}

      start().catch((error) => {{
        statusEl.textContent = `error — ${{error.message}}`;
        statusEl.className = 'status-line error';
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
                "The deployment could not access video metadata from the configured provider. "
                "Please try again later."
            ),
        )
    return JSONResponse(hydrate_state(state))


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> JSONResponse:
    state = storage.load_state(job_id)
    if not state:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JSONResponse(recover_stale_active_state(state))


@app.post("/api/jobs/{job_id}/run")
def run_job(job_id: str, payload: JobRunRequest) -> JSONResponse:
    state = storage.load_state(job_id)
    if not state:
        raise HTTPException(status_code=404, detail="Job not found.")
    hydrated = recover_stale_active_state(state)
    if hydrated["status"] in TERMINAL_STATUSES or hydrated["status"] in ACTIVE_STATUSES:
        if hydrated["status"] in ACTIVE_STATUSES:
            return JSONResponse(hydrate_state(hydrated))
    return JSONResponse(queue_requested_work(hydrated, payload))


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
