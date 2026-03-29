from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse


def sanitize_filename_component(value: str, fallback: str = "untitled") -> str:
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", value)
    sanitized = re.sub(r"\s+", " ", sanitized).strip().strip(".")
    if not sanitized:
        sanitized = fallback
    return sanitized[:160].rstrip()


def is_youtube_url(url: str) -> bool:
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    return hostname in {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}


def extract_youtube_video_id(url: str) -> str | None:
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()

    if hostname == "youtu.be":
        path = parsed.path.strip("/")
        return path or None

    if hostname in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        if parsed.path == "/watch":
            return parse_qs(parsed.query).get("v", [None])[0]
        path = parsed.path.strip("/")
        if path.startswith("shorts/"):
            return path.split("/", 1)[1] or None
        if path.startswith("live/"):
            return path.split("/", 1)[1] or None

    return None


def classify_youtube_live_status(metadata: dict) -> str:
    live_status = metadata.get("live_status")
    if live_status:
        return live_status
    if metadata.get("is_live"):
        return "is_live"
    if metadata.get("was_live"):
        return "was_live"
    return "not_live"


def describe_source_for_summary(source_context: dict) -> str:
    if source_context["source_type"] != "youtube":
        return "Source type: local file. This source is not a YouTube live video."

    live_status = source_context.get("youtube_live_status")
    if live_status == "is_live":
        outcome_map = {
            "stream_ended": "The source was a live video and the capture ended because the stream ended.",
            "reached_timeout": "The source was a live video and the capture stopped because it reached the requested timeout.",
            "manually_stopped": "The source was a live video and the capture stopped because it was manually stopped.",
            "captured_now": "The source was a live video and this transcript covers the audio that was available at request time.",
        }
        return outcome_map.get(
            source_context.get("youtube_live_capture_outcome"),
            "The source was a live video and the capture stop reason is unknown.",
        )

    if live_status in {"was_live", "post_live"}:
        return "Source type: archived YouTube live recording. It is not currently live."

    return "Source type: standard YouTube video. It is not a live video."
