from __future__ import annotations

import html
import json
import os
import re
import xml.etree.ElementTree as ET
from typing import Any, Optional

import requests


RAPIDAPI_HOST = os.getenv("CILL_RAPIDAPI_HOST", "youtube-media-downloader.p.rapidapi.com")
RAPIDAPI_BASE_URL = f"https://{RAPIDAPI_HOST}"
DEFAULT_TIMEOUT_SECONDS = float(os.getenv("CILL_RAPIDAPI_TIMEOUT_SECONDS", "20"))
KEY_ENV_NAMES = ("X_RAPIDAPI_KEY", "CILL_RAPIDAPI_KEY", "RAPIDAPI_KEY")
TIMECODE_RE = re.compile(r"^\d{1,2}:\d{2}(?::\d{2})?[,.]\d{3}\s+-->\s+\d{1,2}:\d{2}(?::\d{2})?[,.]\d{3}")
TAG_RE = re.compile(r"<[^>]+>")
WORD_RE = re.compile(r"\b[\w']+\b", flags=re.UNICODE)
MIN_VALID_SUBTITLE_WORD_COUNT = int(os.getenv("CILL_MIN_VALID_SUBTITLE_WORD_COUNT", "120"))
MIN_VALID_SUBTITLE_LINE_COUNT = int(os.getenv("CILL_MIN_VALID_SUBTITLE_LINE_COUNT", "20"))
MIN_VALID_SUBTITLE_CHAR_COUNT = int(os.getenv("CILL_MIN_VALID_SUBTITLE_CHAR_COUNT", "400"))
MIN_VALID_SUBTITLE_WORDS_PER_MINUTE = float(os.getenv("CILL_MIN_VALID_SUBTITLE_WPM", "60"))
MAX_VALID_SUBTITLE_WORDS_PER_MINUTE = float(os.getenv("CILL_MAX_VALID_SUBTITLE_WPM", "260"))
MIN_VALID_SUBTITLE_COVERAGE_RATIO = float(os.getenv("CILL_MIN_VALID_SUBTITLE_COVERAGE", "0.45"))
MAX_VALID_SUBTITLE_COVERAGE_RATIO = float(os.getenv("CILL_MAX_VALID_SUBTITLE_COVERAGE", "2.5"))


class RapidAPIYoutubeError(RuntimeError):
    pass


def get_api_key(explicit_key: Optional[str] = None) -> Optional[str]:
    if explicit_key:
        return explicit_key
    for env_name in KEY_ENV_NAMES:
        value = os.getenv(env_name)
        if value:
            return value.strip()
    return None


def is_configured(explicit_key: Optional[str] = None) -> bool:
    return bool(get_api_key(explicit_key))


class RapidAPIYoutubeClient:
    def __init__(
        self,
        api_key: str,
        *,
        host: str = RAPIDAPI_HOST,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        session: Optional[requests.Session] = None,
    ) -> None:
        if not api_key:
            raise RapidAPIYoutubeError("RapidAPI key is not configured.")
        self.api_key = api_key
        self.host = host
        self.base_url = f"https://{host}"
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()

    @classmethod
    def from_env(cls) -> "RapidAPIYoutubeClient":
        api_key = get_api_key()
        if not api_key:
            raise RapidAPIYoutubeError("RapidAPI key is not configured.")
        return cls(api_key)

    def _headers(self) -> dict[str, str]:
        return {
            "x-rapidapi-host": self.host,
            "x-rapidapi-key": self.api_key,
        }

    def _get(self, path: str, params: dict[str, Any]) -> requests.Response:
        response = self.session.get(
            f"{self.base_url}{path}",
            headers=self._headers(),
            params=params,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response

    def get_video_details(
        self,
        video_id: str,
        *,
        lang: str = "en",
        subtitles: bool = True,
        videos: str = "auto",
        audios: str = "auto",
        url_access: str = "normal",
    ) -> dict[str, Any]:
        response = self._get(
            "/v2/video/details",
            {
                "videoId": video_id,
                "lang": lang,
                "subtitles": json.dumps(subtitles).lower(),
                "videos": videos,
                "audios": audios,
                "urlAccess": url_access,
            },
        )
        try:
            return response.json()
        except ValueError as exc:  # pragma: no cover - provider dependent
            raise RapidAPIYoutubeError("RapidAPI returned invalid JSON for video details.") from exc

    def get_subtitle_text(
        self,
        subtitle_url: str,
        *,
        format: str = "srt",
        fix_overlap: bool = True,
        target_lang: Optional[str] = None,
    ) -> str:
        params: dict[str, Any] = {
            "subtitleUrl": subtitle_url,
            "format": format,
            "fixOverlap": json.dumps(fix_overlap).lower(),
        }
        if target_lang:
            params["targetLang"] = target_lang
        response = self._get("/v2/video/subtitles", params)
        return response.text


def extract_subtitle_tracks(details: dict[str, Any]) -> list[dict[str, Any]]:
    items = ((details.get("subtitles") or {}).get("items") or [])
    tracks: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        url = item.get("url")
        if not url:
            continue
        tracks.append(
            {
                "url": url,
                "code": item.get("code") or item.get("languageCode"),
                "label": item.get("text") or item.get("label") or item.get("name"),
                "kind": item.get("kind"),
                "auto_generated": is_auto_generated_track(item),
            }
        )
    return tracks


def is_auto_generated_track(track: dict[str, Any]) -> bool:
    text = " ".join(
        str(track.get(field, ""))
        for field in ("kind", "text", "label", "name")
        if track.get(field)
    ).lower()
    return "auto" in text or "generated" in text


def choose_subtitle_track(
    tracks: list[dict[str, Any]],
    *,
    preferred_languages: tuple[str, ...] = ("en", "en-US", "en-GB"),
) -> Optional[dict[str, Any]]:
    if not tracks:
        return None

    def sort_key(track: dict[str, Any]) -> tuple[int, int, str]:
        code = track.get("code") or ""
        language_rank = len(preferred_languages) + 1
        for index, preferred in enumerate(preferred_languages):
            if code == preferred:
                language_rank = index
                break
            if code.startswith(f"{preferred}-"):
                language_rank = index
                break
        auto_rank = 1 if track.get("auto_generated") else 0
        return (language_rank, auto_rank, code)

    return sorted(tracks, key=sort_key)[0]


def normalize_subtitle_text(raw_text: str) -> str:
    content = raw_text.strip()
    if not content:
        return ""
    if content.startswith("<") and "<text" in content:
        return _normalize_xml_subtitles(content)
    return _normalize_timed_text_subtitles(content)


def _normalize_xml_subtitles(raw_text: str) -> str:
    root = ET.fromstring(raw_text)
    lines: list[str] = []
    for node in root.iter():
        if node.tag != "text":
            continue
        text = "".join(node.itertext()).strip()
        if not text:
            continue
        lines.append(html.unescape(text))
    return "\n".join(lines).strip()


def _normalize_timed_text_subtitles(raw_text: str) -> str:
    lines: list[str] = []
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        upper_line = line.upper()
        if upper_line == "WEBVTT" or upper_line.startswith("NOTE"):
            continue
        if line.isdigit():
            continue
        if TIMECODE_RE.match(line):
            continue
        clean = html.unescape(TAG_RE.sub("", line)).strip()
        if clean:
            lines.append(clean)
    return "\n".join(lines).strip()


def build_subtitle_stats(text: str, duration_seconds: Optional[float]) -> dict[str, Any]:
    words = WORD_RE.findall(text)
    non_empty_lines = [line for line in text.splitlines() if line.strip()]
    duration_minutes = round(duration_seconds / 60, 2) if duration_seconds and duration_seconds > 0 else None
    estimated_minutes = round(len(words) / 150, 2) if words else None
    coverage_ratio = None
    if duration_minutes and estimated_minutes is not None:
        coverage_ratio = round(estimated_minutes / duration_minutes, 2)
    stats: dict[str, Any] = {
        "char_count": len(text),
        "word_count": len(words),
        "line_count": len(non_empty_lines),
        "duration_seconds": duration_seconds,
        "duration_minutes": duration_minutes,
        "words_per_minute": None,
        "estimated_minutes_from_words": estimated_minutes,
        "coverage_ratio": coverage_ratio,
    }
    if duration_seconds and duration_seconds > 0:
        stats["words_per_minute"] = round(len(words) / (duration_seconds / 60), 1)
    return stats


def validate_subtitle_stats(stats: dict[str, Any]) -> dict[str, Any]:
    word_count = int(stats.get("word_count") or 0)
    line_count = int(stats.get("line_count") or 0)
    char_count = int(stats.get("char_count") or 0)
    words_per_minute = stats.get("words_per_minute")
    coverage_ratio = stats.get("coverage_ratio")

    if char_count < MIN_VALID_SUBTITLE_CHAR_COUNT:
        return {"usable": False, "reason": "subtitle_text_too_short"}
    if line_count < MIN_VALID_SUBTITLE_LINE_COUNT:
        return {"usable": False, "reason": "subtitle_line_count_too_low"}
    if word_count < MIN_VALID_SUBTITLE_WORD_COUNT:
        return {"usable": False, "reason": "subtitle_word_count_too_low"}
    if words_per_minute is not None and not (
        MIN_VALID_SUBTITLE_WORDS_PER_MINUTE <= float(words_per_minute) <= MAX_VALID_SUBTITLE_WORDS_PER_MINUTE
    ):
        return {"usable": False, "reason": "subtitle_words_per_minute_out_of_range"}
    if coverage_ratio is not None and not (
        MIN_VALID_SUBTITLE_COVERAGE_RATIO <= float(coverage_ratio) <= MAX_VALID_SUBTITLE_COVERAGE_RATIO
    ):
        return {"usable": False, "reason": "subtitle_coverage_ratio_out_of_range"}
    return {"usable": True, "reason": "ok"}
