from __future__ import annotations

import json
import mimetypes
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from cill.env import load_project_dotenv

try:  # pragma: no cover - depends on installed extras
    from vercel.blob import get as vercel_get
    from vercel.blob import list_objects as vercel_list_objects
    from vercel.blob import put as vercel_put
except ImportError:  # pragma: no cover - depends on installed extras
    vercel_get = None
    vercel_list_objects = None
    vercel_put = None


LOCAL_WEB_CACHE_DIRNAME = "web-cache"
WEB_STATE_FILENAME = "meta.json"
WEB_TRANSCRIPT_FILENAME = "transcript.txt"
WEB_SUMMARY_FILENAME = "summary.txt"
WEB_DIARIZED_TRANSCRIPT_FILENAME = "diarized_transcript.txt"
WEB_DIARIZED_SUMMARY_FILENAME = "diarized_summary.txt"
WEB_RAPIDAPI_SUBTITLE_METADATA_FILENAME = "rapidapi_subtitle_metadata.json"
WEB_RAPIDAPI_SUBTITLE_TEXT_FILENAME = "rapidapi_subtitle_text.txt"


class StorageBackend(ABC):
    @abstractmethod
    def load_state(self, job_id: str) -> Optional[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def save_state(self, job_id: str, state: dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def read_text(self, job_id: str, filename: str, video_id: Optional[str] = None) -> Optional[str]:
        raise NotImplementedError

    @abstractmethod
    def write_text(self, job_id: str, filename: str, value: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_states(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    def find_cached_audio(self, video_id: str) -> Optional[str]:
        return None


class LocalStorageBackend(StorageBackend):
    def __init__(self, output_dir: str = "./output", audio_dir: str = "./sources/youtube") -> None:
        self.output_dir = Path(output_dir)
        self.audio_dir = Path(audio_dir)
        self.web_cache_dir = self.output_dir / LOCAL_WEB_CACHE_DIRNAME
        self.web_cache_dir.mkdir(parents=True, exist_ok=True)

    def _job_dir(self, job_id: str) -> Path:
        return self.web_cache_dir / job_id

    def _job_path(self, job_id: str, filename: str) -> Path:
        return self._job_dir(job_id) / filename

    def load_state(self, job_id: str) -> Optional[dict[str, Any]]:
        state_path = self._job_path(job_id, WEB_STATE_FILENAME)
        if not state_path.exists():
            return None
        return json.loads(state_path.read_text(encoding="utf-8"))

    def save_state(self, job_id: str, state: dict[str, Any]) -> None:
        job_dir = self._job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        self._job_path(job_id, WEB_STATE_FILENAME).write_text(
            json.dumps(state, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def read_text(self, job_id: str, filename: str, video_id: Optional[str] = None) -> Optional[str]:
        local_path = self._job_path(job_id, filename)
        if local_path.exists():
            return local_path.read_text(encoding="utf-8")

        if not video_id:
            return None

        if filename == WEB_TRANSCRIPT_FILENAME:
            return self._read_legacy_transcript(video_id)
        if filename == WEB_SUMMARY_FILENAME:
            return self._read_legacy_summary(video_id)
        if filename == WEB_DIARIZED_TRANSCRIPT_FILENAME:
            return self._read_legacy_diarized_transcript(video_id)
        if filename == WEB_DIARIZED_SUMMARY_FILENAME:
            return self._read_legacy_diarized_summary(video_id)
        return None

    def write_text(self, job_id: str, filename: str, value: str) -> None:
        job_dir = self._job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        self._job_path(job_id, filename).write_text(value, encoding="utf-8")

    def list_states(self) -> list[dict[str, Any]]:
        if not self.web_cache_dir.exists():
            return []

        states: list[dict[str, Any]] = []
        for state_path in sorted(self.web_cache_dir.glob(f"*/{WEB_STATE_FILENAME}")):
            try:
                states.append(json.loads(state_path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                continue
        return states

    def find_cached_audio(self, video_id: str) -> Optional[str]:
        if not self.audio_dir.exists():
            return None

        marker = f"[{video_id}]"
        matches: list[tuple[float, str]] = []
        for candidate in self.audio_dir.iterdir():
            if marker not in candidate.name or not candidate.is_file():
                continue
            matches.append((candidate.stat().st_mtime, str(candidate)))

        if not matches:
            return None

        matches.sort(key=lambda item: item[0], reverse=True)
        return matches[0][1]

    def _read_legacy_transcript(self, video_id: str) -> Optional[str]:
        return self._read_legacy_variant_output(video_id, mode_tag="plain", summary=False)

    def _read_legacy_summary(self, video_id: str) -> Optional[str]:
        return self._read_legacy_variant_output(video_id, mode_tag="plain", summary=True)

    def _read_legacy_diarized_transcript(self, video_id: str) -> Optional[str]:
        return self._read_legacy_variant_output(video_id, mode_tag="diarized", summary=False)

    def _read_legacy_diarized_summary(self, video_id: str) -> Optional[str]:
        return self._read_legacy_variant_output(video_id, mode_tag="diarized", summary=True)

    def _read_legacy_variant_output(self, video_id: str, *, mode_tag: str, summary: bool) -> Optional[str]:
        marker = f"[{video_id}]"
        suffix = "__summary.txt" if summary else ".txt"
        matches = sorted(
            [
                path
                for path in self.output_dir.iterdir()
                if path.is_file()
                and marker in path.name
                and f"__{mode_tag}__lang-auto" in path.name
                and path.name.endswith(suffix)
                and (summary or not path.name.endswith("__summary.txt"))
            ],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not matches:
            return None

        content = matches[0].read_text(encoding="utf-8").strip()
        if summary:
            summary_marker = "\nSummary:\n"
            if summary_marker in content:
                return content.split(summary_marker, 1)[1].strip()
        else:
            full_transcript_marker = "\nFull transcript:\n"
            if full_transcript_marker in content:
                return content.split(full_transcript_marker, 1)[1].strip()
        parts = content.split("\n\n", 1)
        if len(parts) == 2:
            return parts[1].strip()
        return content


class BlobStorageBackend(StorageBackend):
    def __init__(self, prefix: str = "cill-web", token: Optional[str] = None) -> None:
        if vercel_get is None or vercel_list_objects is None or vercel_put is None:  # pragma: no cover - depends on installed extras
            raise RuntimeError("vercel Blob SDK dependencies are not installed")

        self.prefix = prefix.strip("/ ")
        self.token = token or os.getenv("BLOB_READ_WRITE_TOKEN")
        if not self.token:
            raise RuntimeError("BLOB_READ_WRITE_TOKEN is not set")

    def _job_prefix(self, job_id: str) -> str:
        return f"{self.prefix}/jobs/{job_id}"

    def _pathname(self, job_id: str, filename: str) -> str:
        return f"{self._job_prefix(job_id)}/{filename}"

    def _list_job_blobs(self, job_id: str) -> dict[str, dict[str, Any]]:
        response = vercel_list_objects(token=self.token, prefix=self._job_prefix(job_id))
        return {
            blob.pathname: {
                "pathname": blob.pathname,
                "url": blob.url,
                "uploadedAt": blob.uploaded_at.isoformat() if getattr(blob, "uploaded_at", None) else "",
            }
            for blob in response.blobs
        }

    def _resolve_blob(self, blobs: dict[str, dict[str, Any]], job_id: str, filename: str) -> Optional[dict[str, Any]]:
        exact_pathname = self._pathname(job_id, filename)
        blob = blobs.get(exact_pathname)
        if blob:
            return blob

        path = Path(filename)
        stem_prefix = f"{self._job_prefix(job_id)}/{path.stem}-"
        suffix = path.suffix
        matches = [
            candidate
            for pathname, candidate in blobs.items()
            if pathname.startswith(stem_prefix) and pathname.endswith(suffix)
        ]
        if not matches:
            return None

        matches.sort(key=lambda item: item.get("uploadedAt", ""), reverse=True)
        return matches[0]

    def _read_blob_text(self, pathname: str) -> str:
        result = vercel_get(pathname, access="private", token=self.token, use_cache=False)
        return result.content.decode("utf-8")

    def _put_private_blob(self, pathname: str, data: bytes, content_type: str) -> None:
        vercel_put(
            pathname,
            data,
            access="private",
            content_type=content_type,
            overwrite=True,
            token=self.token,
        )

    def _list_all_job_blobs(self) -> list[dict[str, Any]]:
        response = vercel_list_objects(token=self.token, prefix=f"{self.prefix}/jobs/")
        return [
            {
                "pathname": blob.pathname,
                "url": blob.url,
                "uploadedAt": blob.uploaded_at.isoformat() if getattr(blob, "uploaded_at", None) else "",
            }
            for blob in response.blobs
        ]

    def load_state(self, job_id: str) -> Optional[dict[str, Any]]:
        blobs = self._list_job_blobs(job_id)
        blob = self._resolve_blob(blobs, job_id, WEB_STATE_FILENAME)
        if not blob:
            return None
        return json.loads(self._read_blob_text(blob["pathname"]))

    def save_state(self, job_id: str, state: dict[str, Any]) -> None:
        self._put_private_blob(
            self._pathname(job_id, WEB_STATE_FILENAME),
            json.dumps(state, indent=2, sort_keys=True).encode("utf-8"),
            "application/json",
        )

    def read_text(self, job_id: str, filename: str, video_id: Optional[str] = None) -> Optional[str]:
        blobs = self._list_job_blobs(job_id)
        blob = self._resolve_blob(blobs, job_id, filename)
        if not blob:
            return None
        return self._read_blob_text(blob["pathname"])

    def write_text(self, job_id: str, filename: str, value: str) -> None:
        self._put_private_blob(
            self._pathname(job_id, filename),
            value.encode("utf-8"),
            mimetypes.guess_type(filename)[0] or "text/plain",
        )

    def list_states(self) -> list[dict[str, Any]]:
        candidates: dict[str, dict[str, Any]] = {}
        prefix = f"{self.prefix}/jobs/"
        for blob in self._list_all_job_blobs():
            pathname = blob.get("pathname", "")
            if not pathname.startswith(prefix):
                continue

            remainder = pathname[len(prefix) :]
            parts = remainder.split("/", 1)
            if len(parts) != 2:
                continue

            job_id, filename = parts
            if filename != WEB_STATE_FILENAME and not (
                filename.startswith("meta-") and filename.endswith(".json")
            ):
                continue

            current = candidates.get(job_id)
            if current and current.get("uploadedAt", "") >= blob.get("uploadedAt", ""):
                continue
            candidates[job_id] = blob

        states: list[dict[str, Any]] = []
        for blob in candidates.values():
            try:
                states.append(json.loads(self._read_blob_text(blob["pathname"])))
            except (KeyError, json.JSONDecodeError):
                continue
        return states


def create_storage_backend() -> StorageBackend:
    load_project_dotenv()
    if os.getenv("BLOB_READ_WRITE_TOKEN"):
        return BlobStorageBackend()
    return LocalStorageBackend()
