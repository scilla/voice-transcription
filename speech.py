import argparse
import datetime
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Optional

try:
    import dotenv
except ImportError:  # pragma: no cover - depends on local environment
    dotenv = None

try:
    from openai import APITimeoutError, OpenAI
except ImportError:  # pragma: no cover - depends on local environment
    APITimeoutError = Exception
    OpenAI = None

from cill.shared import (
    classify_youtube_live_status,
    describe_source_for_summary,
    is_youtube_url,
    sanitize_filename_component,
)


if dotenv is not None:
    dotenv.load_dotenv()

REQUEST_TIMEOUT_SECONDS = 300
MAX_RETRIES = 1
RETRY_BACKOFF_SECONDS = 5
DEFAULT_MODEL_NAME = "gpt-4o-transcribe"
DIARIZE_MODEL_NAME = "gpt-4o-transcribe-diarize"
SUMMARY_MODEL_NAME = "gpt-4.1-mini"

MAX_MODEL_DURATION_SECONDS = 1400
CHUNK_TARGET_DURATION_SECONDS = 900
CHUNK_OVERLAP_SECONDS = 5

SOURCES_DIR = "./sources"
YOUTUBE_SOURCES_DIR = os.path.join(SOURCES_DIR, "youtube")
OUTPUT_DIR = "./output"

SUPPORTED_SOURCE_EXTENSIONS = {
    ".mp3",
    ".mp4",
    ".opus",
    ".wav",
    ".m4a",
    ".aac",
    ".flac",
    ".ogg",
    ".webm",
    ".mkv",
    ".mov",
    ".ts",
}
DIRECT_AUDIO_EXTENSIONS = {".mp3", ".opus", ".wav", ".m4a", ".aac", ".flac", ".ogg"}

YOUTUBE_UPCOMING_STATUSES = {"is_upcoming"}
YOUTUBE_ACTIVE_LIVE_STATUSES = {"is_live"}
VERBOSE = False


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe local files or YouTube audio with OpenAI.")
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--file",
        help="Local file to transcribe. Accepts an absolute path, a path relative to the repo, or a path relative to sources/.",
    )
    source_group.add_argument(
        "--url",
        help="YouTube video or live URL to transcribe.",
    )
    parser.add_argument(
        "--language",
        help='Transcription language hint such as "it" or "en". Use "auto" or omit the flag for auto-detect.',
    )
    parser.add_argument(
        "--live-duration-minutes",
        type=float,
        help="For active live capture mode, stop automatically after this many minutes.",
    )
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="Use gpt-4o-transcribe-diarize for speaker-labeled output.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download and re-transcription even if matching artifacts already exist.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force a fresh YouTube download even if a matching local audio file already exists.",
    )
    parser.add_argument(
        "--force-transcribe",
        action="store_true",
        help="Force a fresh transcription even if a matching transcript file already exists.",
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Run a second-step summary after transcription and save it as a separate output file.",
    )
    parser.add_argument(
        "--live-now",
        action="store_true",
        help="For active YouTube lives, download the currently available audio immediately and transcribe it without waiting.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print diagnostic details about downloader, ffmpeg, and OpenAI steps.",
    )
    return parser.parse_args(argv)


def log_verbose(message: str) -> None:
    if VERBOSE:
        print(f"[verbose] {message}")


def format_command(command: list[str]) -> str:
    return shlex.join(command)


def normalize_language_hint(language_hint: Optional[str]) -> Optional[str]:
    if language_hint is None:
        return None
    normalized = language_hint.strip()
    if not normalized or normalized.lower() == "auto":
        return None
    return normalized


def validate_args(args: argparse.Namespace) -> None:
    if args.live_duration_minutes is not None and args.live_duration_minutes <= 0:
        raise SystemExit("--live-duration-minutes must be a positive number")
    if args.url and not is_youtube_url(args.url):
        raise SystemExit("Only YouTube URLs are supported with --url")


def determine_source_type(args: argparse.Namespace) -> str:
    if args.file:
        return "local"
    if args.url:
        return "youtube"
    return prompt_source_type()


def resolve_local_file_path(file_arg: str, sources_dir: str = SOURCES_DIR) -> str:
    candidates = []
    if os.path.isabs(file_arg):
        candidates.append(file_arg)
    else:
        candidates.append(os.path.abspath(file_arg))
        candidates.append(os.path.abspath(os.path.join(sources_dir, file_arg)))

    for candidate in candidates:
        if os.path.isfile(candidate):
            extension = os.path.splitext(candidate)[1].lower()
            if extension not in SUPPORTED_SOURCE_EXTENSIONS:
                raise SystemExit(
                    f"Unsupported local file extension for {candidate}. Supported extensions: "
                    f"{', '.join(sorted(SUPPORTED_SOURCE_EXTENSIONS))}"
                )
            log_verbose(f"Resolved local file argument {file_arg!r} -> {candidate}")
            return candidate

    raise SystemExit(f"Local file not found: {file_arg}")


def get_model_name(diarize: bool) -> str:
    return DIARIZE_MODEL_NAME if diarize else DEFAULT_MODEL_NAME


def get_summary_model_name() -> str:
    return SUMMARY_MODEL_NAME


def get_openai_client() -> OpenAI:
    if OpenAI is None:
        print("Error: openai is not installed. Install dependencies with: pip install -r requirements.txt")
        sys.exit(1)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY is not set. Add it to .env or your shell environment.")
        sys.exit(1)
    log_verbose("Creating OpenAI client")
    return OpenAI(api_key=api_key, timeout=REQUEST_TIMEOUT_SECONDS)


def ensure_command_available(command_name: str, install_hint: str) -> str:
    command_path = shutil.which(command_name)
    if not command_path:
        print(f"Error: {command_name} is not installed or not on PATH. {install_hint}")
        sys.exit(1)
    log_verbose(f"Resolved command {command_name} -> {command_path}")
    return command_path


def is_supported_local_source(file_name: str) -> bool:
    return os.path.splitext(file_name)[1].lower() in SUPPORTED_SOURCE_EXTENSIONS


def get_audio_files_from_sources(sources_dir: str = SOURCES_DIR) -> list[tuple[str, float]]:
    if not os.path.exists(sources_dir):
        print(f"Error: {sources_dir} folder not found")
        sys.exit(1)

    files: list[tuple[str, float]] = []
    for root, _, filenames in os.walk(sources_dir):
        for filename in filenames:
            if not is_supported_local_source(filename):
                continue
            file_path = os.path.join(root, filename)
            relative_path = os.path.relpath(file_path, sources_dir)
            files.append((relative_path, os.path.getmtime(file_path)))

    if not files:
        print("No audio or video files found in sources folder")
        sys.exit(1)

    files.sort(key=lambda item: item[1], reverse=True)
    log_verbose(f"Discovered {len(files)} supported local source files under {sources_dir}")
    return files


def prompt_source_type() -> str:
    print("\nSource type:")
    print("1. Local file")
    print("2. YouTube URL")

    while True:
        choice = input("\nSelect a source type (1-2): ").strip()
        if choice == "1":
            return "local"
        if choice == "2":
            return "youtube"
        print("Please enter 1 or 2.")


def choose_local_file(sources_dir: str = SOURCES_DIR) -> str:
    files = get_audio_files_from_sources(sources_dir)

    print("\nAvailable files (sorted by most recent):")
    for index, (relative_path, mod_time) in enumerate(files, start=1):
        modified_at = datetime.datetime.fromtimestamp(mod_time)
        print(f"{index}. {relative_path} ({modified_at.strftime('%Y-%m-%d %H:%M:%S')})")

    while True:
        try:
            choice = input(f"\nSelect a file (1-{len(files)}): ").strip()
            selected_index = int(choice) - 1
            if 0 <= selected_index < len(files):
                return os.path.join(sources_dir, files[selected_index][0])
            print(f"Please enter a number between 1 and {len(files)}")
        except ValueError:
            print("Invalid input. Please enter a number.")


def prompt_language_hint() -> Optional[str]:
    response = input(
        "\nTranscription language hint (leave blank for auto-detect, e.g. it, en): "
    ).strip()
    return response or None


def prompt_optional_max_duration_minutes() -> Optional[float]:
    while True:
        response = input(
            "Optional max duration for live capture in minutes (leave blank to record until stopped): "
        ).strip()
        if not response:
            return None
        try:
            value = float(response)
        except ValueError:
            print("Please enter a valid number of minutes or leave it blank.")
            continue
        if value <= 0:
            print("Please enter a positive number of minutes.")
            continue
        return value


def format_timestamp(seconds: float) -> str:
    total_ms = int(seconds * 1000)
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds_fraction = remainder / 1000
    return f"{hours:02d}:{minutes:02d}:{seconds_fraction:06.3f}"


def get_audio_duration(audio_path: str) -> float:
    try:
        command = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            audio_path,
        ]
        log_verbose(f"Running ffprobe: {format_command(command)}")
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip())
        log_verbose(f"Audio duration for {audio_path}: {duration:.2f}s")
        return duration
    except subprocess.CalledProcessError as exc:
        print(f"Error determining duration with ffprobe: {exc}")
        sys.exit(1)
    except ValueError:
        print("Unable to parse audio duration.")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: ffprobe is not installed. Please install ffmpeg suite first.")
        sys.exit(1)


def convert_media_to_mp3(media_path: str) -> str:
    base_name, _ = os.path.splitext(media_path)
    output_path = f"{base_name}.mp3"

    if os.path.exists(output_path) and os.path.getmtime(output_path) >= os.path.getmtime(media_path):
        print(f"Reusing extracted audio: {output_path}")
        log_verbose(f"Skipping ffmpeg conversion because extracted audio is up to date: {output_path}")
        return output_path

    try:
        command = [
            "ffmpeg",
            "-y",
            "-i",
            media_path,
            "-vn",
            "-q:a",
            "2",
            output_path,
        ]
        log_verbose(f"Running ffmpeg audio extraction: {format_command(command)}")
        subprocess.run(command, check=True, capture_output=True)
        print(f"Audio extracted: {output_path}")
        return output_path
    except subprocess.CalledProcessError as exc:
        print(f"Error extracting audio: {exc}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: ffmpeg is not installed. Please install it first.")
        sys.exit(1)


def ensure_audio_file(media_path: str) -> str:
    extension = os.path.splitext(media_path)[1].lower()
    if extension in DIRECT_AUDIO_EXTENSIONS:
        return media_path
    return convert_media_to_mp3(media_path)


def split_audio_file(
    audio_path: str,
    segment_length: int = CHUNK_TARGET_DURATION_SECONDS,
    overlap: int = CHUNK_OVERLAP_SECONDS,
) -> tuple[str, list[tuple[str, float]]]:
    if overlap >= segment_length:
        raise ValueError("Overlap must be smaller than segment length.")

    temp_dir = tempfile.mkdtemp(prefix="segments_", dir=os.getcwd())
    base_name, ext = os.path.splitext(os.path.basename(audio_path))
    total_duration = get_audio_duration(audio_path)
    step = segment_length - overlap

    offsets: list[tuple[str, float]] = []
    try:
        index = 0
        start = 0.0
        while start < total_duration:
            duration = min(segment_length, total_duration - start)
            output_file = os.path.join(temp_dir, f"{base_name}_part_{index:03d}{ext}")
            command = [
                "ffmpeg",
                "-ss",
                str(start),
                "-t",
                str(duration),
                "-i",
                audio_path,
                "-c",
                "copy",
                "-map",
                "0",
                "-avoid_negative_ts",
                "1",
                output_file,
            ]
            log_verbose(
                f"Creating chunk {index} from {start:.1f}s for {duration:.1f}s: {format_command(command)}"
            )
            subprocess.run(command, check=True, capture_output=True)
            offsets.append((output_file, start))
            index += 1
            start += step
    except subprocess.CalledProcessError as exc:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"Error splitting audio: {exc}")
        sys.exit(1)
    except FileNotFoundError:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("Error: ffmpeg is not installed. Please install it first.")
        sys.exit(1)

    if not offsets:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("Failed to create audio chunks.")
        sys.exit(1)

    return temp_dir, offsets


def prompt_youtube_url() -> str:
    while True:
        url = input("\nEnter a YouTube video or live URL: ").strip()
        if not url:
            print("Please enter a URL.")
            continue
        if not is_youtube_url(url):
            print("Only YouTube URLs are supported in v1.")
            continue
        return url


def probe_youtube_metadata(url: str, yt_dlp_path: str = "yt-dlp") -> dict:
    command = [
        yt_dlp_path,
        "--ignore-config",
        "--dump-single-json",
        "--skip-download",
        "--no-warnings",
        "--no-playlist",
        url,
    ]
    log_verbose(f"Probing YouTube metadata: {format_command(command)}")
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else str(exc)
        raise RuntimeError(f"yt-dlp metadata probe failed: {stderr}") from exc

    try:
        metadata = json.loads(result.stdout)
        log_verbose(
            f"Metadata probe completed for video_id={metadata.get('id')} live_status={metadata.get('live_status')}"
        )
        return metadata
    except json.JSONDecodeError as exc:
        raise RuntimeError("yt-dlp returned invalid metadata JSON") from exc


def build_youtube_basename(metadata: dict, suffix: Optional[str] = None) -> str:
    title = sanitize_filename_component(metadata.get("title") or "youtube-audio")
    video_id = metadata.get("id") or "unknown"
    base_name = f"{title} [{video_id}]"
    if suffix:
        safe_suffix = sanitize_filename_component(suffix, fallback="capture").replace(" ", "-")
        return f"{base_name}__{safe_suffix}"
    return base_name


def build_youtube_output_template(
    metadata: dict,
    download_dir: str = YOUTUBE_SOURCES_DIR,
    suffix: Optional[str] = None,
) -> str:
    return os.path.join(download_dir, f"{build_youtube_basename(metadata, suffix=suffix)}.%(ext)s")


def build_ytdlp_vod_command(url: str, output_template: str, yt_dlp_path: str = "yt-dlp") -> list[str]:
    return [
        yt_dlp_path,
        "--ignore-config",
        "--no-warnings",
        "--no-playlist",
        "--extract-audio",
        "--audio-format",
        "mp3",
        "--output",
        output_template,
        "--print",
        "after_move:filepath",
        url,
    ]


def build_ytdlp_live_command(url: str, output_template: str, yt_dlp_path: str = "yt-dlp") -> list[str]:
    return [
        yt_dlp_path,
        "--ignore-config",
        "--no-warnings",
        "--no-playlist",
        "--live-from-start",
        "--format",
        "bestaudio/best",
        "--no-part",
        "--output",
        output_template,
        "--print",
        "after_move:filepath",
        url,
    ]


def build_ytdlp_live_section_command(
    url: str,
    output_template: str,
    duration_seconds: int,
    yt_dlp_path: str = "yt-dlp",
) -> list[str]:
    return [
        yt_dlp_path,
        "--ignore-config",
        "--no-warnings",
        "--no-playlist",
        "--live-from-start",
        "--format",
        "bestaudio/best",
        "--no-part",
        "--download-sections",
        f"*0-{duration_seconds}",
        "--output",
        output_template,
        "--print",
        "after_move:filepath",
        url,
    ]


def build_ytdlp_live_now_command(
    url: str,
    output_template: str,
    live_duration_seconds: int,
    yt_dlp_path: str = "yt-dlp",
) -> list[str]:
    return build_ytdlp_live_section_command(
        url,
        output_template,
        duration_seconds=live_duration_seconds,
        yt_dlp_path=yt_dlp_path,
    )


def parse_yt_dlp_output_path(stdout: str) -> Optional[str]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    return lines[-1] if lines else None


def signal_process(process: subprocess.Popen, sig: int) -> None:
    try:
        if hasattr(os, "killpg"):
            os.killpg(process.pid, sig)
        else:
            process.send_signal(sig)
    except ProcessLookupError:
        pass


def wait_for_process_shutdown(process: subprocess.Popen, timeout_seconds: float = 30.0) -> None:
    try:
        process.wait(timeout=timeout_seconds)
        return
    except subprocess.TimeoutExpired:
        print("yt-dlp did not stop in time. Terminating it...")
        signal_process(process, signal.SIGTERM)

    try:
        process.wait(timeout=10.0)
        return
    except subprocess.TimeoutExpired:
        print("yt-dlp did not terminate cleanly. Killing it...")
        signal_process(process, signal.SIGKILL)
        process.wait(timeout=5.0)


def find_downloaded_youtube_file(download_dir: str, video_id: str) -> Optional[str]:
    if not os.path.isdir(download_dir):
        return None

    matches: list[tuple[float, str]] = []
    marker = f"[{video_id}]"
    for filename in os.listdir(download_dir):
        if marker not in filename:
            continue
        if filename.endswith((".part", ".temp", ".ytdl", ".info.json")):
            continue
        path = os.path.join(download_dir, filename)
        if os.path.isfile(path):
            matches.append((os.path.getmtime(path), path))

    if not matches:
        return None

    matches.sort(key=lambda item: item[0], reverse=True)
    return matches[0][1]


def find_download_from_output_template(output_template: str) -> Optional[str]:
    download_dir = os.path.dirname(output_template)
    if not os.path.isdir(download_dir):
        return None

    output_name = os.path.basename(output_template)
    if not output_name.endswith(".%(ext)s"):
        return None
    expected_stem = output_name[: -len(".%(ext)s")]

    matches: list[tuple[float, str]] = []
    for filename in os.listdir(download_dir):
        stem, extension = os.path.splitext(filename)
        if stem != expected_stem:
            continue
        if extension in {".part", ".temp", ".ytdl", ".info.json"}:
            continue
        path = os.path.join(download_dir, filename)
        if os.path.isfile(path):
            matches.append((os.path.getmtime(path), path))

    if not matches:
        return None

    matches.sort(key=lambda item: item[0], reverse=True)
    return matches[0][1]


def get_live_snapshot_duration_seconds(metadata: dict) -> int:
    duration_value = metadata.get("duration")
    if duration_value is None:
        raise RuntimeError("Unable to determine the currently available duration for this live stream")

    try:
        duration_seconds = max(1, math.ceil(float(duration_value)))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Unable to parse the currently available duration for this live stream") from exc

    return duration_seconds


def download_youtube_vod(
    url: str,
    metadata: dict,
    force_download: bool = False,
    yt_dlp_path: str = "yt-dlp",
) -> str:
    os.makedirs(YOUTUBE_SOURCES_DIR, exist_ok=True)
    if not force_download:
        existing_download = find_downloaded_youtube_file(YOUTUBE_SOURCES_DIR, metadata.get("id", ""))
        if existing_download:
            print(f"Reusing previously downloaded YouTube audio: {existing_download}")
            log_verbose(f"Skipping yt-dlp VOD download because cached audio exists: {existing_download}")
            return existing_download

    output_template = build_youtube_output_template(metadata)
    command = build_ytdlp_vod_command(url, output_template, yt_dlp_path=yt_dlp_path)
    log_verbose(f"Running yt-dlp VOD download: {format_command(command)}")

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else str(exc)
        raise RuntimeError(f"yt-dlp download failed: {stderr}") from exc

    output_path = parse_yt_dlp_output_path(result.stdout)
    if output_path and os.path.exists(output_path):
        log_verbose(f"yt-dlp VOD download wrote audio file: {output_path}")
        return output_path

    fallback = find_downloaded_youtube_file(YOUTUBE_SOURCES_DIR, metadata.get("id", ""))
    if fallback:
        log_verbose(f"Resolved yt-dlp VOD download output via fallback scan: {fallback}")
        return fallback

    raise RuntimeError("yt-dlp completed but the downloaded audio file could not be found")


def download_youtube_live_snapshot(
    url: str,
    metadata: dict,
    force_download: bool = False,
    yt_dlp_path: str = "yt-dlp",
) -> tuple[str, str]:
    live_duration_seconds = get_live_snapshot_duration_seconds(metadata)
    return download_youtube_live_section(
        url,
        metadata,
        duration_seconds=live_duration_seconds,
        suffix=f"live-now-{live_duration_seconds}s",
        force_download=force_download,
        capture_outcome="captured_now",
        yt_dlp_path=yt_dlp_path,
    )


def download_youtube_live_section(
    url: str,
    metadata: dict,
    duration_seconds: int,
    suffix: str,
    force_download: bool = False,
    capture_outcome: str = "reached_timeout",
    yt_dlp_path: str = "yt-dlp",
) -> tuple[str, str]:
    os.makedirs(YOUTUBE_SOURCES_DIR, exist_ok=True)
    output_template = build_youtube_output_template(metadata, suffix=suffix)

    if not force_download:
        existing_download = find_download_from_output_template(output_template)
        if existing_download:
            print(f"Reusing previously downloaded bounded live audio: {existing_download}")
            log_verbose(
                "Skipping yt-dlp bounded live download because cached audio exists: "
                f"{existing_download}"
            )
            return existing_download, capture_outcome

    command = build_ytdlp_live_section_command(
        url,
        output_template,
        duration_seconds=duration_seconds,
        yt_dlp_path=yt_dlp_path,
    )
    log_verbose(
        f"Running yt-dlp bounded live download for {duration_seconds}s: {format_command(command)}"
    )

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else str(exc)
        raise RuntimeError(f"yt-dlp bounded live download failed: {stderr}") from exc

    output_path = parse_yt_dlp_output_path(result.stdout)
    if output_path and os.path.exists(output_path):
        log_verbose(f"yt-dlp bounded live download wrote audio file: {output_path}")
        return output_path, capture_outcome

    fallback = find_download_from_output_template(output_template)
    if fallback:
        log_verbose(f"Resolved bounded live output via fallback scan: {fallback}")
        return fallback, capture_outcome

    raise RuntimeError("yt-dlp completed but the bounded live audio file could not be found")


def capture_youtube_live(
    url: str,
    metadata: dict,
    max_duration_minutes: Optional[float],
    force_download: bool = False,
    yt_dlp_path: str = "yt-dlp",
) -> tuple[str, str]:
    if max_duration_minutes is not None:
        duration_seconds = max(1, math.ceil(max_duration_minutes * 60))
        print(
            "\nStarting bounded live capture. "
            f"Downloading the first {duration_seconds} seconds currently available before transcription."
        )
        return download_youtube_live_section(
            url,
            metadata,
            duration_seconds=duration_seconds,
            suffix=f"live-{duration_seconds}s",
            force_download=force_download,
            capture_outcome="reached_timeout",
            yt_dlp_path=yt_dlp_path,
        )

    os.makedirs(YOUTUBE_SOURCES_DIR, exist_ok=True)
    output_template = build_youtube_output_template(metadata)
    command = build_ytdlp_live_command(url, output_template, yt_dlp_path=yt_dlp_path)

    print("\nStarting live capture. Press Ctrl+C to stop and begin transcription.")
    log_verbose(f"Running yt-dlp live capture: {format_command(command)}")
    process = subprocess.Popen(command, start_new_session=True)
    stop_requested = False
    capture_outcome = "stream_ended"

    try:
        while True:
            return_code = process.poll()
            if return_code is not None:
                break

            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping live capture. Finalizing downloaded media...")
        stop_requested = True
        capture_outcome = "manually_stopped"
        signal_process(process, signal.SIGINT)

    if stop_requested and process.poll() is None:
        wait_for_process_shutdown(process)

    output_path = find_downloaded_youtube_file(YOUTUBE_SOURCES_DIR, metadata.get("id", ""))
    if output_path:
        log_verbose(f"Live capture finalized media file: {output_path} ({capture_outcome})")
        return output_path, capture_outcome

    if process.returncode not in (0, None):
        raise RuntimeError("yt-dlp live capture stopped before any usable media file was written")

    raise RuntimeError("Live capture completed but no media file was found")


def print_youtube_metadata_summary(metadata: dict, live_status: str) -> None:
    uploader = metadata.get("channel") or metadata.get("uploader") or "Unknown"
    print("\nYouTube source detected:")
    print(f"Title: {metadata.get('title', 'Unknown')}")
    print(f"Uploader: {uploader}")
    print(f"Video ID: {metadata.get('id', 'Unknown')}")
    print(f"Live status: {live_status}")


def transcribe_chunk_with_retry(chunk_path: str, language_hint: Optional[str], diarize: bool):
    client = get_openai_client()
    last_error = None
    model_name = get_model_name(diarize)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            request_kwargs = {
                "model": model_name,
                "timeout": REQUEST_TIMEOUT_SECONDS,
            }
            if diarize:
                request_kwargs["response_format"] = "diarized_json"
                request_kwargs["chunking_strategy"] = "auto"
            if language_hint:
                request_kwargs["language"] = language_hint
            log_verbose(
                "Submitting transcription request "
                f"model={model_name} diarize={diarize} language={language_hint or 'auto'} "
                f"chunk={chunk_path}"
            )

            with open(chunk_path, "rb") as audio_file:
                return client.audio.transcriptions.create(file=audio_file, **request_kwargs)
        except APITimeoutError as exc:  # pragma: no cover - network dependent
            last_error = exc
            if attempt == MAX_RETRIES:
                raise
            wait_for = RETRY_BACKOFF_SECONDS * attempt
            print(f"Timeout on attempt {attempt}/{MAX_RETRIES}, retrying in {wait_for}s...")
            time.sleep(wait_for)
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc
            if attempt == MAX_RETRIES:
                raise
            wait_for = RETRY_BACKOFF_SECONDS * attempt
            print(f"Error on attempt {attempt}/{MAX_RETRIES}, retrying in {wait_for}s... ({exc})")
            time.sleep(wait_for)

    raise last_error


def build_output_header(
    source_context: dict,
    audio_file_path: str,
    language_hint: Optional[str],
    diarize: bool,
) -> list[str]:
    model_name = get_model_name(diarize)
    header = [f"\n#########\n{datetime.datetime.now()}"]
    header.append(f"Source file: {source_context['selected_source_path']}")
    header.append(f"Processed file: {audio_file_path}")
    header.append(f"Model: {model_name}")
    header.append(f"Diarization: {'enabled' if diarize else 'disabled'}")
    header.append(f"Language hint: {language_hint or 'auto'}")

    if source_context["source_type"] == "youtube":
        header.append(f"YouTube URL: {source_context['youtube_url']}")
        header.append(f"YouTube title: {source_context['youtube_title'] or 'Unknown'}")
        header.append(f"YouTube uploader: {source_context['youtube_uploader'] or 'Unknown'}")
        header.append(f"YouTube video id: {source_context['youtube_video_id'] or 'Unknown'}")
        header.append(f"YouTube live status: {source_context['youtube_live_status'] or 'Unknown'}")
        if source_context.get("youtube_live_capture_outcome"):
            header.append(
                f"YouTube live capture outcome: {source_context['youtube_live_capture_outcome']}"
            )
        header.append(f"Local audio path: {audio_file_path}")

    return header


def build_transcription_output_path(
    source_context: dict,
    language_hint: Optional[str],
    diarize: bool,
) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(source_context["selected_source_path"]))[0]
    mode_tag = "diarized" if diarize else "plain"
    language_tag = sanitize_filename_component(language_hint or "auto", fallback="auto").replace(" ", "-")
    output_filename = f"{base_name}__{mode_tag}__lang-{language_tag}.txt"
    return os.path.join(OUTPUT_DIR, output_filename)


def build_summary_output_path(
    source_context: dict,
    language_hint: Optional[str],
    diarize: bool,
) -> str:
    transcript_output_path = build_transcription_output_path(source_context, language_hint, diarize)
    return transcript_output_path[:-4] + "__summary.txt"


def should_skip_transcription(
    source_context: dict,
    output_path: str,
    force_transcribe: bool,
) -> bool:
    if force_transcribe:
        return False
    if source_context["source_type"] != "youtube":
        return False
    return os.path.exists(output_path)


def should_skip_summary(output_path: str, force_transcribe: bool) -> bool:
    if force_transcribe:
        return False
    return os.path.exists(output_path)


def extract_transcript_text_from_output(output_path: str) -> str:
    with open(output_path, "r", encoding="utf-8") as output_file:
        content = output_file.read().strip()

    full_transcript_marker = "\nFull transcript:\n"
    if full_transcript_marker in content:
        return content.split(full_transcript_marker, 1)[1].strip()

    parts = content.split("\n\n", 1)
    if len(parts) == 2:
        return parts[1].strip()
    return content


def summarize_transcript(transcript_text: str, source_context: dict) -> str:
    client = get_openai_client()
    log_verbose(
        "Submitting summary request "
        f"model={get_summary_model_name()} transcript_chars={len(transcript_text)}"
    )
    response = client.chat.completions.create(
        model=get_summary_model_name(),
        messages=[
            {
                "role": "system",
                "content": (
                    "You summarize video transcripts. Return a concise summary with these headings: "
                    "Overview, Key Points, and Notable Takeaways."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{describe_source_for_summary(source_context)}\n\n"
                    "Summarize the following transcript.\n\n"
                    f"{transcript_text}"
                ),
            },
        ],
    )
    content = response.choices[0].message.content
    if isinstance(content, list):
        return "\n".join(
            part.text for part in content if getattr(part, "type", None) == "text" and getattr(part, "text", None)
        ).strip()
    return (content or "").strip()


def write_summary_output(
    source_context: dict,
    transcript_output_path: str,
    language_hint: Optional[str],
    diarize: bool,
    summary_text: str,
) -> str:
    output_path = build_summary_output_path(source_context, language_hint, diarize)
    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write(f"\n#########\n{datetime.datetime.now()}\n")
        output_file.write(f"Transcript source: {transcript_output_path}\n")
        output_file.write(f"Summary model: {get_summary_model_name()}\n")
        output_file.write(f"Diarization: {'enabled' if diarize else 'disabled'}\n")
        output_file.write(f"Language hint: {language_hint or 'auto'}\n")
        if source_context["source_type"] == "youtube":
            output_file.write(f"YouTube URL: {source_context['youtube_url']}\n")
            output_file.write(f"YouTube live status: {source_context['youtube_live_status'] or 'Unknown'}\n")
            if source_context.get("youtube_live_capture_outcome"):
                output_file.write(
                    f"YouTube live capture outcome: {source_context['youtube_live_capture_outcome']}\n"
                )
        output_file.write("\nSummary:\n")
        output_file.write(summary_text)
        output_file.write("\n")
    return output_path


def write_transcription_output(
    source_context: dict,
    audio_file_path: str,
    language_hint: Optional[str],
    diarize: bool,
    transcription,
    segments_output: list[str],
    full_text_output: str,
) -> str:
    output_path = build_transcription_output_path(source_context, language_hint, diarize)

    with open(output_path, "w", encoding="utf-8") as output_file:
        for line in build_output_header(source_context, audio_file_path, language_hint, diarize):
            output_file.write(line + "\n")
        output_file.write("\n")
        if diarize and transcription and hasattr(transcription, "segments"):
            output_file.write("Segments:\n")
            for line in segments_output:
                output_file.write(line + "\n")
            output_file.write("\nFull transcript:\n")
            output_file.write(full_text_output)
        else:
            output_file.write(full_text_output)
        output_file.write("\n\n")

    return output_path


def transcribe_audio_file(
    audio_file_path: str,
    language_hint: Optional[str],
    diarize: bool,
) -> tuple[object, list[str], str]:
    print(f"Transcribing: {audio_file_path}")

    total_duration = get_audio_duration(audio_file_path)
    needs_chunking = total_duration > MAX_MODEL_DURATION_SECONDS

    if needs_chunking:
        print(
            f"Audio duration {total_duration:.1f}s exceeds model limit ({MAX_MODEL_DURATION_SECONDS}s). Splitting into chunks."
        )
        chunk_dir, chunks_with_offsets = split_audio_file(audio_file_path)
    else:
        chunk_dir = None
        chunks_with_offsets = [(audio_file_path, 0.0)]

    segments_output: list[str] = []
    full_text_parts: list[str] = []
    transcription = None

    try:
        for chunk_path, offset in chunks_with_offsets:
            print(f"Processing chunk: {os.path.basename(chunk_path)} (offset {offset:.1f}s)")
            transcription = transcribe_chunk_with_retry(chunk_path, language_hint, diarize)

            if diarize and hasattr(transcription, "segments"):
                for segment in transcription.segments:
                    segment_text = segment.text.strip()
                    start_time = segment.start + offset
                    end_time = segment.end + offset
                    segments_output.append(
                        f"[{format_timestamp(start_time)} - {format_timestamp(end_time)}] {segment.speaker}: {segment_text}"
                    )
                full_text_parts.append(transcription.text.strip())
            else:
                if hasattr(transcription, "text"):
                    text_value = transcription.text.strip()
                else:
                    text_value = str(transcription).strip()
                segments_output.append(text_value)
                full_text_parts.append(text_value)
    finally:
        if chunk_dir:
            shutil.rmtree(chunk_dir, ignore_errors=True)

    full_text_output = "\n".join([part for part in full_text_parts if part])
    return transcription, segments_output, full_text_output


def main() -> None:
    global VERBOSE
    args = parse_args()
    VERBOSE = args.verbose
    validate_args(args)
    log_verbose(f"Parsed CLI args: {args}")
    diarize = args.diarize
    summarize = args.summarize
    live_now = args.live_now
    force_download = args.force or args.force_download
    force_transcribe = args.force or args.force_transcribe
    source_type = determine_source_type(args)
    language_hint = normalize_language_hint(args.language)

    if source_type == "local":
        if args.file:
            selected_source_path = resolve_local_file_path(args.file)
        else:
            selected_source_path = choose_local_file()
        if language_hint is None and args.language is None:
            language_hint = prompt_language_hint()
        source_context = {
            "source_type": "local",
            "selected_source_path": selected_source_path,
            "youtube_url": None,
            "youtube_title": None,
            "youtube_uploader": None,
            "youtube_video_id": None,
            "youtube_live_status": None,
            "youtube_live_capture_outcome": None,
        }
    else:
        yt_dlp_path = ensure_command_available("yt-dlp", "Install it with: brew install yt-dlp")
        url = args.url if args.url else prompt_youtube_url()
        metadata = probe_youtube_metadata(url, yt_dlp_path=yt_dlp_path)
        live_status = classify_youtube_live_status(metadata)
        print_youtube_metadata_summary(metadata, live_status)

        if live_status in YOUTUBE_UPCOMING_STATUSES:
            print("Scheduled/upcoming YouTube lives are not supported in v1.")
            sys.exit(1)

        if language_hint is None and args.language is None:
            language_hint = prompt_language_hint()

        if live_status in YOUTUBE_ACTIVE_LIVE_STATUSES:
            if live_now:
                selected_source_path, live_capture_outcome = download_youtube_live_snapshot(
                    url,
                    metadata,
                    force_download=force_download,
                    yt_dlp_path=yt_dlp_path,
                )
            else:
                if args.live_duration_minutes is not None:
                    max_duration_minutes = args.live_duration_minutes
                else:
                    max_duration_minutes = prompt_optional_max_duration_minutes()
                selected_source_path, live_capture_outcome = capture_youtube_live(
                    url,
                    metadata,
                    max_duration_minutes=max_duration_minutes,
                    force_download=force_download,
                    yt_dlp_path=yt_dlp_path,
                )
        else:
            if args.live_duration_minutes is not None:
                log_verbose("--live-duration-minutes ignored because the source is not an active live stream")
            live_capture_outcome = None
            selected_source_path = download_youtube_vod(
                url,
                metadata,
                force_download=force_download,
                yt_dlp_path=yt_dlp_path,
            )

        source_context = {
            "source_type": "youtube",
            "selected_source_path": selected_source_path,
            "youtube_url": url,
            "youtube_title": metadata.get("title"),
            "youtube_uploader": metadata.get("channel") or metadata.get("uploader"),
            "youtube_video_id": metadata.get("id"),
            "youtube_live_status": live_status,
            "youtube_live_capture_outcome": live_capture_outcome,
        }

    audio_file_path = ensure_audio_file(selected_source_path)
    transcript_output_path = build_transcription_output_path(source_context, language_hint, diarize)
    if should_skip_transcription(source_context, transcript_output_path, force_transcribe):
        print(f"Skipping transcription; existing transcript found at {transcript_output_path}")
        full_text_output = extract_transcript_text_from_output(transcript_output_path)
        output_path = transcript_output_path
        transcription = None
        segments_output = []
    else:
        transcription, segments_output, full_text_output = transcribe_audio_file(
            audio_file_path,
            language_hint,
            diarize,
        )
        output_path = write_transcription_output(
            source_context,
            audio_file_path,
            language_hint,
            diarize,
            transcription,
            segments_output,
            full_text_output,
        )

    if diarize and segments_output:
        print("Segments:")
        print("\n".join(segments_output))
    elif transcription is not None:
        print("Transcript:")
        print(full_text_output)
    print(f"\nFull transcript saved to {output_path}")

    if summarize:
        summary_output_path = build_summary_output_path(source_context, language_hint, diarize)
        if should_skip_summary(summary_output_path, force_transcribe):
            print(f"Skipping summary; existing summary found at {summary_output_path}")
            return

        summary_text = summarize_transcript(full_text_output, source_context)
        written_summary_path = write_summary_output(
            source_context,
            output_path,
            language_hint,
            diarize,
            summary_text,
        )
        print("\nSummary:")
        print(summary_text)
        print(f"\nSummary saved to {written_summary_path}")


if __name__ == "__main__":
    main()
