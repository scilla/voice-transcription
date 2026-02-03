import os
import time
from openai import APITimeoutError, OpenAI
import dotenv
import datetime
import subprocess
import sys
import tempfile
import shutil

dotenv.load_dotenv()

REQUEST_TIMEOUT_SECONDS = 300
MAX_RETRIES = 1
RETRY_BACKOFF_SECONDS = 5

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=REQUEST_TIMEOUT_SECONDS)

MAX_MODEL_DURATION_SECONDS = 1400
CHUNK_TARGET_DURATION_SECONDS = 900
CHUNK_OVERLAP_SECONDS = 5

def extract_audio_from_mp4(mp4_file):
    """Extract audio from MP4 file and save as MP3"""
    output_file = mp4_file.replace(".mp4", ".mp3").replace(".MP4", ".mp3")
    
    try:
        subprocess.run([
            "ffmpeg", "-i", mp4_file, "-q:a", "9", "-n", output_file
        ], check=True, capture_output=True)
        print(f"Audio extracted: {output_file}")
        return output_file
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: ffmpeg is not installed. Please install it first.")
        sys.exit(1)

def get_audio_duration(audio_path):
    """Return audio duration in seconds using ffprobe"""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                audio_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f"Error determining duration with ffprobe: {e}")
        sys.exit(1)
    except ValueError:
        print("Unable to parse audio duration.")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: ffprobe is not installed. Please install ffmpeg suite first.")
        sys.exit(1)

def split_audio_file(
    audio_path,
    segment_length=CHUNK_TARGET_DURATION_SECONDS,
    overlap=CHUNK_OVERLAP_SECONDS,
):
    """Split audio into overlapping chunks suitable for the diarization model"""
    if overlap >= segment_length:
        raise ValueError("Overlap must be smaller than segment length.")

    temp_dir = tempfile.mkdtemp(prefix="segments_", dir=os.getcwd())
    base_name, ext = os.path.splitext(os.path.basename(audio_path))
    total_duration = get_audio_duration(audio_path)
    step = segment_length - overlap

    offsets = []
    try:
        index = 0
        start = 0.0
        while start < total_duration:
            duration = min(segment_length, total_duration - start)
            output_file = os.path.join(
                temp_dir, f"{base_name}_part_{index:03d}{ext}"
            )
            subprocess.run(
                [
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
                ],
                check=True,
                capture_output=True,
            )
            offsets.append((output_file, start))
            index += 1
            start += step
    except subprocess.CalledProcessError as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"Error splitting audio: {e}")
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

def get_audio_files_from_sources():
    """Get audio/video files from sources folder sorted by last modified time"""
    supported_formats = [".mp3", ".mp4", ".opus", ".wav", ".m4a"]
    sources_dir = "./sources"
    
    if not os.path.exists(sources_dir):
        print(f"Error: {sources_dir} folder not found")
        sys.exit(1)
    
    files = []
    for file in os.listdir(sources_dir):
        if any(file.lower().endswith(fmt) for fmt in supported_formats):
            file_path = os.path.join(sources_dir, file)
            mod_time = os.path.getmtime(file_path)
            files.append((file, mod_time))
    
    if not files:
        print("No audio or video files found in sources folder")
        sys.exit(1)
    
    # Sort by modification time (newest first)
    files.sort(key=lambda x: x[1], reverse=True)
    return [f[0] for f in files]

def choose_file():
    """Display files and let user choose one"""
    files = get_audio_files_from_sources()
    
    print("\nAvailable files (sorted by most recent):")
    for i, file in enumerate(files, 1):
        file_path = os.path.join("./sources", file)
        mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
        print(f"{i}. {file} ({mod_time.strftime('%Y-%m-%d %H:%M:%S')})")
    
    while True:
        try:
            choice = input(f"\nSelect a file (1-{len(files)}): ").strip()
            index = int(choice) - 1
            if 0 <= index < len(files):
                return os.path.join("./sources", files[index])
            else:
                print(f"Please enter a number between 1 and {len(files)}")
        except ValueError:
            print("Invalid input. Please enter a number.")


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm format"""
    total_ms = int(seconds * 1000)
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds_fraction = remainder / 1000
    return f"{hours:02d}:{minutes:02d}:{seconds_fraction:06.3f}"


def transcribe_chunk_with_retry(chunk_path: str):
    """Send a chunk to the API with simple retries on timeout/network errors."""
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with open(chunk_path, "rb") as audio_file:
                return client.audio.transcriptions.create(
                    model="gpt-4o-transcribe-diarize",
                    file=audio_file,
                    response_format="diarized_json",
                    chunking_strategy="auto",
                    language="it",
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
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


# Let user choose a file
selected_source_path = choose_file()
audio_file_path = selected_source_path

# Extract audio from MP4 if needed
if audio_file_path.lower().endswith(".mp4"):
    print(f"MP4 file detected: {audio_file_path}")
    audio_file_path = extract_audio_from_mp4(audio_file_path)

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

# Transcribe the audio (handling chunks if necessary)
segments_output = []
full_text_parts = []
transcription = None

try:
    for chunk_path, offset in chunks_with_offsets:
        print(f"Processing chunk: {os.path.basename(chunk_path)} (offset {offset:.1f}s)")
        transcription = transcribe_chunk_with_retry(chunk_path)

        if hasattr(transcription, "segments"):
            for segment in transcription.segments:
                segment_text = segment.text.strip()
                start_time = segment.start + offset
                end_time = segment.end + offset
                segments_output.append(
                    f"[{format_timestamp(start_time)} - {format_timestamp(end_time)}] {segment.speaker}: {segment_text}"
                )
            full_text_parts.append(transcription.text.strip())
        else:
            text_value = str(transcription).strip()
            segments_output.append(text_value)
            full_text_parts.append(text_value)
finally:
    if chunk_dir:
        shutil.rmtree(chunk_dir, ignore_errors=True)

full_text_output = "\n".join([part for part in full_text_parts if part])

output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

output_filename = os.path.splitext(os.path.basename(selected_source_path))[0] + ".txt"
output_path = os.path.join(output_dir, output_filename)

with open(output_path, "w") as f:
    f.write(f"\n#########\n{datetime.datetime.now()}\n")
    f.write(f"Source file: {selected_source_path}\n")
    if selected_source_path != audio_file_path:
        f.write(f"Processed file: {audio_file_path}\n")
    f.write("Model: gpt-4o-transcribe-diarize\n\n")
    if transcription and hasattr(transcription, "segments"):
        f.write("Segments:\n")
        for line in segments_output:
            f.write(line + "\n")
        f.write("\nFull transcript:\n")
        f.write(full_text_output)
    else:
        f.write(full_text_output)
    f.write("\n\n")

print("Segments:")
print("\n".join(segments_output))
print(f"\nFull transcript saved to {output_path}")
