# Voice Transcription

A small Python utility to transcribe local audio/video files or YouTube videos/lives using OpenAI's transcription models. It can scan the `sources/` folder, download YouTube media into `sources/youtube/`, extract audio when needed, split long recordings into chunks, and save the transcript to `output/<source-name>.txt`.

The repo also includes a small FastAPI app for `youtube.cill.app`. It rewrites `youtube.cill.app/...` or `www.youtube.cill.app/...` back to the original `youtube.com` URL, reuses cached transcript/summary artifacts when possible, queues cache misses for a local worker to process, and renders plain and diarized transcript/summary outputs side by side.

## Features

- Interactive source selection for local files or YouTube URLs
- Recursive file discovery from `sources/` (supports `.mp3`, `.mp4`, `.opus`, `.wav`, `.m4a`, `.aac`, `.flac`, `.ogg`, `.webm`, `.mkv`, `.mov`, `.ts`)
- YouTube video download and direct audio extraction via `yt-dlp`
- YouTube live capture from the currently available live window, with manual or timed stop
- Optional live snapshot mode with `--live-now` to transcribe what is available immediately
- Media-to-audio conversion with `ffmpeg` when needed
- Automatic chunking for long audio files to stay within model duration limits
- Plain transcription by default using the `gpt-4o-transcribe` model
- Optional diarized transcription (speaker-labeled segments) with `--diarize`, using `gpt-4o-transcribe-diarize`
- Optional second-step summary with `--summarize`
- Saves the full transcript and, when diarization is enabled, speaker-labeled segments
- FastAPI web app for `youtube.cill.app` with side-by-side plain and diarized rendering plus polling job states
- Queue-based web processing for cache misses, consumed by a local worker
- Parallel worker execution for plain and diarized transcript+summary branches
- Optional RapidAPI YouTube helper layer for subtitle retrieval and future media download fallback
- Local filesystem cache for web development plus Vercel Blob support in production

## Requirements

- Python 3.9+ (recommended)
- `ffmpeg` and `ffprobe` (for extracting audio, converting media, and splitting)
- `yt-dlp` (for YouTube video and live ingestion)
- The Python packages listed in `requirements.txt`:

```
openai
python-dotenv
fastapi
uvicorn
yt-dlp
vercel-blob
vercel
requests
httpx
# Note: ffmpeg must be installed separately (brew install ffmpeg on macOS)
```

Install system dependencies on macOS:

```bash
brew install ffmpeg yt-dlp
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Environment

Create a `.env` file in the project root (or set environment variables) with your OpenAI API key:

```
OPENAI_API_KEY=sk-...
```

Optional environment variables:

```
X_RAPIDAPI_KEY=...
BLOB_READ_WRITE_TOKEN=...
```

## Usage

1. Run the script:

```bash
python3 speech.py
```

Use diarization only when you need speaker labels:

```bash
python3 speech.py --diarize
```

Add a second-step summary after transcription when needed:

```bash
python3 speech.py --summarize
python3 speech.py --diarize --summarize
```

Enable diagnostic logging when debugging downloader, ffmpeg, or OpenAI calls:

```bash
python3 speech.py --verbose
```

Run non-interactively without `printf` by passing parameters as flags:

```bash
python3 speech.py --url "https://www.youtube.com/watch?v=VIDEO_ID" --language auto --summarize
python3 speech.py --file "sources/example.mp3" --language en
python3 speech.py --url "https://www.youtube.com/watch?v=LIVE_ID" --live-duration-minutes 15
python3 speech.py --url "https://www.youtube.com/watch?v=LIVE_ID" --live-duration-minutes 3 --force --summarize
```

For an active YouTube live, transcribe the currently available audio immediately:

```bash
python3 speech.py --live-now
python3 speech.py --live-now --summarize
```

2. Choose a source type:

- `1` for a local file under `sources/`
- `2` for a YouTube video or live URL

3. If you choose a local file, put your audio or video files in the `sources/` directory. Example files are already present:

```
sources/
  ├─ Google meeting recording.mp4
  ├─ Podcast Episode 42.mp3
  └─ WhatsApp Audio 2025-07-25 at 14.10.15.opus
```

4. Follow the interactive prompts:

- Enter a transcription language hint (`it`, `en`, etc.) or leave it blank for auto-detect.
- For YouTube live inputs, optionally enter a max capture duration in minutes or leave it blank to record until `Ctrl+C`.
- When a live duration is provided, the script downloads only that bounded section instead of relying on a later interrupt.
- With `--live-now`, active live streams skip the wait/stop flow and download the currently available live window immediately.
- If you pass `--file`, `--url`, `--language`, or `--live-duration-minutes`, the matching prompts are skipped.

The script will then:

- Reuse a local file or download/capture media from YouTube.
- Reuse an existing downloaded YouTube audio file when the same video ID was already fetched, unless forced.
- Extract or convert media to audio when needed.
- Check audio duration and split into chunks if longer than the model limit.
- Call OpenAI's model to produce the transcript.
- Save the full transcript to `output/<source-name>__<mode>__lang-<hint>.txt`. When `--diarize` is enabled, the output also includes speaker-labeled segments. The folder is created automatically if missing.
- When `--summarize` is enabled, run a second OpenAI step and save the summary to `output/<source-name>__<mode>__lang-<hint>__summary.txt`.

After completion you'll see the output printed to the terminal and find the transcript under `output/` with tags for transcription mode and language hint.

## Web App

The web app is FastAPI-based and is intended for Vercel deployment. v1 is intentionally narrow:

- YouTube VOD only
- automatic plain and diarized transcription
- automatic summary after each transcript branch
- no live streams
- no local file uploads
- no ffmpeg/chunking in the web request path
- hard limit: 25 minutes per video
- hard limit: 20 MB downloaded audio size
- optional RapidAPI subtitle lookup when `X_RAPIDAPI_KEY` is configured

### Local Run

Start the app locally:

```bash
.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Then open:

```bash
http://youtube.localhost:8000/watch?v=zFcyWFK1q8I
```

Any `*.localhost` hostname resolves locally in modern browsers, so you can test the host-replacement flow without editing `/etc/hosts`.

### How the Web Flow Works

- `youtube.cill.app/watch?v=...` becomes `https://youtube.com/watch?v=...`
- `www.youtube.cill.app/watch?v=...` becomes `https://www.youtube.com/watch?v=...`
- the page creates or reuses a deterministic job
- the page renders two columns: `Plain` and `Diarized`
- if both branches already have transcript and summary, the job returns immediately
- if one branch is cached and the other is missing, the cached branch appears immediately while the worker fills the missing branch
- the worker downloads audio once, then computes the plain and diarized transcript+summary branches in parallel
- if `X_RAPIDAPI_KEY` is configured, the worker also attempts a YouTube subtitle lookup and stores subtitle metadata for later quality checks

The page polls three internal endpoints:

- `POST /api/jobs`
- `POST /api/jobs/{job_id}/run`
- `GET /api/jobs/{job_id}`

The browser polls with increasing wait intervals from 5 seconds up to 60 seconds.

### Local Worker

Queue processing happens outside the Vercel request path. Run the worker locally against the same backend that the app uses:

```bash
.venv/bin/python -m cill.worker
```

Process a single job:

```bash
.venv/bin/python -m cill.worker --job-id <job_id>
```

Keep polling continuously:

```bash
.venv/bin/python -m cill.worker --loop --interval-seconds 10
```

The app and worker automatically load the project `.env` file before choosing a storage backend. In local development, the worker uses the filesystem cache. If `.env` includes `BLOB_READ_WRITE_TOKEN`, it consumes queued jobs from Vercel Blob instead.

### Local Cache Reuse

In local development the web app reuses the repo's existing CLI artifacts:

- plain or diarized transcript and summary from `output/`
- downloaded YouTube audio from `sources/youtube/`

That means previously processed videos like `zFcyWFK1q8I` can render immediately in the web app without a new download or OpenAI call.

### Vercel Deployment

The Vercel entrypoint is `app.py`, which exposes the FastAPI app from `cill.app`. The repo includes `vercel.json` with Fluid Compute enabled and a function `maxDuration` of 300 seconds.

Required environment variables in Vercel:

```bash
OPENAI_API_KEY=...
BLOB_READ_WRITE_TOKEN=...
X_RAPIDAPI_KEY=... # optional
```

Production persistence uses Vercel Blob:

- `meta.json`
- `transcript.txt`
- `summary.txt`
- `diarized_transcript.txt`
- `diarized_summary.txt`
- `rapidapi_subtitle_metadata.json`
- `rapidapi_subtitle_text.txt`

Each job is stored under a deterministic cache key derived from `youtube:<video_id>:plain:auto`.

### Domain Setup

Add these domains to the Vercel project:

- `youtube.cill.app`
- `www.youtube.cill.app`

Then create matching Cloudflare DNS records that point to the Vercel target shown in the Vercel domain panel. Typical setup is one CNAME for `youtube` and one for `www.youtube`.

Relevant Vercel docs:

- [Python runtime](https://vercel.com/docs/functions/runtimes/python)
- [Function duration](https://vercel.com/docs/functions/configuring-functions/duration)
- [Custom domains](https://vercel.com/docs/domains/set-up-custom-domain)

### Force reprocessing

Use these flags when you want to bypass reuse checks:

```bash
python3 speech.py --force
python3 speech.py --force-download
python3 speech.py --force-transcribe
```

- `--force`: force both a fresh YouTube download and a fresh transcription
- `--force-download`: redownload YouTube media even if a matching file already exists
- `--force-transcribe`: rerun the transcription and summary even if matching output files already exist
- `--verbose`: print diagnostic details about subprocess commands, cache reuse, and OpenAI requests
- `--file`: select a local input file directly
- `--url`: select a YouTube input URL directly
- `--language`: provide the transcription language hint directly. Use `auto` to skip the hint.
- `--live-duration-minutes`: provide the live timeout directly instead of answering the prompt

## Notes on behavior

- `ffmpeg` and `ffprobe` (from the ffmpeg suite) must be on your PATH. If they're missing the script will exit with a helpful message.
- `yt-dlp` must be on your PATH for YouTube inputs. Local file transcription does not require it.
- The script uses chunking when audio duration exceeds ~1400 seconds to avoid model limits. The chunk size is tuned in the script (constants `MAX_MODEL_DURATION_SECONDS` and `CHUNK_TARGET_DURATION_SECONDS`).
- Active YouTube lives are captured first and transcribed only after capture stops. v1 does not provide rolling or real-time transcript updates.
- `--live-duration-minutes` uses a bounded yt-dlp section download so the requested duration is enforced up front.
- `--live-now` is the exception: for active lives it downloads the audio available so far right away, then transcribes and optionally summarizes it immediately.
- Scheduled/upcoming YouTube lives are not supported in v1.
- Completed YouTube videos reuse previously downloaded audio and previously generated transcripts for the same mode/language unless you pass a force flag.
- Summary prompts include whether the source is a live video or not. For captured live videos they also include whether the capture ended naturally, hit the timeout, or was manually stopped.
- The default model is `gpt-4o-transcribe`. Use `--diarize` to switch to `gpt-4o-transcribe-diarize`.

## Troubleshooting

- ffmpeg not found: Install it (macOS):

```bash
brew install ffmpeg
```

- No files found in `sources/`: Make sure you placed audio/video files in the `sources/` folder and they use a supported extension.

- yt-dlp not found for YouTube mode: Install it (macOS):

```bash
brew install yt-dlp
```

- API errors: Check `OPENAI_API_KEY` and your network connection. If the OpenAI client returns an error, examine the error printed to the console.

## Extending

- Add more CLI flags for non-interactive runs (pass filename, language, or model).
- Add optional output formats (SRT, VTT) for use in subtitles.
- Add tests that mock the OpenAI client for faster local development.

## License

MIT-style — feel free to reuse and adapt.
