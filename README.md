# Voice Transcription

A small Python utility to transcribe audio/video files using OpenAI's transcription (diarized) model. It scans the `sources/` folder, lets you choose a file, extracts audio when needed, splits long recordings into chunks, and saves a diarized transcription to `output/<source-name>.txt`.

## Features

- Automatic file discovery from `sources/` (supports .mp3, .mp4, .opus, .wav, .m4a)
- MP4 audio extraction (requires ffmpeg)
- Automatic chunking for long audio files to stay within model duration limits
- Diarized transcription (speaker-labeled segments) using the `gpt-4o-transcribe-diarize` model
- Saves both segmented output and the full transcript to `output/<source-name>.txt`

## Requirements

- Python 3.9+ (recommended)
- `ffmpeg` (for extracting audio and splitting). On macOS: `brew install ffmpeg`
- The Python packages listed in `requirements.txt`:

```
openai
python-dotenv
# Note: ffmpeg must be installed separately (brew install ffmpeg on macOS)
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

## Usage

1. Put your audio or video files in the `sources/` directory. Example files are already present:

```
sources/
  ├─ Google meeting recording.mp4
  ├─ Podcast Episode 42.mp3
  └─ WhatsApp Audio 2025-07-25 at 14.10.15.opus
```

2. Run the script:

```bash
python3 speech.py
```

3. Follow the interactive prompt to select a file. The script will:

- Detect if the file is an MP4 and extract audio to an MP3 using `ffmpeg`.
- Check audio duration and split into chunks if longer than the model limit.
- Call OpenAI's model to produce a diarized JSON response.
- Save segment lines and the full transcript to `output/<source-name>.txt`. The folder is created automatically if missing.

After completion you'll see the segmented output printed to the terminal and find the transcript under `output/` with the same base filename as the source.

## Notes on behavior

- `ffmpeg` and `ffprobe` (from the ffmpeg suite) must be on your PATH. If they're missing the script will exit with a helpful message.
- The script uses chunking when audio duration exceeds ~1400 seconds to avoid model limits. The chunk size is tuned in the script (constants `MAX_MODEL_DURATION_SECONDS` and `CHUNK_TARGET_DURATION_SECONDS`).
- The sample model name used in the script is `gpt-4o-transcribe-diarize`. Adjust model and request options in `speech.py` if you need different behavior (language, format, etc.).

## Troubleshooting

- ffmpeg not found: Install it (macOS):

```bash
brew install ffmpeg
```

- No files found in `sources/`: Make sure you placed audio/video files in the `sources/` folder and they use a supported extension.

- API errors: Check `OPENAI_API_KEY` and your network connection. If the OpenAI client returns an error, examine the error printed to the console.

## Extending

- Add CLI flags for non-interactive runs (pass filename, language, or model).
- Add optional output formats (SRT, VTT) for use in subtitles.
- Add tests that mock the OpenAI client for faster local development.

## License

MIT-style — feel free to reuse and adapt.

---

If you want, I can also:

- Add a small example `sources/` test file and a non-interactive CLI flag to pick a file by name.
- Add unit tests for timestamp formatting and chunking logic.

Reply what you'd like next.
