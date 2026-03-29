import json
import tempfile
import threading
import unittest
from datetime import datetime, timezone
import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from fastapi.testclient import TestClient

from cill.app import (
    JobRunRequest,
    MAX_VIDEO_SECONDS,
    VariantRunRequest,
    app,
    create_or_reuse_job,
    process_job,
    queue_requested_work,
    reconstruct_source_url,
)
from cill.shared import extract_youtube_video_id
from cill.storage import (
    BlobStorageBackend,
    LocalStorageBackend,
    WEB_DIARIZED_SUMMARY_FILENAME,
    WEB_DIARIZED_TRANSCRIPT_FILENAME,
    WEB_RAPIDAPI_SUBTITLE_METADATA_FILENAME,
    WEB_RAPIDAPI_SUBTITLE_TEXT_FILENAME,
    WEB_SUMMARY_FILENAME,
    WEB_TRANSCRIPT_FILENAME,
    create_storage_backend,
)
from cill.worker import process_pending_jobs


class WebAppTests(unittest.TestCase):
    VIDEO_ID = "abc123xyz89"

    def test_reconstruct_source_url_for_supported_hosts(self) -> None:
        self.assertEqual(
            reconstruct_source_url("youtube.cill.app", "/watch", "v=abc123"),
            "https://youtube.com/watch?v=abc123",
        )
        self.assertEqual(
            reconstruct_source_url("www.youtube.localhost:8000", "/watch", "v=abc123"),
            "https://www.youtube.com/watch?v=abc123",
        )

    def test_reconstruct_source_url_rejects_non_youtube_host(self) -> None:
        with self.assertRaises(ValueError):
            reconstruct_source_url("google.cill.app", "/search", "q=test")

    def test_extract_youtube_video_id(self) -> None:
        self.assertEqual(
            extract_youtube_video_id("https://youtube.com/watch?v=abc123"),
            "abc123",
        )
        self.assertEqual(
            extract_youtube_video_id("https://youtu.be/xyz789"),
            "xyz789",
        )

    def test_local_storage_reads_legacy_transcript_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            transcript_path = output_dir / "Video [abc123]__plain__lang-auto.txt"
            summary_path = output_dir / "Video [abc123]__plain__lang-auto__summary.txt"
            diarized_transcript_path = output_dir / "Video [abc123]__diarized__lang-auto.txt"
            diarized_summary_path = output_dir / "Video [abc123]__diarized__lang-auto__summary.txt"
            transcript_path.write_text("header\n\nLegacy transcript", encoding="utf-8")
            summary_path.write_text("header\nSummary:\nLegacy summary", encoding="utf-8")
            diarized_transcript_path.write_text(
                "header\n\nLegacy diarized transcript",
                encoding="utf-8",
            )
            diarized_summary_path.write_text(
                "header\nSummary:\nLegacy diarized summary",
                encoding="utf-8",
            )

            storage = LocalStorageBackend(output_dir=str(output_dir), audio_dir=str(Path(temp_dir) / "audio"))

            self.assertEqual(storage.read_text("job-1", WEB_TRANSCRIPT_FILENAME, video_id="abc123"), "Legacy transcript")
            self.assertEqual(storage.read_text("job-1", WEB_SUMMARY_FILENAME, video_id="abc123"), "Legacy summary")
            self.assertEqual(
                storage.read_text("job-1", WEB_DIARIZED_TRANSCRIPT_FILENAME, video_id="abc123"),
                "Legacy diarized transcript",
            )
            self.assertEqual(
                storage.read_text("job-1", WEB_DIARIZED_SUMMARY_FILENAME, video_id="abc123"),
                "Legacy diarized summary",
            )

    @mock.patch("cill.storage.vercel_put")
    @mock.patch("cill.storage.vercel_get")
    @mock.patch("cill.storage.vercel_list_objects")
    def test_blob_storage_reads_and_writes_job_artifacts(
        self,
        vercel_list_objects_mock: mock.Mock,
        vercel_get_mock: mock.Mock,
        vercel_put_mock: mock.Mock,
    ) -> None:
        vercel_get_mock.return_value = SimpleNamespace(content=b'{"status":"idle"}')
        vercel_list_objects_mock.return_value = SimpleNamespace(
            blobs=[
                SimpleNamespace(
                    pathname="prefix/jobs/job-1/meta.json",
                    url="https://blob.example/meta.json",
                    uploaded_at=datetime(2026, 3, 29, 10, 0, 0, tzinfo=timezone.utc),
                )
            ]
        )

        storage = BlobStorageBackend(prefix="prefix", token="test-token")
        state = storage.load_state("job-1")

        self.assertEqual(state, {"status": "idle"})
        storage.write_text("job-1", WEB_TRANSCRIPT_FILENAME, "Transcript")

        vercel_put_mock.assert_called_with(
            "prefix/jobs/job-1/transcript.txt",
            b"Transcript",
            access="private",
            content_type="text/plain",
            overwrite=True,
            token="test-token",
        )

    @mock.patch("cill.storage.vercel_get")
    @mock.patch("cill.storage.vercel_list_objects")
    def test_blob_storage_reads_suffixed_private_blob_pathnames(
        self,
        vercel_list_objects_mock: mock.Mock,
        vercel_get_mock: mock.Mock,
    ) -> None:
        vercel_get_mock.return_value = SimpleNamespace(content=b'{"status":"cache_hit"}')
        vercel_list_objects_mock.return_value = SimpleNamespace(
            blobs=[
                SimpleNamespace(
                    pathname="prefix/jobs/job-1/meta-AbCd1234.json",
                    url="https://blob.example/meta-suffixed.json",
                    uploaded_at=datetime(2026, 3, 29, 10, 0, 0, tzinfo=timezone.utc),
                )
            ]
        )

        storage = BlobStorageBackend(prefix="prefix", token="test-token")
        state = storage.load_state("job-1")

        self.assertEqual(state, {"status": "cache_hit"})

    def test_create_storage_backend_loads_blob_token_from_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            previous_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                Path(".env").write_text('BLOB_READ_WRITE_TOKEN="blob-token"\n', encoding="utf-8")
                with mock.patch.dict("os.environ", {}, clear=True):
                    with mock.patch("cill.storage.BlobStorageBackend", return_value="blob-backend") as blob_mock:
                        backend = create_storage_backend()
            finally:
                os.chdir(previous_cwd)

        self.assertEqual(backend, "blob-backend")
        blob_mock.assert_called_once_with()

    @mock.patch("cill.app.probe_youtube_metadata")
    def test_create_or_reuse_job_uses_cached_local_outputs(self, probe_mock: mock.Mock) -> None:
        probe_mock.return_value = {
            "id": self.VIDEO_ID,
            "title": "Video",
            "uploader": "Uploader",
            "duration": 120,
            "live_status": "not_live",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "output"
            audio_dir = Path(temp_dir) / "audio"
            output_dir.mkdir(parents=True, exist_ok=True)
            audio_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / f"Video [{self.VIDEO_ID}]__plain__lang-auto.txt").write_text(
                "header\n\nCached transcript",
                encoding="utf-8",
            )
            (output_dir / f"Video [{self.VIDEO_ID}]__plain__lang-auto__summary.txt").write_text(
                "header\nSummary:\nCached summary",
                encoding="utf-8",
            )
            (output_dir / f"Video [{self.VIDEO_ID}]__diarized__lang-auto.txt").write_text(
                "header\n\nCached diarized transcript",
                encoding="utf-8",
            )
            (output_dir / f"Video [{self.VIDEO_ID}]__diarized__lang-auto__summary.txt").write_text(
                "header\nSummary:\nCached diarized summary",
                encoding="utf-8",
            )

            storage = LocalStorageBackend(output_dir=str(output_dir), audio_dir=str(audio_dir))
            with mock.patch("cill.app.storage", storage):
                state = create_or_reuse_job(f"https://youtube.com/watch?v={self.VIDEO_ID}")

        self.assertEqual(state["status"], "cache_hit")
        self.assertEqual(state["transcript"], "Cached transcript")
        self.assertEqual(state["summary"], "Cached summary")
        self.assertEqual(state["variants"]["diarized"]["transcript"], "Cached diarized transcript")
        self.assertEqual(state["variants"]["diarized"]["summary"], "Cached diarized summary")
        probe_mock.assert_called_once()

    @mock.patch("cill.app.probe_youtube_metadata", side_effect=RuntimeError("blocked"))
    def test_process_job_returns_error_state_when_metadata_probe_fails(self, probe_mock: mock.Mock) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = LocalStorageBackend(
                output_dir=str(Path(temp_dir) / "output"),
                audio_dir=str(Path(temp_dir) / "audio"),
            )
            with mock.patch("cill.app.storage", storage):
                state = create_or_reuse_job(f"https://youtube.com/watch?v={self.VIDEO_ID}")
                state = queue_requested_work(
                    state,
                    JobRunRequest(plain=VariantRunRequest(transcript=True)),
                )
                result = process_job(state)

        self.assertEqual(result["status"], "error")
        self.assertIn("blocked", result["error"])
        self.assertGreaterEqual(probe_mock.call_count, 1)

    @mock.patch("cill.app.probe_youtube_metadata")
    def test_create_or_reuse_job_probes_cache_miss_without_queueing(self, probe_mock: mock.Mock) -> None:
        probe_mock.return_value = {
            "id": self.VIDEO_ID,
            "title": "Video",
            "uploader": "Uploader",
            "duration": 120,
            "live_status": "not_live",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = LocalStorageBackend(
                output_dir=str(Path(temp_dir) / "output"),
                audio_dir=str(Path(temp_dir) / "audio"),
            )
            with mock.patch("cill.app.storage", storage):
                state = create_or_reuse_job(f"https://youtube.com/watch?v={self.VIDEO_ID}")

        self.assertEqual(state["status"], "idle")
        self.assertEqual(state["video_id"], self.VIDEO_ID)
        probe_mock.assert_called_once()

    @mock.patch("cill.app.probe_youtube_metadata")
    def test_create_or_reuse_job_returns_partial_cache_state_when_only_plain_exists(self, probe_mock: mock.Mock) -> None:
        probe_mock.return_value = {
            "id": self.VIDEO_ID,
            "title": "Video",
            "uploader": "Uploader",
            "duration": 120,
            "live_status": "not_live",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "output"
            audio_dir = Path(temp_dir) / "audio"
            output_dir.mkdir(parents=True, exist_ok=True)
            audio_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / f"Video [{self.VIDEO_ID}]__plain__lang-auto.txt").write_text(
                "header\n\nCached transcript",
                encoding="utf-8",
            )
            (output_dir / f"Video [{self.VIDEO_ID}]__plain__lang-auto__summary.txt").write_text(
                "header\nSummary:\nCached summary",
                encoding="utf-8",
            )

            storage = LocalStorageBackend(output_dir=str(output_dir), audio_dir=str(audio_dir))
            with mock.patch("cill.app.storage", storage):
                state = create_or_reuse_job(f"https://youtube.com/watch?v={self.VIDEO_ID}")

        self.assertEqual(state["status"], "idle")
        self.assertEqual(state["variants"]["plain"]["status"], "cache_hit")
        self.assertEqual(state["variants"]["diarized"]["status"], "idle")

    @mock.patch("cill.app.probe_youtube_metadata")
    def test_process_job_rejects_live(self, probe_mock: mock.Mock) -> None:
        probe_mock.return_value = {
            "id": "live123",
            "title": "Live",
            "uploader": "Uploader",
            "duration": 120,
            "live_status": "is_live",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            storage = LocalStorageBackend(
                output_dir=str(Path(temp_dir) / "output"),
                audio_dir=str(Path(temp_dir) / "audio"),
            )
            with mock.patch("cill.app.storage", storage):
                state = create_or_reuse_job("https://youtube.com/watch?v=live123")

        self.assertEqual(state["status"], "unsupported_live")

    @mock.patch("cill.app.maybe_fetch_rapidapi_subtitles", side_effect=lambda state, duration_seconds: state)
    @mock.patch("cill.app.probe_youtube_metadata")
    def test_process_job_rejects_over_duration(self, probe_mock: mock.Mock, subtitle_mock: mock.Mock) -> None:
        probe_mock.return_value = {
            "id": "long123",
            "title": "Long Video",
            "uploader": "Uploader",
            "duration": MAX_VIDEO_SECONDS + 1,
            "live_status": "not_live",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            storage = LocalStorageBackend(
                output_dir=str(Path(temp_dir) / "output"),
                audio_dir=str(Path(temp_dir) / "audio"),
            )
            with mock.patch("cill.app.storage", storage):
                state = create_or_reuse_job("https://youtube.com/watch?v=long123")

        self.assertEqual(state["status"], "unsupported_duration")

    @mock.patch("cill.app.download_audio")
    @mock.patch("cill.app.maybe_fetch_rapidapi_subtitles")
    @mock.patch("cill.app.probe_youtube_metadata")
    @mock.patch("cill.app.speech.summarize_transcript", return_value="Summary from subtitles")
    @mock.patch("cill.app.speech.transcribe_audio_file")
    def test_process_job_uses_valid_subtitles_for_long_plain_variant(
        self,
        transcribe_mock: mock.Mock,
        summarize_mock: mock.Mock,
        probe_mock: mock.Mock,
        fetch_subtitles_mock: mock.Mock,
        download_audio_mock: mock.Mock,
    ) -> None:
        probe_mock.return_value = {
            "id": "long123",
            "title": "Long Video",
            "uploader": "Uploader",
            "duration": MAX_VIDEO_SECONDS + 300,
            "live_status": "not_live",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            storage = LocalStorageBackend(
                output_dir=str(Path(temp_dir) / "output"),
                audio_dir=str(Path(temp_dir) / "audio"),
            )

            def subtitle_side_effect(state: dict, duration_seconds: float | None) -> dict:
                subtitle_text = "Subtitle transcript line one.\nSubtitle transcript line two."
                subtitle_metadata = {
                    "selected_track": {"code": "en", "kind": None},
                    "stats": {
                        "char_count": len(subtitle_text),
                        "word_count": 240,
                        "line_count": 24,
                        "duration_seconds": duration_seconds,
                        "duration_minutes": 30.0,
                        "words_per_minute": 120.0,
                        "estimated_minutes_from_words": 28.0,
                        "coverage_ratio": 0.93,
                    },
                    "validation": {"usable": True, "reason": "ok"},
                }
                storage.write_text(state["job_id"], WEB_RAPIDAPI_SUBTITLE_TEXT_FILENAME, subtitle_text)
                storage.write_text(
                    state["job_id"],
                    WEB_RAPIDAPI_SUBTITLE_METADATA_FILENAME,
                    json.dumps(subtitle_metadata),
                )
                rapidapi_state = state["rapidapi"]
                rapidapi_state.update(
                    {
                        "configured": True,
                        "subtitle_available": True,
                        "subtitle_fetched": True,
                        "subtitle_language": "en",
                        "subtitle_kind": None,
                        "subtitle_word_count": 240,
                        "subtitle_chars": len(subtitle_text),
                        "subtitle_duration_seconds": duration_seconds,
                        "subtitle_usable": True,
                        "subtitle_validation_reason": "ok",
                        "error": None,
                    }
                )
                return state

            fetch_subtitles_mock.side_effect = subtitle_side_effect

            with mock.patch("cill.app.storage", storage):
                state = create_or_reuse_job("https://youtube.com/watch?v=long123")
                state = queue_requested_work(
                    state,
                    JobRunRequest(plain=VariantRunRequest(transcript=True, summary=True)),
                )
                final_state = process_job(state)

        self.assertEqual(final_state["status"], "complete")
        self.assertEqual(final_state["variants"]["plain"]["transcript"], "Subtitle transcript line one.\nSubtitle transcript line two.")
        self.assertEqual(final_state["variants"]["plain"]["summary"], "Summary from subtitles")
        self.assertEqual(final_state["variants"]["diarized"]["status"], "idle")
        self.assertTrue(final_state["rapidapi"]["subtitle_usable"])
        transcribe_mock.assert_not_called()
        download_audio_mock.assert_not_called()
        summarize_mock.assert_called_once()

    @mock.patch("cill.app.speech.summarize_transcript", return_value="Fresh summary")
    @mock.patch("cill.app.speech.transcribe_audio_file")
    def test_process_job_generates_only_missing_summary(
        self,
        transcribe_mock: mock.Mock,
        summarize_mock: mock.Mock,
    ) -> None:
        transcribe_mock.return_value = (None, [], "should not be used")

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "output"
            audio_dir = Path(temp_dir) / "audio"
            output_dir.mkdir(parents=True, exist_ok=True)
            audio_dir.mkdir(parents=True, exist_ok=True)
            storage = LocalStorageBackend(output_dir=str(output_dir), audio_dir=str(audio_dir))
            state = {
                "job_id": "job123",
                "cache_key": "youtube:abc123:plain:auto",
                "source_url": "https://youtube.com/watch?v=abc123",
                "video_id": "abc123",
                "title": "Video",
                "uploader": "Uploader",
                "status": "idle",
                "error": None,
                "transcript_ready": True,
                "summary_ready": False,
                "created_at": "now",
                "updated_at": "now",
                "variants": {
                    "plain": {
                        "name": "plain",
                        "label": "Plain transcript",
                        "diarize": False,
                        "status": "queued",
                        "error": None,
                        "requested_transcript": False,
                        "requested_summary": True,
                        "transcript_ready": True,
                        "summary_ready": False,
                        "transcript_source": "legacy_cache",
                        "summary_basis": None,
                    },
                    "diarized": {
                        "name": "diarized",
                        "label": "Diarized transcript",
                        "diarize": True,
                        "status": "cache_hit",
                        "error": None,
                        "requested_transcript": False,
                        "requested_summary": False,
                        "transcript_ready": True,
                        "summary_ready": True,
                        "transcript_source": "legacy_cache",
                        "summary_basis": "legacy_cache",
                    },
                },
            }
            storage.save_state("job123", state)
            storage.write_text("job123", WEB_TRANSCRIPT_FILENAME, "Existing transcript")
            storage.write_text("job123", WEB_DIARIZED_TRANSCRIPT_FILENAME, "Existing diarized transcript")
            storage.write_text("job123", WEB_DIARIZED_SUMMARY_FILENAME, "Existing diarized summary")
            observed_statuses: list[str] = []

            def save_state_spy(job_id: str, payload: dict) -> None:
                observed_statuses.append(payload["status"])
                LocalStorageBackend.save_state(storage, job_id, payload)

            with mock.patch("cill.app.storage", storage):
                with mock.patch.object(storage, "save_state", side_effect=save_state_spy):
                    final_state = process_job(state)

        self.assertEqual(final_state["status"], "complete")
        self.assertEqual(final_state["summary"], "Fresh summary")
        transcribe_mock.assert_not_called()
        summarize_mock.assert_called_once()
        self.assertIn("summarizing", observed_statuses)
        self.assertEqual(observed_statuses[-1], "complete")

    @mock.patch("cill.app.download_audio")
    @mock.patch("cill.app.maybe_fetch_rapidapi_subtitles", side_effect=lambda state, duration_seconds: state)
    @mock.patch("cill.app.probe_youtube_metadata")
    @mock.patch("cill.app.speech.summarize_transcript")
    @mock.patch("cill.app.speech.transcribe_audio_file")
    def test_process_job_computes_plain_and_diarized_in_parallel(
        self,
        transcribe_mock: mock.Mock,
        summarize_mock: mock.Mock,
        probe_mock: mock.Mock,
        subtitle_mock: mock.Mock,
        download_mock: mock.Mock,
    ) -> None:
        probe_mock.return_value = {
            "id": self.VIDEO_ID,
            "title": "Video",
            "uploader": "Uploader",
            "duration": 120,
            "live_status": "not_live",
        }
        barrier = threading.Barrier(2)

        def transcribe_side_effect(audio_path: str, language_hint: str | None, diarize: bool):
            barrier.wait(timeout=2)
            text = "Diarized transcript" if diarize else "Plain transcript"
            return (None, [], text)

        def summarize_side_effect(transcript_text: str, source_context: dict):
            return f"Summary for {transcript_text}"

        transcribe_mock.side_effect = transcribe_side_effect
        summarize_mock.side_effect = summarize_side_effect

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_dir = Path(temp_dir) / "audio"
            output_dir = Path(temp_dir) / "output"
            audio_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            audio_path = audio_dir / f"Video [{self.VIDEO_ID}].m4a"
            audio_path.write_bytes(b"test-audio")
            download_mock.return_value = (str(audio_path), False)
            storage = LocalStorageBackend(output_dir=str(output_dir), audio_dir=str(audio_dir))

            with mock.patch("cill.app.storage", storage):
                state = create_or_reuse_job(f"https://youtube.com/watch?v={self.VIDEO_ID}")
                state = queue_requested_work(
                    state,
                    JobRunRequest(
                        plain=VariantRunRequest(transcript=True, summary=True),
                        diarized=VariantRunRequest(transcript=True, summary=True),
                    ),
                )
                final_state = process_job(state)

        self.assertEqual(final_state["status"], "complete")
        self.assertEqual(final_state["variants"]["plain"]["transcript"], "Plain transcript")
        self.assertEqual(final_state["variants"]["plain"]["summary"], "Summary for Plain transcript")
        self.assertEqual(final_state["variants"]["diarized"]["transcript"], "Diarized transcript")
        self.assertEqual(final_state["variants"]["diarized"]["summary"], "Summary for Diarized transcript")
        self.assertEqual(transcribe_mock.call_count, 2)
        self.assertEqual(summarize_mock.call_count, 2)
        self.assertTrue(final_state["variants"]["diarized"]["transcript_ready"])

    @mock.patch("cill.app.create_or_reuse_job")
    def test_processing_page_route_supports_localhost_host_rewrite(self, create_job_mock: mock.Mock) -> None:
        create_job_mock.return_value = {
            "job_id": "job123",
            "status": "cache_hit",
            "title": "Video",
            "uploader": "Uploader",
            "transcript": "Transcript",
            "summary": "Summary",
            "error": None,
        }

        client = TestClient(app)
        response = client.get(
            "/watch?v=abc123",
            headers={"host": "youtube.localhost"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("Queue work", response.text)
        self.assertIn("https://youtube.com/watch?v=abc123", response.text)

    @mock.patch("cill.app.create_or_reuse_job")
    def test_create_job_endpoint_returns_probe_state(self, create_job_mock: mock.Mock) -> None:
        create_job_mock.return_value = {
            "job_id": "job123",
            "status": "idle",
            "video_id": self.VIDEO_ID,
            "title": "Unknown",
            "uploader": "Unknown",
            "error": None,
            "transcript_ready": False,
            "summary_ready": False,
            "created_at": "now",
            "updated_at": "now",
        }
        client = TestClient(app)
        response = client.post(
            "/api/jobs",
            json={"source_url": "https://youtube.com/watch?v=abc123"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "idle")
        self.assertEqual(payload["video_id"], self.VIDEO_ID)
        self.assertIsNone(payload["error"])

    def test_run_endpoint_queues_selected_plain_variant_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = LocalStorageBackend(
                output_dir=str(Path(temp_dir) / "output"),
                audio_dir=str(Path(temp_dir) / "audio"),
            )
            initial_state = {
                "job_id": "job123",
                "cache_key": f"youtube:{self.VIDEO_ID}:plain:auto",
                "source_url": f"https://youtube.com/watch?v={self.VIDEO_ID}",
                "video_id": self.VIDEO_ID,
                "title": "Video",
                "uploader": "Uploader",
                "duration_seconds": 120,
                "live_status": "not_live",
                "status": "idle",
                "error": None,
                "transcript_ready": False,
                "summary_ready": False,
                "created_at": "now",
                "updated_at": "now",
            }
            storage.save_state("job123", initial_state)

            with mock.patch("cill.app.storage", storage):
                client = TestClient(app)
                response = client.post(
                    "/api/jobs/job123/run",
                    json={
                        "plain": {"transcript": True, "summary": True},
                        "diarized": {"transcript": False, "summary": False},
                    },
                )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "queued")
        self.assertTrue(payload["variants"]["plain"]["requested_transcript"])
        self.assertTrue(payload["variants"]["plain"]["requested_summary"])
        self.assertEqual(payload["variants"]["diarized"]["status"], "idle")

    def test_processing_page_renders_manual_queue_controls(self) -> None:
        client = TestClient(app)
        response = client.get("/watch?v=abc123xyz89", headers={"host": "youtube.localhost"})

        self.assertEqual(response.status_code, 200)
        self.assertIn("Queue work", response.text)
        self.assertIn("/api/jobs/${currentJobId}/run", response.text)
        self.assertIn("Nothing queued yet.", response.text)

    def test_instruction_page_for_root(self) -> None:
        client = TestClient(app)
        response = client.get("/", headers={"host": "localhost"})

        self.assertEqual(response.status_code, 200)
        self.assertIn("Replace", response.text)
        self.assertIn("/favicon.svg", response.text)

    def test_favicon_routes_return_svg(self) -> None:
        client = TestClient(app)

        ico_response = client.get("/favicon.ico")
        svg_response = client.get("/favicon.svg")

        self.assertEqual(ico_response.status_code, 200)
        self.assertEqual(svg_response.status_code, 200)
        self.assertEqual(ico_response.headers["content-type"], "image/svg+xml")
        self.assertEqual(svg_response.headers["content-type"], "image/svg+xml")
        self.assertIn("<svg", ico_response.text)
        self.assertIn("<svg", svg_response.text)

    def test_processing_page_uses_backoff_polling(self) -> None:
        client = TestClient(app)
        response = client.get("/watch?v=abc123", headers={"host": "youtube.localhost"})

        self.assertEqual(response.status_code, 200)
        self.assertIn("<h2>Plain</h2>", response.text)
        self.assertIn("<h2>Diarized</h2>", response.text)
        self.assertIn("let delayMs = 5000;", response.text)
        self.assertIn("Math.min(60000, delayMs + 5000)", response.text)
        self.assertIn("/api/jobs/${currentJobId}/run", response.text)

    @mock.patch("cill.worker.process_job")
    def test_worker_processes_only_queued_jobs(self, process_job_mock: mock.Mock) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = LocalStorageBackend(
                output_dir=str(Path(temp_dir) / "output"),
                audio_dir=str(Path(temp_dir) / "audio"),
            )
            storage.save_state(
                "queued-job",
                {
                    "job_id": "queued-job",
                    "status": "queued",
                    "source_url": f"https://youtube.com/watch?v={self.VIDEO_ID}",
                    "variants": {
                        "plain": {
                            "name": "plain",
                            "label": "Plain transcript",
                            "diarize": False,
                            "status": "queued",
                            "error": None,
                            "requested_transcript": True,
                            "requested_summary": False,
                            "transcript_ready": False,
                            "summary_ready": False,
                            "transcript_source": None,
                            "summary_basis": None,
                        },
                        "diarized": {
                            "name": "diarized",
                            "label": "Diarized transcript",
                            "diarize": True,
                            "status": "idle",
                            "error": None,
                            "requested_transcript": False,
                            "requested_summary": False,
                            "transcript_ready": False,
                            "summary_ready": False,
                            "transcript_source": None,
                            "summary_basis": None,
                        },
                    },
                },
            )
            storage.save_state("done-job", {"job_id": "done-job", "status": "complete", "source_url": "https://youtube.com/watch?v=def456"})
            process_job_mock.return_value = {"status": "complete"}

            with mock.patch("cill.worker.storage", storage):
                processed = process_pending_jobs()

        self.assertEqual(processed, 1)
        process_job_mock.assert_called_once()
