import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from cill.app import (
    MAX_VIDEO_SECONDS,
    app,
    create_or_reuse_job,
    process_job,
    reconstruct_source_url,
)
from cill.shared import extract_youtube_video_id
from cill.storage import (
    BlobStorageBackend,
    LocalStorageBackend,
    WEB_SUMMARY_FILENAME,
    WEB_TRANSCRIPT_FILENAME,
)
from cill.worker import process_pending_jobs


class WebAppTests(unittest.TestCase):
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
            transcript_path.write_text("header\n\nLegacy transcript", encoding="utf-8")
            summary_path.write_text("header\nSummary:\nLegacy summary", encoding="utf-8")

            storage = LocalStorageBackend(output_dir=str(output_dir), audio_dir=str(Path(temp_dir) / "audio"))

            self.assertEqual(storage.read_text("job-1", WEB_TRANSCRIPT_FILENAME, video_id="abc123"), "Legacy transcript")
            self.assertEqual(storage.read_text("job-1", WEB_SUMMARY_FILENAME, video_id="abc123"), "Legacy summary")

    @mock.patch("cill.storage.requests.get")
    @mock.patch("cill.storage.vercel_blob")
    def test_blob_storage_reads_and_writes_job_artifacts(
        self,
        vercel_blob_mock: mock.Mock,
        requests_get_mock: mock.Mock,
    ) -> None:
        requests_put_mock = mock.Mock()
        requests_get_mock.return_value = mock.Mock(
            text='{"status":"idle"}',
            raise_for_status=mock.Mock(),
        )
        vercel_blob_mock.list.return_value = {
            "blobs": [
                {
                    "pathname": "prefix/jobs/job-1/meta.json",
                    "url": "https://blob.example/meta.json",
                }
            ]
        }

        with mock.patch("cill.storage.requests.put", requests_put_mock):
            requests_put_mock.return_value = mock.Mock(raise_for_status=mock.Mock())
            storage = BlobStorageBackend(prefix="prefix", token="test-token")
            state = storage.load_state("job-1")

            self.assertEqual(state, {"status": "idle"})
            storage.write_text("job-1", WEB_TRANSCRIPT_FILENAME, "Transcript")

        requests_put_mock.assert_called_with(
            "https://blob.vercel-storage.com/",
            params={"pathname": "prefix/jobs/job-1/transcript.txt"},
            headers={
                "access": "private",
                "authorization": "Bearer test-token",
                "x-api-version": "10",
                "x-content-type": "text/plain",
                "x-allow-overwrite": "1",
            },
            data=b"Transcript",
            timeout=30,
        )

    @mock.patch("cill.storage.requests.get")
    @mock.patch("cill.storage.vercel_blob")
    def test_blob_storage_reads_suffixed_private_blob_pathnames(
        self,
        vercel_blob_mock: mock.Mock,
        requests_get_mock: mock.Mock,
    ) -> None:
        requests_get_mock.return_value = mock.Mock(
            text='{"status":"cache_hit"}',
            raise_for_status=mock.Mock(),
        )
        vercel_blob_mock.list.return_value = {
            "blobs": [
                {
                    "pathname": "prefix/jobs/job-1/meta-AbCd1234.json",
                    "url": "https://blob.example/meta-suffixed.json",
                    "uploadedAt": "2026-03-29T10:00:00.000Z",
                }
            ]
        }

        storage = BlobStorageBackend(prefix="prefix", token="test-token")
        state = storage.load_state("job-1")

        self.assertEqual(state, {"status": "cache_hit"})

    @mock.patch("cill.app.probe_youtube_metadata")
    def test_create_or_reuse_job_uses_cached_local_outputs(self, probe_mock: mock.Mock) -> None:
        probe_mock.return_value = {
            "id": "abc123",
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
            (output_dir / "Video [abc123]__plain__lang-auto.txt").write_text(
                "header\n\nCached transcript",
                encoding="utf-8",
            )
            (output_dir / "Video [abc123]__plain__lang-auto__summary.txt").write_text(
                "header\nSummary:\nCached summary",
                encoding="utf-8",
            )

            storage = LocalStorageBackend(output_dir=str(output_dir), audio_dir=str(audio_dir))
            with mock.patch("cill.app.storage", storage):
                state = create_or_reuse_job("https://youtube.com/watch?v=abc123")

        self.assertEqual(state["status"], "cache_hit")
        self.assertEqual(state["transcript"], "Cached transcript")
        self.assertEqual(state["summary"], "Cached summary")
        probe_mock.assert_not_called()

    @mock.patch("cill.app.probe_youtube_metadata", side_effect=RuntimeError("blocked"))
    def test_process_job_returns_error_state_when_metadata_probe_fails(self, probe_mock: mock.Mock) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = LocalStorageBackend(
                output_dir=str(Path(temp_dir) / "output"),
                audio_dir=str(Path(temp_dir) / "audio"),
            )
            with mock.patch("cill.app.storage", storage):
                state = create_or_reuse_job("https://youtube.com/watch?v=abc123")
                result = process_job(state)

        self.assertEqual(result["status"], "error")
        self.assertIn("blocked", result["error"])
        probe_mock.assert_called_once()

    @mock.patch("cill.app.probe_youtube_metadata")
    def test_create_or_reuse_job_enqueues_cache_miss_without_probe(self, probe_mock: mock.Mock) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = LocalStorageBackend(
                output_dir=str(Path(temp_dir) / "output"),
                audio_dir=str(Path(temp_dir) / "audio"),
            )
            with mock.patch("cill.app.storage", storage):
                state = create_or_reuse_job("https://youtube.com/watch?v=abc123")

        self.assertEqual(state["status"], "queued")
        self.assertEqual(state["video_id"], "abc123")
        probe_mock.assert_not_called()

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
                state = process_job(state)

        self.assertEqual(state["status"], "unsupported_live")

    @mock.patch("cill.app.probe_youtube_metadata")
    def test_process_job_rejects_over_duration(self, probe_mock: mock.Mock) -> None:
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
                state = process_job(state)

        self.assertEqual(state["status"], "unsupported_duration")

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
            }
            storage.save_state("job123", state)
            storage.write_text("job123", WEB_TRANSCRIPT_FILENAME, "Existing transcript")
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
        self.assertIn("Creating job", response.text)
        self.assertIn("https://youtube.com/watch?v=abc123", response.text)

    @mock.patch("cill.app.create_or_reuse_job")
    def test_create_job_endpoint_returns_queued_state(self, create_job_mock: mock.Mock) -> None:
        create_job_mock.return_value = {
            "job_id": "job123",
            "status": "queued",
            "video_id": "abc123",
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
        self.assertEqual(payload["status"], "queued")
        self.assertEqual(payload["video_id"], "abc123")
        self.assertIsNone(payload["error"])

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
        self.assertIn("let delayMs = 5000;", response.text)
        self.assertIn("Math.min(60000, delayMs + 5000)", response.text)
        self.assertNotIn("/api/jobs/${state.job_id}/run", response.text)

    @mock.patch("cill.worker.process_job")
    def test_worker_processes_only_queued_jobs(self, process_job_mock: mock.Mock) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = LocalStorageBackend(
                output_dir=str(Path(temp_dir) / "output"),
                audio_dir=str(Path(temp_dir) / "audio"),
            )
            storage.save_state("queued-job", {"job_id": "queued-job", "status": "queued", "source_url": "https://youtube.com/watch?v=abc123"})
            storage.save_state("done-job", {"job_id": "done-job", "status": "complete", "source_url": "https://youtube.com/watch?v=def456"})
            process_job_mock.return_value = {"status": "complete"}

            with mock.patch("cill.worker.storage", storage):
                processed = process_pending_jobs()

        self.assertEqual(processed, 1)
        process_job_mock.assert_called_once()
