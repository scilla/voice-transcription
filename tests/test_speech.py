import os
import io
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import speech


class SpeechTests(unittest.TestCase):
    def test_parse_args_supports_verbose_flag(self) -> None:
        args = speech.parse_args(["--verbose"])

        self.assertTrue(args.verbose)

    def test_parse_args_supports_non_interactive_flags(self) -> None:
        args = speech.parse_args(
            [
                "--url",
                "https://www.youtube.com/watch?v=abc123",
                "--language",
                "en",
                "--live-duration-minutes",
                "15",
            ]
        )

        self.assertEqual(args.url, "https://www.youtube.com/watch?v=abc123")
        self.assertEqual(args.language, "en")
        self.assertEqual(args.live_duration_minutes, 15.0)

    def test_log_verbose_prints_only_when_enabled(self) -> None:
        original_verbose = speech.VERBOSE
        try:
            speech.VERBOSE = True
            with mock.patch("sys.stdout", new_callable=io.StringIO) as stdout:
                speech.log_verbose("hello")
            self.assertIn("[verbose] hello", stdout.getvalue())

            speech.VERBOSE = False
            with mock.patch("sys.stdout", new_callable=io.StringIO) as stdout:
                speech.log_verbose("hidden")
            self.assertEqual(stdout.getvalue(), "")
        finally:
            speech.VERBOSE = original_verbose

    def test_normalize_language_hint_handles_auto(self) -> None:
        self.assertIsNone(speech.normalize_language_hint(None))
        self.assertIsNone(speech.normalize_language_hint(""))
        self.assertIsNone(speech.normalize_language_hint("auto"))
        self.assertEqual(speech.normalize_language_hint("en"), "en")

    def test_validate_args_rejects_non_youtube_url(self) -> None:
        args = speech.parse_args(["--url", "https://example.com/video"])

        with self.assertRaises(SystemExit):
            speech.validate_args(args)

    def test_determine_source_type_prefers_file_or_url_flags(self) -> None:
        self.assertEqual(speech.determine_source_type(speech.parse_args(["--file", "clip.mp3"])), "local")
        self.assertEqual(
            speech.determine_source_type(
                speech.parse_args(["--url", "https://www.youtube.com/watch?v=abc123"])
            ),
            "youtube",
        )

    def test_resolve_local_file_path_accepts_repo_relative_and_sources_relative(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            sources_dir = os.path.join(temp_dir, "sources")
            nested_dir = os.path.join(sources_dir, "nested")
            os.makedirs(nested_dir, exist_ok=True)

            direct_file = os.path.join(temp_dir, "direct.mp3")
            nested_file = os.path.join(nested_dir, "clip.mp3")
            with open(direct_file, "wb") as file_handle:
                file_handle.write(b"data")
            with open(nested_file, "wb") as file_handle:
                file_handle.write(b"data")

            resolved_direct = speech.resolve_local_file_path(direct_file, sources_dir=sources_dir)
            resolved_nested = speech.resolve_local_file_path("nested/clip.mp3", sources_dir=sources_dir)

        self.assertEqual(resolved_direct, direct_file)
        self.assertEqual(resolved_nested, nested_file)

    def test_recursive_source_discovery_includes_nested_youtube_downloads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_dir = os.path.join(temp_dir, "youtube")
            os.makedirs(nested_dir, exist_ok=True)

            top_level_file = os.path.join(temp_dir, "older.mp3")
            nested_file = os.path.join(nested_dir, "newer.mp3")

            with open(top_level_file, "wb") as file_handle:
                file_handle.write(b"old")
            with open(nested_file, "wb") as file_handle:
                file_handle.write(b"new")

            os.utime(top_level_file, (1, 1))
            os.utime(nested_file, (2, 2))

            files = speech.get_audio_files_from_sources(temp_dir)

        self.assertEqual(files[0][0], "youtube/newer.mp3")
        self.assertEqual(files[1][0], "older.mp3")

    def test_classify_youtube_live_status(self) -> None:
        self.assertEqual(speech.classify_youtube_live_status({"live_status": "is_live"}), "is_live")
        self.assertEqual(speech.classify_youtube_live_status({"live_status": "post_live"}), "post_live")
        self.assertEqual(speech.classify_youtube_live_status({"is_live": True}), "is_live")
        self.assertEqual(speech.classify_youtube_live_status({"was_live": True}), "was_live")
        self.assertEqual(speech.classify_youtube_live_status({}), "not_live")

    def test_build_ytdlp_vod_command(self) -> None:
        command = speech.build_ytdlp_vod_command(
            "https://www.youtube.com/watch?v=abc123",
            "./sources/youtube/Test [abc123].%(ext)s",
        )

        self.assertIn("--extract-audio", command)
        self.assertIn("--audio-format", command)
        self.assertIn("mp3", command)
        self.assertIn("./sources/youtube/Test [abc123].%(ext)s", command)

    def test_build_ytdlp_live_command(self) -> None:
        command = speech.build_ytdlp_live_command(
            "https://www.youtube.com/watch?v=abc123",
            "./sources/youtube/Test [abc123].%(ext)s",
        )

        self.assertIn("--live-from-start", command)
        self.assertIn("--format", command)
        self.assertIn("bestaudio/best", command)
        self.assertIn("--no-part", command)

    def test_build_ytdlp_live_section_command(self) -> None:
        command = speech.build_ytdlp_live_section_command(
            "https://www.youtube.com/watch?v=abc123",
            "./sources/youtube/Test [abc123]__live-180s.%(ext)s",
            duration_seconds=180,
        )

        self.assertIn("--live-from-start", command)
        self.assertIn("--download-sections", command)
        self.assertIn("*0-180", command)
        self.assertIn("./sources/youtube/Test [abc123]__live-180s.%(ext)s", command)

    def test_build_ytdlp_live_now_command(self) -> None:
        command = speech.build_ytdlp_live_now_command(
            "https://www.youtube.com/watch?v=abc123",
            "./sources/youtube/Test [abc123]__live-now-120s.%(ext)s",
            live_duration_seconds=120,
        )

        self.assertIn("--live-from-start", command)
        self.assertIn("--download-sections", command)
        self.assertIn("*0-120", command)
        self.assertIn("./sources/youtube/Test [abc123]__live-now-120s.%(ext)s", command)

    @mock.patch("speech.wait_for_process_shutdown")
    @mock.patch("speech.find_downloaded_youtube_file")
    @mock.patch("speech.signal_process")
    @mock.patch("speech.time.sleep", side_effect=KeyboardInterrupt)
    @mock.patch("speech.subprocess.Popen")
    def test_live_capture_interrupt_keeps_downloaded_media(
        self,
        popen_mock: mock.Mock,
        _sleep_mock: mock.Mock,
        signal_process_mock: mock.Mock,
        find_downloaded_mock: mock.Mock,
        wait_for_shutdown_mock: mock.Mock,
    ) -> None:
        process = mock.Mock()
        process.pid = 1234
        process.poll.side_effect = [None, None]
        process.returncode = 130
        popen_mock.return_value = process
        find_downloaded_mock.return_value = "./sources/youtube/Test [abc123].webm"

        output_path = speech.capture_youtube_live(
            "https://www.youtube.com/watch?v=abc123",
            {"id": "abc123", "title": "Test"},
            max_duration_minutes=None,
            yt_dlp_path="yt-dlp",
        )

        self.assertEqual(output_path, ("./sources/youtube/Test [abc123].webm", "manually_stopped"))
        signal_process_mock.assert_called()
        wait_for_shutdown_mock.assert_called_once_with(process)

    @mock.patch("speech.download_youtube_live_section")
    def test_live_capture_timeout_uses_bounded_download(
        self,
        download_section_mock: mock.Mock,
    ) -> None:
        download_section_mock.return_value = ("./sources/youtube/Test [abc123]__live-60s.mp3", "reached_timeout")

        output_path = speech.capture_youtube_live(
            "https://www.youtube.com/watch?v=abc123",
            {"id": "abc123", "title": "Test"},
            max_duration_minutes=1,
            force_download=True,
            yt_dlp_path="yt-dlp",
        )

        self.assertEqual(output_path, ("./sources/youtube/Test [abc123]__live-60s.mp3", "reached_timeout"))
        download_section_mock.assert_called_once_with(
            "https://www.youtube.com/watch?v=abc123",
            {"id": "abc123", "title": "Test"},
            duration_seconds=60,
            suffix="live-60s",
            force_download=True,
            capture_outcome="reached_timeout",
            yt_dlp_path="yt-dlp",
        )

    @mock.patch("speech.find_downloaded_youtube_file", return_value="./sources/youtube/Test [abc123].webm")
    @mock.patch("speech.subprocess.Popen")
    def test_live_capture_stream_end_returns_stream_ended(
        self,
        popen_mock: mock.Mock,
        _find_downloaded_mock: mock.Mock,
    ) -> None:
        process = mock.Mock()
        process.pid = 1234
        process.poll.side_effect = [0]
        process.returncode = 0
        popen_mock.return_value = process

        output_path = speech.capture_youtube_live(
            "https://www.youtube.com/watch?v=abc123",
            {"id": "abc123", "title": "Test"},
            max_duration_minutes=None,
            yt_dlp_path="yt-dlp",
        )

        self.assertEqual(output_path, ("./sources/youtube/Test [abc123].webm", "stream_ended"))

    @mock.patch("speech.subprocess.run")
    @mock.patch("speech.find_downloaded_youtube_file", return_value="./sources/youtube/Test [abc123].mp3")
    def test_download_youtube_vod_reuses_existing_audio(
        self,
        find_downloaded_mock: mock.Mock,
        run_mock: mock.Mock,
    ) -> None:
        output_path = speech.download_youtube_vod(
            "https://www.youtube.com/watch?v=abc123",
            {"id": "abc123", "title": "Test"},
            force_download=False,
            yt_dlp_path="yt-dlp",
        )

        self.assertEqual(output_path, "./sources/youtube/Test [abc123].mp3")
        find_downloaded_mock.assert_called_once()
        run_mock.assert_not_called()

    @mock.patch(
        "speech.find_download_from_output_template",
        return_value="./sources/youtube/Test [abc123]__live-now-120s.mp3",
    )
    @mock.patch("speech.subprocess.run")
    def test_download_youtube_live_snapshot_reuses_existing_audio(
        self,
        run_mock: mock.Mock,
        find_downloaded_mock: mock.Mock,
    ) -> None:
        output_path = speech.download_youtube_live_snapshot(
            "https://www.youtube.com/watch?v=abc123",
            {"id": "abc123", "title": "Test", "duration": 120},
            force_download=False,
            yt_dlp_path="yt-dlp",
        )

        self.assertEqual(
            output_path,
            ("./sources/youtube/Test [abc123]__live-now-120s.mp3", "captured_now"),
        )
        find_downloaded_mock.assert_called_once()
        run_mock.assert_not_called()

    def test_blank_language_hint_omits_language_parameter(self) -> None:
        client_mock = mock.Mock()
        create_mock = client_mock.audio.transcriptions.create
        create_mock.return_value = SimpleNamespace(text="ciao", segments=[])

        with tempfile.NamedTemporaryFile(suffix=".mp3") as temp_audio, mock.patch(
            "speech.get_openai_client", return_value=client_mock
        ):
            speech.transcribe_chunk_with_retry(temp_audio.name, None, diarize=False)

        _, kwargs = create_mock.call_args
        self.assertNotIn("language", kwargs)

    def test_explicit_language_hint_is_forwarded(self) -> None:
        client_mock = mock.Mock()
        create_mock = client_mock.audio.transcriptions.create
        create_mock.return_value = SimpleNamespace(text="hello", segments=[])

        with tempfile.NamedTemporaryFile(suffix=".mp3") as temp_audio, mock.patch(
            "speech.get_openai_client", return_value=client_mock
        ):
            speech.transcribe_chunk_with_retry(temp_audio.name, "en", diarize=False)

        _, kwargs = create_mock.call_args
        self.assertEqual(kwargs["language"], "en")

    def test_build_transcription_output_path_adds_config_tags(self) -> None:
        output_path = speech.build_transcription_output_path(
            {
                "source_type": "youtube",
                "selected_source_path": "./sources/youtube/Test [abc123].mp3",
            },
            "en",
            diarize=True,
        )

        self.assertTrue(output_path.endswith("Test [abc123]__diarized__lang-en.txt"))

    def test_build_summary_output_path_adds_summary_suffix(self) -> None:
        output_path = speech.build_summary_output_path(
            {
                "source_type": "youtube",
                "selected_source_path": "./sources/youtube/Test [abc123].mp3",
            },
            "en",
            diarize=False,
        )

        self.assertTrue(output_path.endswith("Test [abc123]__plain__lang-en__summary.txt"))

    def test_should_skip_transcription_for_existing_youtube_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "existing.txt")
            with open(output_path, "w", encoding="utf-8") as file_handle:
                file_handle.write("done")

            should_skip = speech.should_skip_transcription(
                {"source_type": "youtube"},
                output_path,
                force_transcribe=False,
            )

        self.assertTrue(should_skip)

    def test_extract_transcript_text_from_output_prefers_full_transcript_section(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "transcript.txt")
            with open(output_path, "w", encoding="utf-8") as file_handle:
                file_handle.write("Header\n\nSegments:\nfoo\n\nFull transcript:\nclean body")

            transcript_text = speech.extract_transcript_text_from_output(output_path)

        self.assertEqual(transcript_text, "clean body")

    def test_describe_source_for_summary_mentions_live_stop_reason(self) -> None:
        description = speech.describe_source_for_summary(
            {
                "source_type": "youtube",
                "youtube_live_status": "is_live",
                "youtube_live_capture_outcome": "manually_stopped",
            }
        )

        self.assertIn("live video", description)
        self.assertIn("manually stopped", description)

    def test_describe_source_for_summary_mentions_live_now_snapshot(self) -> None:
        description = speech.describe_source_for_summary(
            {
                "source_type": "youtube",
                "youtube_live_status": "is_live",
                "youtube_live_capture_outcome": "captured_now",
            }
        )

        self.assertIn("live video", description)
        self.assertIn("available at request time", description)

    def test_summarize_transcript_includes_live_context_in_prompt(self) -> None:
        client_mock = mock.Mock()
        create_mock = client_mock.chat.completions.create
        create_mock.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Summary text"))]
        )

        with mock.patch("speech.get_openai_client", return_value=client_mock):
            summary = speech.summarize_transcript(
                "Transcript body",
                {
                    "source_type": "youtube",
                    "youtube_live_status": "is_live",
                    "youtube_live_capture_outcome": "reached_timeout",
                },
            )

        self.assertEqual(summary, "Summary text")
        _, kwargs = create_mock.call_args
        user_prompt = kwargs["messages"][1]["content"]
        self.assertIn("reached the requested timeout", user_prompt)
        self.assertIn("Transcript body", user_prompt)

    def test_output_header_contains_youtube_metadata(self) -> None:
        header = speech.build_output_header(
            {
                "source_type": "youtube",
                "selected_source_path": "./sources/youtube/Test [abc123].webm",
                "youtube_url": "https://www.youtube.com/watch?v=abc123",
                "youtube_title": "Test stream",
                "youtube_uploader": "Channel",
                "youtube_video_id": "abc123",
                "youtube_live_status": "is_live",
                "youtube_live_capture_outcome": "stream_ended",
            },
            "./sources/youtube/Test [abc123].mp3",
            "it",
            diarize=True,
        )

        joined_header = "\n".join(header)
        self.assertIn("YouTube URL: https://www.youtube.com/watch?v=abc123", joined_header)
        self.assertIn("YouTube title: Test stream", joined_header)
        self.assertIn("YouTube uploader: Channel", joined_header)
        self.assertIn("YouTube video id: abc123", joined_header)
        self.assertIn("YouTube live status: is_live", joined_header)
        self.assertIn("YouTube live capture outcome: stream_ended", joined_header)
        self.assertIn("Language hint: it", joined_header)
        self.assertIn("Diarization: enabled", joined_header)


if __name__ == "__main__":
    unittest.main()
