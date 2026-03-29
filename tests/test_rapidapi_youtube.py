import unittest
from unittest import mock

from cill import rapidapi_youtube


class RapidAPIYoutubeTests(unittest.TestCase):
    def test_client_builds_video_details_request(self) -> None:
        session = mock.Mock()
        response = mock.Mock()
        response.json.return_value = {"ok": True}
        response.raise_for_status.return_value = None
        session.get.return_value = response
        client = rapidapi_youtube.RapidAPIYoutubeClient("secret-key", session=session)

        payload = client.get_video_details("abc123")

        self.assertEqual(payload, {"ok": True})
        session.get.assert_called_once_with(
            "https://youtube-media-downloader.p.rapidapi.com/v2/video/details",
            headers={
                "x-rapidapi-host": "youtube-media-downloader.p.rapidapi.com",
                "x-rapidapi-key": "secret-key",
            },
            params={
                "videoId": "abc123",
                "lang": "en",
                "subtitles": "true",
                "videos": "auto",
                "audios": "auto",
                "urlAccess": "normal",
            },
            timeout=rapidapi_youtube.DEFAULT_TIMEOUT_SECONDS,
        )

    def test_client_builds_subtitle_request(self) -> None:
        session = mock.Mock()
        response = mock.Mock()
        response.text = "subtitle text"
        response.raise_for_status.return_value = None
        session.get.return_value = response
        client = rapidapi_youtube.RapidAPIYoutubeClient("secret-key", session=session)

        subtitle_text = client.get_subtitle_text("https://example.com/subtitles")

        self.assertEqual(subtitle_text, "subtitle text")
        session.get.assert_called_once_with(
            "https://youtube-media-downloader.p.rapidapi.com/v2/video/subtitles",
            headers={
                "x-rapidapi-host": "youtube-media-downloader.p.rapidapi.com",
                "x-rapidapi-key": "secret-key",
            },
            params={
                "subtitleUrl": "https://example.com/subtitles",
                "format": "srt",
                "fixOverlap": "true",
            },
            timeout=rapidapi_youtube.DEFAULT_TIMEOUT_SECONDS,
        )

    def test_client_uses_x_rapidapi_key_from_env(self) -> None:
        with mock.patch.dict("os.environ", {"X_RAPIDAPI_KEY": "secret-key"}, clear=True):
            client = rapidapi_youtube.RapidAPIYoutubeClient.from_env()

        self.assertEqual(client.api_key, "secret-key")

    def test_extract_and_choose_subtitle_track_prefers_english_manual(self) -> None:
        details = {
            "subtitles": {
                "items": [
                    {"url": "https://example.com/auto", "code": "en", "text": "English (auto-generated)"},
                    {"url": "https://example.com/manual", "code": "en", "text": "English"},
                    {"url": "https://example.com/it", "code": "it", "text": "Italian"},
                ]
            }
        }

        track = rapidapi_youtube.choose_subtitle_track(
            rapidapi_youtube.extract_subtitle_tracks(details)
        )

        self.assertIsNotNone(track)
        self.assertEqual(track["url"], "https://example.com/manual")
        self.assertFalse(track["auto_generated"])

    def test_normalize_xml_subtitles(self) -> None:
        raw = '<transcript><text start="0" dur="1">Hello &amp; welcome</text><text start="1" dur="1">to cill</text></transcript>'

        normalized = rapidapi_youtube.normalize_subtitle_text(raw)

        self.assertEqual(normalized, "Hello & welcome\nto cill")

    def test_normalize_srt_subtitles(self) -> None:
        raw = """1
00:00:00,000 --> 00:00:02,000
Hello there

2
00:00:02,100 --> 00:00:04,000
<i>General Kenobi</i>
"""

        normalized = rapidapi_youtube.normalize_subtitle_text(raw)

        self.assertEqual(normalized, "Hello there\nGeneral Kenobi")

    def test_build_subtitle_stats(self) -> None:
        stats = rapidapi_youtube.build_subtitle_stats("one two three", 60)

        self.assertEqual(stats["word_count"], 3)
        self.assertEqual(stats["char_count"], len("one two three"))
        self.assertEqual(stats["duration_seconds"], 60)
        self.assertEqual(stats["words_per_minute"], 3.0)
