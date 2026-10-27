import json
from unittest.mock import MagicMock, patch
import pytest

from app.services.ytdlp_service import (
    YtDlpSubtitleResult,
    _vtt_time_to_seconds,
    extract_subtitles_with_ytdlp,
    parse_json3_subtitles,
    parse_vtt_subtitles,
)


def test_vtt_time_to_seconds() -> None:
    assert _vtt_time_to_seconds("00:01:23.450") == pytest.approx(83.45)
    assert _vtt_time_to_seconds("02:15.500") == pytest.approx(135.5)
    assert _vtt_time_to_seconds("invalid") == 0.0


def test_parse_vtt_subtitles() -> None:
    sample_vtt = """WEBVTT

00:00:01.000 --> 00:00:04.200
Welcome to this dictation practice session.

00:00:05.100 --> 00:00:08.500
<c>Listen</c> and type what you hear.
"""
    segments = parse_vtt_subtitles(sample_vtt)
    assert len(segments) == 2
    assert segments[0]["text"] == "Welcome to this dictation practice session."
    assert segments[0]["start"] == 1.0
    assert segments[0]["duration"] == 3.2
    assert segments[1]["text"] == "Listen and type what you hear."
    assert segments[1]["start"] == 5.1
    assert segments[1]["duration"] == 3.4


def test_parse_json3_subtitles() -> None:
    sample_json = {
        "events": [
            {
                "tStartMs": 1500,
                "dDurationMs": 3000,
                "segs": [{"utf8": "Hello "}, {"utf8": "engineers!"}],
            },
            {
                "tStartMs": 5000,
                "dDurationMs": 2500,
                "segs": [{"utf8": "[Music]"}],
            },
            {
                "tStartMs": 8000,
                "dDurationMs": 4000,
                "segs": [{"utf8": "This is our second sentence."}],
            },
        ]
    }
    segments = parse_json3_subtitles(sample_json)
    # [Music] should be filtered out
    assert len(segments) == 2
    assert segments[0]["text"] == "Hello engineers!"
    assert segments[0]["start"] == 1.5
    assert segments[0]["duration"] == 3.0
    assert segments[1]["text"] == "This is our second sentence."
    assert segments[1]["start"] == 8.0
    assert segments[1]["duration"] == 4.0


@patch("app.services.ytdlp_service.fetch_subtitle_content")
@patch("yt_dlp.YoutubeDL")
def test_extract_subtitles_with_ytdlp_success_json3(mock_ydl_cls, mock_fetch) -> None:
    mock_ydl_instance = MagicMock()
    mock_ydl_cls.return_value.__enter__.return_value = mock_ydl_instance

    mock_ydl_instance.extract_info.return_value = {
        "title": "Tutorial Video Title",
        "subtitles": {
            "en": [
                {"ext": "json3", "url": "https://example.com/subs.json3"},
            ]
        },
    }

    mock_fetch.return_value = json.dumps(
        {
            "events": [
                {
                    "tStartMs": 500,
                    "dDurationMs": 2500,
                    "segs": [{"utf8": "Fast and free subtitle extraction with yt-dlp."}],
                }
            ]
        }
    )

    result = extract_subtitles_with_ytdlp("https://www.youtube.com/watch?v=sample123")

    assert isinstance(result, YtDlpSubtitleResult)
    assert result.title == "Tutorial Video Title"
    assert len(result.segments) == 1
    assert result.segments[0]["text"] == "Fast and free subtitle extraction with yt-dlp."
    assert result.segments[0]["start"] == 0.5


@patch("yt_dlp.YoutubeDL")
def test_extract_subtitles_with_ytdlp_raises_when_no_subtitles(mock_ydl_cls) -> None:
    mock_ydl_instance = MagicMock()
    mock_ydl_cls.return_value.__enter__.return_value = mock_ydl_instance
    mock_ydl_instance.extract_info.return_value = {
        "title": "Video Without Subs",
        "subtitles": {},
        "automatic_captions": {},
    }

    with pytest.raises(ValueError, match="No se encontraron subtítulos"):
        extract_subtitles_with_ytdlp("https://www.youtube.com/watch?v=nosubs")

