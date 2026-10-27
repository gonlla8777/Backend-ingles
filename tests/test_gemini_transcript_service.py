import json
from unittest.mock import MagicMock
import pytest

from app.core.config import Settings
from app.services.curated_transcripts import CURATED_TRANSCRIPTS
from app.services.gemini_transcript_service import (
    _clean_json_text,
    _normalize_youtube_url,
    transcribe_youtube_with_gemini,
)
from app.services.youtube_extractor import YouTubeExtractorService


def test_normalize_youtube_url() -> None:
    assert (
        _normalize_youtube_url("dQw4w9WgXcQ")
        == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    )
    assert (
        _normalize_youtube_url("https://youtu.be/dQw4w9WgXcQ")
        == "https://youtu.be/dQw4w9WgXcQ"
    )


def test_clean_json_text_strips_markdown_fences() -> None:
    raw = '```json\n[{"text": "Hello", "start": 0.0, "duration": 2.0}]\n```'
    assert _clean_json_text(raw) == '[{"text": "Hello", "start": 0.0, "duration": 2.0}]'


def test_transcribe_youtube_with_gemini_success() -> None:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps(
        [
            {"text": "Welcome to our channel.", "start": 0.5, "duration": 3.0},
            {"text": "Today we will study English listening.", "start": 4.0, "duration": 4.5},
        ]
    )
    mock_client.models.generate_content.return_value = mock_response

    segments = transcribe_youtube_with_gemini(
        video_url="https://www.youtube.com/watch?v=sample12345",
        client=mock_client,
        model="gemini-3.8-flash",
    )

    assert len(segments) == 2
    assert segments[0]["text"] == "Welcome to our channel."
    assert segments[0]["start"] == 0.5
    assert segments[0]["duration"] == 3.0


def test_transcribe_youtube_with_gemini_wrapped_in_dict() -> None:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps(
        {
            "segments": [
                {"text": "First sentence.", "start": 1.0, "duration": 2.0},
            ]
        }
    )
    mock_client.models.generate_content.return_value = mock_response

    segments = transcribe_youtube_with_gemini(
        video_url="sample12345",
        client=mock_client,
    )

    assert len(segments) == 1
    assert segments[0]["text"] == "First sentence."


def test_transcribe_youtube_with_gemini_raises_without_api_key(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "")
    with pytest.raises(ValueError, match="GEMINI_API_KEY no configurada"):
        transcribe_youtube_with_gemini("dQw4w9WgXcQ")


def test_cascade_step1_curated_transcript_resolution() -> None:
    settings = Settings(enable_mock_fallback=True)
    service = YouTubeExtractorService(settings=settings)

    # jNQXAC9IVRw está en CURATED_TRANSCRIPTS
    result = service.fetch_transcript("jNQXAC9IVRw")

    assert result.video_id == "jNQXAC9IVRw"
    assert len(result.chunks) == len(CURATED_TRANSCRIPTS["jNQXAC9IVRw"])
    assert result.chunks[0].text == CURATED_TRANSCRIPTS["jNQXAC9IVRw"][0]["text"]
    assert result.is_auto_generated is False


def test_cascade_step2_gemini_transcription_when_not_in_curated() -> None:
    mock_gemini_transcriber = MagicMock()
    mock_gemini_transcriber.return_value = [
        {"text": "Gemini extracted this live video speech.", "start": 0.0, "duration": 3.5},
        {"text": "It bypasses YouTube datacenter blocks cleanly.", "start": 4.0, "duration": 4.0},
    ]

    settings = Settings(gemini_api_key="valid_key", enable_mock_fallback=True)
    service = YouTubeExtractorService(
        settings=settings,
        gemini_transcriber=mock_gemini_transcriber,
    )

    result = service.fetch_transcript("sample12345")

    assert mock_gemini_transcriber.called
    assert result.video_id == "sample12345"
    assert len(result.chunks) == 2
    assert result.chunks[0].text == "Gemini extracted this live video speech."
    assert result.is_auto_generated is True


def test_cascade_step4_fallback_when_gemini_and_scraping_fail() -> None:
    def failing_gemini(video_url: str):
        raise RuntimeError("Gemini quota exceeded or video restricted")

    class FailingApi:
        @classmethod
        def list_transcripts(cls, video_id, **kwargs):
            raise IpBlocked(video_id)

    settings = Settings(gemini_api_key="valid_key", enable_mock_fallback=True)
    service = YouTubeExtractorService(
        api_factory=FailingApi,
        settings=settings,
        gemini_transcriber=failing_gemini,
    )

    result = service.fetch_transcript("sample67890")

    assert result.video_id == "sample67890"
    assert len(result.chunks) == 6
    assert "Hello and welcome" in result.chunks[0].text
