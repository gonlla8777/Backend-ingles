import logging
import os
from unittest.mock import MagicMock
import pytest
from youtube_transcript_api._errors import (
    CouldNotRetrieveTranscript,
    IpBlocked,
    RequestBlocked,
)

from app.core.config import Settings
from app.core.exceptions import TranscriptUnavailableError
from app.services.youtube_extractor import (
    YouTubeExtractorService,
    create_authenticated_session,
    extract_video_id,
    get_cookies_path,
    is_ip_blocked_or_retrieval_error,
)


def test_extract_video_id_parses_various_formats() -> None:
    assert extract_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert (
        extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        == "dQw4w9WgXcQ"
    )
    assert extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert (
        extract_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ")
        == "dQw4w9WgXcQ"
    )


def test_get_cookies_path_from_content_writes_and_closes_tempfile(
    monkeypatch,
) -> None:
    raw_cookie_line = ".youtube.com\tTRUE\t/\tTRUE\t2147483647\tSID\tsample_secret_cookie\n"
    monkeypatch.setenv("YOUTUBE_COOKIES_CONTENT", raw_cookie_line)

    cookies_path = get_cookies_path()
    assert cookies_path is not None
    assert os.path.exists(cookies_path)

    # Verificar que el archivo está cerrado y se puede leer
    with open(cookies_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "# Netscape HTTP Cookie File" in content
    assert "sample_secret_cookie" in content

    # Comprobar que create_authenticated_session lee las cookies
    session = create_authenticated_session(cookies_path)
    assert session is not None
    assert session.cookies.get("SID") == "sample_secret_cookie"
    assert "Mozilla/5.0" in session.headers.get("User-Agent", "")


def test_extractor_passes_cookies_kwargs_to_api_factory(monkeypatch) -> None:
    monkeypatch.setenv(
        "YOUTUBE_COOKIES_CONTENT",
        "# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tTRUE\t2147483647\tLOGIN_INFO\taf_info\n",
    )
    settings = Settings(enable_mock_fallback=False)

    mock_transcript_list = MagicMock()
    mock_transcript = MagicMock()
    mock_transcript.language_code = "en"
    mock_transcript.is_generated = False
    mock_transcript.fetch.return_value.to_raw_data.return_value = [
        {"text": "Authenticated caption", "start": 0.0, "duration": 2.0}
    ]
    mock_transcript_list.find_manually_created_transcript.return_value = (
        mock_transcript
    )

    captured_kwargs = {}

    class MockApi:
        @classmethod
        def list_transcripts(cls, video_id, **kwargs):
            captured_kwargs.update(kwargs)
            return mock_transcript_list

    service = YouTubeExtractorService(
        api_factory=MockApi,
        settings=settings,
    )
    result = service.fetch_transcript("dQw4w9WgXcQ")

    assert "cookies" in captured_kwargs
    assert os.path.exists(captured_kwargs["cookies"])
    assert result.chunks[0].text == "Authenticated caption"


def test_extractor_logs_warning_when_failing_with_cookies_path(
    monkeypatch, caplog
) -> None:
    monkeypatch.setenv(
        "YOUTUBE_COOKIES_CONTENT",
        "# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tTRUE\t2147483647\tSID\t123\n",
    )
    settings = Settings(enable_mock_fallback=True)

    class FailingApi:
        @classmethod
        def list_transcripts(cls, video_id, **kwargs):
            raise IpBlocked(video_id)

    service = YouTubeExtractorService(
        api_factory=FailingApi,
        settings=settings,
    )

    with caplog.at_level(logging.WARNING):
        result = service.fetch_transcript("dQw4w9WgXcQ")

    # Verificamos que se activa el fallback pedagógico
    assert result.video_id == "dQw4w9WgXcQ"
    assert len(result.chunks) == 6

    # Verificamos el log de advertencia con cookies_path
    assert any(
        "Extracción falló aún con cookies_path=" in record.message
        for record in caplog.records
    )


def test_extractor_activates_mock_fallback_on_ip_blocked_under_35s() -> None:
    settings = Settings(enable_mock_fallback=True)

    class FailingApi:
        @classmethod
        def list_transcripts(cls, video_id, **kwargs):
            raise IpBlocked(video_id)

    service = YouTubeExtractorService(
        api_factory=FailingApi,
        settings=settings,
    )

    result = service.fetch_transcript("dQw4w9WgXcQ")
    assert result.video_id == "dQw4w9WgXcQ"
    assert len(result.chunks) == 6
    assert "Hello and welcome" in result.chunks[0].text
    assert result.language_code == "en"

    # Verificar que los timestamps están estrictamente entre 0s y 35s
    last_chunk = result.chunks[-1]
    assert last_chunk.start >= 0.0
    assert (last_chunk.start + last_chunk.duration) <= 35.0


def test_extractor_raises_transcript_unavailable_when_fallback_disabled() -> None:
    settings = Settings(enable_mock_fallback=False)

    class FailingApi:
        @classmethod
        def list_transcripts(cls, video_id, **kwargs):
            raise RequestBlocked(video_id)

    service = YouTubeExtractorService(
        api_factory=FailingApi,
        settings=settings,
    )

    with pytest.raises(TranscriptUnavailableError) as exc_info:
        service.fetch_transcript("dQw4w9WgXcQ")

    assert "dQw4w9WgXcQ" in str(exc_info.value)


def test_is_ip_blocked_or_retrieval_error_detection() -> None:
    assert is_ip_blocked_or_retrieval_error(IpBlocked("vid")) is True
    assert is_ip_blocked_or_retrieval_error(RequestBlocked("vid")) is True
    assert is_ip_blocked_or_retrieval_error(CouldNotRetrieveTranscript("vid")) is True
    assert (
        is_ip_blocked_or_retrieval_error(
            RuntimeError("YouTube is blocking requests from your IP.")
        )
        is True
    )
    assert (
        is_ip_blocked_or_retrieval_error(
            Exception("IP belonging to a cloud provider blocked by YouTube")
        )
        is True
    )
