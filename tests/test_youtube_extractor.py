import tempfile
from unittest.mock import MagicMock
import pytest
from youtube_transcript_api._errors import IpBlocked, RequestBlocked, NoTranscriptFound

from app.core.config import Settings
from app.core.exceptions import TranscriptUnavailableError
from app.services.youtube_extractor import (
    YouTubeExtractorService,
    extract_video_id,
    resolve_cookie_file,
    create_authenticated_session,
    get_demo_transcript,
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


def test_resolve_cookie_file_from_content_prepends_netscape_header_if_missing() -> None:
    raw_cookie_line = ".youtube.com\tTRUE\t/\tTRUE\t2147483647\tSID\tsample_secret_cookie\n"
    settings = Settings(youtube_cookies_content=raw_cookie_line)

    file_path = resolve_cookie_file(settings=settings)
    assert file_path is not None

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "# Netscape HTTP Cookie File" in content
    assert "sample_secret_cookie" in content

    # Test create_authenticated_session successfully loads it
    session = create_authenticated_session(file_path)
    assert session is not None
    assert session.cookies.get("SID") == "sample_secret_cookie"
    assert "Mozilla/5.0" in session.headers.get("User-Agent", "")


def test_extractor_passes_authenticated_session_to_api_factory() -> None:
    raw_cookie_line = (
        "# Netscape HTTP Cookie File\n"
        ".youtube.com\tTRUE\t/\tTRUE\t2147483647\tLOGIN_INFO\taf_info\n"
    )
    settings = Settings(
        youtube_cookies_content=raw_cookie_line,
        allow_mock_fallback=False,
    )

    mock_api_instance = MagicMock()
    mock_transcript_list = MagicMock()
    mock_transcript = MagicMock()
    mock_transcript.language_code = "en"
    mock_transcript.is_generated = False
    mock_fetched = MagicMock()
    mock_fetched.to_raw_data.return_value = [
        {"text": "Authenticated caption", "start": 0.0, "duration": 2.0}
    ]
    mock_transcript.fetch.return_value = mock_fetched
    mock_transcript_list.find_manually_created_transcript.return_value = mock_transcript
    mock_api_instance.list.return_value = mock_transcript_list

    captured_http_client = []

    def mock_api_factory(http_client=None):
        captured_http_client.append(http_client)
        return mock_api_instance

    service = YouTubeExtractorService(
        api_factory=mock_api_factory,
        settings=settings,
    )
    result = service.fetch_transcript("dQw4w9WgXcQ")

    assert len(captured_http_client) == 1
    session = captured_http_client[0]
    assert session is not None
    assert session.cookies.get("LOGIN_INFO") == "af_info"
    assert result.chunks[0].text == "Authenticated caption"


def test_extractor_activates_mock_fallback_on_ip_blocked_when_enabled() -> None:
    settings = Settings(allow_mock_fallback=True)

    mock_api_instance = MagicMock()
    mock_api_instance.list.side_effect = IpBlocked("dQw4w9WgXcQ")

    service = YouTubeExtractorService(
        api_factory=lambda **_: mock_api_instance,
        settings=settings,
    )

    result = service.fetch_transcript("dQw4w9WgXcQ")
    assert result.video_id == "dQw4w9WgXcQ"
    assert len(result.chunks) >= 5
    assert "Hello and welcome" in result.chunks[0].text
    assert result.language_code == "en"


def test_extractor_raises_transcript_unavailable_when_fallback_disabled() -> None:
    settings = Settings(allow_mock_fallback=False)

    mock_api_instance = MagicMock()
    mock_api_instance.list.side_effect = RequestBlocked("dQw4w9WgXcQ")

    service = YouTubeExtractorService(
        api_factory=lambda **_: mock_api_instance,
        settings=settings,
    )

    with pytest.raises(TranscriptUnavailableError) as exc_info:
        service.fetch_transcript("dQw4w9WgXcQ")

    assert "dQw4w9WgXcQ" in str(exc_info.value)

