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
    is_ip_blocked_or_retrieval_error,
    resolve_cookie_source,
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


def test_resolve_cookie_source_from_content_creates_tempfile_with_netscape_header() -> None:
    raw_cookie_line = ".youtube.com\tTRUE\t/\tTRUE\t2147483647\tSID\tsample_secret_cookie\n"
    settings = Settings(youtube_cookies_content=raw_cookie_line)

    file_path, is_temp = resolve_cookie_source(settings=settings)
    assert file_path is not None
    assert is_temp is True
    assert os.path.exists(file_path)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "# Netscape HTTP Cookie File" in content
        assert "sample_secret_cookie" in content

        # Verificar que requests.Session carga las cookies con User-Agent
        session = create_authenticated_session(file_path)
        assert session is not None
        assert session.cookies.get("SID") == "sample_secret_cookie"
        assert "Mozilla/5.0" in session.headers.get("User-Agent", "")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


def test_extractor_passes_cookiefile_and_http_client_to_api_factory() -> None:
    raw_cookie_line = (
        "# Netscape HTTP Cookie File\n"
        ".youtube.com\tTRUE\t/\tTRUE\t2147483647\tLOGIN_INFO\taf_info\n"
    )
    settings = Settings(
        youtube_cookies_content=raw_cookie_line,
        enable_mock_fallback=False,
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

    captured_kwargs = {}

    def mock_api_factory(**kwargs):
        captured_kwargs.update(kwargs)
        return mock_api_instance

    service = YouTubeExtractorService(
        api_factory=mock_api_factory,
        settings=settings,
    )
    result = service.fetch_transcript("dQw4w9WgXcQ")

    assert "cookiefile" in captured_kwargs
    assert "http_client" in captured_kwargs
    assert captured_kwargs["http_client"].cookies.get("LOGIN_INFO") == "af_info"
    assert result.chunks[0].text == "Authenticated caption"


def test_extractor_activates_mock_fallback_on_ip_blocked_and_yields_6_sentences_under_35s() -> None:
    settings = Settings(enable_mock_fallback=True)

    mock_api_instance = MagicMock()
    mock_api_instance.list.side_effect = IpBlocked("dQw4w9WgXcQ")

    service = YouTubeExtractorService(
        api_factory=lambda **_: mock_api_instance,
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

    mock_api_instance = MagicMock()
    mock_api_instance.list.side_effect = RequestBlocked("dQw4w9WgXcQ")

    service = YouTubeExtractorService(
        api_factory=lambda **_: mock_api_instance,
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


def test_extractor_cleans_up_temp_cookie_file_in_finally() -> None:
    raw_cookie_line = (
        "# Netscape HTTP Cookie File\n"
        ".youtube.com\tTRUE\t/\tTRUE\t2147483647\tSID\ttemp_check\n"
    )
    settings = Settings(
        youtube_cookies_content=raw_cookie_line,
        enable_mock_fallback=True,
    )

    created_paths = []

    def mock_api_factory(**kwargs):
        if "cookiefile" in kwargs:
            created_paths.append(kwargs["cookiefile"])
            # El archivo debe existir mientras se ejecuta la invocación
            assert os.path.exists(kwargs["cookiefile"])
        raise IpBlocked("test_vid")

    service = YouTubeExtractorService(
        api_factory=mock_api_factory,
        settings=settings,
    )

    service.fetch_transcript("dQw4w9WgXcQ")

    assert len(created_paths) == 1
    # Tras finalizar la ejecución, el archivo temporal debe haber sido limpiado
    assert not os.path.exists(created_paths[0])
