from fastapi.testclient import TestClient
from youtube_transcript_api._errors import IpBlocked

from app.api.v1.endpoints.videos import get_extractor
from app.core.config import Settings, get_settings
from app.core.exceptions import TranscriptUnavailableError
from app.main import app
from app.schemas.transcript_schemas import RawTranscriptChunk
from app.services.youtube_extractor import (
    ExtractedTranscript,
    YouTubeExtractorService,
    extract_video_id,
)


class _FakeYouTubeExtractor:
    def fetch_transcript(
        self, url_or_id: str, preferred_languages: list[str] | None = None
    ) -> ExtractedTranscript:
        vid = extract_video_id(url_or_id)
        return ExtractedTranscript(
            video_id=vid,
            language_code="en",
            is_auto_generated=False,
            chunks=[
                RawTranscriptChunk(text="[Music]", start=0.0, duration=1.5),
                RawTranscriptChunk(
                    text="Hello everyone and welcome", start=1.5, duration=2.0
                ),
                RawTranscriptChunk(
                    text="to our technical interview prep!", start=3.5, duration=2.5
                ),
                RawTranscriptChunk(
                    text=">> Let's start with your background.",
                    start=6.5,
                    duration=3.0,
                ),
            ],
        )


def test_healthcheck_returns_cookie_and_fallback_status() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "youtube_cookies_configured" in data
    assert "mock_fallback_enabled" in data


def test_process_video_endpoint_returns_clean_coherent_segments() -> None:
    app.dependency_overrides[get_extractor] = lambda: _FakeYouTubeExtractor()
    try:
        client = TestClient(app)
        response = client.post(
            "/api/v1/videos/process-video",
            json={"video_url_or_id": "https://youtu.be/dQw4w9WgXcQ"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["video_id"] == "dQw4w9WgXcQ"
        assert body["segment_count"] == 2
        assert (
            body["segments"][0]["text"]
            == "Hello everyone and welcome to our technical interview prep!"
        )
        assert (
            body["segments"][1]["text"] == "Let's start with your background."
        )
        assert body["segments"][1]["is_speaker_turn_change"] is True
    finally:
        app.dependency_overrides.clear()


def test_process_video_endpoint_activates_mock_fallback_on_ip_block() -> None:
    class _FailingIpBlockedExtractor:
        def fetch_transcript(self, url_or_id: str, preferred_languages=None):
            raise TranscriptUnavailableError(
                "dQw4w9WgXcQ",
                reason="YouTube is blocking requests from your IP",
            )

    app.dependency_overrides[get_extractor] = lambda: _FailingIpBlockedExtractor()
    app.dependency_overrides[get_settings] = lambda: Settings(allow_mock_fallback=True)

    try:
        client = TestClient(app)
        response = client.post(
            "/api/v1/videos/process-video",
            json={"video_url_or_id": "https://youtu.be/dQw4w9WgXcQ"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["video_id"] == "dQw4w9WgXcQ"
        assert body["segment_count"] >= 5
        assert "Hello and welcome" in body["segments"][0]["text"]
    finally:
        app.dependency_overrides.clear()


def test_process_video_endpoint_returns_404_when_fallback_disabled() -> None:
    class _FailingIpBlockedExtractor:
        def fetch_transcript(self, url_or_id: str, preferred_languages=None):
            raise TranscriptUnavailableError(
                "dQw4w9WgXcQ",
                reason="YouTube is blocking requests from your IP",
            )

    app.dependency_overrides[get_extractor] = lambda: _FailingIpBlockedExtractor()
    app.dependency_overrides[get_settings] = lambda: Settings(allow_mock_fallback=False)

    try:
        client = TestClient(app)
        response = client.post(
            "/api/v1/videos/process-video",
            json={"video_url_or_id": "https://youtu.be/dQw4w9WgXcQ"},
        )
        assert response.status_code == 404
        assert "No hay subtítulos en inglés disponibles" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()


def test_process_video_rejects_invalid_url_with_422() -> None:
    client = TestClient(app)
    response = client.post(
        "/process-video",
        json={"video_url_or_id": "https://example.com/not-youtube"},
    )
    assert response.status_code == 422


def test_evaluate_transcription_accepts_canonical_contractions() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/v1/evaluation/evaluate-transcription",
        json={
            "segment_index": 0,
            "reference_text": "I am going to review the architecture today.",
            "user_input": "I'm gonna review the architecture today",
            "combo_multiplier": 1.5,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] == "semantic_equivalent"
    assert body["is_passed"] is True
    assert body["xp_earned"] > 0


def test_quiz_generator_returns_contextual_questions() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/v1/quiz/quiz-generator",
        json={
            "video_id": "dQw4w9WgXcQ",
            "transcript_sentences": [
                "Caching reduces database latency significantly.",
                "Always measure hit ratio in production.",
            ],
            "question_count": 2,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["questions"]) == 2
