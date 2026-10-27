from fastapi.testclient import TestClient

from app.api.v1.endpoints.videos import get_extractor
from app.core.config import Settings, get_settings
from app.core.exceptions import TranscriptUnavailableError
from app.main import app
from app.schemas.transcript_schemas import RawTranscriptChunk
from app.services.youtube_extractor import (
    ExtractedTranscript,
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
    assert "youtube_proxy_configured" in data
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
                reason="YouTube is blocking requests from your IP. This usually is due to... IP belonging to a cloud provider",
            )

    app.dependency_overrides[get_extractor] = lambda: _FailingIpBlockedExtractor()
    app.dependency_overrides[get_settings] = lambda: Settings(
        enable_mock_fallback=True
    )

    try:
        client = TestClient(app)
        response = client.post(
            "/api/v1/videos/process-video",
            json={"video_url_or_id": "https://youtu.be/dQw4w9WgXcQ"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["video_id"] == "dQw4w9WgXcQ"
        assert body["segment_count"] == 6
        assert "Hello and welcome" in body["segments"][0]["text"]
        assert body["segments"][-1]["end_time"] <= 35.0
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
    app.dependency_overrides[get_settings] = lambda: Settings(
        enable_mock_fallback=False,
        allow_mock_fallback=False,
    )

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


def test_process_transcript_endpoint_segments_client_raw_chunks() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/v1/videos/process-transcript",
        json={
            "video_id": "dQw4w9WgXcQ",
            "title": "Client Extracted Video",
            "raw_transcript": [
                {"text": "Hello world and welcome", "start": 0.5, "duration": 2.5},
                {"text": "to our interactive English player.", "start": 3.0, "duration": 3.0},
            ],
            "language_code": "en",
            "is_auto_generated": False,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["video_id"] == "dQw4w9WgXcQ"
    assert body["segment_count"] == 1
    assert body["segments"][0]["text"] == "Hello world and welcome to our interactive English player."
    assert body["segments"][0]["start_time"] == 0.5
    assert body["segments"][0]["end_time"] == 6.0


def test_process_transcript_endpoint_activates_fallback_on_empty_list() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/v1/videos/process-transcript",
        json={
            "video_id": "dQw4w9WgXcQ",
            "raw_transcript": [],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["video_id"] == "dQw4w9WgXcQ"
    assert body["segment_count"] == 6


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


def test_process_video_curated_project_zomboid() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/v1/videos/process-video",
        json={"url": "https://www.youtube.com/watch?v=project_zomboid"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["video_id"] == "project_zomboid"
    assert "Project Zomboid" in data["title"]
    assert data["segment_count"] >= 5
    assert "Project Zomboid" in data["segments"][0]["text"]


def test_process_video_curated_q7ecb76yboo() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/v1/videos/process-video",
        json={"url": "https://www.youtube.com/watch?v=Q7ecB76Yboo"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["video_id"] == "Q7ecB76Yboo"
    assert "Project Zomboid" in data["title"] or "Ultimate Beginners Guide" in data["title"]
    assert data["segment_count"] >= 5
    assert "Project Zomboid" in data["segments"][0]["text"]



def test_process_video_route_aliases_and_methods_prevent_405() -> None:
    client = TestClient(app)
    payload = {"url": "https://www.youtube.com/watch?v=jNQXAC9IVRw"}

    # 1. Rutas con prefijo /api/v1/videos
    res1 = client.post("/api/v1/videos/process-video", json=payload)
    assert res1.status_code == 200

    # 2. Rutas con trailing slash
    res2 = client.post("/api/v1/videos/process-video/", json=payload)
    assert res2.status_code == 200

    # 3. Alias /api/v1/process-video
    res3 = client.post("/api/v1/process-video", json=payload)
    assert res3.status_code == 200

    # 4. Prefijo directo /videos/process-video
    res4 = client.post("/videos/process-video", json=payload)
    assert res4.status_code == 200

    # 5. Alias raíz /process-video
    res5 = client.post("/process-video", json=payload)
    assert res5.status_code == 200

    # 6. Método GET con query param no arroja 405
    res6 = client.get("/api/v1/videos/process-video?url=project_zomboid")
    assert res6.status_code == 200
    assert "project_zomboid" in res6.json()["video_id"]


def test_process_video_ytdlp_extraction_success() -> None:
    from unittest.mock import patch
    from app.services.ytdlp_service import YtDlpSubtitleResult

    fake_result = YtDlpSubtitleResult(
        segments=[
            {"text": "Extracted via yt-dlp in milliseconds.", "start": 0.5, "duration": 3.0},
            {"text": "Zero token consumption on YouTube official captions.", "start": 3.5, "duration": 4.0},
        ],
        title="Custom YouTube Video Title",
    )

    with patch("app.api.v1.endpoints.videos.extract_subtitles_with_ytdlp", return_value=fake_result):
        client = TestClient(app)
        response = client.post(
            "/api/v1/videos/process-video",
            json={"url": "https://www.youtube.com/watch?v=ytdlpTest123"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["video_id"] == "ytdlpTest12"
        assert data["title"] == "Custom YouTube Video Title"
        assert data["segment_count"] >= 1
        assert "yt-dlp" in data["segments"][0]["text"]


