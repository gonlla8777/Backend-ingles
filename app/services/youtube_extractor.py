import re
from dataclasses import dataclass
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import YouTubeTranscriptApiException

from app.core.exceptions import InvalidYouTubeUrlError, TranscriptUnavailableError
from app.schemas.transcript_schemas import RawTranscriptChunk

_DIRECT_VIDEO_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{11}$")
_YOUTUBE_URL_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?.*?v=|embed/|shorts/|live/)|youtu\.be/)([a-zA-Z0-9_-]{11})"
)


@dataclass(frozen=True)
class ExtractedTranscript:
    video_id: str
    language_code: str
    is_auto_generated: bool
    chunks: list[RawTranscriptChunk]


def extract_video_id(url_or_id: str) -> str:
    """Extrae el ID de 11 caracteres desde cualquier formato de URL de YouTube."""
    candidate = url_or_id.strip()
    if _DIRECT_VIDEO_ID_ID := _DIRECT_VIDEO_ID_RE.match(candidate):
        return _DIRECT_VIDEO_ID_ID.group(0)

    match = _YOUTUBE_URL_RE.search(candidate)
    if match:
        return match.group(1)

    raise InvalidYouTubeUrlError(url_or_id)


class YouTubeExtractorService:
    """Servicio thread-safe de extracción de subtítulos usando `youtube-transcript-api` v1.x."""

    def __init__(self, api_factory: type[YouTubeTranscriptApi] = YouTubeTranscriptApi) -> None:
        self._api_factory = api_factory

    def fetch_transcript(
        self,
        url_or_id: str,
        preferred_languages: list[str] | None = None,
    ) -> ExtractedTranscript:
        video_id = extract_video_id(url_or_id)
        languages = preferred_languages or ["en", "en-US", "en-GB", "en-AU", "en-CA"]

        # Se instancia por llamada porque YouTubeTranscriptApi usa requests.Session no thread-safe.
        ytt_api = self._api_factory()

        try:
            transcript_list = ytt_api.list(video_id)

            selected_transcript = None
            try:
                selected_transcript = transcript_list.find_manually_created_transcript(
                    languages
                )
            except YouTubeTranscriptApiException:
                selected_transcript = transcript_list.find_generated_transcript(
                    languages
                )

            fetched = selected_transcript.fetch()
            raw_items = fetched.to_raw_data()

            chunks = [
                RawTranscriptChunk(
                    text=str(item.get("text", "")),
                    start=float(item.get("start", 0.0)),
                    duration=max(0.0, float(item.get("duration", 0.0))),
                )
                for item in raw_items
            ]

            return ExtractedTranscript(
                video_id=video_id,
                language_code=selected_transcript.language_code,
                is_auto_generated=bool(selected_transcript.is_generated),
                chunks=chunks,
            )
        except YouTubeTranscriptApiException as exc:
            raise TranscriptUnavailableError(video_id, reason=str(exc)) from exc

