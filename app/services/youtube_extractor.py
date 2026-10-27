import atexit
import http.cookiejar
import logging
import os
from pathlib import Path
import re
import tempfile
from dataclasses import dataclass
from typing import Optional

import requests
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import YouTubeTranscriptApiException

from app.core.config import Settings, get_settings
from app.core.exceptions import InvalidYouTubeUrlError, TranscriptUnavailableError
from app.schemas.transcript_schemas import RawTranscriptChunk

logger = logging.getLogger(__name__)

_DIRECT_VIDEO_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{11}$")
_YOUTUBE_URL_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?.*?v=|embed/|shorts/|live/)|youtu\.be/)([a-zA-Z0-9_-]{11})"
)

_CACHED_TEMP_COOKIE_FILE: Optional[str] = None


@dataclass(frozen=True)
class ExtractedTranscript:
    video_id: str
    language_code: str
    is_auto_generated: bool
    chunks: list[RawTranscriptChunk]


def extract_video_id(url_or_id: str) -> str:
    """Extrae el ID de 11 caracteres desde cualquier formato de URL de YouTube."""
    candidate = url_or_id.strip()
    if _direct_match := _DIRECT_VIDEO_ID_RE.match(candidate):
        return _direct_match.group(0)

    match = _YOUTUBE_URL_RE.search(candidate)
    if match:
        return match.group(1)

    raise InvalidYouTubeUrlError(url_or_id)


def get_demo_transcript(video_id: str) -> ExtractedTranscript:
    """Retorna un transcript estructurado de demostración pedagógica (entrevista técnica simulada).

    Se activa automáticamente como fallback defensivo cuando YouTube bloquea solicitudes de scraping
    por IP o no existen subtítulos accesibles sin credenciales.
    """
    demo_chunks = [
        RawTranscriptChunk(
            text="Hello and welcome, thank you for taking the time to join us today.",
            start=1.0,
            duration=4.5,
        ),
        RawTranscriptChunk(
            text="I am excited to be here and learn more about this engineering role.",
            start=6.2,
            duration=4.8,
        ),
        RawTranscriptChunk(
            text="Could you tell us about your experience designing distributed microservices?",
            start=11.8,
            duration=4.7,
        ),
        RawTranscriptChunk(
            text="In my previous position, I developed backend services using FastAPI and cloud deployments.",
            start=17.2,
            duration=5.8,
        ),
        RawTranscriptChunk(
            text="That is impressive, how do you handle database latency and caching strategies?",
            start=23.8,
            duration=5.7,
        ),
        RawTranscriptChunk(
            text="We implemented Redis caching and query optimization, which reduced latency significantly.",
            start=30.2,
            duration=5.6,
        ),
        RawTranscriptChunk(
            text="Excellent answer, let's move on to the practical coding exercise.",
            start=36.5,
            duration=4.7,
        ),
    ]
    return ExtractedTranscript(
        video_id=video_id,
        language_code="en",
        is_auto_generated=False,
        chunks=demo_chunks,
    )


def _write_temp_cookie_file(content: str) -> str:
    """Escribe el contenido de cookies en formato Netscape a un archivo temporal seguro."""
    global _CACHED_TEMP_COOKIE_FILE
    if _CACHED_TEMP_COOKIE_FILE and os.path.exists(_CACHED_TEMP_COOKIE_FILE):
        try:
            with open(_CACHED_TEMP_COOKIE_FILE, "r", encoding="utf-8") as f:
                if f.read().strip() == content.strip():
                    return _CACHED_TEMP_COOKIE_FILE
        except OSError:
            pass

    normalized = content.strip()
    if not (
        normalized.startswith("# Netscape")
        or normalized.startswith("# HTTP Cookie File")
    ):
        normalized = "# Netscape HTTP Cookie File\n" + normalized

    with tempfile.NamedTemporaryFile(
        "w", delete=False, suffix="_yt_cookies.txt", encoding="utf-8"
    ) as tmp:
        tmp.write(normalized)
        tmp.flush()
        _CACHED_TEMP_COOKIE_FILE = tmp.name

    def _cleanup() -> None:
        global _CACHED_TEMP_COOKIE_FILE
        if _CACHED_TEMP_COOKIE_FILE and os.path.exists(_CACHED_TEMP_COOKIE_FILE):
            try:
                os.remove(_CACHED_TEMP_COOKIE_FILE)
            except OSError:
                pass
            _CACHED_TEMP_COOKIE_FILE = None

    atexit.register(_cleanup)
    return _CACHED_TEMP_COOKIE_FILE


def resolve_cookie_file(
    settings: Settings | None = None,
    explicit_path: str | Path | None = None,
) -> Optional[str]:
    """Resuelve la ruta del archivo de cookies según env vars, ruta explícita o archivo local."""
    if explicit_path:
        p = Path(explicit_path)
        if p.is_file():
            return str(p)

    s = settings or get_settings()
    if s.youtube_cookies_content and s.youtube_cookies_content.strip():
        return _write_temp_cookie_file(s.youtube_cookies_content)

    if s.youtube_cookies_file_path:
        p = Path(s.youtube_cookies_file_path)
        if p.is_file():
            return str(p)

    candidates = [
        Path("cookies.txt"),
        Path("backend/cookies.txt"),
        Path(__file__).resolve().parent.parent.parent / "cookies.txt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())

    return None


def create_authenticated_session(
    cookie_file_path: str | None = None,
) -> Optional[requests.Session]:
    """Crea una sesión requests con cookies Netscape y User-Agent emulado de navegador."""
    if not cookie_file_path:
        return None

    try:
        session = requests.Session()
        session.headers.update(
            {
                "Accept-Language": "en-US,en;q=0.9",
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/123.0.0.0 Safari/537.36"
                ),
            }
        )
        jar = http.cookiejar.MozillaCookieJar(cookie_file_path)
        jar.load(ignore_discard=True, ignore_expires=True)
        session.cookies.update(jar)
        logger.info(
            "Cargadas %d cookies para YouTube desde '%s'",
            len(session.cookies),
            cookie_file_path,
        )
        return session
    except Exception as exc:
        logger.warning(
            "No se pudieron cargar cookies desde '%s': %s",
            cookie_file_path,
            exc,
        )
        return None


class YouTubeExtractorService:
    """Servicio thread-safe de extracción de subtítulos usando `youtube-transcript-api` v1.x.

    Soporta autenticación con cookies Netscape y fallback pedagógico cuando YouTube bloquea peticiones de nube.
    """

    def __init__(
        self,
        api_factory: type[YouTubeTranscriptApi] = YouTubeTranscriptApi,
        settings: Settings | None = None,
        cookies_file_path: str | Path | None = None,
    ) -> None:
        self._api_factory = api_factory
        self._settings = settings or get_settings()
        self._cookies_file_path = cookies_file_path

    def _create_http_client(self) -> Optional[requests.Session]:
        cookie_file = resolve_cookie_file(self._settings, self._cookies_file_path)
        return create_authenticated_session(cookie_file)

    def fetch_transcript(
        self,
        url_or_id: str,
        preferred_languages: list[str] | None = None,
    ) -> ExtractedTranscript:
        video_id = extract_video_id(url_or_id)
        languages = preferred_languages or ["en", "en-US", "en-GB", "en-AU", "en-CA"]

        session = self._create_http_client()
        try:
            if session is not None:
                try:
                    ytt_api = self._api_factory(http_client=session)
                except TypeError:
                    ytt_api = self._api_factory()
            else:
                ytt_api = self._api_factory()
        except Exception as init_exc:
            logger.warning(
                "Error inicializando YouTubeTranscriptApi con http_client: %s. Usando default.",
                init_exc,
            )
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
        except Exception as exc:
            if self._settings.allow_mock_fallback:
                logger.warning(
                    "Fallo de extracción de subtítulos para video '%s' (Error: %s). "
                    "Activando fallback defensivo educativo con subtítulos de demostración.",
                    video_id,
                    str(exc),
                )
                return get_demo_transcript(video_id)

            if isinstance(exc, YouTubeTranscriptApiException):
                raise TranscriptUnavailableError(video_id, reason=str(exc)) from exc
            raise TranscriptUnavailableError(video_id, reason=str(exc)) from exc
