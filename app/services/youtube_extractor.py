import http.cookiejar
import inspect
import logging
import os
from pathlib import Path
import re
import tempfile
from dataclasses import dataclass
from typing import Optional, Tuple

import requests
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    CouldNotRetrieveTranscript,
    IpBlocked,
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    YouTubeTranscriptApiException,
)

from app.core.config import Settings, get_settings
from app.core.exceptions import InvalidYouTubeUrlError, TranscriptUnavailableError
from app.schemas.transcript_schemas import RawTranscriptChunk

logger = logging.getLogger(__name__)

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
    if _direct_match := _DIRECT_VIDEO_ID_RE.match(candidate):
        return _direct_match.group(0)

    match = _YOUTUBE_URL_RE.search(candidate)
    if match:
        return match.group(1)

    raise InvalidYouTubeUrlError(url_or_id)


def get_demo_transcript(video_id: str) -> ExtractedTranscript:
    """Retorna una transcripción estructurada de demostración pedagógica (entrevista técnica laboral).

    Simula una conversación en inglés de 6 oraciones con timestamps calculados entre 0s y 35s,
    diseñada para que el reproductor de Flutter active el Auto-Pause, calcule el diff y permita
    practicar la transcripción sin bloqueos por IP de YouTube.
    """
    demo_chunks = [
        RawTranscriptChunk(
            text="Hello and welcome, thank you for joining us today.",
            start=0.5,
            duration=4.5,
        ),
        RawTranscriptChunk(
            text="I am excited to be here and discuss this engineering role.",
            start=6.0,
            duration=4.8,
        ),
        RawTranscriptChunk(
            text="Could you tell us about your experience designing backend systems?",
            start=11.5,
            duration=5.0,
        ),
        RawTranscriptChunk(
            text="In my previous position, I developed microservices using FastAPI.",
            start=17.5,
            duration=5.0,
        ),
        RawTranscriptChunk(
            text="How do you handle database latency and caching strategies?",
            start=23.5,
            duration=4.8,
        ),
        RawTranscriptChunk(
            text="We implemented Redis caching, which reduced response times significantly.",
            start=29.5,
            duration=5.2,
        ),
    ]
    return ExtractedTranscript(
        video_id=video_id,
        language_code="en",
        is_auto_generated=False,
        chunks=demo_chunks,
    )


def create_temp_cookie_file(content: str) -> str:
    """Escribe el contenido de cookies Netscape en un archivo temporal con limpieza segura."""
    normalized = content.strip()
    if not (
        normalized.startswith("# Netscape")
        or normalized.startswith("# HTTP Cookie File")
    ):
        normalized = "# Netscape HTTP Cookie File\n" + normalized

    temp_file = tempfile.NamedTemporaryFile(
        mode="w+", delete=False, suffix=".txt", encoding="utf-8"
    )
    try:
        temp_file.write(normalized)
        temp_file.flush()
        return temp_file.name
    finally:
        temp_file.close()


def resolve_cookie_source(
    settings: Settings | None = None,
    explicit_path: str | Path | None = None,
) -> Tuple[Optional[str], bool]:
    """Resuelve la ruta del archivo de cookies.

    Retorna una tupla: (ruta_del_archivo, es_archivo_temporal).
    """
    if explicit_path:
        p = Path(explicit_path)
        if p.is_file():
            return str(p), False

    s = settings or get_settings()

    # 1. Variable de entorno YOUTUBE_COOKIES_CONTENT
    content = s.youtube_cookies_content or os.getenv("YOUTUBE_COOKIES_CONTENT")
    if content and content.strip():
        temp_path = create_temp_cookie_file(content)
        return temp_path, True

    # 2. Variable de entorno YOUTUBE_COOKIES_FILE_PATH
    if s.youtube_cookies_file_path:
        p = Path(s.youtube_cookies_file_path)
        if p.is_file():
            return str(p), False

    # 3. Archivo físico cookies.txt en backend/ o raíz del proyecto
    candidates = [
        Path("cookies.txt"),
        Path("backend/cookies.txt"),
        Path(__file__).resolve().parent.parent.parent / "cookies.txt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve()), False

    return None, False


def create_authenticated_session(
    cookie_file_path: str | None = None,
) -> Optional[requests.Session]:
    """Crea una sesión requests con cookies Netscape y User-Agent de navegador legítimo."""
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


def instantiate_transcript_api(
    api_factory: type[YouTubeTranscriptApi],
    session: Optional[requests.Session],
    cookie_file_path: Optional[str],
) -> YouTubeTranscriptApi:
    """Instancia la API pasando cookiefile y/o http_client según la firma admitida."""
    try:
        sig = inspect.signature(api_factory)
        kwargs = {}

        if "cookiefile" in sig.parameters and cookie_file_path:
            kwargs["cookiefile"] = cookie_file_path
        if "cookies" in sig.parameters and cookie_file_path:
            kwargs["cookies"] = cookie_file_path
        if "http_client" in sig.parameters and session:
            kwargs["http_client"] = session

        has_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if has_var_kw:
            if cookie_file_path:
                kwargs["cookiefile"] = cookie_file_path
            if session:
                kwargs["http_client"] = session

        try:
            return api_factory(**kwargs)
        except TypeError:
            # Si api_factory no acepta cookiefile, intentar solo con http_client
            kwargs.pop("cookiefile", None)
            kwargs.pop("cookies", None)
            return api_factory(**kwargs)
    except Exception:
        # Fallback de instanciación sin argumentos
        try:
            return api_factory()
        except Exception:
            return YouTubeTranscriptApi()


def is_ip_blocked_or_retrieval_error(exc: Exception) -> bool:
    """Determina si la excepción se debe a bloqueo de IP o indisponibilidad en YouTube."""
    if isinstance(
        exc,
        (
            IpBlocked,
            RequestBlocked,
            CouldNotRetrieveTranscript,
            TranscriptsDisabled,
            NoTranscriptFound,
            YouTubeTranscriptApiException,
        ),
    ):
        return True

    msg = str(exc).lower()
    blocking_patterns = [
        "blocking requests from your ip",
        "cloud provider",
        "ipblocked",
        "requestblocked",
        "could not retrieve a transcript",
        "too many requests",
        "youtube is blocking",
    ]
    return any(pattern in msg for pattern in blocking_patterns)


class YouTubeExtractorService:
    """Servicio thread-safe de extracción de subtítulos usando `youtube-transcript-api` v1.x.

    Soporta autenticación con cookies Netscape y fallback pedagógico cuando YouTube bloquea IPs de datacenters.
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

    def fetch_transcript(
        self,
        url_or_id: str,
        preferred_languages: list[str] | None = None,
    ) -> ExtractedTranscript:
        video_id = extract_video_id(url_or_id)
        languages = preferred_languages or ["en", "en-US", "en-GB", "en-AU", "en-CA"]

        cookie_path, is_temp = resolve_cookie_source(
            self._settings, self._cookies_file_path
        )
        session = create_authenticated_session(cookie_path)

        try:
            ytt_api = instantiate_transcript_api(
                self._api_factory, session, cookie_path
            )

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
            if self._settings.is_fallback_enabled and is_ip_blocked_or_retrieval_error(
                exc
            ):
                logger.warning(
                    "YouTube IP block detectado para el video '%s' (Error: %s). "
                    "Activando fallback defensivo educativo con transcripción de demostración.",
                    video_id,
                    str(exc),
                )
                return get_demo_transcript(video_id)

            if isinstance(exc, YouTubeTranscriptApiException):
                raise TranscriptUnavailableError(video_id, reason=str(exc)) from exc
            raise TranscriptUnavailableError(video_id, reason=str(exc)) from exc
        finally:
            # Limpieza y cierre adecuado del archivo temporal tras la ejecución
            if is_temp and cookie_path and os.path.exists(cookie_path):
                try:
                    os.remove(cookie_path)
                except OSError:
                    pass
