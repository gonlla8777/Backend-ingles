import atexit
import http.cookiejar
import logging
import os
from pathlib import Path
import re
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, Optional

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
from youtube_transcript_api.proxies import GenericProxyConfig

from app.core.config import Settings, get_settings
from app.core.exceptions import InvalidYouTubeUrlError, TranscriptUnavailableError
from app.schemas.transcript_schemas import RawTranscriptChunk
from app.services.curated_transcripts import CURATED_TRANSCRIPTS
from app.services.gemini_transcript_service import transcribe_youtube_with_gemini
from app.services.ytdlp_service import extract_subtitles_with_ytdlp

logger = logging.getLogger(__name__)

_DIRECT_VIDEO_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{11}$")
_YOUTUBE_URL_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?.*?v=|embed/|shorts/|live/)|youtu\.be/)([a-zA-Z0-9_-]{11})"
)

_ACTIVE_TEMP_COOKIE_FILE: Optional[str] = None


@dataclass(frozen=True)
class ExtractedTranscript:
    video_id: str
    language_code: str
    is_auto_generated: bool
    chunks: list[RawTranscriptChunk]


def extract_video_id(url_or_id: str) -> str:
    """Extrae el ID de 11 caracteres o identificador curado desde una URL o ID."""
    candidate = (url_or_id or "").strip()
    if candidate in CURATED_TRANSCRIPTS:
        return candidate
    if "zomboid" in candidate.lower():
        return "project_zomboid"
    for key in CURATED_TRANSCRIPTS:
        if key in candidate:
            return key
    if _direct_match := _DIRECT_VIDEO_ID_RE.match(candidate):
        return _direct_match.group(0)

    match = _YOUTUBE_URL_RE.search(candidate)
    if match:
        return match.group(1)

    raise InvalidYouTubeUrlError(url_or_id)


def get_cookies_path() -> Optional[str]:
    """Obtiene la ruta persistida del archivo de cookies.

    1. Si existe cookies.txt local en la raíz o en backend/.
    2. Si existe la variable de entorno YOUTUBE_COOKIES_CONTENT, escribe su contenido
       en un archivo temporal, hace flush() y close() para garantizar su persistencia.
    """
    global _ACTIVE_TEMP_COOKIE_FILE

    # 1. Si existe cookies.txt local
    for candidate in (
        Path("cookies.txt"),
        Path("backend/cookies.txt"),
        Path(__file__).resolve().parent.parent.parent / "cookies.txt",
    ):
        if candidate.is_file():
            return str(candidate.resolve())

    # 2. Si viene por variable de entorno YOUTUBE_COOKIES_CONTENT
    cookies_content = os.getenv("YOUTUBE_COOKIES_CONTENT")
    if cookies_content and cookies_content.strip():
        if _ACTIVE_TEMP_COOKIE_FILE and os.path.exists(_ACTIVE_TEMP_COOKIE_FILE):
            return _ACTIVE_TEMP_COOKIE_FILE

        normalized = cookies_content.strip()
        if not (
            normalized.startswith("# Netscape")
            or normalized.startswith("# HTTP Cookie File")
        ):
            normalized = "# Netscape HTTP Cookie File\n" + normalized

        temp = tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".txt", encoding="utf-8"
        )
        temp.write(normalized + "\n")
        temp.flush()
        temp.close()
        _ACTIVE_TEMP_COOKIE_FILE = temp.name
        return _ACTIVE_TEMP_COOKIE_FILE

    return None


def _cleanup_temp_cookies() -> None:
    global _ACTIVE_TEMP_COOKIE_FILE
    if _ACTIVE_TEMP_COOKIE_FILE and os.path.exists(_ACTIVE_TEMP_COOKIE_FILE):
        try:
            os.remove(_ACTIVE_TEMP_COOKIE_FILE)
        except OSError:
            pass
        _ACTIVE_TEMP_COOKIE_FILE = None


atexit.register(_cleanup_temp_cookies)


def create_authenticated_session(
    cookie_file_path: Optional[str] = None,
    proxies: Optional[dict[str, str]] = None,
) -> Optional[requests.Session]:
    """Crea una sesión requests con cookies Netscape, proxies configurados y User-Agent legítimo."""
    if not cookie_file_path and not proxies:
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

        if proxies:
            session.proxies.update(proxies)

        if cookie_file_path and os.path.exists(cookie_file_path):
            jar = http.cookiejar.MozillaCookieJar(cookie_file_path)
            jar.load(ignore_discard=True, ignore_expires=True)
            session.cookies.update(jar)

        return session
    except Exception as exc:
        logger.warning(
            "Error configurando sesión requests (cookies=%s, proxies=%s): %s",
            cookie_file_path,
            bool(proxies),
            exc,
        )
        return None


# Retrocompatibilidad para youtube-transcript-api v1.x:
if not hasattr(YouTubeTranscriptApi, "get_transcript"):

    @classmethod
    def _poly_get_transcript(cls, video_id: str, languages=("en",), **kwargs):
        cookies = kwargs.get("cookies")
        proxies_dict = kwargs.get("proxies")
        session = create_authenticated_session(cookies, proxies=proxies_dict)
        proxy_config = (
            GenericProxyConfig(
                http_url=proxies_dict.get("http"),
                https_url=proxies_dict.get("https"),
            )
            if proxies_dict
            else None
        )
        ytt = cls(http_client=session, proxy_config=proxy_config)
        t_list = ytt.list(video_id)
        try:
            selected = t_list.find_manually_created_transcript(languages)
        except Exception:
            selected = t_list.find_generated_transcript(languages)
        return selected.fetch().to_raw_data()

    YouTubeTranscriptApi.get_transcript = _poly_get_transcript

if not hasattr(YouTubeTranscriptApi, "list_transcripts"):

    @classmethod
    def _poly_list_transcripts(cls, video_id: str, **kwargs):
        cookies = kwargs.get("cookies")
        proxies_dict = kwargs.get("proxies")
        session = create_authenticated_session(cookies, proxies=proxies_dict)
        proxy_config = (
            GenericProxyConfig(
                http_url=proxies_dict.get("http"),
                https_url=proxies_dict.get("https"),
            )
            if proxies_dict
            else None
        )
        ytt = cls(http_client=session, proxy_config=proxy_config)
        return ytt.list(video_id)

    YouTubeTranscriptApi.list_transcripts = _poly_list_transcripts


def get_demo_transcript(video_id: str) -> ExtractedTranscript:
    """Retorna una transcripción estructurada de demostración pedagógica (entrevista técnica laboral).

    Simula una conversación en inglés de 6 oraciones con timestamps calculados entre 0s y 35s,
    diseñada para que el reproductor de Flutter active el Auto-Pause, calcule el diff y permita
    practicar la transcripción sin bloqueos por IP de YouTube.
    """
    raw_curated = CURATED_TRANSCRIPTS.get(video_id) or CURATED_TRANSCRIPTS.get("demo")
    if raw_curated:
        demo_chunks = [
            RawTranscriptChunk(
                text=str(c["text"]),
                start=float(c["start"]),
                duration=float(c["duration"]),
            )
            for c in raw_curated
        ]
    else:
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
        "ip blocked",
        "requestblocked",
        "could not retrieve a transcript",
        "too many requests",
        "youtube is blocking",
    ]
    return any(pattern in msg for pattern in blocking_patterns)


class YouTubeExtractorService:
    """Servicio thread-safe de extracción de transcripciones de YouTube.

    Implementa una cascada de resolución de 4 niveles para eludir bloqueos de red:
    1. Catálogo curado local (`CURATED_TRANSCRIPTS`).
    2. Transcripción con Google Gemini API (`Part.from_uri(file_uri=video_url, mime_type='video/*')`).
    3. Extracción tradicional con `youtube-transcript-api` (cookies y proxy Webshare).
    4. Fallback defensivo educativo estructurado si todas las fuentes externas fallan.
    """

    def __init__(
        self,
        api_factory: type[YouTubeTranscriptApi] = YouTubeTranscriptApi,
        settings: Settings | None = None,
        cookies_file_path: str | Path | None = None,
        proxy_url: str | None = None,
        gemini_transcriber: Optional[Callable[..., list[dict]]] = transcribe_youtube_with_gemini,
        ytdlp_extractor: Optional[Callable[..., Any]] = extract_subtitles_with_ytdlp,
    ) -> None:
        self._api_factory = api_factory
        self._settings = settings or get_settings()
        self._cookies_file_path = cookies_file_path
        self._proxy_url = proxy_url
        self._gemini_transcriber = gemini_transcriber
        self._ytdlp_extractor = ytdlp_extractor

    def fetch_transcript(
        self,
        url_or_id: str,
        preferred_languages: list[str] | None = None,
    ) -> ExtractedTranscript:
        video_id = extract_video_id(url_or_id)
        video_url = (
            url_or_id
            if url_or_id.startswith("http://") or url_or_id.startswith("https://")
            else f"https://www.youtube.com/watch?v={video_id}"
        )
        languages = preferred_languages or ["en", "en-US", "en-GB", "en-AU", "en-CA"]

        # -------------------------------------------------------------
        # Paso 1: Catálogo curado local
        # -------------------------------------------------------------
        if video_id in CURATED_TRANSCRIPTS and self._api_factory is YouTubeTranscriptApi:
            logger.info(
                f"[YouTubeExtractor] Video '{video_id}' localizado en CURATED_TRANSCRIPTS local. "
                "Cargando segmentos curados instantáneamente."
            )
            chunks = [
                RawTranscriptChunk(
                    text=str(item.get("text", "")),
                    start=float(item.get("start", 0.0)),
                    duration=max(0.1, float(item.get("duration", 0.0))),
                )
                for item in CURATED_TRANSCRIPTS[video_id]
            ]
            return ExtractedTranscript(
                video_id=video_id,
                language_code="en",
                is_auto_generated=False,
                chunks=chunks,
            )

        # -------------------------------------------------------------
        # Paso 2: Extracción con yt-dlp (subtítulos oficiales sin coste de LLM)
        # -------------------------------------------------------------
        if self._api_factory is YouTubeTranscriptApi and self._ytdlp_extractor:
            try:
                logger.info(
                    f"[YouTubeExtractor] Intentando extracción con yt-dlp para '{video_id}' ({video_url})..."
                )
                ytdlp_result = self._ytdlp_extractor(
                    video_url,
                    preferred_languages=languages,
                )
                raw_chunks_list = (
                    ytdlp_result.segments
                    if hasattr(ytdlp_result, "segments")
                    else ytdlp_result[0]
                )
                if raw_chunks_list:
                    chunks = [
                        RawTranscriptChunk(
                            text=str(item.get("text", "")),
                            start=float(item.get("start", 0.0)),
                            duration=max(0.1, float(item.get("duration", 0.0))),
                        )
                        for item in raw_chunks_list
                    ]
                    logger.info(
                        f"[YouTubeExtractor] Éxito con yt-dlp: {len(chunks)} fragmentos extraídos."
                    )
                    return ExtractedTranscript(
                        video_id=video_id,
                        language_code="en",
                        is_auto_generated=False,
                        chunks=chunks,
                    )
            except Exception as ytdlp_err:
                logger.warning(
                    f"[YouTubeExtractor] Falló yt-dlp para '{video_id}': {ytdlp_err}. "
                    "Continuando con Paso 3 (Gemini)..."
                )

        # -------------------------------------------------------------
        # Paso 3: Transcripción directa con Gemini API (multimodal audio/video)
        # -------------------------------------------------------------
        if (
            self._api_factory is YouTubeTranscriptApi
            and self._settings.has_gemini_key
            and self._gemini_transcriber
        ):
            try:
                logger.info(
                    f"[YouTubeExtractor] Intentando transcripción directa con Gemini para '{video_id}' ({video_url})..."
                )
                gemini_segments = self._gemini_transcriber(
                    video_url,
                    models_list=["gemini-3.5-flash-lite", "gemini-3.6-flash"],
                )
                if gemini_segments:
                    chunks = [
                        RawTranscriptChunk(
                            text=str(item.get("text", "")),
                            start=float(item.get("start", 0.0)),
                            duration=max(0.1, float(item.get("duration", 0.0))),
                        )
                        for item in gemini_segments
                    ]
                    logger.info(
                        f"[YouTubeExtractor] Éxito transcribiendo con Gemini: {len(chunks)} fragmentos generados."
                    )
                    return ExtractedTranscript(
                        video_id=video_id,
                        language_code="en",
                        is_auto_generated=True,
                        chunks=chunks,
                    )
            except Exception as gemini_err:
                logger.warning(
                    f"[YouTubeExtractor] Falló transcripción con Gemini para '{video_id}': {gemini_err}. "
                    "Continuando con scraping tradicional / fallback de contingencia."
                )

        # -------------------------------------------------------------
        # Paso 4: Extracción tradicional vía youtube-transcript-api (o api_factory mockeada)
        # -------------------------------------------------------------
        raw_proxy = (
            self._proxy_url
            or self._settings.youtube_proxy_url
            or os.getenv("YOUTUBE_PROXY_URL")
        )
        proxies = (
            {
                "http": raw_proxy.strip(),
                "https": raw_proxy.strip(),
            }
            if raw_proxy and raw_proxy.strip()
            else None
        )

        cookies_path = (
            str(self._cookies_file_path)
            if self._cookies_file_path
            else get_cookies_path()
        )

        kwargs = {}
        if proxies:
            kwargs["proxies"] = proxies
        if cookies_path:
            kwargs["cookies"] = cookies_path

        logger.info(
            f"Descargando subtítulos para video={video_id} usando proxies={'SÍ' if proxies else 'NO'}"
        )

        try:
            # 1. Si api_factory tiene list_transcripts
            if hasattr(self._api_factory, "list_transcripts"):
                transcript_list = self._api_factory.list_transcripts(
                    video_id, **kwargs
                )
                try:
                    selected_transcript = (
                        transcript_list.find_manually_created_transcript(languages)
                    )
                except Exception:
                    selected_transcript = (
                        transcript_list.find_generated_transcript(languages)
                    )

                fetched = selected_transcript.fetch()
                raw_items = fetched.to_raw_data()
                language_code = getattr(
                    selected_transcript, "language_code", "en"
                )
                is_auto_generated = bool(
                    getattr(selected_transcript, "is_generated", False)
                )

            # 2. Si api_factory tiene get_transcript
            elif hasattr(self._api_factory, "get_transcript"):
                raw_items = self._api_factory.get_transcript(
                    video_id, languages=languages, **kwargs
                )
                language_code = "en"
                is_auto_generated = False

            # 3. Fallback de instanciación directa
            else:
                session = create_authenticated_session(cookies_path, proxies=proxies)
                proxy_config = (
                    GenericProxyConfig(
                        http_url=proxies.get("http"),
                        https_url=proxies.get("https"),
                    )
                    if proxies
                    else None
                )
                try:
                    ytt_api = self._api_factory(
                        http_client=session, proxy_config=proxy_config
                    )
                except TypeError:
                    try:
                        ytt_api = self._api_factory(http_client=session)
                    except TypeError:
                        ytt_api = self._api_factory()
                transcript_list = ytt_api.list(video_id)
                try:
                    selected_transcript = (
                        transcript_list.find_manually_created_transcript(languages)
                    )
                except Exception:
                    selected_transcript = (
                        transcript_list.find_generated_transcript(languages)
                    )
                fetched = selected_transcript.fetch()
                raw_items = fetched.to_raw_data()
                language_code = getattr(
                    selected_transcript, "language_code", "en"
                )
                is_auto_generated = bool(
                    getattr(selected_transcript, "is_generated", False)
                )

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
                language_code=language_code,
                is_auto_generated=is_auto_generated,
                chunks=chunks,
            )
        except Exception as exc:
            if cookies_path:
                logger.warning(
                    f"Extracción falló aún con cookies_path={cookies_path}: {exc}"
                )

            # -------------------------------------------------------------
            # Paso 4: Fallback de contingencia ante fallas de YouTube
            # -------------------------------------------------------------
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
