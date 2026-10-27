"""Servicio de extracción de subtítulos oficiales de YouTube usando yt-dlp.

Proporciona extracción directa sin consumo de cuota de LLM, con soporte de
formatos json3 y WebVTT, y fallback integrado para cookies y proxies residenciales.
"""

import json
import logging
import os
import re
from typing import Any, Optional

import requests
import yt_dlp

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_VTT_TIMESTAMP_RE = re.compile(
    r"(\d+:\d+(?::\d+)?[\.,]\d+)\s*-->\s*(\d+:\d+(?::\d+)?[\.,]\d+)"
)


class YtDlpSubtitleResult(tuple):
    """Tupla especializada (segments, title) que permite acceso por índice o propiedades."""

    def __new__(cls, segments: list[dict[str, Any]], title: str = "YouTube Video"):
        return super().__new__(cls, (segments, title))

    @property
    def segments(self) -> list[dict[str, Any]]:
        return self[0]

    @property
    def title(self) -> str:
        return self[1]


def _vtt_time_to_seconds(ts: str) -> float:
    """Convierte marcas de tiempo VTT/SRT (HH:MM:SS.mmm o MM:SS.mmm) a segundos flotantes."""
    cleaned = ts.replace(",", ".").strip()
    parts = cleaned.split(":")
    if len(parts) == 3:
        return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
    if len(parts) == 2:
        return float(parts[0]) * 60.0 + float(parts[1])
    return 0.0


def parse_vtt_subtitles(content: str) -> list[dict[str, Any]]:
    """Parsea contenido en formato WebVTT a lista de diccionarios {text, start, duration}."""
    lines = content.splitlines()
    segments: list[dict[str, Any]] = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        match = _VTT_TIMESTAMP_RE.search(line)
        if match:
            start = _vtt_time_to_seconds(match.group(1))
            end = _vtt_time_to_seconds(match.group(2))
            duration = max(0.1, end - start)
            i += 1

            text_lines: list[str] = []
            while i < len(lines):
                cur = lines[i].strip()
                if not cur or _VTT_TIMESTAMP_RE.search(cur):
                    break
                text_lines.append(cur)
                i += 1

            raw_text = " ".join(text_lines)
            cleaned = re.sub(r"<[^>]+>", "", raw_text).strip()
            cleaned = re.sub(r"\s+", " ", cleaned).strip()

            if cleaned and cleaned not in (
                "[Music]",
                "[Applause]",
                "[Laughter]",
                "[music]",
            ):
                segments.append(
                    {
                        "text": cleaned,
                        "start": round(start, 3),
                        "duration": round(duration, 3),
                    }
                )
            continue
        i += 1

    return segments


def parse_json3_subtitles(raw_data: str | dict) -> list[dict[str, Any]]:
    """Parsea el formato interno json3 de YouTube a lista de {text, start, duration}."""
    data = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
    segments: list[dict[str, Any]] = []

    for event in data.get("events", []):
        segs = event.get("segs", [])
        text = "".join(s.get("utf8", "") for s in segs)
        text = re.sub(r"\s+", " ", text).strip()

        if not text or text in ("[Music]", "[Applause]", "[Laughter]", "[music]"):
            continue

        start_ms = event.get("tStartMs", 0)
        dur_ms = event.get("dDurationMs", 0)

        start = max(0.0, float(start_ms) / 1000.0)
        duration = max(0.1, float(dur_ms) / 1000.0)

        segments.append(
            {
                "text": text,
                "start": round(start, 3),
                "duration": round(duration, 3),
            }
        )

    return segments


def fetch_subtitle_content(
    subtitle_url: str,
    proxy_url: Optional[str] = None,
    timeout: float = 10.0,
) -> str:
    """Descarga el contenido de la URL de subtítulos con User-Agent de navegador y proxies opcionales."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    proxies = (
        {"http": proxy_url, "https": proxy_url}
        if proxy_url and proxy_url.strip()
        else None
    )

    response = requests.get(
        subtitle_url,
        headers=headers,
        proxies=proxies,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.text


def extract_subtitles_with_ytdlp(
    video_url: str,
    cookies_path: Optional[str] = None,
    proxy_url: Optional[str] = None,
    preferred_languages: Optional[list[str]] = None,
) -> YtDlpSubtitleResult:
    """Extrae y parsea los subtítulos en inglés de YouTube usando yt-dlp.

    Retorna un YtDlpSubtitleResult (tupla desempaquetable de `(segments, title)`).
    Lanza ValueError si no se encuentran subtítulos válidos.
    """
    settings = get_settings()
    if cookies_path:
        active_cookies = cookies_path
    else:
        try:
            from app.services.youtube_extractor import get_cookies_path

            active_cookies = get_cookies_path()
        except ImportError:
            active_cookies = None
    active_proxy = (
        proxy_url
        or os.getenv("YOUTUBE_PROXY_URL")
        or settings.youtube_proxy_url
    )
    langs = preferred_languages or ["en", "en-US", "en-orig", "en-GB", "en-CA"]

    ydl_opts: dict[str, Any] = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": langs,
        "quiet": True,
        "no_warnings": True,
    }

    if active_cookies and os.path.exists(active_cookies):
        ydl_opts["cookiefile"] = active_cookies
    if active_proxy:
        ydl_opts["proxy"] = active_proxy

    logger.info(
        f"[yt-dlp] Extrayendo metadatos de subtítulos para '{video_url}' "
        f"(proxy={bool(active_proxy)}, cookies={bool(active_cookies)})..."
    )

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
    except Exception as exc:
        logger.warning(f"[yt-dlp] Falló extract_info para '{video_url}': {exc}")
        raise ValueError(f"yt-dlp no pudo inspeccionar el video '{video_url}': {exc}") from exc

    video_title = info.get("title") or "YouTube Video"
    subtitles = info.get("subtitles") or info.get("automatic_captions") or {}

    if not subtitles:
        raise ValueError(f"No se encontraron subtítulos disponibles para '{video_url}'.")

    # Localizar pista en idioma inglés
    matched_formats: Optional[list[dict[str, Any]]] = None
    for lang in langs:
        if lang in subtitles and subtitles[lang]:
            matched_formats = subtitles[lang]
            break

    if not matched_formats:
        # Fallback a cualquier código que comience con 'en'
        for k, v in subtitles.items():
            if str(k).startswith("en") and v:
                matched_formats = v
                break

    if not matched_formats:
        raise ValueError(
            f"No se encontró pista en inglés en los subtítulos de '{video_url}'."
        )

    # Priorizar formato json3 o el primero disponible (vtt/srv3/ttml)
    chosen_fmt = next(
        (f for f in matched_formats if f.get("ext") == "json3"),
        matched_formats[0],
    )
    fmt_url = chosen_fmt.get("url")
    fmt_ext = chosen_fmt.get("ext", "json3")

    if not fmt_url:
        raise ValueError(f"La pista seleccionada de '{video_url}' no contiene URL de descarga.")

    logger.info(f"[yt-dlp] Descargando contenido de subtítulos ({fmt_ext}) para '{video_url}'...")
    raw_content = fetch_subtitle_content(
        fmt_url,
        proxy_url=active_proxy,
        timeout=12.0,
    )

    # Parsear según el formato obtenido
    segments: list[dict[str, Any]] = []
    if fmt_ext == "json3" or raw_content.strip().startswith("{"):
        try:
            segments = parse_json3_subtitles(raw_content)
        except Exception as json_err:
            logger.warning(f"[yt-dlp] Error parseando json3: {json_err}, intentando VTT")
            segments = parse_vtt_subtitles(raw_content)
    else:
        segments = parse_vtt_subtitles(raw_content)

    if not segments:
        raise ValueError(
            f"Los subtítulos descargados de '{video_url}' no contienen segmentos válidos."
        )

    logger.info(
        f"[yt-dlp] Éxito: {len(segments)} segmentos extraídos de '{video_url}'."
    )
    return YtDlpSubtitleResult(segments, video_title)
