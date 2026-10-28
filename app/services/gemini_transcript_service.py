"""Servicio de transcripción directa de videos de YouTube usando Google GenAI SDK (Gemini)."""

import concurrent.futures
import json
import logging
import os
import re
from typing import Any, Optional

from google import genai
from google.genai import types

from app.core.config import get_settings

logger = logging.getLogger(__name__)

FALLBACK_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.8-flash",
]

PROMPT_YOUTUBE_TRANSCRIPTION = """
Transcribe EVERY SINGLE spoken word verbatim. Do NOT summarize. Do NOT rephrase or replace words with synonyms. Capture the exact words heard in the audio from start to finish.
Divide the speech into chronological, complete sentences suitable for an English dictation and listening practice app.
For each segment, provide:
- "text": The exact, verbatim English transcript of that spoken segment without omitting or altering any words.
- "start": Timestamp in seconds (float) when the segment begins.
- "duration": Duration in seconds (float) of that spoken segment.

Return ONLY a valid JSON array of objects with the keys 'text', 'start', and 'duration'.
"""


def _normalize_youtube_url(url_or_id: str) -> str:
    candidate = url_or_id.strip()
    if candidate.startswith("http://") or candidate.startswith("https://"):
        return candidate
    return f"https://www.youtube.com/watch?v={candidate}"


def _clean_json_text(raw_text: str) -> str:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)
    return cleaned.strip()


def transcribe_youtube_with_gemini(
    video_url: str,
    client: Optional[genai.Client] = None,
    model: Optional[str] = None,
    models_list: Optional[list[str]] = None,
    timeout_seconds: float = 80.0,
) -> list[dict[str, Any]]:
    """Transcribe el audio de un video de YouTube llamando a Gemini con `Part.from_uri`.

    Implementa una cascada multimodelo ordenada con tolerancia a 404/503 y límite de timeout
    realista (80 segundos por intento) para permitir inferencia multimodal y generación estructurada.
    """
    settings = get_settings()
    if client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key is None:
            api_key = settings.gemini_api_key
        if not api_key or api_key == "your_gemini_api_key_here":
            raise ValueError("GEMINI_API_KEY no configurada")
        gemini_client = genai.Client(api_key=api_key)
    else:
        gemini_client = client

    normalized_url = _normalize_youtube_url(video_url)

    # Configuración de lista ordenada de modelos candidatos
    models_to_try: list[str] = []
    if models_list:
        models_to_try.extend(models_list)
    preferred_model = (
        model or os.getenv("GEMINI_MODEL") or settings.gemini_model or FALLBACK_MODELS[0]
    )
    if preferred_model and preferred_model not in models_to_try:
        models_to_try.append(preferred_model)
    for fb in FALLBACK_MODELS:
        if fb not in models_to_try:
            models_to_try.append(fb)

    last_error: Optional[Exception] = None

    for candidate_model in models_to_try:
        try:
            logger.info(
                f"[GeminiTranscriptService] Intentando transcripción de '{normalized_url}' con modelo '{candidate_model}' (timeout={timeout_seconds}s)..."
            )

            # Ejecución con timeout acotado para evitar cuelgues HTTP
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    gemini_client.models.generate_content,
                    model=candidate_model,
                    contents=[
                        types.Part.from_uri(
                            file_uri=normalized_url,
                            mime_type="video/*",
                        ),
                        PROMPT_YOUTUBE_TRANSCRIPTION,
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                    ),
                )
                response = future.result(timeout=timeout_seconds)

            raw_text = getattr(response, "text", "") or ""
            cleaned_text = _clean_json_text(raw_text)
            if not cleaned_text:
                raise ValueError(f"Modelo '{candidate_model}' devolvió texto vacío")

            parsed = json.loads(cleaned_text)

            # Normalizar si viene envuelto en un objeto {'segments': [...]} o {'transcript': [...]}
            if isinstance(parsed, dict):
                for key in ("segments", "transcript", "items", "data"):
                    if key in parsed and isinstance(parsed[key], list):
                        parsed = parsed[key]
                        break

            if not isinstance(parsed, list):
                raise ValueError(f"Estructura inesperada devuelta por {candidate_model}: {type(parsed).__name__}")

            sanitized_segments: list[dict[str, Any]] = []
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text", "")).strip()
                if not text:
                    continue
                start = max(0.0, float(item.get("start", 0.0)))
                duration = max(0.1, float(item.get("duration", 1.0)))
                sanitized_segments.append(
                    {
                        "text": text,
                        "start": start,
                        "duration": duration,
                    }
                )

            if sanitized_segments:
                logger.info(
                    f"[GeminiTranscriptService] Transcripción completada exitosamente: "
                    f"{len(sanitized_segments)} segmentos generados con '{candidate_model}'."
                )
                return sanitized_segments

        except concurrent.futures.TimeoutError as timeout_err:
            last_error = timeout_err
            logger.warning(
                f"[GeminiTranscriptService] Timeout de {timeout_seconds}s excedido para modelo '{candidate_model}'. "
                "Intentando siguiente modelo de la cascada..."
            )
            continue
        except Exception as exc:
            last_error = exc
            logger.warning(
                f"[GeminiTranscriptService] Error en modelo '{candidate_model}': {exc}. "
                "Intentando siguiente modelo de la cascada..."
            )
            continue

    if last_error:
        raise last_error

    return []
