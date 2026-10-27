"""Servicio de transcripción directa de videos de YouTube usando Google GenAI SDK (Gemini)."""

import json
import logging
import os
import re
from typing import Any, Optional

from google import genai
from google.genai import types

from app.core.config import get_settings

logger = logging.getLogger(__name__)

PROMPT_YOUTUBE_TRANSCRIPTION = """
Listen carefully to the audio of this YouTube video and generate a precise, chronological transcript in English.
Divide the speech into natural, coherent sentences or pedagogical learning chunks suitable for a dictation and listening practice app.
For each segment, provide:
- "text": The exact English transcript of that spoken segment.
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
) -> list[dict[str, Any]]:
    """Transcribe el audio de un video de YouTube llamando a Gemini con `Part.from_uri`.

    Gemini procesa el video directamente a través de la infraestructura interna de Google,
    evitando bloqueos de red de YouTube que sufren los datacenters en la nube.
    """
    settings = get_settings()
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key is None:
        api_key = settings.gemini_api_key
    if not api_key or api_key == "your_gemini_api_key_here":
        raise ValueError("GEMINI_API_KEY no configurada")

    gemini_client = client or genai.Client(api_key=api_key)
    normalized_url = _normalize_youtube_url(video_url)

    # Cascada de modelos candidatos (el preferido primero, con degradación elegante si está retirado u ocupado)
    preferred_model = (
        model or os.getenv("GEMINI_MODEL") or settings.gemini_model or "gemini-3.8-flash"
    )
    candidate_models = [
        preferred_model,
        "gemini-3.8-flash",
        "gemini-3.5-flash",
        "gemini-2.5-flash",
        "gemini-flash-latest",
    ]
    models_to_try: list[str] = []
    for cand in candidate_models:
        if cand and cand not in models_to_try:
            models_to_try.append(cand)

    last_error: Optional[Exception] = None

    for candidate_model in models_to_try:
        try:
            logger.info(
                f"[GeminiTranscriptService] Transcribiendo video '{normalized_url}' usando modelo '{candidate_model}'..."
            )
            response = gemini_client.models.generate_content(
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

            raw_text = response.text or ""
            cleaned_text = _clean_json_text(raw_text)
            if not cleaned_text:
                raise ValueError("Gemini devolvió una respuesta de texto vacía")

            parsed = json.loads(cleaned_text)

            # Normalizar si viene envuelto en un objeto {'segments': [...]} o {'transcript': [...]}
            if isinstance(parsed, dict):
                for key in ("segments", "transcript", "items", "data"):
                    if key in parsed and isinstance(parsed[key], list):
                        parsed = parsed[key]
                        break

            if not isinstance(parsed, list):
                raise ValueError(f"Formato inesperado devuelto por Gemini: {type(parsed).__name__}")

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
                    f"[GeminiTranscriptService] Transcripción completada con éxito: "
                    f"{len(sanitized_segments)} segmentos generados con '{candidate_model}'."
                )
                return sanitized_segments

        except Exception as exc:
            last_error = exc
            logger.warning(
                f"[GeminiTranscriptService] Falló transcripción con modelo '{candidate_model}': {exc}"
            )
            continue

    if last_error:
        raise last_error

    raise RuntimeError(f"No fue posible transcribir el video '{normalized_url}' con Gemini.")
