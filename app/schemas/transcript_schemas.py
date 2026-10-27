from datetime import datetime, timezone
from typing import Any, Optional
from pydantic import BaseModel, Field, model_validator


class RawTranscriptChunk(BaseModel):
    """Fragmento crudo extraído desde YouTube antes de limpiar/agrupar."""

    text: str
    start: float = Field(ge=0.0)
    duration: float = Field(ge=0.0)

    @property
    def end(self) -> float:
        return self.start + self.duration


# Alias para clientes y servicios externos
RawTranscriptItem = RawTranscriptChunk


class CleanedChunk(BaseModel):
    """Fragmento limpio con metadatos de turno de diálogo."""

    text: str
    start: float = Field(ge=0.0)
    end: float = Field(ge=0.0)
    is_speaker_turn_change: bool = False


class ProcessVideoRequest(BaseModel):
    """Payload de entrada para `/process-video`."""

    video_url_or_id: Optional[str] = Field(
        default=None,
        description="URL completa de YouTube (watch, youtu.be, shorts) o ID de 11 caracteres.",
    )
    url: Optional[str] = Field(
        default=None,
        description="Alias opcional de video_url_or_id para compatibilidad con clientes.",
    )
    preferred_languages: list[str] = Field(
        default_factory=lambda: ["en", "en-US", "en-GB", "en-AU", "en-CA"],
        description="Orden de prioridad de códigos de idioma en inglés.",
    )
    use_ai_punctuation_fallback: bool = Field(
        default=False,
        description="Si es True y los auto-captions carecen de puntuación, intenta restaurarla con Gemini Flash.",
    )

    @model_validator(mode="before")
    @classmethod
    def resolve_url_field(cls, data: Any) -> Any:
        if isinstance(data, dict):
            val = data.get("video_url_or_id") or data.get("url")
            if not val or not str(val).strip():
                raise ValueError("Se requiere 'video_url_or_id' o 'url'.")
            data["video_url_or_id"] = str(val).strip()
            data["url"] = str(val).strip()
        return data


class ProcessTranscriptPayload(BaseModel):
    """Payload de entrada para `/process-transcript` extraído directamente en el cliente."""

    video_id: str = Field(
        ...,
        min_length=1,
        description="ID de 11 caracteres del video de YouTube.",
    )
    title: Optional[str] = Field(
        default="YouTube Practice Video",
        description="Título descriptivo del video.",
    )
    raw_transcript: list[RawTranscriptItem] = Field(
        default_factory=list,
        description="Lista de fragmentos crudos {text, start, duration} extraídos por el cliente.",
    )
    language_code: str = Field(
        default="en",
        description="Código ISO del idioma de la transcripción.",
    )
    is_auto_generated: bool = Field(
        default=False,
        description="Indica si la transcripción fue generada automáticamente por ASR.",
    )
    use_ai_punctuation_fallback: bool = Field(
        default=False,
        description="Si es True y carece de puntuación, restaura con Gemini Flash.",
    )


class TranscriptSegmentOut(BaseModel):
    """Oración gramaticalmente coherente lista para sincronizar con el reproductor."""

    index: int = Field(ge=0)
    text: str
    start_time: float = Field(ge=0.0)
    end_time: float = Field(ge=0.0)
    word_count: int = Field(ge=1)
    is_speaker_turn_change: bool = False


class ProcessVideoResponse(BaseModel):
    """Respuesta estructurada de `/process-video` y `/process-transcript`."""

    video_id: str
    title: Optional[str] = Field(
        default="YouTube Practice Video",
        description="Título pedagógico o título del video de YouTube.",
    )
    language_code: str
    is_auto_generated: bool
    segment_count: int
    total_words: int
    segments: list[TranscriptSegmentOut]
    processed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# Alias semántico
VideoDetailResponse = ProcessVideoResponse
