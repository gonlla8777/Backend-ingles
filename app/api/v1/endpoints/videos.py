import logging
import time
from typing import Any, Optional
from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.core.exceptions import EmptyTranscriptError, TranscriptUnavailableError
from app.schemas.transcript_schemas import (
    ProcessTranscriptPayload,
    ProcessVideoRequest,
    ProcessVideoResponse,
    RawTranscriptChunk,
    VideoDetailResponse,
)
from app.services.coherent_segmenter import CoherentSegmenter
from app.services.curated_transcripts import CURATED_TRANSCRIPTS
from app.services.gemini_mentor_service import GeminiMentorService
from app.services.text_cleaner import TranscriptTextCleaner
from app.services.youtube_extractor import (
    YouTubeExtractorService,
    extract_video_id,
    get_demo_transcript,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def get_extractor(
    settings: Settings = Depends(get_settings),
) -> YouTubeExtractorService:
    return YouTubeExtractorService(settings=settings)


def get_cleaner() -> TranscriptTextCleaner:
    return TranscriptTextCleaner()


def get_segmenter(settings: Settings = Depends(get_settings)) -> CoherentSegmenter:
    return CoherentSegmenter(settings=settings)


def get_mentor(settings: Settings = Depends(get_settings)) -> GeminiMentorService:
    return GeminiMentorService(settings=settings)


def build_video_response(
    video_id: str,
    title: str,
    raw_transcript: list[dict[str, Any]],
    cleaner: TranscriptTextCleaner,
    segmenter: CoherentSegmenter,
    is_auto_generated: bool = False,
) -> VideoDetailResponse:
    """Construye un VideoDetailResponse / ProcessVideoResponse a partir de fragmentos curados."""
    chunks = [
        RawTranscriptChunk(
            text=str(item.get("text", "")).strip(),
            start=float(item.get("start", 0.0)),
            duration=max(0.1, float(item.get("duration", 0.0))),
        )
        for item in raw_transcript
        if str(item.get("text", "")).strip()
    ]
    cleaned = cleaner.clean_chunks(chunks)
    if not cleaned:
        raise EmptyTranscriptError(video_id)

    segments = segmenter.segment(cleaned)
    if not segments:
        raise EmptyTranscriptError(video_id)

    total_words = sum(seg.word_count for seg in segments)
    return VideoDetailResponse(
        video_id=video_id,
        title=title,
        language_code="en",
        is_auto_generated=is_auto_generated,
        segment_count=len(segments),
        total_words=total_words,
        segments=segments,
    )


@router.post(
    "/process-video",
    response_model=VideoDetailResponse,
    summary="Extrae, limpia y segmenta subtítulos de YouTube en oraciones coherentes",
)
@router.post(
    "/process-video/",
    response_model=VideoDetailResponse,
    include_in_schema=False,
)
@router.get(
    "/process-video",
    response_model=VideoDetailResponse,
    include_in_schema=False,
)
@router.get(
    "/process-video/",
    response_model=VideoDetailResponse,
    include_in_schema=False,
)
def process_video_endpoint(
    payload: Optional[ProcessVideoRequest] = None,
    url: Optional[str] = None,
    video_url_or_id: Optional[str] = None,
    extractor: YouTubeExtractorService = Depends(get_extractor),
    cleaner: TranscriptTextCleaner = Depends(get_cleaner),
    segmenter: CoherentSegmenter = Depends(get_segmenter),
    mentor: GeminiMentorService = Depends(get_mentor),
    settings: Settings = Depends(get_settings),
) -> VideoDetailResponse:
    start_time = time.perf_counter()

    video_url = ""
    if payload:
        video_url = payload.url or payload.video_url_or_id or ""
    if not video_url:
        video_url = url or video_url_or_id or "jNQXAC9IVRw"

    video_id = extract_video_id(video_url)

    # 1. Comprobación prioritaria de CURATED_TRANSCRIPTS
    if video_id in CURATED_TRANSCRIPTS:
        logger.info(f"[Curated] Video {video_id} resuelto desde catálogo curado.")
        curated = CURATED_TRANSCRIPTS[video_id]
        return build_video_response(
            video_id=video_id,
            title=curated["title"],
            raw_transcript=curated["raw_transcript"],
            cleaner=cleaner,
            segmenter=segmenter,
            is_auto_generated=False,
        )

    # 2. Si no está en catálogo curado, procesar con pipeline completo (Gemini / scraping / fallback)
    pref_langs = (
        payload.preferred_languages
        if payload and payload.preferred_languages
        else ["en", "en-US", "en-GB"]
    )
    use_ai = (
        payload.use_ai_punctuation_fallback
        if payload
        else False
    )

    try:
        extracted = extractor.fetch_transcript(
            url_or_id=video_url,
            preferred_languages=pref_langs,
        )
    except TranscriptUnavailableError as exc:
        if settings.is_fallback_enabled:
            logger.warning(
                "Fallback defensivo en endpoint /process-video para video '%s': %s",
                video_id,
                exc,
            )
            extracted = get_demo_transcript(video_id)
        else:
            raise

    cleaned_chunks = cleaner.clean_chunks(extracted.chunks)
    if not cleaned_chunks:
        raise EmptyTranscriptError(extracted.video_id)

    segments = segmenter.segment(cleaned_chunks)
    if not segments:
        raise EmptyTranscriptError(extracted.video_id)

    if use_ai and extracted.is_auto_generated:
        segments = mentor.restore_punctuation_if_needed(segments)

    total_words = sum(seg.word_count for seg in segments)
    elapsed = time.perf_counter() - start_time
    logger.info(
        "[process-video] Video '%s' procesado en %.2fs con %d segmentos (%d palabras).",
        extracted.video_id,
        elapsed,
        len(segments),
        total_words,
    )

    return VideoDetailResponse(
        video_id=extracted.video_id,
        title=f"YouTube Practice ({extracted.video_id})",
        language_code=extracted.language_code,
        is_auto_generated=extracted.is_auto_generated,
        segment_count=len(segments),
        total_words=total_words,
        segments=segments,
    )


@router.post(
    "/process-transcript",
    response_model=VideoDetailResponse,
    summary="Recibe subtítulos crudos extraídos por el cliente y los segmenta pedagógicamente",
)
@router.post(
    "/process-transcript/",
    response_model=VideoDetailResponse,
    include_in_schema=False,
)
@router.get(
    "/process-transcript",
    response_model=VideoDetailResponse,
    include_in_schema=False,
)
@router.get(
    "/process-transcript/",
    response_model=VideoDetailResponse,
    include_in_schema=False,
)
def process_transcript_endpoint(
    payload: Optional[ProcessTranscriptPayload] = None,
    cleaner: TranscriptTextCleaner = Depends(get_cleaner),
    segmenter: CoherentSegmenter = Depends(get_segmenter),
    mentor: GeminiMentorService = Depends(get_mentor),
    settings: Settings = Depends(get_settings),
) -> VideoDetailResponse:
    start_time = time.perf_counter()
    raw_items = payload.raw_transcript if payload else []
    video_id = payload.video_id if payload else "demo"
    title = payload.title if (payload and payload.title) else "YouTube Practice Video"

    # Si la lista viene vacía y el fallback está activado
    if not raw_items:
        if settings.is_fallback_enabled:
            logger.warning(
                "process-transcript recibió transcripción vacía para '%s'. Activando demo fallback.",
                video_id,
            )
            demo = get_demo_transcript(video_id)
            raw_items = [
                RawTranscriptChunk(
                    text=c.text, start=c.start, duration=c.duration
                )
                for c in demo.chunks
            ]
        else:
            raise EmptyTranscriptError(video_id)

    chunks = [
        RawTranscriptChunk(
            text=item.text, start=item.start, duration=item.duration
        )
        for item in raw_items
    ]

    cleaned_chunks = cleaner.clean_chunks(chunks)
    if not cleaned_chunks:
        raise EmptyTranscriptError(video_id)

    segments = segmenter.segment(cleaned_chunks)
    if not segments:
        raise EmptyTranscriptError(video_id)

    if payload and payload.use_ai_punctuation_fallback and payload.is_auto_generated:
        segments = mentor.restore_punctuation_if_needed(segments)

    total_words = sum(seg.word_count for seg in segments)
    elapsed = time.perf_counter() - start_time
    logger.info(
        "[process-transcript] Video '%s' procesado en %.2fs con %d segmentos (%d palabras).",
        video_id,
        elapsed,
        len(segments),
        total_words,
    )

    return VideoDetailResponse(
        video_id=video_id,
        title=title,
        language_code=payload.language_code if payload else "en",
        is_auto_generated=payload.is_auto_generated if payload else False,
        segment_count=len(segments),
        total_words=total_words,
        segments=segments,
    )
