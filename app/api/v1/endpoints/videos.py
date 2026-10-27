import logging
import time
from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.core.exceptions import EmptyTranscriptError, TranscriptUnavailableError
from app.schemas.transcript_schemas import (
    ProcessTranscriptPayload,
    ProcessVideoRequest,
    ProcessVideoResponse,
    RawTranscriptChunk,
)
from app.services.coherent_segmenter import CoherentSegmenter
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


@router.post(
    "/process-video",
    response_model=ProcessVideoResponse,
    summary="Extrae, limpia y segmenta subtítulos de YouTube en oraciones coherentes",
)
def process_video_endpoint(
    payload: ProcessVideoRequest,
    extractor: YouTubeExtractorService = Depends(get_extractor),
    cleaner: TranscriptTextCleaner = Depends(get_cleaner),
    segmenter: CoherentSegmenter = Depends(get_segmenter),
    mentor: GeminiMentorService = Depends(get_mentor),
    settings: Settings = Depends(get_settings),
) -> ProcessVideoResponse:
    start_time = time.perf_counter()
    try:
        extracted = extractor.fetch_transcript(
            url_or_id=payload.video_url_or_id,
            preferred_languages=payload.preferred_languages,
        )
    except TranscriptUnavailableError as exc:
        if settings.is_fallback_enabled:
            video_id = extract_video_id(payload.video_url_or_id)
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

    if payload.use_ai_punctuation_fallback and extracted.is_auto_generated:
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

    return ProcessVideoResponse(
        video_id=extracted.video_id,
        language_code=extracted.language_code,
        is_auto_generated=extracted.is_auto_generated,
        segment_count=len(segments),
        total_words=total_words,
        segments=segments,
    )


@router.post(
    "/process-transcript",
    response_model=ProcessVideoResponse,
    summary="Recibe subtítulos crudos extraídos por el cliente y los segmenta pedagógicamente",
)
def process_transcript_endpoint(
    payload: ProcessTranscriptPayload,
    cleaner: TranscriptTextCleaner = Depends(get_cleaner),
    segmenter: CoherentSegmenter = Depends(get_segmenter),
    mentor: GeminiMentorService = Depends(get_mentor),
    settings: Settings = Depends(get_settings),
) -> ProcessVideoResponse:
    start_time = time.perf_counter()
    raw_items = payload.raw_transcript

    # Si la lista viene vacía y el fallback está activado
    if not raw_items:
        if settings.is_fallback_enabled:
            logger.warning(
                "process-transcript recibió transcripción vacía para '%s'. Activando demo fallback.",
                payload.video_id,
            )
            demo = get_demo_transcript(payload.video_id)
            raw_items = [
                RawTranscriptChunk(
                    text=c.text, start=c.start, duration=c.duration
                )
                for c in demo.chunks
            ]
        else:
            raise EmptyTranscriptError(payload.video_id)

    chunks = [
        RawTranscriptChunk(
            text=item.text, start=item.start, duration=item.duration
        )
        for item in raw_items
    ]

    cleaned_chunks = cleaner.clean_chunks(chunks)
    if not cleaned_chunks:
        raise EmptyTranscriptError(payload.video_id)

    segments = segmenter.segment(cleaned_chunks)
    if not segments:
        raise EmptyTranscriptError(payload.video_id)

    if payload.use_ai_punctuation_fallback and payload.is_auto_generated:
        segments = mentor.restore_punctuation_if_needed(segments)

    total_words = sum(seg.word_count for seg in segments)
    elapsed = time.perf_counter() - start_time
    logger.info(
        "[process-transcript] Video '%s' procesado en %.2fs con %d segmentos (%d palabras).",
        payload.video_id,
        elapsed,
        len(segments),
        total_words,
    )

    return ProcessVideoResponse(
        video_id=payload.video_id,
        language_code=payload.language_code,
        is_auto_generated=payload.is_auto_generated,
        segment_count=len(segments),
        total_words=total_words,
        segments=segments,
    )
