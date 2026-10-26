from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.core.exceptions import EmptyTranscriptError
from app.schemas.transcript_schemas import ProcessVideoRequest, ProcessVideoResponse
from app.services.coherent_segmenter import CoherentSegmenter
from app.services.gemini_mentor_service import GeminiMentorService
from app.services.text_cleaner import TranscriptTextCleaner
from app.services.youtube_extractor import YouTubeExtractorService

router = APIRouter()


def get_extractor() -> YouTubeExtractorService:
    return YouTubeExtractorService()


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
) -> ProcessVideoResponse:
    extracted = extractor.fetch_transcript(
        url_or_id=payload.video_url_or_id,
        preferred_languages=payload.preferred_languages,
    )

    cleaned_chunks = cleaner.clean_chunks(extracted.chunks)
    if not cleaned_chunks:
        raise EmptyTranscriptError(extracted.video_id)

    segments = segmenter.segment(cleaned_chunks)
    if not segments:
        raise EmptyTranscriptError(extracted.video_id)

    if payload.use_ai_punctuation_fallback and extracted.is_auto_generated:
        segments = mentor.restore_punctuation_if_needed(segments)

    total_words = sum(seg.word_count for seg in segments)

    return ProcessVideoResponse(
        video_id=extracted.video_id,
        language_code=extracted.language_code,
        is_auto_generated=extracted.is_auto_generated,
        segment_count=len(segments),
        total_words=total_words,
        segments=segments,
    )

