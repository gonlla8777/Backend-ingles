from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.schemas.evaluation_schemas import (
    EvaluateTranscriptionRequest,
    EvaluateTranscriptionResponse,
)
from app.services.gemini_mentor_service import GeminiMentorService

router = APIRouter()


def get_mentor(settings: Settings = Depends(get_settings)) -> GeminiMentorService:
    return GeminiMentorService(settings=settings)


@router.post(
    "/evaluate-transcription",
    response_model=EvaluateTranscriptionResponse,
    summary="Evalúa la transcripción del usuario (Fast-Path local + Gemini Flash)",
)
def evaluate_transcription_endpoint(
    payload: EvaluateTranscriptionRequest,
    mentor: GeminiMentorService = Depends(get_mentor),
) -> EvaluateTranscriptionResponse:
    return mentor.evaluate_transcription(payload)

