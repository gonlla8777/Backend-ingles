from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.schemas.quiz_schemas import QuizGeneratorRequest, QuizGeneratorResponse
from app.services.gemini_mentor_service import GeminiMentorService

router = APIRouter()


def get_mentor(settings: Settings = Depends(get_settings)) -> GeminiMentorService:
    return GeminiMentorService(settings=settings)


@router.post(
    "/quiz-generator",
    response_model=QuizGeneratorResponse,
    summary="Genera 2 a 3 preguntas contextuales al finalizar el video",
)
def quiz_generator_endpoint(
    payload: QuizGeneratorRequest,
    mentor: GeminiMentorService = Depends(get_mentor),
) -> QuizGeneratorResponse:
    return mentor.generate_quiz(payload)

