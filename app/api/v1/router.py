from fastapi import APIRouter

from app.api.v1.endpoints import evaluation, quiz, videos

api_v1_router = APIRouter()
api_v1_router.include_router(videos.router, prefix="/videos", tags=["Videos"])
api_v1_router.include_router(videos.router, tags=["Videos (alias)"])
api_v1_router.include_router(
    evaluation.router, prefix="/evaluation", tags=["AI Mentor & Evaluation"]
)
api_v1_router.include_router(evaluation.router, tags=["Evaluation (alias)"])
api_v1_router.include_router(quiz.router, prefix="/quiz", tags=["Contextual Quiz"])
api_v1_router.include_router(quiz.router, tags=["Quiz (alias)"])
