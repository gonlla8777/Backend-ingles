from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.endpoints import dictionary, evaluation, quiz, videos
from app.api.v1.router import api_v1_router
from app.core.config import get_settings
from app.core.exceptions import EchoFlowError
from app.services.dictionary_proxy_service import dictionary_proxy_service

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await dictionary_proxy_service.close()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
    description=(
        "Backend de EchoFlow para extracción y segmentación coherente de subtítulos "
        "de YouTube, evaluación semántica con Gemini Flash y generación de quizzes."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_origin_regex=r"https://.*\.web\.app|https://.*\.firebaseapp\.com|http://localhost(:\d+)?|http://127\.0\.0\.1(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(EchoFlowError)
async def echoflow_error_handler(_: Request, exc: EchoFlowError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message},
    )


@app.get("/health", tags=["Health"])
def healthcheck() -> dict[str, object]:
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.app_version,
        "release": "dictionary-lemmatization-v1",
        "gemini_configured": settings.has_gemini_key,
        "youtube_cookies_configured": settings.has_youtube_cookies,
        "youtube_proxy_configured": settings.has_youtube_proxy,
        "mock_fallback_enabled": settings.is_fallback_enabled,
    }


# Rutas versionadas consumidas por el cliente Flutter (/api/v1/...)
app.include_router(api_v1_router, prefix="/api/v1")

# Rutas con prefijo directo /videos/... y /dictionary/...
app.include_router(videos.router, prefix="/videos", tags=["Videos Direct"])
app.include_router(dictionary.router, prefix="/dictionary", tags=["Dictionary Direct"])

# Alias directos en raíz (/process-video, /evaluate-transcription, /quiz-generator, /dictionary)
app.include_router(videos.router, tags=["Direct Aliases"])
app.include_router(evaluation.router, tags=["Direct Aliases"])
app.include_router(quiz.router, tags=["Direct Aliases"])
app.include_router(dictionary.router, tags=["Direct Aliases"])
