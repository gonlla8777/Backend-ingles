import logging
from fastapi import APIRouter, HTTPException, status

from app.schemas.dictionary_schemas import WordDefinitionOut
from app.services.dictionary_proxy_service import dictionary_proxy_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/{word}",
    response_model=WordDefinitionOut,
    summary="Consulta la definición y fonética de una palabra en inglés vía proxy",
    description=(
        "Proxy backend con lematización automática (plurales, conjugaciones) "
        "y garantía de respuesta con fallback de IA/traducción. "
        "Evita restricciones de CORS desde Flutter Web."
    ),
)
async def get_word_definition(word: str) -> WordDefinitionOut:
    cleaned = dictionary_proxy_service.clean_word(word)
    if not cleaned or len(cleaned) < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La palabra consultada debe contener al menos un carácter válido.",
        )

    return await dictionary_proxy_service.lookup(cleaned)
