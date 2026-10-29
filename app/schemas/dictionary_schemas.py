from typing import Optional
from pydantic import BaseModel, Field


class WordMeaningOut(BaseModel):
    part_of_speech: str = Field(
        default="general",
        description="Parte de la oración (ej. noun, verb, adjective).",
    )
    definition: str = Field(
        ...,
        description="Definición concisa en inglés.",
    )
    example: Optional[str] = Field(
        default=None,
        description="Oración de ejemplo en inglés.",
    )


class WordDefinitionOut(BaseModel):
    word: str = Field(
        ...,
        description="Palabra o lema consultado.",
    )
    phonetic: Optional[str] = Field(
        default=None,
        description="Transcripción fonética (IPA).",
    )
    part_of_speech: Optional[str] = Field(
        default=None,
        description="Categoría gramatical principal.",
    )
    definition: Optional[str] = Field(
        default=None,
        description="Definición concisa principal.",
    )
    translation_es: Optional[str] = Field(
        default=None,
        description="Traducción contextual al español.",
    )
    meanings: list[WordMeaningOut] = Field(
        default_factory=list,
        description="Lista de significados ordenados por relevancia.",
    )
