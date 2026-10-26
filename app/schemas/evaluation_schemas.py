from typing import Literal
from pydantic import BaseModel, Field


VerdictType = Literal["exact", "semantic_equivalent", "minor_typo", "needs_retry"]


class EvaluateTranscriptionRequest(BaseModel):
    segment_index: int = Field(ge=0)
    reference_text: str = Field(..., min_length=1)
    user_input: str = Field(default="")
    combo_multiplier: float = Field(default=1.0, ge=1.0, le=5.0)


class GrammarCorrectionOut(BaseModel):
    original_fragment: str
    corrected_fragment: str
    explanation_es: str


class GeminiSemanticEvaluationSchema(BaseModel):
    """Esquema Pydantic usado como `response_schema` en Gemini Flash."""

    accuracy_score: float = Field(ge=0.0, le=1.0)
    verdict: VerdictType
    is_passed: bool
    corrections: list[GrammarCorrectionOut] = Field(default_factory=list)
    pronunciation_tip: str | None = None
    formality_register: str | None = None
    idiom_note: str | None = None


class EvaluateTranscriptionResponse(BaseModel):
    segment_index: int
    user_input: str
    reference_text: str
    accuracy_score: float = Field(ge=0.0, le=1.0)
    verdict: VerdictType
    is_passed: bool
    xp_earned: int = Field(ge=0)
    combo_multiplier: float
    corrections: list[GrammarCorrectionOut] = Field(default_factory=list)
    pronunciation_tip: str | None = None
    formality_register: str | None = None
    idiom_note: str | None = None

