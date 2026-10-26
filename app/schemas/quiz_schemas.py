from pydantic import BaseModel, Field


class QuizGeneratorRequest(BaseModel):
    video_id: str = Field(..., min_length=1)
    transcript_sentences: list[str] = Field(..., min_length=1)
    question_count: int = Field(default=3, ge=2, le=3)


class QuizQuestionOut(BaseModel):
    id: str
    question: str
    options: list[str] = Field(..., min_length=4, max_length=4)
    correct_option_index: int = Field(ge=0, le=3)
    explanation_es: str


class QuizGeneratorResponse(BaseModel):
    video_id: str
    questions: list[QuizQuestionOut]

