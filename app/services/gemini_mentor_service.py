import difflib
import re
from google import genai
from google.genai import types
from pydantic import BaseModel

from app.core.config import Settings, get_settings
from app.schemas.evaluation_schemas import (
    EvaluateTranscriptionRequest,
    EvaluateTranscriptionResponse,
    GeminiSemanticEvaluationSchema,
    GrammarCorrectionOut,
)
from app.schemas.quiz_schemas import (
    QuizGeneratorRequest,
    QuizGeneratorResponse,
    QuizQuestionOut,
)
from app.schemas.transcript_schemas import TranscriptSegmentOut

# Contracciones y reducciones coloquiales frecuentes equivalentes en contexto oral
_CANONICAL_CONTRACTIONS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bi'm\b"), "i am"),
    (re.compile(r"\byou're\b"), "you are"),
    (re.compile(r"\bwe're\b"), "we are"),
    (re.compile(r"\bthey're\b"), "they are"),
    (re.compile(r"\bhe's\b"), "he is"),
    (re.compile(r"\bshe's\b"), "she is"),
    (re.compile(r"\bit's\b"), "it is"),
    (re.compile(r"\bthat's\b"), "that is"),
    (re.compile(r"\bthere's\b"), "there is"),
    (re.compile(r"\bwhat's\b"), "what is"),
    (re.compile(r"\blet's\b"), "let us"),
    (re.compile(r"\bdon't\b"), "do not"),
    (re.compile(r"\bdoesn't\b"), "does not"),
    (re.compile(r"\bdidn't\b"), "did not"),
    (re.compile(r"\bcan't\b"), "cannot"),
    (re.compile(r"\bwon't\b"), "will not"),
    (re.compile(r"\bwouldn't\b"), "would not"),
    (re.compile(r"\bshouldn't\b"), "should not"),
    (re.compile(r"\bcouldn't\b"), "could not"),
    (re.compile(r"\bi've\b"), "i have"),
    (re.compile(r"\bwe've\b"), "we have"),
    (re.compile(r"\bthey've\b"), "they have"),
    (re.compile(r"\bi'll\b"), "i will"),
    (re.compile(r"\bwe'll\b"), "we will"),
    (re.compile(r"\bgonna\b"), "going to"),
    (re.compile(r"\bwanna\b"), "want to"),
    (re.compile(r"\bgotta\b"), "got to"),
    (re.compile(r"\bkinda\b"), "kind of"),
    (re.compile(r"\bsorta\b"), "sort of"),
]

_NON_WORD_PUNCT_RE = re.compile(r"[^\w\s']")
_MULTI_SPACE_RE = re.compile(r"\s+")

_BASE_XP_PER_WORD = 5


def normalize_for_comparison(text: str, expand_contractions: bool = True) -> str:
    lowered = text.lower().strip().replace("’", "'")
    lowered = _NON_WORD_PUNCT_RE.sub(" ", lowered)
    if expand_contractions:
        for pattern, replacement in _CANONICAL_CONTRACTIONS:
            lowered = pattern.sub(replacement, lowered)
    lowered = lowered.replace("'", "")
    return _MULTI_SPACE_RE.sub(" ", lowered).strip()


class _QuizListSchema(BaseModel):
    questions: list[QuizQuestionOut]


class _PunctuatedSegmentsSchema(BaseModel):
    punctuated_sentences: list[str]


class GeminiMentorService:
    """
    Juez y Mentor Pedagógico Híbrido:
    1. Fast-Path determinista en 0ms para coincidencias exactas o contracciones equivalentes
       (protegiendo el rate limit del Free Tier de Google Gemini).
    2. Evaluación semántica tolerante y feedback con `google-genai` (Gemini Flash).
    3. Fallback determinista local si aún no se configuró `GEMINI_API_KEY`.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def evaluate_transcription(
        self, req: EvaluateTranscriptionRequest
    ) -> EvaluateTranscriptionResponse:
        ref_strict = normalize_for_comparison(
            req.reference_text, expand_contractions=False
        )
        usr_strict = normalize_for_comparison(
            req.user_input, expand_contractions=False
        )
        ref_canonical = normalize_for_comparison(
            req.reference_text, expand_contractions=True
        )
        usr_canonical = normalize_for_comparison(
            req.user_input, expand_contractions=True
        )

        ref_word_count = max(1, len(ref_strict.split()))

        # 1. Fast-Path: Coincidencia exacta o contracción equivalente (0 gasto de cuota IA)
        if usr_strict and usr_strict == ref_strict:
            xp = round(ref_word_count * _BASE_XP_PER_WORD * req.combo_multiplier)
            return EvaluateTranscriptionResponse(
                segment_index=req.segment_index,
                user_input=req.user_input,
                reference_text=req.reference_text,
                accuracy_score=1.0,
                verdict="exact",
                is_passed=True,
                xp_earned=xp,
                combo_multiplier=req.combo_multiplier,
                corrections=[],
                pronunciation_tip=self._quick_local_pronunciation_tip(
                    req.reference_text
                ),
                formality_register=self._infer_local_register(req.reference_text),
            )

        if usr_canonical and usr_canonical == ref_canonical:
            xp = round(ref_word_count * _BASE_XP_PER_WORD * req.combo_multiplier)
            return EvaluateTranscriptionResponse(
                segment_index=req.segment_index,
                user_input=req.user_input,
                reference_text=req.reference_text,
                accuracy_score=0.98,
                verdict="semantic_equivalent",
                is_passed=True,
                xp_earned=xp,
                combo_multiplier=req.combo_multiplier,
                corrections=[],
                pronunciation_tip=(
                    "¡Excelente! Usaste una variante equivalente (contracción/forma completa) "
                    "totalmente natural en inglés hablado."
                ),
                formality_register=self._infer_local_register(req.reference_text),
            )

        # 2. Evaluación Semántica con Gemini Flash si hay API Key configurada
        if self._settings.has_gemini_key and req.user_input.strip():
            try:
                return self._evaluate_with_gemini(req, ref_word_count)
            except Exception:
                # Degrada con elegancia al motor determinista si se alcanza el límite RPM del Free Tier
                pass

        # 3. Evaluación determinista de respaldo (offline o sin API key)
        return self._evaluate_deterministic_fallback(
            req, ref_canonical, usr_canonical, ref_word_count
        )

    def _evaluate_with_gemini(
        self, req: EvaluateTranscriptionRequest, ref_word_count: int
    ) -> EvaluateTranscriptionResponse:
        client = genai.Client(api_key=self._settings.gemini_api_key)
        prompt = (
            "Actúa como un juez y tutor experto en inglés para hispanohablantes.\n"
            f"Oración original del video: \"{req.reference_text}\"\n"
            f"Transcripción del estudiante: \"{req.user_input}\"\n\n"
            "Evalúa con tolerancia a variantes coloquiales/formales válidas y contracciones. "
            "Devuelve correcciones puntuales en español, un tip breve de pronunciación "
            "(connected speech / linking), el registro de formalidad y explicación de modismos si existen."
        )
        response = client.models.generate_content(
            model=self._settings.gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=GeminiSemanticEvaluationSchema,
                temperature=0.15,
            ),
        )
        parsed: GeminiSemanticEvaluationSchema = response.parsed
        matched_words = round(ref_word_count * parsed.accuracy_score)
        xp = (
            round(matched_words * _BASE_XP_PER_WORD * req.combo_multiplier)
            if parsed.is_passed
            else 0
        )

        return EvaluateTranscriptionResponse(
            segment_index=req.segment_index,
            user_input=req.user_input,
            reference_text=req.reference_text,
            accuracy_score=round(parsed.accuracy_score, 3),
            verdict=parsed.verdict,
            is_passed=parsed.is_passed,
            xp_earned=xp,
            combo_multiplier=req.combo_multiplier,
            corrections=parsed.corrections,
            pronunciation_tip=parsed.pronunciation_tip,
            formality_register=parsed.formality_register,
            idiom_note=parsed.idiom_note,
        )

    def _evaluate_deterministic_fallback(
        self,
        req: EvaluateTranscriptionRequest,
        ref_canonical: str,
        usr_canonical: str,
        ref_word_count: int,
    ) -> EvaluateTranscriptionResponse:
        if not usr_canonical:
            return EvaluateTranscriptionResponse(
                segment_index=req.segment_index,
                user_input=req.user_input,
                reference_text=req.reference_text,
                accuracy_score=0.0,
                verdict="needs_retry",
                is_passed=False,
                xp_earned=0,
                combo_multiplier=1.0,
                corrections=[
                    GrammarCorrectionOut(
                        original_fragment="(vacío)",
                        corrected_fragment=req.reference_text,
                        explanation_es="Escucha nuevamente el fragmento e intenta transcribirlo.",
                    )
                ],
                pronunciation_tip=self._quick_local_pronunciation_tip(
                    req.reference_text
                ),
                formality_register=self._infer_local_register(req.reference_text),
            )

        ratio = difflib.SequenceMatcher(None, ref_canonical, usr_canonical).ratio()
        accuracy = round(max(0.0, min(1.0, ratio)), 3)
        is_passed = accuracy >= 0.70
        verdict = "minor_typo" if accuracy >= 0.85 else "needs_retry"
        xp = (
            round(ref_word_count * accuracy * _BASE_XP_PER_WORD * req.combo_multiplier)
            if is_passed
            else 0
        )

        corrections: list[GrammarCorrectionOut] = []
        if accuracy < 1.0:
            corrections.append(
                GrammarCorrectionOut(
                    original_fragment=req.user_input.strip(),
                    corrected_fragment=req.reference_text,
                    explanation_es="Revisa las diferencias ortográficas o palabras faltantes respecto al audio.",
                )
            )

        return EvaluateTranscriptionResponse(
            segment_index=req.segment_index,
            user_input=req.user_input,
            reference_text=req.reference_text,
            accuracy_score=accuracy,
            verdict=verdict,
            is_passed=is_passed,
            xp_earned=xp,
            combo_multiplier=req.combo_multiplier if is_passed else 1.0,
            corrections=corrections,
            pronunciation_tip=self._quick_local_pronunciation_tip(req.reference_text),
            formality_register=self._infer_local_register(req.reference_text),
        )

    def generate_quiz(self, req: QuizGeneratorRequest) -> QuizGeneratorResponse:
        excerpt = "\n".join(req.transcript_sentences[:40])
        if self._settings.has_gemini_key:
            try:
                client = genai.Client(api_key=self._settings.gemini_api_key)
                prompt = (
                    f"Genera exactamente {req.question_count} preguntas de opción múltiple (4 opciones cada una) "
                    "en inglés sobre comprensión contextual, vocabulario y modismos del siguiente video, "
                    "con explicación pedagógica breve en español (`explanation_es`):\n\n"
                    f"{excerpt}"
                )
                response = client.models.generate_content(
                    model=self._settings.gemini_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=_QuizListSchema,
                        temperature=0.2,
                    ),
                )
                parsed: _QuizListSchema = response.parsed
                return QuizGeneratorResponse(
                    video_id=req.video_id,
                    questions=parsed.questions[: req.question_count],
                )
            except Exception:
                pass

        # Respaldo determinista para pruebas sin GEMINI_API_KEY
        sample_sentence = req.transcript_sentences[0]
        fallback_questions = [
            QuizQuestionOut(
                id=f"{req.video_id}_q1",
                question=f'Which statement best reflects the opening idea: "{sample_sentence[:80]}"?',
                options=[
                    sample_sentence,
                    "The speaker cancels the discussion immediately.",
                    "No topic is introduced in the segment.",
                    "The speaker switches to an unrelated language.",
                ],
                correct_option_index=0,
                explanation_es="Corresponde a la idea central expresada al inicio del video.",
            ),
            QuizQuestionOut(
                id=f"{req.video_id}_q2",
                question="What is the primary communicative register of this transcript?",
                options=[
                    self._infer_local_register(excerpt),
                    "Medieval poetic verse",
                    "Legal courtroom verdict only",
                    "Silent pantomime",
                ],
                correct_option_index=0,
                explanation_es="El registro se deduce del vocabulario y estructura de las oraciones del video.",
            ),
        ]
        return QuizGeneratorResponse(
            video_id=req.video_id,
            questions=fallback_questions,
        )

    def restore_punctuation_if_needed(
        self, segments: list[TranscriptSegmentOut]
    ) -> list[TranscriptSegmentOut]:
        """Restaura puntuación básica con Gemini Flash si los auto-captions no tienen puntos."""
        if not self._settings.has_gemini_key or not segments:
            return segments

        punctuated_count = sum(
            1 for seg in segments if seg.text.endswith((".", "?", "!"))
        )
        if punctuated_count / len(segments) >= 0.25:
            return segments

        try:
            client = genai.Client(api_key=self._settings.gemini_api_key)
            raw_lines = [seg.text for seg in segments]
            prompt = (
                "Agrega mayúsculas y signos de puntuación naturales a cada una de estas "
                f"{len(raw_lines)} líneas en inglés SIN cambiar, agregar ni quitar palabras, "
                "manteniendo exactamente la misma cantidad y orden de elementos:\n"
                + "\n".join(f"{i}: {line}" for i, line in enumerate(raw_lines))
            )
            response = client.models.generate_content(
                model=self._settings.gemini_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=_PunctuatedSegmentsSchema,
                    temperature=0.0,
                ),
            )
            parsed: _PunctuatedSegmentsSchema = response.parsed
            if len(parsed.punctuated_sentences) != len(segments):
                return segments

            return [
                seg.model_copy(update={"text": new_text.strip() or seg.text})
                for seg, new_text in zip(segments, parsed.punctuated_sentences)
            ]
        except Exception:
            return segments

    @staticmethod
    def _quick_local_pronunciation_tip(text: str) -> str:
        lower = text.lower()
        if "going to" in lower or "gonna" in lower:
            return 'En habla fluida, "going to" suele enlazarse como "gonna" /ˈɡʌnə/.'
        if "want to" in lower or "wanna" in lower:
            return 'En conversación natural, "want to" suele reducirse a "wanna" /ˈwɑːnə/.'
        return "Presta atención al enlace (linking) entre consonantes finales y vocales iniciales."

    @staticmethod
    def _infer_local_register(text: str) -> str:
        lower = text.lower()
        if any(tok in lower for tok in ("gonna", "wanna", "yeah", "kinda", "gotta")):
            return "Casual / Conversational"
        if any(
            tok in lower
            for tok in ("furthermore", "therefore", "architecture", "strategy")
        ):
            return "Professional / Technical"
        return "Neutral / Standard Spoken English"

