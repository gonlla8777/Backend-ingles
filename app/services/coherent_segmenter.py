import re
from dataclasses import dataclass

from app.core.config import Settings, get_settings
from app.schemas.transcript_schemas import CleanedChunk, TranscriptSegmentOut

# Abreviaturas frecuentes que NO deben disparar corte de oración al terminar en punto
_ABBREVIATIONS = {
    "mr.",
    "mrs.",
    "ms.",
    "dr.",
    "prof.",
    "sr.",
    "jr.",
    "st.",
    "vs.",
    "etc.",
    "e.g.",
    "i.e.",
    "u.s.",
    "u.k.",
    "u.s.a.",
    "ph.d.",
    "inc.",
    "ltd.",
    "co.",
    "corp.",
    "fig.",
    "no.",
}

_TERMINAL_PUNCT_RE = re.compile(r'[.?!]+["\')]?$')
_SOFT_CLAUSE_END_RE = re.compile(r'[,;:—–-]+$')
_CONJUNCTION_START_RE = re.compile(
    r"^(?:and|but|so|because|which|while|although|though|when|if|however|therefore|then)\b",
    re.IGNORECASE,
)
_SENTENCE_BOUNDARY_RE = re.compile(r'([.?!]+["\')]?)\s+(?=[A-Z0-9"\'])')
_SINGLE_INITIAL_RE = re.compile(r"^[A-Z]\.$")


def _word_count(text: str) -> int:
    stripped = text.strip()
    return len(stripped.split()) if stripped else 0


def _ends_with_abbreviation(text: str) -> bool:
    tokens = text.strip().split()
    if not tokens:
        return False
    last_token = tokens[-1].strip('"\'()')
    if last_token.lower() in _ABBREVIATIONS:
        return True
    if _SINGLE_INITIAL_RE.match(last_token):
        return True
    return False


def _is_true_sentence_end(text: str) -> bool:
    stripped = text.strip()
    if not stripped or not _TERMINAL_PUNCT_RE.search(stripped):
        return False
    return not _ends_with_abbreviation(stripped)


@dataclass
class _WorkingSegment:
    text: str
    start: float
    end: float
    is_speaker_turn_change: bool


class CoherentSegmenter:
    """
    Agrupa fragmentos rotos de YouTube en oraciones gramaticalmente completas,
    naturales y dimensionadas para transcripción/dictado interactivo.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def segment(self, chunks: list[CleanedChunk]) -> list[TranscriptSegmentOut]:
        if not chunks:
            return []

        # Paso 1: Subdividir fragmentos que contengan más de una oración interna
        # o fragmentos de auto-captions excesivamente largos, interpolando timestamps.
        atomic_chunks: list[CleanedChunk] = []
        for chunk in chunks:
            for sub in self._split_intra_chunk_sentences(chunk):
                atomic_chunks.extend(self._split_oversized_single_chunk(sub))

        # Paso 2: Acumular en oraciones completas respetando puntuación, pausas y longitud.
        merged: list[_WorkingSegment] = []
        buf_texts: list[str] = []
        buf_start: float = 0.0
        buf_end: float = 0.0
        buf_speaker_change: bool = False
        buf_words: int = 0

        def flush_buffer() -> None:
            nonlocal buf_texts, buf_start, buf_end, buf_speaker_change, buf_words
            if not buf_texts:
                return
            combined = re.sub(r"\s+", " ", " ".join(buf_texts)).strip()
            if combined:
                merged.append(
                    _WorkingSegment(
                        text=combined,
                        start=buf_start,
                        end=max(buf_start, buf_end),
                        is_speaker_turn_change=buf_speaker_change,
                    )
                )
            buf_texts = []
            buf_words = 0
            buf_speaker_change = False

        for piece in atomic_chunks:
            piece_words = _word_count(piece.text)
            if piece_words == 0:
                continue

            if buf_texts:
                acoustic_gap = max(0.0, piece.start - buf_end)
                projected_words = buf_words + piece_words
                projected_duration = piece.end - buf_start
                last_text = buf_texts[-1]

                # 2a. Cambio explícito de interlocutor -> cierra el turno previo
                if piece.is_speaker_turn_change:
                    flush_buffer()
                # 2b. Silencio acústico largo -> pausa natural de discurso
                elif acoustic_gap >= 2.0 or (
                    acoustic_gap >= self._settings.segment_acoustic_gap_seconds
                    and buf_words >= self._settings.segment_min_words
                ):
                    flush_buffer()
                # 2c. Guardarraíl pedagógico para oraciones largas o auto-captions sin puntos
                elif (
                    buf_words >= self._settings.segment_target_max_words
                    or projected_duration > self._settings.segment_max_duration_seconds
                ) and (
                    _SOFT_CLAUSE_END_RE.search(last_text)
                    or _CONJUNCTION_START_RE.search(piece.text)
                    or projected_words > self._settings.segment_hard_max_words
                ):
                    flush_buffer()
                elif projected_words > self._settings.segment_hard_max_words:
                    flush_buffer()

            if not buf_texts:
                buf_start = piece.start
                buf_speaker_change = piece.is_speaker_turn_change

            buf_texts.append(piece.text)
            buf_end = max(buf_end, piece.end)
            buf_words += piece_words

            # Cierre inmediato tras puntuación terminal real (. ? !)
            if _is_true_sentence_end(piece.text):
                flush_buffer()
            # Cierre suave si ya superamos el objetivo óptimo y hay coma/punto y coma
            elif (
                buf_words >= self._settings.segment_target_max_words
                and _SOFT_CLAUSE_END_RE.search(piece.text)
            ):
                flush_buffer()

        flush_buffer()

        # Paso 3: Fusión suave de micro-fragmentos (< 3 palabras) si están pegados en audio
        refined = self._merge_adjacent_micro_segments(merged)

        # Paso 4: Construcción final indexada
        output: list[TranscriptSegmentOut] = []
        for idx, seg in enumerate(refined):
            normalized_text = self._polish_segment_text(seg.text)
            words = _word_count(normalized_text)
            if words == 0:
                continue
            output.append(
                TranscriptSegmentOut(
                    index=len(output),
                    text=normalized_text,
                    start_time=round(seg.start, 3),
                    end_time=round(max(seg.start, seg.end), 3),
                    word_count=words,
                    is_speaker_turn_change=seg.is_speaker_turn_change or idx == 0,
                )
            )

        return output

    def _split_intra_chunk_sentences(self, chunk: CleanedChunk) -> list[CleanedChunk]:
        """
        Divide un mismo bloque de YouTube cuando contiene el fin de una oración y el
        inicio de otra (ej. 'That works well. Next we deploy'), interpolando tiempos.
        """
        text = chunk.text.strip()
        if not text:
            return []

        parts: list[str] = []
        cursor = 0
        for match in _SENTENCE_BOUNDARY_RE.finditer(text):
            end_pos = match.end(1)
            candidate_left = text[cursor:end_pos].strip()
            if not _ends_with_abbreviation(candidate_left):
                parts.append(candidate_left)
                cursor = match.end()

        remainder = text[cursor:].strip()
        if remainder:
            parts.append(remainder)

        if len(parts) <= 1:
            return [chunk]

        return self._interpolate_subchunks(
            parts=parts,
            start=chunk.start,
            end=chunk.end,
            first_is_speaker_change=chunk.is_speaker_turn_change,
        )

    def _split_oversized_single_chunk(self, chunk: CleanedChunk) -> list[CleanedChunk]:
        """Divide bloques individuales sin puntuación que excedan `segment_hard_max_words`."""
        words = chunk.text.split()
        target = self._settings.segment_target_max_words
        hard_max = self._settings.segment_hard_max_words

        if len(words) <= hard_max:
            return [chunk]

        parts: list[str] = []
        idx = 0
        while idx < len(words):
            remaining = len(words) - idx
            if remaining <= hard_max:
                parts.append(" ".join(words[idx:]))
                break

            # Busca una conjunción cercana a `target` para cortar con naturalidad
            split_at = idx + target
            best_conj = None
            window_start = max(idx + self._settings.segment_min_words, idx + target - 4)
            window_end = min(len(words) - 2, idx + hard_max)
            for pos in range(window_start, window_end):
                if _CONJUNCTION_START_RE.match(words[pos]):
                    best_conj = pos
                    break
            if best_conj is not None:
                split_at = best_conj

            parts.append(" ".join(words[idx:split_at]))
            idx = split_at

        return self._interpolate_subchunks(
            parts=parts,
            start=chunk.start,
            end=chunk.end,
            first_is_speaker_change=chunk.is_speaker_turn_change,
        )

    @staticmethod
    def _interpolate_subchunks(
        parts: list[str],
        start: float,
        end: float,
        first_is_speaker_change: bool,
    ) -> list[CleanedChunk]:
        total_chars = max(1, sum(len(p) for p in parts))
        total_duration = max(0.0, end - start)
        current_start = start
        result: list[CleanedChunk] = []

        for idx, part in enumerate(parts):
            weight = len(part) / total_chars
            part_duration = total_duration * weight
            part_end = end if idx == len(parts) - 1 else current_start + part_duration
            result.append(
                CleanedChunk(
                    text=part,
                    start=round(current_start, 3),
                    end=round(part_end, 3),
                    is_speaker_turn_change=(first_is_speaker_change and idx == 0),
                )
            )
            current_start = part_end

        return result

    def _merge_adjacent_micro_segments(
        self, segments: list[_WorkingSegment]
    ) -> list[_WorkingSegment]:
        if len(segments) <= 1:
            return segments

        result: list[_WorkingSegment] = []
        idx = 0
        while idx < len(segments):
            current = segments[idx]
            curr_words = _word_count(current.text)

            # Si es una micro-interjección (< 3 palabras) muy pegada al siguiente bloque
            if (
                curr_words < 3
                and idx + 1 < len(segments)
                and not segments[idx + 1].is_speaker_turn_change
            ):
                nxt = segments[idx + 1]
                gap = max(0.0, nxt.start - current.end)
                if (
                    gap <= self._settings.segment_micro_merge_gap_seconds
                    and (curr_words + _word_count(nxt.text))
                    <= self._settings.segment_target_max_words
                ):
                    merged = _WorkingSegment(
                        text=f"{current.text} {nxt.text}",
                        start=current.start,
                        end=nxt.end,
                        is_speaker_turn_change=current.is_speaker_turn_change,
                    )
                    segments[idx + 1] = merged
                    idx += 1
                    continue

            result.append(current)
            idx += 1

        return result

    @staticmethod
    def _polish_segment_text(text: str) -> str:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return cleaned
        # Asegura mayúscula inicial cuando proviene de auto-captions en minúsculas
        first_alpha_match = re.search(r"[a-zA-Z]", cleaned)
        if first_alpha_match and first_alpha_match.start() == 0 and cleaned[0].islower():
            cleaned = cleaned[0].upper() + cleaned[1:]
        return cleaned

