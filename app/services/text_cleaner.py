import html
import re

from app.schemas.transcript_schemas import CleanedChunk, RawTranscriptChunk

# Pistas no verbales entre corchetes: [Music], [Applause], [Laughter], [Inaudible]
_BRACKET_NOISE_RE = re.compile(r"\[[^\]]*\]")

# Pistas acústicas entre paréntesis: (music), (laughs), (applause), (upbeat music)
_PAREN_SOUND_CUE_RE = re.compile(
    r"\(\s*(?:music|applause|laughter|laughs|laughing|sighs|cheering|cheers|"
    r"inaudible|crosstalk|coughing|coughs|gasps|groans|beeping|bell[^)]*|"
    r"upbeat[^)]*|dramatic[^)]*|soft[^)]*|audience[^)]*|speaking[^)]*|sound[^)]*)\s*\)",
    re.IGNORECASE,
)

# Notas musicales envolventes o sueltas
_MUSIC_NOTES_Block_RE = re.compile(r"[♪♫]+.*?[♪♫]+|[♪♫]+")

# Etiquetas HTML residuales (<font>, <i>, <b>)
_HTML_TAGS_RE = re.compile(r"<[^>]+>")

# Marcadores de cambio de interlocutor al inicio del bloque: ">> ", "- ", "INTERVIEWER: "
_SPEAKER_ARROW_RE = re.compile(r"^\s*(?:>>+|-\s+(?=[A-Z\"']))\s*")
_SPEAKER_LABEL_RE = re.compile(r"^\s*([A-Z][A-Z0-9\s]{1,20}):\s+")

_ALPHANUM_RE = re.compile(r"[A-Za-z0-9]")
_WHITESPACE_RE = re.compile(r"\s+")


class TranscriptTextCleaner:
    """Limpia artefactos de subtítulos de YouTube preservando cambios de hablante."""

    def clean_chunks(self, raw_chunks: list[RawTranscriptChunk]) -> list[CleanedChunk]:
        cleaned: list[CleanedChunk] = []

        for chunk in raw_chunks:
            result = self.clean_single(chunk)
            if result is not None:
                cleaned.append(result)

        return cleaned

    def clean_single(self, chunk: RawTranscriptChunk) -> CleanedChunk | None:
        # Doble unescape por casos frecuentes de YouTube como "&amp;#39;"
        text = html.unescape(html.unescape(chunk.text))
        text = _HTML_TAGS_RE.sub(" ", text)
        text = text.replace("\n", " ").replace("\r", " ")

        # Normalización tipográfica para facilitar escritura por teclado estándar
        text = (
            text.replace("’", "'")
            .replace("‘", "'")
            .replace("“", '"')
            .replace("”", '"')
            .replace("\u200b", "")
        )

        is_speaker_change = False
        if _SPEAKER_ARROW_RE.match(text):
            is_speaker_change = True
            text = _SPEAKER_ARROW_RE.sub("", text, count=1)

        if _SPEAKER_LABEL_RE.match(text):
            is_speaker_change = True
            text = _SPEAKER_LABEL_RE.sub("", text, count=1)

        # Eliminación de acotaciones sonoras y musicales
        text = _MUSIC_NOTES_Block_RE.sub(" ", text)
        text = _BRACKET_NOISE_RE.sub(" ", text)
        text = _PAREN_SOUND_CUE_RE.sub(" ", text)

        text = _WHITESPACE_RE.sub(" ", text).strip()

        # Descarta fragmentos vacíos o compuestos únicamente por signos
        if not text or not _ALPHANUM_RE.search(text):
            return None

        end_time = max(chunk.start, chunk.end)
        return CleanedChunk(
            text=text,
            start=round(chunk.start, 3),
            end=round(end_time, 3),
            is_speaker_turn_change=is_speaker_change,
        )

