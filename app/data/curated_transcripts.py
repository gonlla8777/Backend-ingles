"""Re-export del catálogo curado para app.data.curated_transcripts."""

from app.services.curated_transcripts import (
    CURATED_TRANSCRIPTS,
    CuratedTranscriptEntry,
    ELEPHANT_CHUNKS,
    INTERVIEW_CHUNKS,
    PROJECT_ZOMBOID_CHUNKS,
    Q7ECB76YBOO_FULL_CHUNKS,
)

__all__ = [
    "CURATED_TRANSCRIPTS",
    "CuratedTranscriptEntry",
    "ELEPHANT_CHUNKS",
    "INTERVIEW_CHUNKS",
    "PROJECT_ZOMBOID_CHUNKS",
    "Q7ECB76YBOO_FULL_CHUNKS",
]

