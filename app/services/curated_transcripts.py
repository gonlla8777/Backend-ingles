"""Catálogo curado local de transcripciones para videos pedagógicos de práctica."""

import json
from pathlib import Path
from typing import Any

_DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "q7ecb76yboo_transcript.json"

if _DATA_FILE.is_file():
    try:
        with open(_DATA_FILE, "r", encoding="utf-8") as _f:
            Q7ECB76YBOO_FULL_CHUNKS = json.load(_f)
    except Exception:
        Q7ECB76YBOO_FULL_CHUNKS = []
else:
    Q7ECB76YBOO_FULL_CHUNKS = []


class CuratedTranscriptEntry(list):
    """Representa una transcripción curada compatible con dict y list."""

    def __init__(
        self,
        raw_transcript: list[dict[str, Any]],
        title: str = "YouTube Practice Video",
    ):
        super().__init__(raw_transcript)
        self.title = title
        self.raw_transcript = raw_transcript

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, str):
            if item == "title":
                return self.title
            if item in ("raw_transcript", "chunks"):
                return self.raw_transcript
        return super().__getitem__(item)

    def get(self, key: str, default: Any = None) -> Any:
        if key == "title":
            return self.title
        if key in ("raw_transcript", "chunks"):
            return self.raw_transcript
        return default


PROJECT_ZOMBOID_CHUNKS = [
    {
        "text": "Welcome to Project Zomboid, a hardcore zombie survival role-playing game.",
        "start": 0.5,
        "duration": 4.5,
    },
    {
        "text": "This is the story of how you died, so every decision you make matters.",
        "start": 5.2,
        "duration": 5.0,
    },
    {
        "text": "First, sneak quietly around your starter house to avoid alerting nearby walkers.",
        "start": 10.5,
        "duration": 5.2,
    },
    {
        "text": "Gather essential supplies like fresh water, canned food, and a sturdy weapon.",
        "start": 16.0,
        "duration": 5.0,
    },
    {
        "text": "Remember to close all curtains to prevent zombies from spotting you through windows.",
        "start": 21.5,
        "duration": 5.5,
    },
    {
        "text": "Always keep an eye on your character's fatigue, hunger, and panic moodles.",
        "start": 27.5,
        "duration": 5.0,
    },
]

ELEPHANT_CHUNKS = [
    {
        "text": "All right, so here we are in front of the elephants.",
        "start": 1.2,
        "duration": 3.5,
    },
    {
        "text": "The cool thing about these guys is that they have really, really, really long trunks.",
        "start": 5.0,
        "duration": 5.5,
    },
    {
        "text": "And that's cool.",
        "start": 11.0,
        "duration": 2.5,
    },
    {
        "text": "And that's pretty much all there is to say.",
        "start": 14.0,
        "duration": 3.5,
    },
]

INTERVIEW_CHUNKS = [
    {
        "text": "Hello and welcome, thank you for joining us today.",
        "start": 0.5,
        "duration": 4.5,
    },
    {
        "text": "I am excited to be here and discuss this engineering role.",
        "start": 6.0,
        "duration": 4.8,
    },
    {
        "text": "Could you tell us about your experience designing backend systems?",
        "start": 11.5,
        "duration": 5.0,
    },
    {
        "text": "In my previous position, I developed microservices using FastAPI.",
        "start": 17.5,
        "duration": 5.0,
    },
    {
        "text": "How do you handle database latency and caching strategies?",
        "start": 23.5,
        "duration": 4.8,
    },
    {
        "text": "We implemented Redis caching, which reduced response times significantly.",
        "start": 29.5,
        "duration": 5.2,
    },
]

CURATED_TRANSCRIPTS: dict[str, CuratedTranscriptEntry] = {
    "Q7ecB76Yboo": CuratedTranscriptEntry(
        Q7ECB76YBOO_FULL_CHUNKS or PROJECT_ZOMBOID_CHUNKS,
        title="How to Start Project Zomboid | Ultimate Beginners Guide",
    ),
    "project_zomboid": CuratedTranscriptEntry(
        Q7ECB76YBOO_FULL_CHUNKS or PROJECT_ZOMBOID_CHUNKS,
        title="Project Zomboid - Ultimate Survival & Beginner Guide",
    ),
    "zomboid": CuratedTranscriptEntry(
        Q7ECB76YBOO_FULL_CHUNKS or PROJECT_ZOMBOID_CHUNKS,
        title="Project Zomboid - Ultimate Survival & Beginner Guide",
    ),
    "Project Zomboid": CuratedTranscriptEntry(
        PROJECT_ZOMBOID_CHUNKS,
        title="Project Zomboid - Ultimate Survival & Beginner Guide",
    ),
    "jNQXAC9IVRw": CuratedTranscriptEntry(
        ELEPHANT_CHUNKS,
        title="Me at the zoo",
    ),
    "curated_interview": CuratedTranscriptEntry(
        INTERVIEW_CHUNKS,
        title="Software Engineer Job Interview",
    ),
    "demo": CuratedTranscriptEntry(
        INTERVIEW_CHUNKS,
        title="Technical Demo Session",
    ),
}
