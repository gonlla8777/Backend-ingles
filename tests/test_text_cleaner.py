from app.schemas.transcript_schemas import RawTranscriptChunk
from app.services.text_cleaner import TranscriptTextCleaner


def test_cleans_html_entities_music_and_sound_cues() -> None:
    cleaner = TranscriptTextCleaner()
    raw = [
        RawTranscriptChunk(text="[Music]", start=0.0, duration=2.0),
        RawTranscriptChunk(
            text="We&amp;#39;re excited\nto launch (applause) ♪ la la ♪ today!",
            start=2.0,
            duration=3.5,
        ),
        RawTranscriptChunk(
            text="&gt;&gt; INTERVIEWER: Tell me about your system.",
            start=6.0,
            duration=2.5,
        ),
    ]

    cleaned = cleaner.clean_chunks(raw)
    assert len(cleaned) == 2

    assert cleaned[0].text == "We're excited to launch today!"
    assert cleaned[0].start == 2.0
    assert cleaned[0].end == 5.5
    assert cleaned[0].is_speaker_turn_change is False

    assert cleaned[1].text == "Tell me about your system."
    assert cleaned[1].is_speaker_turn_change is True

