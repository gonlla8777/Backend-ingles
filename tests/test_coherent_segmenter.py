from app.schemas.transcript_schemas import CleanedChunk
from app.services.coherent_segmenter import CoherentSegmenter


def test_merges_broken_youtube_fragments_into_complete_sentence() -> None:
    segmenter = CoherentSegmenter()
    chunks = [
        CleanedChunk(text="welcome back to", start=0.5, end=2.0),
        CleanedChunk(text="the channel today we're going", start=2.0, end=4.2),
        CleanedChunk(text="to build a clean architecture.", start=4.2, end=6.8),
    ]

    segments = segmenter.segment(chunks)
    assert len(segments) == 1
    assert (
        segments[0].text
        == "Welcome back to the channel today we're going to build a clean architecture."
    )
    assert segments[0].start_time == 0.5
    assert segments[0].end_time == 6.8
    assert segments[0].word_count == 13


def test_preserves_abbreviations_without_false_sentence_split() -> None:
    segmenter = CoherentSegmenter()
    chunks = [
        CleanedChunk(text="Dr. Smith worked in the U.S.", start=1.0, end=3.5),
        CleanedChunk(text="for ten years before retiring.", start=3.6, end=6.0),
    ]

    segments = segmenter.segment(chunks)
    assert len(segments) == 1
    assert (
        segments[0].text
        == "Dr. Smith worked in the U.S. for ten years before retiring."
    )


def test_splits_intra_chunk_multiple_sentences_with_interpolated_timestamps() -> None:
    segmenter = CoherentSegmenter()
    chunks = [
        CleanedChunk(
            text="That solves the latency bottleneck. Next we configure Firestore caching.",
            start=10.0,
            end=20.0,
        )
    ]

    segments = segmenter.segment(chunks)
    assert len(segments) == 2
    assert segments[0].text == "That solves the latency bottleneck."
    assert segments[1].text == "Next we configure Firestore caching."
    assert segments[0].start_time == 10.0
    assert 13.0 <= segments[0].end_time <= 17.0
    assert segments[1].start_time == segments[0].end_time
    assert segments[1].end_time == 20.0


def test_pedagogical_guardrail_splits_long_unpunctuated_autocaptions() -> None:
    segmenter = CoherentSegmenter()
    chunks = [
        CleanedChunk(
            text="so today we are going to talk about distributed systems",
            start=0.0,
            end=4.0,
        ),
        CleanedChunk(
            text="and how caching layers reduce database load dramatically",
            start=4.0,
            end=8.0,
        ),
        CleanedChunk(
            text="because otherwise every single user request hits the primary node",
            start=8.0,
            end=12.5,
        ),
        CleanedChunk(
            text="and causes cascading timeouts across the entire cluster",
            start=12.5,
            end=16.5,
        ),
    ]

    segments = segmenter.segment(chunks)
    assert len(segments) >= 2
    for seg in segments:
        assert seg.word_count <= 26
        assert seg.text[0].isupper()

