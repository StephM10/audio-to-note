from pathlib import Path

from audio_to_note.audio import plan_chunks
from audio_to_note.config import load_config
from audio_to_note.hotwords import format_for_asr, format_for_llm, parse_hotwords
from audio_to_note.merge import align_speakers, merge_chunk_segments, normalize_segment
from audio_to_note.models import Segment, SpeakerTurn, Transcript


def test_short_recording_stays_single_pass():
    assert plan_chunks(2815.3) == [(0.0, 2815.3)]


def test_long_recording_uses_bounded_overlapping_chunks():
    chunks = plan_chunks(7500.0, chunk_seconds=3000.0, overlap_seconds=45.0)
    assert chunks == [(0.0, 3000.0), (2955.0, 5955.0), (5910.0, 7500.0)]
    assert all(end - start <= 3000.0 for start, end in chunks)
    assert all(left[1] >= right[0] for left, right in zip(chunks, chunks[1:]))


def test_normalize_segment_accepts_runtime_key_variants():
    segment = normalize_segment({"Start": 1.25, "End": 3.5, "Speaker": 2, "Content": "  hello world  "})
    assert segment == Segment(1.25, 3.5, "SPEAKER_2", "hello world")


def test_merge_deduplicates_overlap_and_reconciles_speakers():
    merged = merge_chunk_segments(
        [
            (0.0, [{"Start": 0, "End": 4, "Speaker": 0, "Content": "hello"}, {"Start": 4, "End": 8, "Speaker": 1, "Content": "world"}]),
            (6.0, [{"Start": 0, "End": 3, "Speaker": 0, "Content": "world"}, {"Start": 3, "End": 7, "Speaker": 1, "Content": "next"}]),
        ]
    )
    assert [(s.start, s.end, s.speaker, s.text) for s in merged] == [
        (0.0, 4.0, "SPEAKER_0", "hello"),
        (4.0, 8.0, "SPEAKER_1", "world"),
        (9.0, 13.0, "SPEAKER_2", "next"),
    ]


def test_merge_trims_plain_transcript_overlap_at_chunk_boundary():
    merged = merge_chunk_segments(
        [
            (0.0, [Segment(0.0, 120.0, "SPEAKER_0", "alpha beta gamma")]),
            (105.0, [Segment(0.0, 120.0, "SPEAKER_0", "beta gamma delta")]),
        ]
    )
    assert [s.text for s in merged] == ["alpha beta gamma", "delta"]


def test_merge_trims_cjk_overlap_at_chunk_boundary():
    merged = merge_chunk_segments(
        [
            (0.0, [Segment(0.0, 60.0, "SPEAKER_0", "我们今天讨论模型量化方案和推理速度")]),
            (45.0, [Segment(0.0, 60.0, "SPEAKER_0", "量化方案和推理速度，接下来看显存占用")]),
        ]
    )
    assert [s.text for s in merged] == ["我们今天讨论模型量化方案和推理速度", "接下来看显存占用"]


def test_hotwords_ignores_comments_and_parses_aliases():
    raw = """
    # This comment must be ignored
    VibeVoice

    cohesion -> cohesin: protein complex
    PBC: periodic boundary conditions
    """
    entries = parse_hotwords(raw)
    assert [e.term for e in entries] == ["VibeVoice", "cohesin", "PBC"]
    assert entries[1].aliases == ["cohesion"]
    assert entries[1].note == "protein complex"
    asr_str = format_for_asr(entries)
    assert "comment" not in asr_str
    assert asr_str == "VibeVoice, cohesin, PBC"
    llm_str = format_for_llm(entries)
    assert "cohesion" in llm_str and "protein complex" in llm_str


def test_align_speakers_uses_max_overlap_duration():
    segments = [
        Segment(0.0, 5.0, "SPEAKER_UNKNOWN", "first part"),
        Segment(5.0, 10.0, "SPEAKER_UNKNOWN", "second part"),
    ]
    turns = [
        SpeakerTurn(0.0, 4.5, "SPEAKER_1"),
        SpeakerTurn(4.5, 11.0, "SPEAKER_0"),
    ]
    aligned = align_speakers(segments, turns)
    assert [s.speaker for s in aligned] == ["SPEAKER_1", "SPEAKER_0"]


def test_transcript_roundtrip_markdown_and_aliases():
    tr = Transcript(
        segments=[Segment(1.0, 4.0, "SPEAKER_0", "Hello team")],
        speaker_aliases={"SPEAKER_0": "Alice"},
    )
    md = tr.to_markdown(title="Sync")
    assert "Alice" in md
    parsed = Transcript.from_markdown(md)
    assert parsed.segments[0].speaker == "Alice"
    assert parsed.segments[0].text == "Hello team"


def test_load_default_config_expands_structure(tmp_path: Path):
    cfg = load_config(project_root=tmp_path)
    assert "asr" in cfg and "diarization" in cfg and "llm" in cfg
