import json
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from audio_to_note.asr import ASRResult
from audio_to_note.cli import main as cli_main
from audio_to_note.config import load_config
from audio_to_note.llm import LLMResponse
from audio_to_note.models import Segment, SpeakerTurn
from audio_to_note.pipeline import JobManager, PipelineOptions, run_pipeline


def test_full_pipeline_and_speaker_rename_and_resummarize(tmp_path: Path):
    audio = tmp_path / "meeting.mp4"
    audio.write_bytes(b"fake-media")
    cfg = load_config(project_root=tmp_path)
    cfg["pipeline"]["outputs_dir"] = str(tmp_path / "outputs")
    jm = JobManager(tmp_path / "outputs")

    class FakeASR:
        id = "fake_asr"
        label = "FakeASR"
        sample_rate = 24000
        long_audio_settings = {"single_pass_seconds": 3300, "chunk_seconds": 3000, "overlap_seconds": 45}

        def transcribe_chunk(self, wav, workdir, **kwargs):
            return ASRResult(segments=[Segment(0.0, 5.0, "SPEAKER_0", "Hello team")], raw_text="Hello team")

    class FakeDiarizer:
        id = "fake_diar"
        label = "FakeDiar"

        def diarize(self, wav, workdir, **kwargs):
            return [SpeakerTurn(0.0, 5.0, "SPEAKER_1")]

    class FakeLLM:
        id = "fake_llm"
        label = "FakeLLM"

        def complete(self, sys_p, usr_p, **kwargs):
            return LLMResponse(text="## Executive Summary\n- Team greeted.", model="fake-m")

    with (
        patch("audio_to_note.pipeline.probe_duration", return_value=5.0),
        patch("audio_to_note.pipeline.extract_wav", return_value=tmp_path / "x.wav"),
        patch("audio_to_note.pipeline.get_asr_provider", return_value=FakeASR()),
        patch("audio_to_note.pipeline.get_diarizer_provider", return_value=FakeDiarizer()),
        patch("audio_to_note.pipeline.get_llm_provider", return_value=FakeLLM()),
    ):
        opts = PipelineOptions(
            title="Weekly Sync",
            asr_provider="crispasr",
            diarizer_provider="sherpa_onnx",
            run_polish=False,
            run_summarize=True,
            llm_model="fake-m",
        )
        rec = run_pipeline(audio, opts, cfg, tmp_path, jm)

    assert rec.status == "completed"
    jdir = jm.job_dir(rec.id)
    assert (jdir / "transcript.json").exists()
    assert (jdir / "summary.md").exists()
    assert "SPEAKER_1" in (jdir / "transcript.md").read_text(encoding="utf-8")

    # Rename SPEAKER_1 -> Bob
    jm.update_speaker_aliases(rec.id, {"SPEAKER_1": "Bob"})
    assert "Bob" in (jdir / "transcript.md").read_text(encoding="utf-8")

    # Stage 5 re-summarize without re-running ASR
    with patch("audio_to_note.pipeline.get_llm_provider", return_value=FakeLLM()):
        rec2 = run_pipeline(
            None,
            PipelineOptions(run_asr=False, run_summarize=True, prompt_id="research_discussion", llm_model="fake-m"),
            cfg,
            tmp_path,
            jm,
            existing_job_id=rec.id,
        )
    assert rec2.status == "completed"


def test_cli_doctor_and_summarize_standalone_transcript(tmp_path: Path):
    runner = CliRunner()
    res_help = runner.invoke(cli_main, ["--help"])
    assert res_help.exit_code == 0
    assert "doctor" in res_help.output

    res_doc = runner.invoke(cli_main, ["doctor"])
    assert res_doc.exit_code == 0
    assert "audio-to-note" in res_doc.output

    tr_file = tmp_path / "sample.json"
    tr_file.write_text(
        json.dumps({"segments": [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_0", "text": "Test note."}]}),
        encoding="utf-8",
    )

    class FakeLLM:
        id = "fake"
        label = "Fake"

        def complete(self, sys_p, usr_p, **kwargs):
            return LLMResponse(text="## Summary\n- Test note.", model="m")

    with (
        patch("audio_to_note.cli.find_project_root", return_value=tmp_path),
        patch("audio_to_note.pipeline.get_llm_provider", return_value=FakeLLM()),
    ):
        res_sum = runner.invoke(cli_main, ["summarize", str(tr_file), "--model", "m"])
    assert res_sum.exit_code == 0
    assert "Summary saved to" in res_sum.output


def test_pipeline_preserves_transcript_when_llm_offline(tmp_path: Path):
    audio = tmp_path / "meeting.mp4"
    audio.write_bytes(b"fake-media")
    cfg = load_config(project_root=tmp_path)
    cfg["pipeline"]["outputs_dir"] = str(tmp_path / "outputs")
    jm = JobManager(tmp_path / "outputs")

    class FakeASR:
        id = "fake_asr"
        label = "FakeASR"
        sample_rate = 24000
        long_audio_settings = {"single_pass_seconds": 3300, "chunk_seconds": 3000, "overlap_seconds": 45}

        def transcribe_chunk(self, wav, workdir, **kwargs):
            return ASRResult(segments=[Segment(0.0, 4.0, "SPEAKER_0", "Kept safe")], raw_text="Kept safe")

    class OfflineLLM:
        id = "fake_llm"
        label = "OfflineLLM"

        def complete(self, sys_p, usr_p, **kwargs):
            raise ConnectionRefusedError("LLM server offline")

    with (
        patch("audio_to_note.pipeline.probe_duration", return_value=4.0),
        patch("audio_to_note.pipeline.extract_wav", return_value=tmp_path / "x.wav"),
        patch("audio_to_note.pipeline.get_asr_provider", return_value=FakeASR()),
        patch("audio_to_note.pipeline.get_llm_provider", return_value=OfflineLLM()),
    ):
        rec = run_pipeline(
            audio,
            PipelineOptions(title="Offline Test", asr_provider="crispasr", diarizer_provider="none", run_summarize=True),
            cfg,
            tmp_path,
            jm,
        )

    assert rec.status == "partial"
    assert "LLM server offline" in rec.error
    jdir = jm.job_dir(rec.id)
    assert (jdir / "transcript.json").exists()
    assert "Kept safe" in (jdir / "transcript.md").read_text(encoding="utf-8")
