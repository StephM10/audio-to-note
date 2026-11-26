import json
import subprocess
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from audio_to_note.asr import get_asr_provider
from audio_to_note.config import load_config
from audio_to_note.hotwords import parse_hotwords
from audio_to_note.models import Segment


def _make_wav(path: Path, seconds: float = 1.0, rate: int = 24000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\0\0" * int(rate * seconds))
    return path


def test_crispasr_runner_parses_structured_json_and_strips_hotword_comments(tmp_path: Path):
    bin_path = tmp_path / "crispasr"
    bin_path.write_text("#!/bin/sh\n", encoding="utf-8")
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")

    cfg = load_config(project_root=tmp_path)
    cfg["asr"]["providers"]["crispasr"]["runtime_bin"] = str(bin_path)
    cfg["asr"]["providers"]["crispasr"]["model"] = str(model_path)
    provider = get_asr_provider("crispasr", tmp_path, cfg)

    def fake_run(command, **kwargs):
        output_base = Path(command[command.index("--output-file") + 1])
        output_base.with_suffix(".json").write_text(
            json.dumps(
                {
                    "transcription": [
                        {
                            "timestamps": {"from": "00:01:02,500", "to": "00:01:05,250"},
                            "offsets": {"from": 62500, "to": 65250},
                            "speaker": "(Speaker 3) ",
                            "text": "structured hello",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    hw = parse_hotwords("# ignore me\nVibeVoice\n")
    wav = _make_wav(tmp_path / "clip.wav")
    with patch("audio_to_note.asr.crispasr.subprocess.run", side_effect=fake_run):
        res = provider.transcribe_chunk(wav, tmp_path / "work", hotwords=hw)

    assert res.segments == [Segment(62.5, 65.25, "SPEAKER_3", "structured hello")]
    ctx_idx = res.command.index("--context")
    assert res.command[ctx_idx + 1] == "VibeVoice"


def test_audio_cpp_normalizes_sample_indices_and_turns(tmp_path: Path):
    bin_path = tmp_path / "audiocpp_cli"
    bin_path.write_text("#!/bin/sh\n", encoding="utf-8")
    model_path = tmp_path / "q8.gguf"
    model_path.write_bytes(b"gguf")

    cfg = load_config(project_root=tmp_path)
    cfg["asr"]["providers"]["audio_cpp"]["runtime_bin"] = str(bin_path)
    cfg["asr"]["providers"]["audio_cpp"]["model"] = str(model_path)
    provider = get_asr_provider("audio_cpp", tmp_path, cfg)

    workdir = tmp_path / "work"
    workdir.mkdir()
    (workdir / "text.txt").write_text("hello\n", encoding="utf-8")
    (workdir / "segments.json").write_text(
        '[{"start_sample": 0, "end_sample": 24000, "speaker_id": "0", "text": "hello"}]',
        encoding="utf-8",
    )
    (workdir / "turns.json").write_text(
        '[{"start_sample": 0, "end_sample": 24000, "speaker_id": "1", "text": ""}]',
        encoding="utf-8",
    )

    with patch("audio_to_note.asr.audio_cpp.subprocess.run") as run_mock:
        run_mock.return_value.stdout = ""
        res = provider.transcribe_chunk(_make_wav(tmp_path / "clip.wav"), workdir)

    assert res.segments == [Segment(0.0, 1.0, "SPEAKER_0", "hello")]
    assert res.turns[0].speaker == "SPEAKER_1"
    assert "--session-option" in res.command


def test_vibeasr_cpp_wraps_plain_stdout_with_wav_duration(tmp_path: Path):
    bin_path = tmp_path / "asr_infer"
    bin_path.write_text("#!/bin/sh\n", encoding="utf-8")
    vae = tmp_path / "vae.gguf"
    lm = tmp_path / "lm.gguf"
    vae.write_bytes(b"v")
    lm.write_bytes(b"l")

    cfg = load_config(project_root=tmp_path)
    cfg["asr"]["providers"]["vibeasr_cpp"]["runtime_bin"] = str(bin_path)
    cfg["asr"]["providers"]["vibeasr_cpp"]["vae_model"] = str(vae)
    cfg["asr"]["providers"]["vibeasr_cpp"]["lm_model"] = str(lm)
    provider = get_asr_provider("vibeasr_cpp", tmp_path, cfg)

    wav = _make_wav(tmp_path / "plain.wav", seconds=1.0)
    with patch("audio_to_note.asr.vibeasr_cpp.subprocess.run") as run_mock:
        run_mock.return_value.stdout = "hello plain transcript\n"
        res = provider.transcribe_chunk(wav, tmp_path / "work")

    assert res.segments == [Segment(0.0, 1.0, "SPEAKER_UNKNOWN", "hello plain transcript")]


def test_openai_compat_asr_parses_verbose_json(tmp_path: Path):
    cfg = load_config(project_root=tmp_path)
    cfg["asr"]["providers"]["openai_compat"]["base_url"] = "http://localhost:9999/v1"
    cfg["asr"]["providers"]["openai_compat"]["api_key"] = "test-key"
    provider = get_asr_provider("openai_compat", tmp_path, cfg)
    wav = _make_wav(tmp_path / "clip.wav", rate=16000)

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "text": "Hello world",
        "segments": [{"start": 0.0, "end": 2.5, "text": "Hello world"}],
    }
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.post.return_value = mock_resp

    with patch("audio_to_note.asr.openai_compat.httpx.Client", return_value=mock_client):
        res = provider.transcribe_chunk(
            wav,
            tmp_path / "work",
            language="en",
            hotwords=parse_hotwords("VibeVoice"),
        )

    assert res.segments == [Segment(0.0, 2.5, "SPEAKER_UNKNOWN", "Hello world")]
    call_kwargs = mock_client.post.call_args.kwargs
    assert call_kwargs["data"]["prompt"] == "VibeVoice"


def test_command_asr_provider_runs_template_and_reads_output_json(tmp_path: Path):
    cfg = load_config(project_root=tmp_path)
    cfg["asr"]["providers"]["command"]["command"] = "my-asr --in {audio} --out {output_json}"
    provider = get_asr_provider("command", tmp_path, cfg)
    wav = _make_wav(tmp_path / "clip.wav", rate=16000)

    def fake_run(cmd, **kwargs):
        out_path = Path(cmd[cmd.index("--out") + 1])
        out_path.write_text(json.dumps([{"start": 0.5, "end": 2.0, "speaker": "0", "text": "hi"}]), encoding="utf-8")
        return SimpleNamespace(stdout="", stderr="")

    with patch("audio_to_note.asr.command.subprocess.run", side_effect=fake_run):
        res = provider.transcribe_chunk(wav, tmp_path / "work")

    assert res.segments == [Segment(0.5, 2.0, "SPEAKER_0", "hi")]
