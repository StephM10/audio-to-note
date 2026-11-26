import json
import time
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from audio_to_note.asr import ASRResult
from audio_to_note.llm import LLMResponse
from audio_to_note.models import Segment
from audio_to_note.web.app import create_app


def test_web_api_endpoints_end_to_end(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname="audio-to-note"\n', encoding="utf-8")
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "hotwords.local.txt").write_text("# comment\nVibeVoice\n", encoding="utf-8")

    app = create_app(project_root=tmp_path)
    client = TestClient(app)

    # Index & environment
    r_idx = client.get("/")
    assert r_idx.status_code == 200
    assert "audio-to-note" in r_idx.text

    r_env = client.get("/api/environment")
    assert r_env.status_code == 200
    env = r_env.json()
    assert any(item["id"] == "crispasr" for item in env["asr"])
    assert any(item["id"] == "sherpa_onnx" for item in env["diarization"])

    # Hotwords CRUD
    r_hw = client.get("/api/hotwords")
    assert r_hw.status_code == 200
    assert "VibeVoice" in r_hw.json()["text"]
    r_hw_put = client.put("/api/hotwords", json={"text": "VibeVoice\nPyannote\n"})
    assert r_hw_put.status_code == 200

    # Prompts CRUD
    r_pr = client.get("/api/prompts")
    assert r_pr.status_code == 200
    assert len(r_pr.json()) >= 4

    # Submit uploaded audio job (mocked ASR & LLM)
    class FakeASR:
        id = "fake"
        label = "Fake"
        sample_rate = 24000
        long_audio_settings = {"single_pass_seconds": 3300, "chunk_seconds": 3000, "overlap_seconds": 45}

        def transcribe_chunk(self, wav, workdir, **kwargs):
            return ASRResult(segments=[Segment(0.0, 3.0, "SPEAKER_0", "API test segment")], raw_text="API test segment")

    class FakeLLM:
        id = "fake"
        label = "Fake"

        def complete(self, sys_p, usr_p, **kwargs):
            return LLMResponse(text="## Summary\n- API test summary", model="mock-m")

    with (
        patch("audio_to_note.pipeline.probe_duration", return_value=3.0),
        patch("audio_to_note.pipeline.extract_wav", return_value=tmp_path / "x.wav"),
        patch("audio_to_note.pipeline.get_asr_provider", return_value=FakeASR()),
        patch("audio_to_note.pipeline.get_llm_provider", return_value=FakeLLM()),
    ):
        r_job = client.post(
            "/api/jobs",
            data={"options": json.dumps({"title": "API Job", "asr_provider": "crispasr", "llm_model": "mock-m"})},
            files={"file": ("clip.wav", b"RIFF0000WAVE", "audio/wav")},
        )
        assert r_job.status_code == 200
        job_id = r_job.json()["job_id"]

        for _ in range(30):
            detail = client.get(f"/api/jobs/{job_id}").json()
            if detail["status"] in ("completed", "failed"):
                break
            time.sleep(0.05)

    assert detail["status"] == "completed"
    assert "API test segment" in detail["content"]["transcript_md"]
    assert "API test summary" in detail["content"]["summary_md"]

    # Speaker alias update
    r_spk = client.put(f"/api/jobs/{job_id}/speakers", json={"aliases": {"SPEAKER_0": "Dr. Lin"}})
    assert r_spk.status_code == 200
    assert "Dr. Lin" in r_spk.json()["transcript_md"]

    # Per-segment speaker & text editing
    r_seg_edit = client.put(
        f"/api/jobs/{job_id}/transcript",
        json={
            "target": "raw",
            "segments": [{"start": 0.0, "end": 3.0, "speaker": "SPEAKER_1", "text": "Edited segment text"}],
        },
    )
    assert r_seg_edit.status_code == 200
    assert r_seg_edit.json()["speakers"] == ["SPEAKER_1"]
    assert "Edited segment text" in r_seg_edit.json()["transcript_md"]

    # Markdown transcript editing
    r_md_edit = client.put(
        f"/api/jobs/{job_id}/transcript",
        json={
            "target": "raw",
            "markdown": "# Meeting\n\n**[00:00:05] Alice**: Markdown edited line\n",
        },
    )
    assert r_md_edit.status_code == 200
    assert r_md_edit.json()["speakers"] == ["Alice"]
    persisted = client.get(f"/api/jobs/{job_id}").json()
    assert persisted["segments"][0]["text"] == "Markdown edited line"

    # Export download
    r_dl = client.get(f"/api/jobs/{job_id}/export/summary.md")
    assert r_dl.status_code == 200
    assert "API test summary" in r_dl.text
