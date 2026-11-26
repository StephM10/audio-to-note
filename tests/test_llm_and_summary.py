from pathlib import Path
from unittest.mock import MagicMock, patch

from audio_to_note.config import load_config
from audio_to_note.hotwords import parse_hotwords
from audio_to_note.llm import LLMResponse, get_llm_provider
from audio_to_note.models import Segment, Transcript
from audio_to_note.prompts import PromptTemplate, get_prompt, list_prompts, save_prompt
from audio_to_note.summarizer import polish_transcript, summarize_transcript


def test_builtin_prompts_discoverable_and_custom_save(tmp_path: Path):
    prompts = list_prompts(tmp_path)
    ids = {p.id for p in prompts}
    assert {"meeting_minutes", "research_discussion", "lecture_notes", "transcript_polish"} <= ids

    custom = PromptTemplate(
        id="weekly_sync",
        name="Weekly Sync",
        description="Custom",
        stage="summarize",
        system_prompt="Sys {output_language_instruction}",
        user_prompt="Body: {transcript}",
    )
    save_prompt(tmp_path, custom)
    loaded = get_prompt(tmp_path, "weekly_sync")
    assert loaded.name == "Weekly Sync"
    assert not loaded.builtin


def test_openai_compat_llm_non_streaming_and_streaming(tmp_path: Path):
    cfg = load_config(project_root=tmp_path)
    cfg["llm"]["model"] = "test-model"
    llm = get_llm_provider("openai_compat", tmp_path, cfg)

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "model": "test-model",
        "choices": [{"message": {"content": "Summary body"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.post.return_value = mock_resp

    with patch("audio_to_note.llm.openai_compat.httpx.Client", return_value=mock_client):
        res = llm.complete("sys", "user")
    assert res.text == "Summary body"
    assert res.prompt_tokens == 10

    # Streaming SSE
    stream_resp = MagicMock()
    stream_resp.__enter__.return_value = stream_resp
    stream_resp.iter_lines.return_value = [
        'data: {"choices":[{"delta":{"content":"Hello "}}]}',
        'data: {"choices":[{"delta":{"content":"world"}}]}',
        "data: [DONE]",
    ]
    mock_client.stream.return_value = stream_resp
    tokens = []
    with patch("audio_to_note.llm.openai_compat.httpx.Client", return_value=mock_client):
        res_stream = llm.complete("sys", "user", on_token=tokens.append)
    assert res_stream.text == "Hello world"
    assert tokens == ["Hello ", "world"]


def test_polish_and_map_reduce_summarization(tmp_path: Path):
    class FakeLLM:
        id = "fake"
        label = "Fake"

        def __init__(self):
            self.calls = []

        def complete(self, system_prompt, user_prompt, **kwargs):
            self.calls.append((system_prompt, user_prompt))
            if "Clean up speech-to-text" in system_prompt:
                return LLMResponse(text="[00:00:00] SPEAKER_0: Cleaned text.", model="fake")
            return LLMResponse(text="- Bullet summary", model="fake")

    fake = FakeLLM()
    tr = Transcript(segments=[Segment(0.0, 5.0, "SPEAKER_0", "uh raw text")] * 40, duration_seconds=200.0)
    polish_tpl = get_prompt(tmp_path, "transcript_polish")
    polished, p_meta = polish_transcript(tr, fake, polish_tpl, parse_hotwords("VibeVoice"))
    assert polished.segments[0].text == "Cleaned text."
    assert p_meta["chunks"] >= 1

    # Force map-reduce with small max_context_tokens
    long_text = "\n".join(f"[00:0{i % 9}:00] SPEAKER_0: Discussion line number {i} with technical content." for i in range(120))
    sum_tpl = get_prompt(tmp_path, "meeting_minutes")
    summary, s_meta = summarize_transcript(
        long_text,
        fake,
        sum_tpl,
        [],
        max_tokens=500,
        max_context_tokens=2000,
    )
    assert s_meta["mode"] == "map_reduce"
    assert s_meta["map_chunks"] >= 2
    assert summary == "- Bullet summary"
