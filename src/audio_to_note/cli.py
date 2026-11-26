"""Command-line interface for audio-to-note."""

from __future__ import annotations

import json
import socket
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.markup import escape as rich_escape
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .config import (
    LLM_PRESETS,
    find_project_root,
    load_config,
    resolve_hotwords_path,
    resolve_path,
    save_user_config,
)
from .discovery import discover_environment
from .llm import get_llm_provider
from .models import Transcript
from .pipeline import JobManager, PipelineOptions, run_pipeline
from .prompts import list_prompts

console = Console()


def _rich_event_handler(event: dict[str, Any]) -> None:
    etype = event.get("type")
    if etype == "stage_start":
        console.print(f"[bold cyan]▶ {event.get('stage', '').upper()}[/]: {event.get('message', '')}")
    elif etype == "chunk_progress":
        console.print(f"  [dim]• {event.get('message', '')}[/]")
    elif etype == "stage_done":
        console.print(f"[green]✓[/] {event.get('message', '')}")
    elif etype == "job_done":
        console.print("[bold green]✓ Pipeline completed[/]")
    elif etype == "job_error":
        console.print(f"[bold red]✗ Error:[/] {event.get('message', '')}")


def _load_runtime(config_path: Path | None) -> tuple[Path, dict[str, Any], JobManager]:
    root = find_project_root()
    cfg = load_config(config_path, project_root=root)
    out_dir = resolve_path(root, cfg.get("pipeline", {}).get("outputs_dir", "outputs")) or (root / "outputs")
    return root, cfg, JobManager(out_dir)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="audio-to-note")
def main() -> None:
    """audio-to-note: Audio/Video -> Transcript -> Diarization -> AI Notes."""


@main.command()
@click.argument("audio", type=click.Path(exists=True, path_type=Path))
@click.option("--asr", "asr_provider", default=None, help="ASR provider (crispasr, audio_cpp, vibeasr_cpp, openai_compat, faster_whisper, command)")
@click.option("--diarize", "diarizer_provider", default=None, help="Diarizer provider (none, sherpa_onnx, pyannote, command)")
@click.option("--num-speakers", type=int, default=None, help="Optional speaker count hint for diarization")
@click.option("--polish/--no-polish", default=None, help="Enable/disable LLM transcript polish pass")
@click.option("--summarize/--no-summarize", default=True, help="Enable/disable LLM summarization")
@click.option("--llm", "llm_provider", default=None, help="LLM provider (openai_compat, llama_cli)")
@click.option("--model", "llm_model", default=None, help="Override LLM model name")
@click.option("--prompt", "prompt_id", default=None, help="Prompt preset ID (meeting_minutes, research_discussion, lecture_notes, ...)")
@click.option("--hotwords", "hotwords_file", type=click.Path(path_type=Path), default=None, help="Path to hotwords file")
@click.option("--language", default=None, help="ASR language code or 'auto'")
@click.option("--output-language", default=None, help="Summary output language or 'auto'")
@click.option("--title", default="", help="Recording title")
@click.option("--output-dir", type=click.Path(path_type=Path), default=None, help="Override outputs root directory")
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=None, help="Path to config.yaml")
def run(
    audio: Path,
    asr_provider: str | None,
    diarizer_provider: str | None,
    num_speakers: int | None,
    polish: bool | None,
    summarize: bool,
    llm_provider: str | None,
    llm_model: str | None,
    prompt_id: str | None,
    hotwords_file: Path | None,
    language: str | None,
    output_language: str | None,
    title: str,
    output_dir: Path | None,
    config_path: Path | None,
) -> None:
    """Run the full pipeline: audio -> ASR -> diarization -> polish -> summary."""
    root, cfg, jm = _load_runtime(config_path)
    if output_dir is not None:
        jm = JobManager(output_dir.expanduser().resolve())

    opts = PipelineOptions(
        title=title or audio.stem,
        run_asr=True,
        asr_provider=asr_provider,
        asr_language=language,
        diarizer_provider=diarizer_provider,
        num_speakers=num_speakers,
        hotwords_file=str(hotwords_file) if hotwords_file else None,
        run_polish=polish,
        run_summarize=summarize,
        llm_provider=llm_provider,
        llm_model=llm_model,
        prompt_id=prompt_id,
        output_language=output_language,
    )
    record = run_pipeline(audio.expanduser().resolve(), opts, cfg, root, jm, on_event=_rich_event_handler)
    jdir = jm.job_dir(record.id)
    if record.status == "partial":
        console.print(
            f"[yellow]Note:[/] Transcript succeeded and was saved, but LLM summarization failed ({record.error}).\n"
            f"Run later: [bold]a2n summarize {record.id}[/]"
        )
    console.print(Panel.fit(f"Job ID: [bold]{record.id}[/]\nDirectory: {jdir}", title="Done"))


@main.command()
@click.argument("audio", type=click.Path(exists=True, path_type=Path))
@click.option("--asr", "asr_provider", default=None, help="ASR provider")
@click.option("--diarize", "diarizer_provider", default=None, help="Diarizer provider (none, sherpa_onnx, pyannote)")
@click.option("--num-speakers", type=int, default=None, help="Optional speaker count hint")
@click.option("--hotwords", "hotwords_file", type=click.Path(path_type=Path), default=None, help="Path to hotwords file")
@click.option("--language", default=None, help="ASR language code or 'auto'")
@click.option("--output-dir", type=click.Path(path_type=Path), default=None, help="Override outputs root directory")
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=None, help="Path to config.yaml")
def transcribe(
    audio: Path,
    asr_provider: str | None,
    diarizer_provider: str | None,
    num_speakers: int | None,
    hotwords_file: Path | None,
    language: str | None,
    output_dir: Path | None,
    config_path: Path | None,
) -> None:
    """Transcribe audio/video (Stages 1-3 only, skips LLM summarization)."""
    root, cfg, jm = _load_runtime(config_path)
    if output_dir is not None:
        jm = JobManager(output_dir.expanduser().resolve())
    opts = PipelineOptions(
        title=audio.stem,
        run_asr=True,
        asr_provider=asr_provider,
        asr_language=language,
        diarizer_provider=diarizer_provider,
        num_speakers=num_speakers,
        hotwords_file=str(hotwords_file) if hotwords_file else None,
        run_polish=False,
        run_summarize=False,
    )
    record = run_pipeline(audio.expanduser().resolve(), opts, cfg, root, jm, on_event=_rich_event_handler)
    console.print(f"Transcript saved to: {jm.job_dir(record.id) / 'transcript.md'}")


@main.command()
@click.argument("target", type=str)
@click.option("--polish/--no-polish", default=None, help="Run transcript polish before summarizing")
@click.option("--llm", "llm_provider", default=None, help="LLM provider")
@click.option("--model", "llm_model", default=None, help="Override LLM model")
@click.option("--prompt", "prompt_id", default=None, help="Prompt preset ID")
@click.option("--hotwords", "hotwords_file", type=click.Path(path_type=Path), default=None, help="Path to hotwords file")
@click.option("--output-language", default=None, help="Summary language or 'auto'")
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=None, help="Path to config.yaml")
def summarize(
    target: str,
    polish: bool | None,
    llm_provider: str | None,
    llm_model: str | None,
    prompt_id: str | None,
    hotwords_file: Path | None,
    output_language: str | None,
    config_path: Path | None,
) -> None:
    """Summarize an existing job ID, job directory, or transcript (.json / .md)."""
    root, cfg, jm = _load_runtime(config_path)
    target_path = Path(target).expanduser()
    existing_job_id: str | None = None
    existing_transcript: Transcript | None = None
    title = ""

    if (jm.outputs_dir / target / "job.json").exists():
        existing_job_id = target
    elif target_path.is_dir() and (target_path / "job.json").exists():
        jm = JobManager(target_path.parent)
        existing_job_id = target_path.name
    elif target_path.is_file():
        title = target_path.stem
        if target_path.suffix.lower() == ".json":
            existing_transcript = Transcript.from_dict(json.loads(target_path.read_text(encoding="utf-8")))
        else:
            existing_transcript = Transcript.from_markdown(target_path.read_text(encoding="utf-8"))
    else:
        raise click.ClickException(f"Target not found as job ID or file: {target}")

    opts = PipelineOptions(
        title=title,
        run_asr=False,
        hotwords_file=str(hotwords_file) if hotwords_file else None,
        run_polish=polish,
        run_summarize=True,
        llm_provider=llm_provider,
        llm_model=llm_model,
        prompt_id=prompt_id,
        output_language=output_language,
    )
    record = run_pipeline(
        None,
        opts,
        cfg,
        root,
        jm,
        existing_job_id=existing_job_id,
        existing_transcript=existing_transcript,
        on_event=_rich_event_handler,
    )
    console.print(f"Summary saved to: {jm.job_dir(record.id) / 'summary.md'}")


def _detect_local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return str(sock.getsockname()[0])
    except OSError:
        return ""


def _open_when_ready(url: str, port: int) -> None:
    def _wait_and_open() -> None:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    webbrowser.open(url)
                    return
            except OSError:
                time.sleep(0.1)

    threading.Thread(target=_wait_and_open, daemon=True).start()


@main.command()
@click.option("--host", default="0.0.0.0", show_default=True, help="Bind host")
@click.option("--port", default=8000, show_default=True, type=int, help="Bind port")
@click.option("--open/--no-open", "open_browser", default=False, help="Open browser automatically")
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=None, help="Path to config.yaml")
def serve(host: str, port: int, open_browser: bool, config_path: Path | None) -> None:
    """Start the local Web GUI and REST/SSE API server."""
    import uvicorn

    from .web.app import create_app

    root = find_project_root()
    app = create_app(project_root=root, config_path=config_path)
    local_url = f"http://127.0.0.1:{port}"
    lines = [
        "[bold green]audio-to-note Web GUI[/]",
        f"Local URL: [link={local_url}]{local_url}[/link]",
    ]
    if host == "0.0.0.0":
        lan_ip = _detect_local_ip()
        if lan_ip and lan_ip != "127.0.0.1":
            wsl_url = f"http://{lan_ip}:{port}"
            lines.append(f"WSL / LAN URL: [link={wsl_url}]{wsl_url}[/link]")
    elif host != "127.0.0.1":
        bind_url = f"http://{host}:{port}"
        lines.append(f"Bind URL: [link={bind_url}]{bind_url}[/link]")
    lines.append(f"Project root: {root}")
    console.print(Panel.fit("\n".join(lines), title="Server"))
    if open_browser:
        _open_when_ready(local_url, port)
    uvicorn.run(app, host=host, port=port, log_level="info")


@main.command()
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=None, help="Path to config.yaml")
def doctor(config_path: Path | None) -> None:
    """Check FFmpeg, GPU, local models, ASR/diarizer/LLM readiness, and prompts."""
    root, cfg, _ = _load_runtime(config_path)
    env = discover_environment(root, cfg)

    table = Table(title="audio-to-note Environment Check", show_lines=True)
    table.add_column("Category", style="cyan", no_wrap=True)
    table.add_column("Component", style="bold")
    table.add_column("Status", no_wrap=True)
    table.add_column("Detail / Hint")

    for section in ("tools", "asr", "diarization", "llm"):
        for item in env[section]:
            badge = "[green]READY[/]" if item["available"] else f"[yellow]{item['status'].upper()}[/]"
            detail = rich_escape(item["detail"])
            if item.get("hint"):
                detail += f"\n[dim]Hint: {rich_escape(item['hint'])}[/]"
            table.add_row(section.upper(), item["name"], badge, detail)

    console.print(table)
    prompts = list_prompts(root)
    hw_path = resolve_hotwords_path(root, cfg.get("pipeline", {}).get("hotwords_file"))
    console.print(f"[bold]Prompts available:[/] {', '.join(p.id for p in prompts)}")
    console.print(f"[bold]Active hotwords file:[/] {hw_path or 'none'}")

    asr_ready = any(item["available"] for item in env["asr"])
    stage1_txt = "[green]READY[/]" if asr_ready else "[yellow]NEEDS_SETUP[/]"
    console.print(f"[bold]Stage 1-3 (Transcription & Diarization):[/] {stage1_txt}")

    llm_id = cfg.get("llm", {}).get("provider", "openai_compat")
    if llm_id == "openai_compat":
        test_res = get_llm_provider("openai_compat", root, cfg).test_connection()
        status_txt = "[green]READY[/]" if test_res["ok"] else "[yellow]OFFLINE[/]"
        console.print(
            f"[bold]Stage 4-5 (LLM Summarization, optional):[/] {status_txt} — {test_res['detail']} "
            "[dim](does not affect transcription)[/]"
        )


@main.command()
def init() -> None:
    """Initialize local gitignored config/config.yaml, .env, and hotwords.local.txt."""
    root = find_project_root()
    cfg = load_config(project_root=root, expand_env=False)
    env = discover_environment(root, load_config(project_root=root, expand_env=True))

    ready_asr = [item["id"] for item in env["asr"] if item["available"]]
    default_asr = ready_asr[0] if ready_asr else "openai_compat"
    asr_choice = click.prompt(
        f"Default ASR provider (detected ready: {ready_asr or ['none']})",
        default=default_asr,
    )
    preset_choice = click.prompt(
        f"LLM preset ({', '.join(LLM_PRESETS.keys())})",
        default="ollama",
    )
    preset = LLM_PRESETS.get(preset_choice, LLM_PRESETS["ollama"])
    base_url = click.prompt("LLM OpenAI-compatible Base URL", default=preset["base_url"])
    model_name = click.prompt("LLM Model name", default=preset["default_model"] or "qwen2.5:7b")

    cfg["asr"]["provider"] = asr_choice
    cfg["llm"]["preset"] = preset_choice
    cfg["llm"]["model"] = model_name
    cfg["llm"]["providers"]["openai_compat"]["base_url"] = base_url

    target_cfg = save_user_config(cfg, root / "config" / "config.yaml")
    hotwords_local = root / "config" / "hotwords.local.txt"
    if not hotwords_local.exists():
        example_hw = root / "config" / "hotwords.example.txt"
        if example_hw.exists():
            hotwords_local.write_text(example_hw.read_text(encoding="utf-8"), encoding="utf-8")

    env_file = root / ".env"
    if not env_file.exists():
        example_env = root / ".env.example"
        if example_env.exists():
            env_file.write_text(example_env.read_text(encoding="utf-8"), encoding="utf-8")

    console.print(f"[green]✓[/] Saved local config to {target_cfg}")
    console.print(f"[green]✓[/] Active hotwords file: {hotwords_local}")
    console.print("Next: run [bold]audio-to-note serve[/] to open the Web GUI, or [bold]audio-to-note doctor[/].")


if __name__ == "__main__":
    main()
