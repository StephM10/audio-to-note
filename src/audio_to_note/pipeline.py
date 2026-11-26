"""End-to-end 5-stage pipeline orchestration and Job workspace persistence."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .asr import get_asr_provider
from .audio import extract_wav, plan_chunks, probe_duration
from .config import resolve_hotwords_path
from .diarize import get_diarizer_provider
from .hotwords import HotwordEntry, load_hotwords, parse_hotwords
from .llm import get_llm_provider
from .merge import align_speakers, merge_chunk_segments
from .models import JobRecord, Segment, Transcript
from .prompts import get_prompt
from .summarizer import format_transcript_for_llm, polish_transcript, summarize_transcript

EventCallback = Callable[[dict[str, Any]], None]


@dataclass
class PipelineOptions:
    title: str = ""
    run_asr: bool = True
    asr_provider: str | None = None
    asr_language: str | None = None
    asr_options: dict[str, Any] = field(default_factory=dict)

    diarizer_provider: str | None = None
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    diarizer_options: dict[str, Any] = field(default_factory=dict)

    hotwords_text: str | None = None
    hotwords_file: str | None = None

    run_polish: bool | None = None
    run_summarize: bool | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_options: dict[str, Any] = field(default_factory=dict)
    prompt_id: str | None = None
    custom_system_prompt: str | None = None
    custom_user_prompt: str | None = None
    output_language: str | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _slugify(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-").lower()
    return cleaned[:48] or "recording"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


class JobManager:
    """Manages persistent job directories under ``outputs/``."""

    def __init__(self, outputs_dir: Path) -> None:
        self.outputs_dir = outputs_dir.resolve()
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    def create_job(self, title: str, input_file: str = "", config_snapshot: dict[str, Any] | None = None) -> JobRecord:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        slug = _slugify(title or Path(input_file).stem or "job")
        job_id = f"{stamp}_{slug}"
        counter = 1
        while (self.outputs_dir / job_id).exists():
            counter += 1
            job_id = f"{stamp}_{slug}_{counter}"
        job_dir = self.outputs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        now = _utc_now()
        record = JobRecord(
            id=job_id,
            title=title or Path(input_file).stem or job_id,
            created_at=now,
            updated_at=now,
            status="queued",
            stage="init",
            progress=0.0,
            input_file=input_file,
            config_snapshot=config_snapshot or {},
        )
        self.save_job(record)
        return record

    def job_dir(self, job_id: str) -> Path:
        safe = Path(job_id).name
        return self.outputs_dir / safe

    def save_job(self, record: JobRecord) -> None:
        record.updated_at = _utc_now()
        path = self.job_dir(record.id) / "job.json"
        _atomic_write_text(path, json.dumps(record.to_dict(), ensure_ascii=False, indent=2) + "\n")

    def get_job(self, job_id: str) -> JobRecord:
        path = self.job_dir(job_id) / "job.json"
        if not path.exists():
            raise FileNotFoundError(f"Job not found: {job_id}")
        return JobRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list_jobs(self) -> list[JobRecord]:
        jobs: list[JobRecord] = []
        if not self.outputs_dir.is_dir():
            return jobs
        for child in self.outputs_dir.iterdir():
            meta = child / "job.json"
            if child.is_dir() and meta.exists():
                try:
                    jobs.append(JobRecord.from_dict(json.loads(meta.read_text(encoding="utf-8"))))
                except Exception:
                    continue
        jobs.sort(key=lambda item: item.created_at, reverse=True)
        return jobs

    def delete_job(self, job_id: str) -> None:
        target = self.job_dir(job_id)
        if target.exists() and target.is_dir():
            shutil.rmtree(target)

    def load_transcript(self, job_id: str, *, prefer_polished: bool = False) -> Transcript:
        jdir = self.job_dir(job_id)
        if prefer_polished and (jdir / "polished.json").exists():
            return Transcript.from_dict(json.loads((jdir / "polished.json").read_text(encoding="utf-8")))
        json_path = jdir / "transcript.json"
        if json_path.exists():
            return Transcript.from_dict(json.loads(json_path.read_text(encoding="utf-8")))
        md_path = jdir / "transcript.md"
        if md_path.exists():
            return Transcript.from_markdown(md_path.read_text(encoding="utf-8"))
        raise FileNotFoundError(f"No transcript found for job {job_id}")

    def update_speaker_aliases(self, job_id: str, aliases: dict[str, str]) -> JobRecord:
        record = self.get_job(job_id)
        clean_aliases = {k: v.strip() for k, v in aliases.items() if v.strip()}
        record.speaker_aliases = clean_aliases
        jdir = self.job_dir(job_id)

        for json_name, md_name in (("transcript.json", "transcript.md"), ("polished.json", "polished.md")):
            jpath = jdir / json_name
            if jpath.exists():
                tr = Transcript.from_dict(json.loads(jpath.read_text(encoding="utf-8")))
                tr.speaker_aliases = clean_aliases
                _atomic_write_text(jpath, json.dumps(tr.to_dict(), ensure_ascii=False, indent=2) + "\n")
                _atomic_write_text(jdir / md_name, tr.to_markdown(title=record.title, apply_aliases=True))

        self.save_job(record)
        return record

    def update_transcript(
        self,
        job_id: str,
        *,
        segments: list[dict[str, Any]] | None = None,
        markdown: str | None = None,
        target: str = "raw",
    ) -> tuple[JobRecord, Transcript]:
        """Persist manual edits (per-segment speaker/text or Markdown) back to disk."""
        record = self.get_job(job_id)
        jdir = self.job_dir(job_id)
        json_name, md_name = (
            ("polished.json", "polished.md") if target == "polished" else ("transcript.json", "transcript.md")
        )
        jpath = jdir / json_name
        mpath = jdir / md_name

        base: Transcript | None = None
        if jpath.exists():
            base = Transcript.from_dict(json.loads(jpath.read_text(encoding="utf-8")))
        elif mpath.exists():
            base = Transcript.from_markdown(mpath.read_text(encoding="utf-8"))

        language = base.language if base else None
        duration = base.duration_seconds if base else 0.0
        aliases = dict(base.speaker_aliases) if base else dict(record.speaker_aliases)
        metadata = dict(base.metadata) if base else {}
        metadata["edited"] = True

        if segments is not None:
            parsed_segments = [
                Segment(
                    start=float(item.get("start", 0.0)),
                    end=max(float(item.get("start", 0.0)), float(item.get("end", item.get("start", 0.0)))),
                    speaker=str(item.get("speaker") or "SPEAKER_UNKNOWN").strip() or "SPEAKER_UNKNOWN",
                    text=str(item.get("text") or "").strip(),
                )
                for item in segments
                if str(item.get("text") or "").strip()
            ]
            tr = Transcript(
                segments=parsed_segments,
                language=language,
                duration_seconds=duration or max((s.end for s in parsed_segments), default=0.0),
                speaker_aliases=aliases,
                metadata=metadata,
            )
            md_text = tr.to_markdown(
                title=f"{record.title} (Polished)" if target == "polished" else record.title,
                apply_aliases=True,
            )
        elif markdown is not None:
            parsed = Transcript.from_markdown(markdown)
            tr = Transcript(
                segments=parsed.segments,
                language=language,
                duration_seconds=duration or max((s.end for s in parsed.segments), default=0.0),
                speaker_aliases=aliases,
                metadata=metadata,
            )
            md_text = markdown if markdown.endswith("\n") else (markdown + "\n")
        else:
            raise ValueError("Provide either 'segments' or 'markdown' to update_transcript.")

        _atomic_write_text(jpath, json.dumps(tr.to_dict(), ensure_ascii=False, indent=2) + "\n")
        _atomic_write_text(mpath, md_text)
        if target == "polished":
            record.artifacts["polished_json"] = json_name
            record.artifacts["polished_md"] = md_name
        else:
            record.artifacts["transcript_json"] = json_name
            record.artifacts["transcript_md"] = md_name
        self.save_job(record)
        return record, tr


def _resolve_hotwords(project_root: Path, config: dict[str, Any], options: PipelineOptions) -> list[HotwordEntry]:
    if options.hotwords_text is not None:
        return parse_hotwords(options.hotwords_text)
    if options.hotwords_file:
        return load_hotwords(options.hotwords_file)
    configured = config.get("pipeline", {}).get("hotwords_file")
    path = resolve_hotwords_path(project_root, configured)
    return load_hotwords(path)


def _emit(cb: EventCallback | None, record: JobRecord, job_manager: JobManager, event_type: str, **data: Any) -> None:
    if "stage" in data:
        record.stage = str(data["stage"])
    if "progress" in data:
        record.progress = float(data["progress"])
    if "message" in data:
        record.message = str(data["message"])
    if event_type in ("stage_start", "stage_done", "job_done", "job_error"):
        job_manager.save_job(record)
    if cb is not None:
        cb({"type": event_type, "job_id": record.id, **data})


def run_pipeline(
    input_path: Path | None,
    options: PipelineOptions,
    config: dict[str, Any],
    project_root: Path,
    job_manager: JobManager,
    *,
    existing_job_id: str | None = None,
    existing_transcript: Transcript | None = None,
    on_event: EventCallback | None = None,
) -> JobRecord:
    """Execute the configured pipeline stages (1..5) and persist all outputs."""
    pipe_cfg = config.get("pipeline", {})
    asr_cfg = config.get("asr", {})
    diar_cfg = config.get("diarization", {})
    llm_cfg = config.get("llm", {})

    asr_id = options.asr_provider or asr_cfg.get("provider", "crispasr")
    diar_id = options.diarizer_provider or diar_cfg.get("provider", "none")
    run_polish = pipe_cfg.get("polish_enabled", False) if options.run_polish is None else options.run_polish
    run_summarize = pipe_cfg.get("summarize_enabled", True) if options.run_summarize is None else options.run_summarize
    llm_id = options.llm_provider or llm_cfg.get("provider", "openai_compat")
    llm_model = options.llm_model or llm_cfg.get("model", "")
    prompt_id = options.prompt_id or pipe_cfg.get("default_prompt", "meeting_minutes")
    language = options.asr_language or asr_cfg.get("language", "auto")
    out_lang = options.output_language or pipe_cfg.get("output_language", "auto")

    if existing_job_id:
        record = job_manager.get_job(existing_job_id)
        record.status = "running"
        record.error = ""
    else:
        title = options.title or (input_path.stem if input_path else "Transcript")
        record = job_manager.create_job(
            title=title,
            input_file=str(input_path) if input_path else "",
            config_snapshot={
                "asr_provider": asr_id,
                "diarizer_provider": diar_id,
                "polish": run_polish,
                "summarize": run_summarize,
                "llm_provider": llm_id,
                "llm_model": llm_model,
                "prompt_id": prompt_id,
            },
        )
        record.status = "running"

    job_dir = job_manager.job_dir(record.id)
    started = time.monotonic()
    metrics: dict[str, Any] = dict(record.metrics)
    hotwords = _resolve_hotwords(project_root, config, options)

    try:
        transcript: Transcript | None = existing_transcript

        # Stages 1-3: Audio Prep -> ASR -> Diarization
        if options.run_asr:
            if input_path is None or not input_path.exists():
                raise FileNotFoundError(f"Input audio/video file not found: {input_path}")

            _emit(on_event, record, job_manager, "stage_start", stage="prepare", progress=0.05, message="Probing audio duration...")
            duration = probe_duration(input_path)
            asr = get_asr_provider(asr_id, project_root, config)
            windows = plan_chunks(duration, **asr.long_audio_settings)
            metrics.update({"duration_seconds": duration, "asr_provider": asr_id, "chunks": windows})

            _emit(on_event, record, job_manager, "stage_start", stage="asr", progress=0.1, message=f"Transcribing {len(windows)} chunk(s) with {asr.label}...")
            chunk_results: list[tuple[float, list[Segment]]] = []
            turns_from_asr = []
            asr_started = time.monotonic()
            with tempfile.TemporaryDirectory(prefix="a2n_") as tmp:
                tmp_root = Path(tmp)
                for idx, (start, end) in enumerate(windows):
                    wav_path = tmp_root / f"chunk_{idx:03d}.wav"
                    extract_wav(input_path, wav_path, start=start, end=end, sample_rate=asr.sample_rate)
                    try:
                        res = asr.transcribe_chunk(
                            wav_path,
                            tmp_root / f"asr_{idx:03d}",
                            language=language,
                            hotwords=hotwords,
                            options=options.asr_options,
                        )
                    except subprocess.CalledProcessError as exc:
                        detail = (exc.stderr or exc.stdout or "").strip()[-3000:]
                        raise RuntimeError(f"ASR failed on chunk {idx + 1}/{len(windows)} ({start:.1f}s-{end:.1f}s): {detail}") from exc
                    chunk_segments = res.segments
                    if res.turns:
                        chunk_segments = align_speakers(chunk_segments, res.turns)
                        turns_from_asr.extend(res.turns)
                    chunk_results.append((start, chunk_segments))
                    frac = 0.1 + 0.45 * ((idx + 1) / max(1, len(windows)))
                    _emit(
                        on_event,
                        record,
                        job_manager,
                        "chunk_progress",
                        stage="asr",
                        progress=frac,
                        chunk=idx + 1,
                        total_chunks=len(windows),
                        message=f"ASR chunk {idx + 1}/{len(windows)} done",
                    )

                merged_segments = merge_chunk_segments(chunk_results)
                metrics["asr_wall_seconds"] = round(time.monotonic() - asr_started, 3)
                metrics["rtf"] = metrics["asr_wall_seconds"] / duration if duration else None

                # Stage 3: Optional external speaker diarization
                if diar_id and diar_id != "none":
                    diarizer = get_diarizer_provider(diar_id, project_root, config)
                    _emit(on_event, record, job_manager, "stage_start", stage="diarize", progress=0.6, message=f"Running speaker diarization ({diarizer.label})...")
                    diar_wav = tmp_root / "diarize_16k.wav"
                    extract_wav(input_path, diar_wav, sample_rate=16000)
                    turns = diarizer.diarize(
                        diar_wav,
                        tmp_root / "diarize",
                        num_speakers=options.num_speakers or diar_cfg.get("num_speakers"),
                        min_speakers=options.min_speakers or diar_cfg.get("min_speakers"),
                        max_speakers=options.max_speakers or diar_cfg.get("max_speakers"),
                        options=options.diarizer_options,
                    )
                    merged_segments = align_speakers(merged_segments, turns)
                    metrics["diarizer_provider"] = diar_id
                    metrics["diarizer_turns"] = len(turns)

            transcript = Transcript(
                segments=merged_segments,
                language=language,
                duration_seconds=duration,
                speaker_aliases=dict(record.speaker_aliases),
                metadata={"asr_provider": asr_id, "diarizer_provider": diar_id},
            )
            _atomic_write_text(
                job_dir / "transcript.json",
                json.dumps(transcript.to_dict(), ensure_ascii=False, indent=2) + "\n",
            )
            _atomic_write_text(
                job_dir / "transcript.md",
                transcript.to_markdown(title=record.title, apply_aliases=True),
            )
            record.artifacts["transcript_json"] = "transcript.json"
            record.artifacts["transcript_md"] = "transcript.md"
            metrics["segments"] = len(merged_segments)
            _emit(on_event, record, job_manager, "stage_done", stage="asr", progress=0.65, message="Transcript ready")
        elif transcript is None:
            transcript = job_manager.load_transcript(record.id, prefer_polished=False)

        # Stage 4 & 5: Optional Transcript Polish & Summarization
        # If ASR succeeded in this run, keep the transcript intact even when the LLM stage fails.
        source_for_summary = transcript
        llm_failed_exc: Exception | None = None
        if run_polish or run_summarize:
            try:
                if run_polish:
                    llm = get_llm_provider(llm_id, project_root, config)
                    polish_prompt = get_prompt(project_root, "transcript_polish")
                    _emit(on_event, record, job_manager, "stage_start", stage="polish", progress=0.7, message="Polishing transcript with LLM & hotwords...")
                    polished, polish_metrics = polish_transcript(
                        transcript,
                        llm,
                        polish_prompt,
                        hotwords,
                        model=llm_model,
                        max_context_tokens=int(llm_cfg.get("max_context_tokens", 12000)),
                        on_progress=lambda cur, tot: _emit(
                            on_event, record, job_manager, "polish_progress", stage="polish", progress=0.7 + 0.1 * (cur / max(1, tot)), message=f"Polishing chunk {cur}/{tot}"
                        ),
                    )
                    _atomic_write_text(
                        job_dir / "polished.json",
                        json.dumps(polished.to_dict(), ensure_ascii=False, indent=2) + "\n",
                    )
                    _atomic_write_text(
                        job_dir / "polished.md",
                        polished.to_markdown(title=f"{record.title} (Polished)", apply_aliases=True),
                    )
                    record.artifacts["polished_json"] = "polished.json"
                    record.artifacts["polished_md"] = "polished.md"
                    metrics["polish"] = polish_metrics
                    source_for_summary = polished
                    _emit(on_event, record, job_manager, "stage_done", stage="polish", progress=0.8, message="Polish complete")

                if run_summarize:
                    llm = get_llm_provider(llm_id, project_root, config)
                    tpl = get_prompt(project_root, prompt_id)
                    if options.custom_system_prompt is not None:
                        tpl.system_prompt = options.custom_system_prompt
                    if options.custom_user_prompt is not None:
                        tpl.user_prompt = options.custom_user_prompt
                    _emit(on_event, record, job_manager, "stage_start", stage="summarize", progress=0.82, message=f"Summarizing with {llm.label} ({tpl.name})...")

                    transcript_text = format_transcript_for_llm(source_for_summary)

                    def _on_tok(tok: str) -> None:
                        if on_event is not None:
                            on_event({"type": "llm_token", "job_id": record.id, "token": tok})

                    summary_md, sum_metrics = summarize_transcript(
                        transcript_text,
                        llm,
                        tpl,
                        hotwords,
                        title=record.title,
                        output_language=out_lang,
                        model=llm_model,
                        temperature=float(llm_cfg.get("temperature", 0.2)),
                        max_tokens=int(llm_cfg.get("max_tokens", 4096)),
                        max_context_tokens=int(llm_cfg.get("max_context_tokens", 12000)),
                        on_token=_on_tok,
                        on_map_progress=lambda cur, tot: _emit(
                            on_event, record, job_manager, "map_progress", stage="summarize", progress=0.82 + 0.12 * (cur / max(1, tot)), message=f"Summarizing part {cur}/{tot}"
                        ),
                    )
                    _atomic_write_text(job_dir / "summary.md", summary_md.rstrip() + "\n")
                    record.artifacts["summary_md"] = "summary.md"
                    metrics["summary"] = sum_metrics
                    _emit(on_event, record, job_manager, "stage_done", stage="summarize", progress=0.98, message="Summary ready")
            except Exception as exc:
                if not options.run_asr:
                    raise
                llm_failed_exc = exc

        metrics["total_wall_seconds"] = round(time.monotonic() - started, 3)
        record.metrics = metrics
        _atomic_write_text(
            job_dir / "metrics.json",
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        )
        record.artifacts["metrics_json"] = "metrics.json"
        if llm_failed_exc is not None:
            record.status = "partial"
            record.error = f"Transcript saved, but LLM stage skipped/failed: {llm_failed_exc}"
            _emit(
                on_event,
                record,
                job_manager,
                "job_done",
                stage="done",
                progress=1.0,
                status="partial",
                message="Transcript ready (LLM stage failed — you can re-summarize later)",
            )
            return record

        record.status = "completed"
        _emit(on_event, record, job_manager, "job_done", stage="done", progress=1.0, message="Completed")
        return record

    except Exception as exc:
        record.status = "failed"
        record.error = str(exc)
        _emit(on_event, record, job_manager, "job_error", stage="error", progress=record.progress, message=str(exc), error=str(exc))
        raise
