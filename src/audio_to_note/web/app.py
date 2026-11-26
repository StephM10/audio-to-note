"""FastAPI application serving the zero-build Web GUI and pipeline REST/SSE API."""

from __future__ import annotations

import asyncio
import json
import queue
import shutil
import threading
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import __version__
from ..config import (
    LLM_PRESETS,
    deep_merge,
    find_project_root,
    load_config,
    mask_secrets,
    resolve_hotwords_path,
    resolve_path,
    save_user_config,
)
from ..discovery import discover_environment
from ..llm import get_llm_provider
from ..models import Transcript
from ..pipeline import JobManager, PipelineOptions, run_pipeline
from ..prompts import PromptTemplate, list_prompts, save_prompt

STATIC_DIR = Path(__file__).resolve().parent / "static"


class HotwordsUpdate(BaseModel):
    text: str


class PromptPayload(BaseModel):
    id: str | None = None
    name: str
    description: str = ""
    stage: str = "summarize"
    system_prompt: str
    user_prompt: str


class SpeakerAliasesUpdate(BaseModel):
    aliases: dict[str, str]


class TranscriptUpdatePayload(BaseModel):
    target: str = "raw"  # "raw" | "polished"
    segments: list[dict[str, Any]] | None = None
    markdown: str | None = None


class LLMTestPayload(BaseModel):
    provider: str = "openai_compat"
    base_url: str = ""
    api_key: str = ""
    model: str = ""


class ResummarizePayload(BaseModel):
    run_polish: bool = False
    run_summarize: bool = True
    llm_provider: str | None = None
    llm_model: str | None = None
    prompt_id: str | None = None
    custom_system_prompt: str | None = None
    custom_user_prompt: str | None = None
    output_language: str | None = None
    hotwords_text: str | None = None


def _preserve_masked_secrets(existing: Any, incoming: Any) -> Any:
    if isinstance(existing, dict) and isinstance(incoming, dict):
        merged = {}
        for k, v in incoming.items():
            if (
                any(tok in k.lower() for tok in ("api_key", "token", "secret"))
                and isinstance(v, str)
                and "****" in v
                and k in existing
            ):
                merged[k] = existing[k]
            else:
                merged[k] = _preserve_masked_secrets(existing.get(k), v)
        return merged
    return incoming


def create_app(project_root: Path | None = None, config_path: Path | None = None) -> FastAPI:
    root = (project_root or find_project_root()).resolve()
    app = FastAPI(title="audio-to-note", version=__version__)
    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    subscribers: dict[str, list[queue.Queue[dict[str, Any]]]] = {}
    sub_lock = threading.Lock()

    def _get_config(expand_env: bool = True) -> dict[str, Any]:
        return load_config(config_path, project_root=root, expand_env=expand_env)

    def _get_job_manager(cfg: dict[str, Any] | None = None) -> JobManager:
        c = cfg or _get_config()
        out_dir = resolve_path(root, c.get("pipeline", {}).get("outputs_dir", "outputs")) or (root / "outputs")
        return JobManager(out_dir)

    def _broadcast(job_id: str, event: dict[str, Any]) -> None:
        with sub_lock:
            queues = list(subscribers.get(job_id, []))
        for q in queues:
            q.put(event)

    def _start_background_job(
        input_path: Path | None,
        options: PipelineOptions,
        cfg: dict[str, Any],
        jm: JobManager,
        *,
        existing_job_id: str | None = None,
        existing_transcript: Transcript | None = None,
    ) -> str:
        if existing_job_id:
            job_id = existing_job_id
        else:
            title = options.title or (input_path.stem if input_path else "Recording")
            rec = jm.create_job(title=title, input_file=str(input_path) if input_path else "")
            job_id = rec.id

        def _worker() -> None:
            try:
                run_pipeline(
                    input_path,
                    options,
                    cfg,
                    root,
                    jm,
                    existing_job_id=job_id,
                    existing_transcript=existing_transcript,
                    on_event=lambda ev: _broadcast(job_id, ev),
                )
            except Exception as exc:
                _broadcast(job_id, {"type": "job_error", "job_id": job_id, "message": str(exc)})

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        return job_id

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        index_file = STATIC_DIR / "index.html"
        if not index_file.exists():
            return HTMLResponse("<h1>audio-to-note</h1><p>Static UI not found.</p>")
        return HTMLResponse(index_file.read_text(encoding="utf-8"))

    @app.get("/api/environment")
    def get_environment() -> dict[str, Any]:
        cfg = _get_config()
        env = discover_environment(root, cfg)
        env["llm_presets"] = LLM_PRESETS
        env["version"] = __version__
        return env

    @app.get("/api/config")
    def get_config() -> dict[str, Any]:
        cfg = _get_config(expand_env=True)
        masked = mask_secrets(cfg)
        masked.pop("_meta", None)
        return masked

    @app.put("/api/config")
    def update_config(payload: dict[str, Any]) -> dict[str, Any]:
        current = _get_config(expand_env=False)
        current.pop("_meta", None)
        clean_incoming = _preserve_masked_secrets(current, payload)
        merged = deep_merge(current, clean_incoming)
        target = config_path if config_path else (root / "config" / "config.yaml")
        save_user_config(merged, target)
        return {"ok": True, "config": mask_secrets(_get_config(expand_env=True))}

    @app.post("/api/llm/test")
    def test_llm_connection(payload: LLMTestPayload) -> dict[str, Any]:
        cfg = _get_config(expand_env=True)
        opts: dict[str, Any] = {}
        if payload.base_url:
            opts["base_url"] = payload.base_url
        if payload.api_key and "****" not in payload.api_key:
            opts["api_key"] = payload.api_key
        provider = get_llm_provider(payload.provider, root, cfg)
        return provider.test_connection(options=opts)

    @app.post("/api/llm/models")
    def fetch_llm_models(payload: LLMTestPayload) -> dict[str, Any]:
        cfg = _get_config(expand_env=True)
        opts: dict[str, Any] = {}
        if payload.base_url:
            opts["base_url"] = payload.base_url
        if payload.api_key and "****" not in payload.api_key:
            opts["api_key"] = payload.api_key
        provider = get_llm_provider(payload.provider, root, cfg)
        try:
            models = provider.list_models(options=opts)
            return {"ok": True, "models": models}
        except Exception as exc:
            return {"ok": False, "models": [], "detail": str(exc)}

    @app.get("/api/hotwords")
    def get_hotwords() -> dict[str, Any]:
        cfg = _get_config()
        hw_path = resolve_hotwords_path(root, cfg.get("pipeline", {}).get("hotwords_file"))
        text = hw_path.read_text(encoding="utf-8") if hw_path and hw_path.exists() else ""
        return {"path": str(hw_path or (root / "config" / "hotwords.local.txt")), "text": text}

    @app.put("/api/hotwords")
    def put_hotwords(payload: HotwordsUpdate) -> dict[str, Any]:
        target = root / "config" / "hotwords.local.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload.text, encoding="utf-8")
        return {"ok": True, "path": str(target)}

    @app.get("/api/prompts")
    def get_prompts() -> list[dict[str, Any]]:
        return [item.to_dict() for item in list_prompts(root)]

    @app.put("/api/prompts/{prompt_id}")
    def put_prompt(prompt_id: str, payload: PromptPayload) -> dict[str, Any]:
        tpl = PromptTemplate(
            id=prompt_id,
            name=payload.name,
            description=payload.description,
            stage=payload.stage,
            system_prompt=payload.system_prompt,
            user_prompt=payload.user_prompt,
        )
        saved_path = save_prompt(root, tpl)
        return {"ok": True, "path": str(saved_path), "prompt": tpl.to_dict()}

    @app.post("/api/prompts")
    def create_prompt(payload: PromptPayload) -> dict[str, Any]:
        pid = payload.id or payload.name.lower().replace(" ", "_")
        tpl = PromptTemplate(
            id=pid,
            name=payload.name,
            description=payload.description,
            stage=payload.stage,
            system_prompt=payload.system_prompt,
            user_prompt=payload.user_prompt,
        )
        saved_path = save_prompt(root, tpl)
        return {"ok": True, "path": str(saved_path), "prompt": tpl.to_dict()}

    @app.get("/api/fs/browse")
    def browse_filesystem(
        path: str = Query(default=""),
        ext: str = Query(default=""),
    ) -> dict[str, Any]:
        target = Path(path).expanduser() if path else root
        if not target.exists():
            target = root
        if target.is_file():
            target = target.parent
        allowed_exts = {e.strip().lower() for e in ext.split(",") if e.strip()}
        dirs: list[dict[str, str]] = []
        files: list[dict[str, Any]] = []
        try:
            for entry in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
                if entry.name.startswith(".") and entry.name not in (".env",):
                    continue
                if entry.is_dir():
                    dirs.append({"name": entry.name, "path": str(entry)})
                elif entry.is_file():
                    if allowed_exts and entry.suffix.lower() not in allowed_exts:
                        continue
                    files.append(
                        {
                            "name": entry.name,
                            "path": str(entry),
                            "size_mb": round(entry.stat().st_size / (1024 * 1024), 2),
                        }
                    )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        return {
            "current": str(target),
            "parent": str(target.parent) if target.parent != target else "",
            "dirs": dirs[:200],
            "files": files[:200],
        }

    @app.get("/api/jobs")
    def list_all_jobs() -> list[dict[str, Any]]:
        jm = _get_job_manager()
        return [job.to_dict() for job in jm.list_jobs()]

    @app.get("/api/jobs/{job_id}")
    def get_job_detail(job_id: str) -> dict[str, Any]:
        jm = _get_job_manager()
        try:
            rec = jm.get_job(job_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        jdir = jm.job_dir(job_id)
        data = rec.to_dict()
        data["content"] = {
            "transcript_md": (jdir / "transcript.md").read_text(encoding="utf-8") if (jdir / "transcript.md").exists() else "",
            "polished_md": (jdir / "polished.md").read_text(encoding="utf-8") if (jdir / "polished.md").exists() else "",
            "summary_md": (jdir / "summary.md").read_text(encoding="utf-8") if (jdir / "summary.md").exists() else "",
        }
        if (jdir / "transcript.json").exists():
            tr = Transcript.from_dict(json.loads((jdir / "transcript.json").read_text(encoding="utf-8")))
            data["speakers"] = tr.unique_speakers()
            data["segments"] = [asdict(seg) for seg in tr.segments]
        else:
            data["speakers"] = []
            data["segments"] = []
        if (jdir / "polished.json").exists():
            ptr = Transcript.from_dict(json.loads((jdir / "polished.json").read_text(encoding="utf-8")))
            data["polished_segments"] = [asdict(seg) for seg in ptr.segments]
        else:
            data["polished_segments"] = []
        return data

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str) -> dict[str, Any]:
        jm = _get_job_manager()
        jm.delete_job(job_id)
        return {"ok": True}

    @app.post("/api/jobs")
    async def create_job(
        options: str = Form(default="{}"),
        local_path: str = Form(default=""),
        file: UploadFile | None = File(default=None),
    ) -> dict[str, Any]:
        cfg = _get_config()
        jm = _get_job_manager(cfg)
        try:
            opt_dict = json.loads(options or "{}")
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid options JSON: {exc}") from exc

        input_path: Path | None = None
        if file is not None and file.filename:
            uploads_dir = root / "uploads"
            uploads_dir.mkdir(parents=True, exist_ok=True)
            safe_name = Path(file.filename).name
            dest = uploads_dir / safe_name
            with dest.open("wb") as handle:
                shutil.copyfileobj(file.file, handle)
            input_path = dest
        elif local_path.strip():
            candidate = Path(local_path.strip()).expanduser().resolve()
            if not candidate.exists() or not candidate.is_file():
                raise HTTPException(status_code=400, detail=f"Local file not found: {candidate}")
            input_path = candidate
        else:
            raise HTTPException(status_code=400, detail="Provide an uploaded file or a local_path.")

        # Apply optional per-job model/API overrides cleanly without mutating global config
        job_cfg = deepcopy(cfg)
        if opt_dict.get("asr_model"):
            asr_id = opt_dict.get("asr_provider") or job_cfg["asr"]["provider"]
            if asr_id in job_cfg["asr"]["providers"]:
                job_cfg["asr"]["providers"][asr_id]["model"] = opt_dict["asr_model"]
        if opt_dict.get("llm_base_url"):
            job_cfg["llm"]["providers"]["openai_compat"]["base_url"] = opt_dict["llm_base_url"]
        if opt_dict.get("llm_api_key") and "****" not in str(opt_dict["llm_api_key"]):
            job_cfg["llm"]["providers"]["openai_compat"]["api_key"] = opt_dict["llm_api_key"]

        pipe_opts = PipelineOptions(
            title=opt_dict.get("title") or input_path.stem,
            run_asr=True,
            asr_provider=opt_dict.get("asr_provider"),
            asr_language=opt_dict.get("asr_language"),
            diarizer_provider=opt_dict.get("diarizer_provider"),
            num_speakers=opt_dict.get("num_speakers") or None,
            hotwords_text=opt_dict.get("hotwords_text"),
            run_polish=opt_dict.get("run_polish"),
            run_summarize=opt_dict.get("run_summarize"),
            llm_provider=opt_dict.get("llm_provider"),
            llm_model=opt_dict.get("llm_model"),
            prompt_id=opt_dict.get("prompt_id"),
            custom_system_prompt=opt_dict.get("custom_system_prompt"),
            custom_user_prompt=opt_dict.get("custom_user_prompt"),
            output_language=opt_dict.get("output_language"),
        )
        job_id = _start_background_job(input_path, pipe_opts, job_cfg, jm)
        return {"job_id": job_id, "status": "queued"}

    @app.post("/api/jobs/{job_id}/summarize")
    def resummarize_job(job_id: str, payload: ResummarizePayload) -> dict[str, Any]:
        cfg = _get_config()
        jm = _get_job_manager(cfg)
        try:
            jm.get_job(job_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        pipe_opts = PipelineOptions(
            run_asr=False,
            hotwords_text=payload.hotwords_text,
            run_polish=payload.run_polish,
            run_summarize=payload.run_summarize,
            llm_provider=payload.llm_provider,
            llm_model=payload.llm_model,
            prompt_id=payload.prompt_id,
            custom_system_prompt=payload.custom_system_prompt,
            custom_user_prompt=payload.custom_user_prompt,
            output_language=payload.output_language,
        )
        _start_background_job(None, pipe_opts, cfg, jm, existing_job_id=job_id)
        return {"job_id": job_id, "status": "queued"}

    @app.put("/api/jobs/{job_id}/speakers")
    def update_speakers(job_id: str, payload: SpeakerAliasesUpdate) -> dict[str, Any]:
        jm = _get_job_manager()
        try:
            rec = jm.update_speaker_aliases(job_id, payload.aliases)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        jdir = jm.job_dir(job_id)
        return {
            "ok": True,
            "speaker_aliases": rec.speaker_aliases,
            "transcript_md": (jdir / "transcript.md").read_text(encoding="utf-8") if (jdir / "transcript.md").exists() else "",
            "polished_md": (jdir / "polished.md").read_text(encoding="utf-8") if (jdir / "polished.md").exists() else "",
        }

    @app.put("/api/jobs/{job_id}/transcript")
    def update_job_transcript(job_id: str, payload: TranscriptUpdatePayload) -> dict[str, Any]:
        jm = _get_job_manager()
        try:
            _, tr = jm.update_transcript(
                job_id,
                segments=payload.segments,
                markdown=payload.markdown,
                target=payload.target,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        jdir = jm.job_dir(job_id)
        return {
            "ok": True,
            "target": payload.target,
            "speakers": tr.unique_speakers(),
            "segments": [asdict(seg) for seg in tr.segments],
            "transcript_md": (jdir / "transcript.md").read_text(encoding="utf-8") if (jdir / "transcript.md").exists() else "",
            "polished_md": (jdir / "polished.md").read_text(encoding="utf-8") if (jdir / "polished.md").exists() else "",
        }

    @app.get("/api/jobs/{job_id}/events")
    async def stream_job_events(job_id: str) -> StreamingResponse:
        jm = _get_job_manager()
        try:
            rec = jm.get_job(job_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        q: queue.Queue[dict[str, Any]] = queue.Queue()
        with sub_lock:
            subscribers.setdefault(job_id, []).append(q)

        async def _event_gen():
            try:
                yield f"data: {json.dumps({'type': 'snapshot', 'job': rec.to_dict()})}\n\n"
                if rec.status in ("completed", "failed"):
                    return
                while True:
                    try:
                        event = await asyncio.to_thread(q.get, True, 0.5)
                        yield f"data: {json.dumps(event)}\n\n"
                        if event.get("type") in ("job_done", "job_error"):
                            break
                    except queue.Empty:
                        latest = jm.get_job(job_id)
                        if latest.status in ("completed", "failed"):
                            yield f"data: {json.dumps({'type': 'snapshot', 'job': latest.to_dict()})}\n\n"
                            break
                        yield ": keepalive\n\n"
            finally:
                with sub_lock:
                    lst = subscribers.get(job_id, [])
                    if q in lst:
                        lst.remove(q)

        return StreamingResponse(_event_gen(), media_type="text/event-stream")

    @app.get("/api/jobs/{job_id}/export/{kind}")
    def export_artifact(job_id: str, kind: str) -> FileResponse:
        jm = _get_job_manager()
        allowed = {"transcript.md", "transcript.json", "polished.md", "polished.json", "summary.md", "metrics.json"}
        if kind not in allowed:
            raise HTTPException(status_code=400, detail=f"Unsupported export kind: {kind}")
        target = jm.job_dir(job_id) / kind
        if not target.exists():
            raise HTTPException(status_code=404, detail=f"Artifact not found: {kind}")
        return FileResponse(path=target, filename=f"{job_id}_{kind}")

    return app
