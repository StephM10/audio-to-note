"""Environment and local model/runtime auto-discovery."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .config import resolve_path
from .models import ProviderStatus


def _scan_files(root: Path, suffixes: tuple[str, ...], max_depth: int = 4) -> list[str]:
    if not root.exists() or not root.is_dir():
        return []
    found: list[str] = []
    base_parts = len(root.parts)
    for path in root.rglob("*"):
        if len(path.parts) - base_parts > max_depth:
            continue
        if path.is_file() and path.suffix.lower() in suffixes and ".part" not in path.name:
            found.append(str(path))
    return sorted(found)


def detect_gpu() -> dict[str, Any]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return {"available": False, "gpus": []}
    try:
        res = subprocess.run(
            [nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
        gpus = [line.strip() for line in res.stdout.splitlines() if line.strip()]
        return {"available": bool(gpus), "gpus": gpus}
    except (subprocess.SubprocessError, OSError):
        return {"available": False, "gpus": []}


def find_binary(project_root: Path, configured: str | None, fallbacks: list[str]) -> Path | None:
    if configured:
        resolved = resolve_path(project_root, configured)
        if resolved and resolved.exists() and resolved.is_file():
            return resolved
        on_path = shutil.which(configured)
        if on_path:
            return Path(on_path)
    for name in fallbacks:
        p = resolve_path(project_root, name)
        if p and p.exists() and p.is_file():
            return p
        on_path = shutil.which(name)
        if on_path:
            return Path(on_path)
    return None


def check_package(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ModuleNotFoundError, ValueError):
        return False


def discover_environment(project_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Inspect tools, GPU, local models, runtimes, and API providers."""
    models_dir = project_root / "models"
    gguf_files = _scan_files(models_dir, (".gguf",), max_depth=3)
    onnx_files = _scan_files(models_dir, (".onnx",), max_depth=4)
    gpu = detect_gpu()

    tools: list[ProviderStatus] = []
    for tool_name in ("ffmpeg", "ffprobe"):
        path = shutil.which(tool_name)
        tools.append(
            ProviderStatus(
                id=tool_name,
                name=tool_name,
                kind="tool",
                available=bool(path),
                status="ready" if path else "missing_dependency",
                detail=path or "Not found on PATH",
                hint="" if path else "Install ffmpeg (e.g. sudo apt install ffmpeg / brew install ffmpeg)",
            )
        )
    tools.append(
        ProviderStatus(
            id="gpu",
            name="NVIDIA GPU",
            kind="tool",
            available=gpu["available"],
            status="ready" if gpu["available"] else "unavailable",
            detail=", ".join(gpu["gpus"]) if gpu["gpus"] else "No CUDA GPU detected (CPU mode available)",
        )
    )

    asr_cfg = config.get("asr", {}).get("providers", {})
    asr_statuses: list[ProviderStatus] = []

    # 1. CrispASR
    crisp_cfg = asr_cfg.get("crispasr", {})
    crisp_bin = find_binary(project_root, crisp_cfg.get("runtime_bin"), ["vendor/CrispASR/build-cuda129/bin/crispasr", "crispasr"])
    crisp_model = resolve_path(project_root, crisp_cfg.get("model"))
    crisp_ok = bool(crisp_bin and crisp_model and crisp_model.exists())
    asr_statuses.append(
        ProviderStatus(
            id="crispasr",
            name="VibeVoice Q4 (CrispASR)",
            kind="asr",
            available=crisp_ok,
            status="ready" if crisp_ok else ("missing_model" if crisp_bin else "unavailable"),
            detail=f"bin={crisp_bin or 'missing'}, model={crisp_model.name if crisp_model and crisp_model.exists() else 'missing'}",
            hint="" if crisp_ok else "Build CrispASR or run: scripts/download_models.sh vibevoice-q4",
            detected_files=[f for f in gguf_files if "q4" in f.lower() or "vibe" in f.lower()],
        )
    )

    # 2. audio.cpp
    acpp_cfg = asr_cfg.get("audio_cpp", {})
    acpp_bin = find_binary(project_root, acpp_cfg.get("runtime_bin"), ["vendor/audio.cpp/build/bin/audiocpp_cli", "audiocpp_cli"])
    acpp_model = resolve_path(project_root, acpp_cfg.get("model"))
    acpp_ok = bool(acpp_bin and acpp_model and acpp_model.exists())
    asr_statuses.append(
        ProviderStatus(
            id="audio_cpp",
            name="VibeVoice Q8 (audio.cpp)",
            kind="asr",
            available=acpp_ok,
            status="ready" if acpp_ok else ("missing_model" if acpp_bin else "unavailable"),
            detail=f"bin={acpp_bin or 'missing'}, model={acpp_model.name if acpp_model and acpp_model.exists() else 'missing'}",
            hint="" if acpp_ok else "Build audio.cpp or run: scripts/download_models.sh vibevoice-q8",
            detected_files=[f for f in gguf_files if "q8" in f.lower()],
        )
    )

    # 3. VibeASR.cpp BitNet
    bit_cfg = asr_cfg.get("vibeasr_cpp", {})
    bit_bin = find_binary(project_root, bit_cfg.get("runtime_bin"), ["vendor/VibeASR.cpp/build/bin/asr_infer", "asr_infer"])
    vae = resolve_path(project_root, bit_cfg.get("vae_model"))
    lm = resolve_path(project_root, bit_cfg.get("lm_model"))
    bit_ok = bool(bit_bin and vae and vae.exists() and lm and lm.exists())
    asr_statuses.append(
        ProviderStatus(
            id="vibeasr_cpp",
            name="VibeVoice BitNet CPU (VibeASR.cpp)",
            kind="asr",
            available=bit_ok,
            status="ready" if bit_ok else ("missing_model" if bit_bin else "unavailable"),
            detail=f"bin={bit_bin or 'missing'}, weights={'ready' if (vae and vae.exists() and lm and lm.exists()) else 'missing'}",
            hint="" if bit_ok else "Build VibeASR.cpp or run: scripts/download_models.sh vibevoice-bitnet",
            detected_files=[f for f in gguf_files if "bitnet" in f.lower() or "vibeasr" in f.lower()],
        )
    )

    # 4. faster-whisper
    fw_installed = check_package("faster_whisper")
    asr_statuses.append(
        ProviderStatus(
            id="faster_whisper",
            name="faster-whisper (Local Python)",
            kind="asr",
            available=fw_installed,
            status="ready" if fw_installed else "missing_dependency",
            detail="faster-whisper installed" if fw_installed else "Optional package not installed",
            hint="" if fw_installed else 'pip install -e ".[whisper]"',
        )
    )

    # 5. OpenAI-compatible Audio API
    oai_asr = asr_cfg.get("openai_compat", {})
    oai_url = str(oai_asr.get("base_url", "")).strip()
    oai_key = str(oai_asr.get("api_key", "")).strip()
    is_local_asr = any(h in oai_url for h in ("127.0.0.1", "localhost", "0.0.0.0"))
    oai_ok = bool(oai_url and (oai_key or is_local_asr))
    asr_statuses.append(
        ProviderStatus(
            id="openai_compat",
            name="OpenAI-Compatible Audio API",
            kind="asr",
            available=oai_ok,
            status="ready" if oai_ok else "needs_config",
            detail=f"{oai_url or 'no URL'} (model={oai_asr.get('model', 'whisper-1')})",
            hint="" if oai_ok else "Set A2N_ASR_API_KEY / OPENAI_API_KEY or configure local Whisper server URL",
        )
    )

    # 6. Generic command
    cmd_asr = asr_cfg.get("command", {}).get("command", "")
    asr_statuses.append(
        ProviderStatus(
            id="command",
            name="Custom CLI Command",
            kind="asr",
            available=bool(cmd_asr),
            status="ready" if cmd_asr else "needs_config",
            detail=str(cmd_asr) if cmd_asr else "No custom CLI template configured",
            hint="" if cmd_asr else "Set asr.providers.command.command in Settings",
        )
    )

    # Diarizers
    diar_cfg = config.get("diarization", {}).get("providers", {})
    diar_statuses: list[ProviderStatus] = [
        ProviderStatus(
            id="none",
            name="Off (Keep ASR Speaker Labels)",
            kind="diarizer",
            available=True,
            status="ready",
            detail="Uses speaker labels emitted by ASR backend",
        )
    ]

    # sherpa-onnx
    sherpa_pkg = check_package("sherpa_onnx")
    sh_cfg = diar_cfg.get("sherpa_onnx", {})
    seg_m = resolve_path(project_root, sh_cfg.get("segmentation_model"))
    emb_m = resolve_path(project_root, sh_cfg.get("embedding_model"))
    sh_models_ok = bool(seg_m and seg_m.exists() and emb_m and emb_m.exists())
    sh_ok = sherpa_pkg and sh_models_ok
    if not sherpa_pkg:
        sh_status = "missing_dependency"
        sh_hint = 'pip install -e ".[diarize-onnx]"'
    elif not sh_models_ok:
        sh_status = "missing_model"
        sh_hint = "scripts/download_models.sh diarization-onnx (or pick .onnx files in GUI)"
    else:
        sh_status = "ready"
        sh_hint = ""
    diar_statuses.append(
        ProviderStatus(
            id="sherpa_onnx",
            name="sherpa-onnx (Local, No Token)",
            kind="diarizer",
            available=sh_ok,
            status=sh_status,
            detail=f"pkg={'ok' if sherpa_pkg else 'missing'}, seg={'ok' if seg_m and seg_m.exists() else 'missing'}, emb={'ok' if emb_m and emb_m.exists() else 'missing'}",
            hint=sh_hint,
            detected_files=onnx_files,
        )
    )

    # pyannote
    pya_pkg = check_package("pyannote.audio")
    hf_cache_tok = Path("~/.cache/huggingface/token").expanduser()
    pya_token = bool(diar_cfg.get("pyannote", {}).get("hf_token", "").strip()) or (
        hf_cache_tok.is_file() and bool(hf_cache_tok.read_text(encoding="utf-8").strip())
    )
    pya_ok = pya_pkg
    diar_statuses.append(
        ProviderStatus(
            id="pyannote",
            name="pyannote.audio (Local High-Accuracy)",
            kind="diarizer",
            available=pya_ok,
            status="ready" if (pya_ok and pya_token) else ("needs_config" if pya_ok else "missing_dependency"),
            detail=f"pkg={'ok' if pya_pkg else 'missing'}, HF_TOKEN={'set' if pya_token else 'unset (needed for first download)'}",
            hint="" if (pya_ok and pya_token) else ('pip install -e ".[diarize-pyannote]"' if not pya_pkg else "Set HF_TOKEN in .env or Settings"),
        )
    )

    cmd_diar = diar_cfg.get("command", {}).get("command", "")
    diar_statuses.append(
        ProviderStatus(
            id="command",
            name="Custom Diarizer CLI",
            kind="diarizer",
            available=bool(cmd_diar),
            status="ready" if cmd_diar else "needs_config",
            detail=str(cmd_diar) if cmd_diar else "No custom CLI template configured",
        )
    )

    # LLMs
    llm_cfg = config.get("llm", {})
    llm_providers = llm_cfg.get("providers", {})
    oai_llm = llm_providers.get("openai_compat", {})
    llm_url = str(oai_llm.get("base_url", "")).strip()
    llm_key = str(oai_llm.get("api_key", "")).strip()
    is_local_llm = any(h in llm_url for h in ("127.0.0.1", "localhost", "0.0.0.0"))
    llm_ok = bool(llm_url and (llm_key or is_local_llm))
    llm_statuses = [
        ProviderStatus(
            id="openai_compat",
            name="OpenAI-Compatible LLM (vLLM / Ollama / llama-server / Cloud)",
            kind="llm",
            available=llm_ok,
            status="ready" if llm_ok else "needs_config",
            detail=f"{llm_url or 'no URL'} (model={llm_cfg.get('model', '')})",
            hint="" if llm_ok else "Configure Base URL & API key in Settings",
        )
    ]

    lcli_cfg = llm_providers.get("llama_cli", {})
    lcli_bin = find_binary(project_root, lcli_cfg.get("runtime_bin"), ["llama-cli", "vendor/llama.cpp/build/bin/llama-cli"])
    lcli_model = resolve_path(project_root, lcli_cfg.get("model"))
    lcli_ok = bool(lcli_bin and lcli_model and lcli_model.exists())
    llm_statuses.append(
        ProviderStatus(
            id="llama_cli",
            name="llama-cli (Local GGUF)",
            kind="llm",
            available=lcli_ok,
            status="ready" if lcli_ok else ("missing_model" if lcli_bin else "unavailable"),
            detail=f"bin={lcli_bin or 'missing'}, model={lcli_model.name if lcli_model and lcli_model.exists() else 'unset'}",
            hint="" if lcli_ok else "Install llama.cpp (llama-cli) and select a local GGUF model",
            detected_files=gguf_files,
        )
    )

    return {
        "project_root": str(project_root),
        "gpu": gpu,
        "tools": [item.to_dict() for item in tools],
        "asr": [item.to_dict() for item in asr_statuses],
        "diarization": [item.to_dict() for item in diar_statuses],
        "llm": [item.to_dict() for item in llm_statuses],
        "models": {
            "gguf": gguf_files,
            "onnx": onnx_files,
        },
    }
