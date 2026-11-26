# Local C++ VibeVoice Runtimes (Optional)

`audio-to-note` supports local GGUF VibeVoice-ASR runtimes when their binaries are placed under `vendor/` or on `$PATH`:

1. **CrispASR (`crispasr` — VibeVoice Q4_K_M)**:
   - Expected default binary: `vendor/CrispASR/build-cuda129/bin/crispasr` (or `crispasr` on `PATH`).
   - Notes:
     - CrispASR's VibeVoice-ASR path allocates the decoder KV cache as `F16`.
     - The `crispasr` adapter clears `CRISPASR_N_GPU_LAYERS` from the child process environment so stale offload settings do not silently spill GPU layers into CPU memory.
     - Tested cleanly on 16 GB CUDA GPUs (`~8.9 GiB` peak VRAM for Q4_K_M).

2. **audio.cpp (`audiocpp_cli` — VibeVoice Q8_0)**:
   - Expected default binary: `vendor/audio.cpp/build/bin/audiocpp_cli`.
   - Configured with `vibevoice_asr.kv_cache_type=bf16` (`f16` / `f32` fallback).

3. **VibeASR.cpp (`asr_infer` — BitNet CPU)**:
   - Expected default binary: `vendor/VibeASR.cpp/build/bin/asr_infer`.
   - Uses shorter 120s chunk windows with 15s overlap by default to keep host VAE memory bounded.

For most new users, `faster_whisper` (`pip install -e ".[whisper]"`) or an OpenAI-compatible audio endpoint requires zero C++ compilation.
