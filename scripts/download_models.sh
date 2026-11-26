#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${ROOT_DIR}/models"

usage() {
    cat <<EOF
Usage: $0 <target>

Targets:
  vibevoice-q4       VibeVoice-ASR Q4_K_M GGUF (~4.5 GB, for CrispASR CUDA)
  vibevoice-q8       VibeVoice-ASR Q8_0 GGUF (~9.2 GB, for audio.cpp CUDA)
  vibevoice-bitnet   VibeVoice-ASR BitNet split GGUFs (~1.6 GB, for VibeASR.cpp CPU)
  diarization-onnx   sherpa-onnx segmentation + 3D-Speaker embedding ONNX (~35 MB, no HF token)
EOF
}

download() {
    local url="$1"
    local output="$2"
    mkdir -p "$(dirname "${output}")"
    curl --http1.1 -L --fail --retry 8 --retry-all-errors --retry-delay 5 \
      --connect-timeout 30 --continue-at - -o "${output}" "${url}"
}

TARGET="${1:-}"
case "${TARGET}" in
    vibevoice-q4)
        download \
          'https://huggingface.co/cstr/vibevoice-asr-GGUF/resolve/main/vibevoice-asr-q4_k.gguf?download=true' \
          "${MODEL_DIR}/vibevoice-asr-q4_k.gguf"
        ;;
    vibevoice-q8)
        download \
          'https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/main/VibeVoice-ASR-GGUF/vibevoice-asr-q8_0.gguf?download=true' \
          "${MODEL_DIR}/vibevoice-asr-q8_0.gguf"
        ;;
    vibevoice-bitnet)
        download \
          'https://huggingface.co/microsoft/VibeVoice-ASR-BitNet/resolve/main/vibeasr-vae-encoder-i8_s.gguf?download=true' \
          "${MODEL_DIR}/vibevoice-bitnet/vibeasr-vae-encoder-i8_s.gguf"
        download \
          'https://huggingface.co/microsoft/VibeVoice-ASR-BitNet/resolve/main/vibeasr-lm-i2_s-embed-q6_k.gguf?download=true' \
          "${MODEL_DIR}/vibevoice-bitnet/vibeasr-lm-i2_s-embed-q6_k.gguf"
        ;;
    diarization-onnx)
        DIAR_DIR="${MODEL_DIR}/diarization"
        mkdir -p "${DIAR_DIR}"
        ARCHIVE="${DIAR_DIR}/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
        download \
          'https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2' \
          "${ARCHIVE}"
        tar -xjf "${ARCHIVE}" -C "${DIAR_DIR}"
        mv -f "${DIAR_DIR}/sherpa-onnx-pyannote-segmentation-3-0" "${DIAR_DIR}/pyannote-segmentation-3-0" 2>/dev/null || true
        rm -f "${ARCHIVE}"
        download \
          'https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx' \
          "${DIAR_DIR}/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
        echo "Diarization ONNX models ready under ${DIAR_DIR}"
        ;;
    *)
        usage
        exit 2
        ;;
esac
