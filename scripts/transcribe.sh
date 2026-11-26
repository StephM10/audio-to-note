#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASR_PROVIDER="${1:-crispasr}"
shift || true

case "${ASR_PROVIDER}" in
    vibevoice_q4) ASR_PROVIDER="crispasr" ;;
    vibevoice_q8) ASR_PROVIDER="audio_cpp" ;;
    vibevoice_bitnet) ASR_PROVIDER="vibeasr_cpp" ;;
esac

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <crispasr|audio_cpp|vibeasr_cpp|faster_whisper|openai_compat> <audio-or-video> [extra audio-to-note flags]" >&2
    exit 2
fi

AUDIO_FILE="$1"
shift

if command -v audio-to-note >/dev/null 2>&1; then
    exec audio-to-note transcribe "${AUDIO_FILE}" --asr "${ASR_PROVIDER}" "$@"
fi

if [[ -x "${ROOT_DIR}/.venv/bin/audio-to-note" ]]; then
    exec "${ROOT_DIR}/.venv/bin/audio-to-note" transcribe "${AUDIO_FILE}" --asr "${ASR_PROVIDER}" "$@"
fi

export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec python3 -m audio_to_note.cli transcribe "${AUDIO_FILE}" --asr "${ASR_PROVIDER}" "$@"
