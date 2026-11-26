#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/vendor/VibeASR.cpp/build"
TOOLS_DIR="${ROOT_DIR}/.tools"
if [[ -x "${TOOLS_DIR}/bin/cmake" ]]; then
    export PYTHONPATH="${TOOLS_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
    export PATH="${TOOLS_DIR}/bin:${PATH}"
fi

cmake -S "${ROOT_DIR}/vendor/VibeASR.cpp" -B "${BUILD_DIR}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" --target asr_infer --parallel "${JOBS:-$(nproc)}"
