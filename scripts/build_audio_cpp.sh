#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/vendor/audio.cpp/build"
TOOLS_DIR="${ROOT_DIR}/.tools"
if [[ -x "${TOOLS_DIR}/bin/cmake" ]]; then
    export PYTHONPATH="${TOOLS_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
    export PATH="${TOOLS_DIR}/bin:${PATH}"
fi

cmake -S "${ROOT_DIR}/vendor/audio.cpp" -B "${BUILD_DIR}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DENGINE_ENABLE_CUDA=ON \
    -DENGINE_ENABLE_CUDA_GRAPHS=ON \
    -DCMAKE_CUDA_ARCHITECTURES=120 \
    -DENGINE_ENABLE_NATIVE_CPU=ON \
    -DAUDIOCPP_MODEL_SET=custom \
    -DAUDIOCPP_MODELS=vibevoice_asr \
    -DAUDIOCPP_DEPLOYMENT_BUILD=ON
cmake --build "${BUILD_DIR}" --target audiocpp_cli --parallel "${JOBS:-$(nproc)}"
