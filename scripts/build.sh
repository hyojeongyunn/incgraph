#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_TYPE=${BUILD_TYPE:-Release}
cmake -S "${ROOT}/src" -B "${ROOT}/build/cpp-${BUILD_TYPE}" -G Ninja -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
cmake --build "${ROOT}/build/cpp-${BUILD_TYPE}" -j
ls "${ROOT}/bin"
