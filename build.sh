#!/usr/bin/env bash
# Configure and build the asmodean C++ extension.
# Requires python 3.13, boost, cmake and the lanfear headers (found
# automatically from an importable lanfear checkout, or via LANFEAR_ROOT).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${here}/build"

if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake not found -- did you load modules first?" >&2
  exit 1
fi

mkdir -p "${build_dir}"
cmake -S "${here}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release "$@"
cmake --build "${build_dir}" -j "$(nproc)"

echo
echo "Built: ${here}/asmodean/_core*.so"
echo "Run the C++ check:   ${here}/bin/test_core"
echo "Run the test suite:  pytest ${here}/tests"
