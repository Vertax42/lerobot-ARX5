#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# build_python.sh — Build arx5_interface Python bindings
#
# Requirements:
#   conda activate lerobot   (spdlog==1.14.1, pybind11, eigen, orocos-kdl,
#                             ros-jazzy-kdl-parser, ninja, soem)
#
# Usage:
#   cd src/lerobot/robots/bi_arx5/ARX5_SDK
#   bash build_python.sh
#
# Output:
#   python/arx5_interface.cpython-<ver>-<arch>-linux-gnu.so
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Verify conda env ────────────────────────────────────────────────────────
if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "ERROR: CONDA_PREFIX is not set. Please activate the arx-py312 conda env first:"
    echo "  conda activate arx-py312"
    exit 1
fi

# Check spdlog version
SPDLOG_VER=$(python -c "
import subprocess, re
r = subprocess.run(['cmake', '--find-package', '-DNAME=spdlog', '-DCOMPILER_ID=GNU',
                    '-DLANGUAGE=CXX', '-DMODE=EXIST'], capture_output=True, text=True)
" 2>/dev/null || true)

echo "Using conda env: ${CONDA_PREFIX}"
echo "spdlog header: $(find "${CONDA_PREFIX}/include" -name 'version.h' -path '*/spdlog/*' 2>/dev/null | head -1)"

# ── Build ───────────────────────────────────────────────────────────────────
BUILD_DIR="${SCRIPT_DIR}/build_py"
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

cmake \
    -S "${SCRIPT_DIR}" \
    -B "${BUILD_DIR}" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="${CONDA_PREFIX}" \
    -DSPDLOG_FMT_EXTERNAL=OFF \
    -DCMAKE_BUILD_RPATH="${CONDA_PREFIX}/lib" \
    -DCMAKE_INSTALL_RPATH="${CONDA_PREFIX}/lib" \
    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
    -DPYTHON_EXECUTABLE="$(which python)"

cmake --build "${BUILD_DIR}" --target arx5_interface -j"$(nproc)"

# The .so is written directly to python/ by CMakeLists
SO_FILE=$(find "${SCRIPT_DIR}/python" -name "arx5_interface*.so" | head -1)
if [[ -n "${SO_FILE}" ]]; then
    echo ""
    echo "✓ Build succeeded: ${SO_FILE}"
    echo ""
    echo "Verify with:"
    echo "  python -c 'import arx5_interface; print(arx5_interface)'"
else
    echo "ERROR: arx5_interface.so not found after build." >&2
    exit 1
fi
