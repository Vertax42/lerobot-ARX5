#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# build_python.sh — Build pyarx Python bindings and install into the active
#                   conda / venv as a pip-editable package.
#
# Requirements:
#   conda activate lerobot   (pybind11, eigen, orocos-kdl,
#                             ros-jazzy-kdl-parser, ninja, soem)
#   System: /usr/local/lib/libspdlog.a (spdlog 1.14.1, fmt::v10 bundled)
#
# Usage:
#   cd src/lerobot/robots/bi_arx5/ARX5_SDK
#   bash build_python.sh
#
# Output:
#   pyarx/_arx5_interface.cpython-<ver>-<arch>-linux-gnu.so
#
# After build, the package is pip-installed as an editable install so that
#   import pyarx
# works in any Python session using the active environment.
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Verify conda env ────────────────────────────────────────────────────────
if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "ERROR: CONDA_PREFIX is not set. Please activate the lerobot conda env first:"
    echo "  conda activate lerobot"
    exit 1
fi

echo "Using conda env: ${CONDA_PREFIX}"

# ── Build ───────────────────────────────────────────────────────────────────
BUILD_DIR="${SCRIPT_DIR}/build"
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

cmake \
    -S "${SCRIPT_DIR}" \
    -B "${BUILD_DIR}" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="/usr/local;${CONDA_PREFIX}" \
    -DSPDLOG_FMT_EXTERNAL=OFF \
    -DCMAKE_BUILD_RPATH="${CONDA_PREFIX}/lib" \
    -DCMAKE_INSTALL_RPATH="${CONDA_PREFIX}/lib" \
    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
    -DPYTHON_EXECUTABLE="$(which python)"

cmake --build "${BUILD_DIR}" --target _arx5_interface -j"$(nproc)"

# The .so is written to pyarx/ by CMakeLists
SO_FILE=$(find "${SCRIPT_DIR}/pyarx" -name "_arx5_interface*.so" | head -1)
if [[ -z "${SO_FILE}" ]]; then
    echo "ERROR: _arx5_interface.so not found after build." >&2
    exit 1
fi

echo ""
echo "✓ Build succeeded: ${SO_FILE}"

# ── Install ──────────────────────────────────────────────────────────────────
echo ""
echo "Installing pyarx as editable package..."
"$(which pip)" install -e "${SCRIPT_DIR}" --no-build-isolation --quiet

echo ""
echo "✓ pyarx installed. Verify with:"
echo "  python -c 'import pyarx; print(pyarx)'"
