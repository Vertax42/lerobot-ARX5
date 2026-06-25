#!/bin/bash
set -euo pipefail

OS_NAME="$(uname -s)"
OS_VERSION=""

if [[ "$OS_NAME" == "Linux" ]]; then
    if command -v lsb_release &>/dev/null; then
        OS_VERSION="$(lsb_release -rs)"
    elif [[ -f /etc/os-release ]]; then
        . /etc/os-release
        OS_VERSION="${VERSION_ID}"
    fi
    if [[ "$OS_VERSION" != "22.04" && "$OS_VERSION" != "24.04" ]]; then
        echo "Warning: this script has only been tested on Ubuntu 22.04 and 24.04"
        echo "Your system is running Ubuntu ${OS_VERSION}."
        read -r -p "Do you want to continue anyway? (y/N): " REPLY
        if [[ ! "${REPLY}" =~ ^[Yy]$ ]]; then
            echo "Installation cancelled."
            exit 1
        fi
    fi
else
    echo "Unsupported operating system: ${OS_NAME}"
    exit 1
fi

echo "Operating system check passed: ${OS_NAME} ${OS_VERSION}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
FRANKA_HAND_ROOT="${WORKSPACE_ROOT}/franka_hand"

mkdir -p "${FRANKA_HAND_ROOT}"

python_importable() {
    local module="$1"
    python - "$module" <<'PY'
import importlib.util
import sys

module = sys.argv[1]
raise SystemExit(0 if importlib.util.find_spec(module) is not None else 1)
PY
}

python_can_import() {
    local module="$1"
    python - "$module" <<'PY'
import importlib
import sys

module = sys.argv[1]
try:
    importlib.import_module(module)
except Exception:
    raise SystemExit(1)
raise SystemExit(0)
PY
}

clone_repo() {
    local name="$1"
    local url="$2"
    local dest="${FRANKA_HAND_ROOT}/${name}"
    local https_url=""

    if [[ -d "${dest}/.git" ]]; then
        echo "[${name}] already cloned at ${dest}"
        return 0
    fi

    if [[ -e "${dest}" && ! -d "${dest}/.git" ]]; then
        echo "Error: ${dest} exists but is not a git checkout."
        exit 1
    fi

    if git clone "${url}" "${dest}"; then
        return 0
    fi

    if [[ "${url}" == git@github.com:* ]]; then
        https_url="https://github.com/${url#git@github.com:}"
        echo "[${name}] SSH clone failed; retrying with HTTPS: ${https_url}"
        if [[ -d "${dest}" ]] && [[ -z "$(ls -A "${dest}")" ]]; then
            rmdir "${dest}"
        fi
        git clone "${https_url}" "${dest}"
        return 0
    fi

    echo "Error: failed to clone ${name} from ${url}"
    exit 1
}

find_pxrea_runtime_lib() {
    local candidate
    local roots=(
        "${FRANKA_HAND_ROOT}"
        "${PROJECT_ROOT}"
        "${PROJECT_ROOT}/.."
        "${HOME:-}"
    )
    local candidates=(
        "${FRANKA_HAND_ROOT}/Xense-Pico-Teleop-Interface/xensevr-pc-service-pybind/lib/libPXREARobotSDK.so"
        "${PROJECT_ROOT}/../Xense-Pico-Teleop-Interface/xensevr-pc-service-pybind/lib/libPXREARobotSDK.so"
        "${PROJECT_ROOT}/third_party/XenseVR-PC-Service/RoboticsService/PXREARobotSDK/build/libPXREARobotSDK.so"
    )

    for candidate in "${candidates[@]}"; do
        if [[ -f "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done

    for root in "${roots[@]}"; do
        [[ -d "${root}" ]] || continue
        while IFS= read -r -d '' candidate; do
            printf '%s\n' "${candidate}"
            return 0
        done < <(
            find "${root}" \
                \( -path '*/xensevr-pc-service-pybind/lib/libPXREARobotSDK.so' \
                -o -path '*/RoboticsService/PXREARobotSDK/build/libPXREARobotSDK.so' \
                -o -path '*/RoboticsService/SDK/linux/64/libPXREARobotSDK.so' \) \
                -print0 2>/dev/null
        )
    done

    return 1
}

install_pxrea_runtime_lib() {
    local lib_path

    if ! lib_path="$(find_pxrea_runtime_lib)"; then
        return 1
    fi

    if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
        cp "${lib_path}" "${CONDA_PREFIX}/lib/libPXREARobotSDK.so"
        echo "[Xense-Pico-Teleop-Interface] installed libPXREARobotSDK.so to ${CONDA_PREFIX}/lib"
        return 0
    fi

    echo "[Xense-Pico-Teleop-Interface] found runtime library at ${lib_path}"
    echo "  Add this directory to LD_LIBRARY_PATH or activate a conda environment and rerun this script."
    return 1
}

find_franka_install_paths() {
    local roots=()
    local root candidate config_dir prefix_root library_dirs
    local canonical_root

    add_root() {
        local candidate_root="$1"

        [[ -n "${candidate_root}" ]] || return 0

        if [[ -f "${candidate_root}" ]]; then
            candidate_root="$(dirname "${candidate_root}")"
        fi

        if command -v readlink >/dev/null 2>&1; then
            canonical_root="$(readlink -f "${candidate_root}" 2>/dev/null || true)"
            if [[ -n "${canonical_root}" ]]; then
                candidate_root="${canonical_root}"
            fi
        fi

        if [[ -d "${candidate_root}" ]]; then
            roots+=("${candidate_root}")
        fi
    }

    add_root "${Franka_DIR:-}"
    add_root "${FRANKA_DIR:-}"

    if [[ -n "${FRANKA_SEARCH_ROOTS:-}" ]]; then
        local IFS=':'
        for root in ${FRANKA_SEARCH_ROOTS}; do
            add_root "${root}"
        done
    fi

    if [[ -n "${CMAKE_PREFIX_PATH:-}" ]]; then
        local IFS=':'
        for root in ${CMAKE_PREFIX_PATH}; do
            add_root "${root}"
        done
    fi

    add_root "/lib/cmake/Franka"
    add_root "/usr/lib/cmake/Franka"
    add_root "/usr/lib/x86_64-linux-gnu/cmake/Franka"
    add_root "/lib"
    add_root "/lib64"
    add_root "/usr/lib"
    add_root "/usr/lib64"
    add_root "/usr/lib/x86_64-linux-gnu"
    add_root "/opt"
    add_root "/usr/local"
    add_root "/usr"
    add_root "${HOME:-}"

    for root in "${roots[@]}"; do
        [[ -n "${root}" ]] || continue
        [[ -d "${root}" ]] || continue

        while IFS= read -r -d '' candidate; do
            config_dir="$(dirname "${candidate}")"
            case "${candidate}" in
                */lib/cmake/Franka/FrankaConfig.cmake)
                    prefix_root="${candidate%/lib/cmake/Franka/FrankaConfig.cmake}"
                    ;;
                */lib64/cmake/Franka/FrankaConfig.cmake)
                    prefix_root="${candidate%/lib64/cmake/Franka/FrankaConfig.cmake}"
                    ;;
                */share/Franka/cmake/FrankaConfig.cmake)
                    prefix_root="${candidate%/share/Franka/cmake/FrankaConfig.cmake}"
                    ;;
                */build/FrankaConfig.cmake)
                    prefix_root="${candidate%/build/FrankaConfig.cmake}"
                    ;;
                */cmake/Franka/FrankaConfig.cmake)
                    prefix_root="${candidate%/cmake/Franka/FrankaConfig.cmake}"
                    ;;
                *)
                    prefix_root="${config_dir}"
                    ;;
            esac

            if [[ -z "${prefix_root}" ]]; then
                prefix_root="/"
            fi

            if library_dirs="$(find_franka_library_dirs "${prefix_root}")"; then
                printf '%s\t%s\t%s\n' "${config_dir}" "${prefix_root}" "${library_dirs}"
                return 0
            fi
        done < <(find "${root}" -type f -name FrankaConfig.cmake -print0 2>/dev/null)
    done

    return 1
}

find_franka_library_dirs() {
    local prefix_root="$1"
    local dirs=()
    local candidate
    local candidates=()

    if [[ "${prefix_root}" == "/" ]]; then
        candidates=(
            "/lib"
            "/lib64"
            "/usr/lib"
            "/usr/lib64"
            "/usr/lib/x86_64-linux-gnu"
            "/usr/local/lib"
            "/usr/local/lib64"
        )
    else
        candidates=(
            "${prefix_root}/lib"
            "${prefix_root}/lib64"
            "${prefix_root}/lib/x86_64-linux-gnu"
            "${prefix_root}/build"
            "${prefix_root}/libfranka/build"
            "${prefix_root}/libfranka/lib"
        )
    fi

    for candidate in "${candidates[@]}"; do
        if [[ -d "${candidate}" ]] && compgen -G "${candidate}/libfranka.so*" >/dev/null; then
            dirs+=("${candidate}")
        fi
    done

    if [[ ${#dirs[@]} -eq 0 ]]; then
        return 1
    fi

    local IFS=:
    printf '%s\n' "${dirs[*]}"
}

find_franka_include_dirs() {
    local prefix_root="$1"
    local dirs=()
    local candidate
    local include_name
    local overlay="${FRANKA_HAND_ROOT}/.include"
    local candidates=()
    local include_names=(
        franka
        research_interface
        fmt
        pinocchio
    )

    candidates=(
        "${prefix_root}/include"
        "${prefix_root}/local/include"
        "/usr/include"
        "/usr/local/include"
    )

    if [[ "${prefix_root}" == "/usr" ]]; then
        candidates+=("/usr/include/x86_64-linux-gnu")
    fi

    for candidate in "${candidates[@]}"; do
        if [[ -f "${candidate}/franka/model.h" ]]; then
            mkdir -p "${overlay}"
            for include_name in "${include_names[@]}"; do
                if [[ -d "${candidate}/${include_name}" ]]; then
                    ln -sfn "${candidate}/${include_name}" "${overlay}/${include_name}"
                fi
            done
            dirs+=("${overlay}")
            break
        fi
    done

    if [[ ${#dirs[@]} -eq 0 ]]; then
        return 1
    fi

    local IFS=:
    printf '%s\n' "${dirs[*]}"
}

pip_install_editable() {
    local name="$1"
    local path="$2"
    local extra_args="${3:-}"

    echo "[${name}] installing from ${path}"
    if [[ -n "${extra_args}" ]]; then
        python -m pip install -e "${path}" ${extra_args}
    else
        python -m pip install -e "${path}"
    fi
}

patch_franky_disable_stubs() {
    local repo="$1"
    local setup_py="${repo}/setup.py"

    if [[ -f "${setup_py}" ]] && grep -q -- "-DBUILD_PYTHON_STUBS=ON" "${setup_py}"; then
        echo "[franky] disabling Python stub generation"
        sed -i 's/-DBUILD_PYTHON_STUBS=ON/-DBUILD_PYTHON_STUBS=OFF/g' "${setup_py}"
    fi
}

install_lerobot_keyboard_deps() {
    if python_can_import "pynput" && python_can_import "evdev"; then
        echo "[lerobot] pynput/evdev already importable, skipping"
        return 0
    fi

    echo "[lerobot] installing pynput/evdev from conda-forge"
    echo "[lerobot] this avoids pip building evdev against mismatched Linux headers"

    if command -v mamba >/dev/null 2>&1; then
        if mamba install -y -c conda-forge pynput evdev; then
            return 0
        fi
        if mamba install -y -c conda-forge pynput python-evdev; then
            return 0
        fi
        if mamba install -y -c conda-forge pynput; then
            return 0
        fi
    elif command -v conda >/dev/null 2>&1; then
        if conda install -y -c conda-forge pynput evdev; then
            return 0
        fi
        if conda install -y -c conda-forge pynput python-evdev; then
            return 0
        fi
        if conda install -y -c conda-forge pynput; then
            return 0
        fi
    else
        echo "[lerobot] WARNING: conda/mamba not found; pip may try to build evdev from source"
        return 0
    fi

    echo "[lerobot] WARNING: failed to install pynput/evdev with conda-forge"
    echo "[lerobot] if pip fails on evdev, run: mamba install -y -c conda-forge pynput evdev"
}

install_lerobot_core() {
    if python_importable "lerobot"; then
        echo "[lerobot] already importable, skipping"
        return 0
    fi

    install_lerobot_keyboard_deps

    echo "[lerobot] installing current repository"
    python -m pip install -e "${PROJECT_ROOT}"
}

install_xense_pico_interface() {
    local repo="${FRANKA_HAND_ROOT}/Xense-Pico-Teleop-Interface"

    if python_can_import "xensevr_pc_service_sdk"; then
        echo "[Xense-Pico-Teleop-Interface] already importable, skipping"
        return 0
    fi

    if python_importable "xensevr_pc_service_sdk"; then
        echo "[Xense-Pico-Teleop-Interface] Python extension found; repairing native runtime library"
        if install_pxrea_runtime_lib && python_can_import "xensevr_pc_service_sdk"; then
            echo "[Xense-Pico-Teleop-Interface] runtime library repaired, skipping rebuild"
            return 0
        fi
    fi

    if [[ ! -d "${repo}" ]]; then
        clone_repo "Xense-Pico-Teleop-Interface" "git@github.com:xensedyl/Xense-Pico-Teleop-Interface.git"
    fi

    echo "[Xense-Pico-Teleop-Interface] installing"
    pushd "${repo}" >/dev/null
    bash setup_env.sh --install
    popd >/dev/null

    install_pxrea_runtime_lib || true
    if ! python_can_import "xensevr_pc_service_sdk"; then
        echo "[Xense-Pico-Teleop-Interface] ERROR: xensevr_pc_service_sdk still cannot be imported"
        echo "  Check that libPXREARobotSDK.so exists and is visible to the dynamic linker."
        exit 1
    fi
}

install_dex_retargeting() {
    local repo="${FRANKA_HAND_ROOT}/dex-retargeting"

    if python_importable "dex_retargeting"; then
        echo "[dex-retargeting] already importable, skipping"
        return 0
    fi

    if [[ ! -d "${repo}" ]]; then
        clone_repo "dex-retargeting" "git@github.com:xensedyl/dex-retargeting.git"
    fi

    pip_install_editable "dex-retargeting" "${repo}"
}

install_franky() {
    local repo="${FRANKA_HAND_ROOT}/franky"
    local franka_config_dir
    local franka_prefix_root
    local franka_include_dirs
    local franka_library_dirs
    local franka_paths
    local cmake_prefix_path
    local cpath
    local cplus_include_path
    local library_path
    local ld_library_path

    if python_importable "franky"; then
        echo "[franky] already importable, skipping"
        return 0
    fi

    if [[ ! -d "${repo}" ]]; then
        clone_repo "franky" "git@github.com:xensedyl/franky.git"
    fi

    pushd "${repo}" >/dev/null
    git submodule update --init --recursive
    patch_franky_disable_stubs "${repo}"

    if ! python -m pybind11 --cmakedir >/dev/null 2>&1; then
        echo "[franky] installing pybind11"
        python -m pip install pybind11
    fi

    if ! franka_paths="$(find_franka_install_paths)"; then
        echo "[franky] ERROR: could not find FrankaConfig.cmake"
        echo "  Set Franka_DIR or FRANKA_SEARCH_ROOTS and rerun."
        echo "  Searched: Franka_DIR, FRANKA_DIR, FRANKA_SEARCH_ROOTS, CMAKE_PREFIX_PATH, \$HOME, /opt, /usr/local, /usr"
        popd >/dev/null
        exit 1
    fi

    IFS=$'\t' read -r franka_config_dir franka_prefix_root franka_library_dirs <<<"${franka_paths}"
    cmake_prefix_path="${franka_prefix_root}"
    if [[ -n "${CMAKE_PREFIX_PATH:-}" ]]; then
        cmake_prefix_path="${franka_prefix_root}:${CMAKE_PREFIX_PATH}"
    fi

    if ! franka_include_dirs="$(find_franka_include_dirs "${franka_prefix_root}")"; then
        echo "[franky] ERROR: found FrankaConfig.cmake but could not locate franka headers"
        echo "  Looked under: ${franka_prefix_root}"
        popd >/dev/null
        exit 1
    fi

    cpath="${franka_include_dirs}"
    if [[ -n "${CPATH:-}" ]]; then
        cpath="${franka_include_dirs}:${CPATH}"
    fi

    cplus_include_path="${franka_include_dirs}"
    if [[ -n "${CPLUS_INCLUDE_PATH:-}" ]]; then
        cplus_include_path="${franka_include_dirs}:${CPLUS_INCLUDE_PATH}"
    fi

    library_path="${franka_library_dirs}"
    if [[ -n "${LIBRARY_PATH:-}" ]]; then
        library_path="${franka_library_dirs}:${LIBRARY_PATH}"
    fi

    ld_library_path="${franka_library_dirs}"
    if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
        ld_library_path="${franka_library_dirs}:${LD_LIBRARY_PATH}"
    fi

    echo "[franky] using Franka config at ${franka_config_dir}"
    echo "[franky] Franka include overlay: ${franka_include_dirs}"
    echo "[franky] Franka library dirs: ${franka_library_dirs}"
    if ! env \
        Franka_DIR="${franka_config_dir}" \
        CMAKE_PREFIX_PATH="${cmake_prefix_path}" \
        CPATH="${cpath}" \
        CPLUS_INCLUDE_PATH="${cplus_include_path}" \
        LIBRARY_PATH="${library_path}" \
        LD_LIBRARY_PATH="${ld_library_path}" \
        python -m pip install -e . --no-build-isolation; then
        echo "[franky] ERROR: failed to build franky-control"
        echo "  Franka config: ${franka_config_dir}"
        echo "  Franka include overlay: ${franka_include_dirs}"
        echo "  Franka library dirs: ${franka_library_dirs}"
        echo "  If this machine needs Franka, make sure libfranka headers, libfranka.so, and FrankaConfig.cmake are the same version."
        popd >/dev/null
        exit 1
    fi
    popd >/dev/null
}

install_lerobot_robot_revo2_hand() {
    local repo="${FRANKA_HAND_ROOT}/lerobot-robot-revo2-hand"

    if python_importable "lerobot_robot_revo2_hand"; then
        echo "[lerobot-robot-revo2-hand] already importable, skipping"
        return 0
    fi

    if [[ ! -d "${repo}" ]]; then
        clone_repo "lerobot-robot-revo2-hand" "git@github.com:xensedyl/lerobot-robot-revo2-hand.git"
    fi

    pip_install_editable "lerobot-robot-revo2-hand" "${repo}"
}

install_lerobot_teleoperator_pico4_hand() {
    local repo="${FRANKA_HAND_ROOT}/lerobot-teleoperator-pico4-hand"

    if python_importable "lerobot_teleoperator_pico4_hand"; then
        echo "[lerobot-teleoperator-pico4-hand] already importable, skipping"
        return 0
    fi

    if [[ ! -d "${repo}" ]]; then
        clone_repo "lerobot-teleoperator-pico4-hand" "git@github.com:xensedyl/lerobot-teleoperator-pico4-hand.git"
    fi

    pip_install_editable "lerobot-teleoperator-pico4-hand" "${repo}"
}

install_lerobot_robot_franka_research3() {
    local repo="${FRANKA_HAND_ROOT}/lerobot-robot-franka-research3"

    if python_importable "lerobot_robot_franka_research3"; then
        echo "[lerobot-robot-franka-research3] already importable, skipping"
        return 0
    fi

    if [[ ! -d "${repo}" ]]; then
        clone_repo "lerobot-robot-franka-research3" "git@github.com:xensedyl/lerobot-robot-franka-research3.git"
    fi

    pip_install_editable "lerobot-robot-franka-research3" "${repo}"
}

install_lerobot_robot_franka_research3_dexhand() {
    local repo="${FRANKA_HAND_ROOT}/lerobot-robot-franka-research3-dexhand"

    if python_importable "lerobot_robot_franka_research3_dexhand"; then
        echo "[lerobot-robot-franka-research3-dexhand] already importable, skipping"
        return 0
    fi

    if [[ ! -d "${repo}" ]]; then
        clone_repo "lerobot-robot-franka-research3-dexhand" "git@github.com:xensedyl/lerobot-robot-franka-research3-dexhand.git"
    fi

    pip_install_editable "lerobot-robot-franka-research3-dexhand" "${repo}"
}

install_xense_pico_interface
install_dex_retargeting
install_franky
install_lerobot_core
install_lerobot_robot_revo2_hand
install_lerobot_teleoperator_pico4_hand
install_lerobot_robot_franka_research3
install_lerobot_robot_franka_research3_dexhand

echo "Done. Installed packages are managed under ${FRANKA_HAND_ROOT}"
