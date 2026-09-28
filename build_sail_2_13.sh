#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="${SOURCE_DIR:-$SCRIPT_DIR}"
SAILIFY_OUTPUT_DIR="${SAILIFY_OUTPUT_DIR:-${SOURCE_DIR}-sail-build}"
LOG_FILE="${LOG_FILE:-$SOURCE_DIR/sail_build_$(date +%Y%m%d).log}"

source /usr/local/PPU_SDK/envsetup.sh

# The container may expose an older system cmake before the Python cmake
# package. Prefer the package binary when available so setup.py sees the
# required version, but retain a valid system cmake as a fallback.
if CMAKE_BIN_DIR="$(python3 -c 'import cmake; from pathlib import Path; print(Path(cmake.__file__).parent / "data" / "bin")' 2>/dev/null)" \
    && [[ -x "$CMAKE_BIN_DIR/cmake" ]]; then
    export PATH="$CMAKE_BIN_DIR:$PATH"
fi
if ! command -v cmake >/dev/null; then
    echo "error: cmake >= 3.27 is required; install it or install the Python cmake package." >&2
    exit 2
fi

# Sailify copies files into the derived output tree without removing old
# wheels. Require an explicit override before reusing such an output tree.
if compgen -G "$SAILIFY_OUTPUT_DIR/dist/*.whl" >/dev/null \
    && [[ "${ALLOW_EXISTING_WHEELS:-0}" != "1" ]]; then
    echo "error: $SAILIFY_OUTPUT_DIR/dist contains existing wheels; use a fresh SAILIFY_OUTPUT_DIR or set ALLOW_EXISTING_WHEELS=1." >&2
    exit 2
fi

export USE_SAIL=1
export USE_FLASH_ATTENTION=1
export USE_MEM_EFF_ATTENTION=1
export USE_PCCL=1
unset USE_NCCL
unset USE_SYSTEM_NCCL
export BUILD_TEST="${BUILD_TEST:-0}"
export PYTORCH_SAIL_ARCH="ppu_10;ppu_15"
export MAX_JOBS=128

SOURCE_VERSION="$(tr -d '[:space:]' < "$SOURCE_DIR/version.txt")"
export PYTORCH_VERSION="${PYTORCH_VERSION:-$SOURCE_VERSION}"
export PYTORCH_BUILD_VERSION="${PYTORCH_BUILD_VERSION:-$SOURCE_VERSION}"
export PYTORCH_BUILD_NUMBER="${PYTORCH_BUILD_NUMBER:-0}"
export SAILIFY_OUTPUT_DIR
# xformers carries an optional CUTLASS2 example symlink whose target is absent;
# it is not part of the PyTorch build and must not abort source conversion.
export SAILIFY_IGNORES="**/third_party/cutlass2/examples/common/accutlass.h"

# SAIL 模式使用 PYTORCH_SAIL_ARCH，避免继承 CUDA 兼容模式的架构变量。
unset TORCH_CUDA_ARCH_LIST

{
    echo "[$(date '+%F %T')] SAIL build started"
    echo "source: $SOURCE_DIR"
    echo "output: $SAILIFY_OUTPUT_DIR"
    echo "architectures: $PYTORCH_SAIL_ARCH"
    echo "version: $PYTORCH_BUILD_VERSION"
    echo "cmake: $(cmake --version | head -1)"

    cd "$SOURCE_DIR"
    echo "[$(date '+%F %T')] Stage 1: sailify conversion"
    python3 setup.py sailify

    cd "$SAILIFY_OUTPUT_DIR"
    echo "[$(date '+%F %T')] Stage 2: wheel build"
    python3 setup.py bdist_wheel

    echo "[$(date '+%F %T')] SAIL build completed"
} 2>&1 | tee -a "$LOG_FILE"
