#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:?REPO_DIR 未设置}"
BUILD_ENVIRONMENT="${BUILD_ENVIRONMENT:-}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
BUILD_ADDITIONAL_PACKAGES="${BUILD_ADDITIONAL_PACKAGES:-}"
MAX_JOBS="${MAX_JOBS:-0}"

source "$(dirname "$0")/sdk_env.sh"
source "$(dirname "$0")/install_pccl.sh"
source "$(dirname "$0")/ccache_env.sh"

cd "$REPO_DIR"
pip install -r requirements.txt

TORCH_VERSION=$(sed -E 's/^([0-9]+\.[0-9]+\.[0-9]+).*/\1/' version.txt)

echo "[build_wheel] 每次编译前清理旧 build 目录"
rm -rf "$REPO_DIR/build"

BUILD_START=$(date +%s)

USE_SAIL=1 \
PYTORCH_SAIL_ARCH="ppu_10;ppu_15" \
USE_PCCL=1 \
USE_FLASH_ATTENTION=1 \
USE_MEM_EFF_ATTENTION=1 \
BUILD_TEST=1 \
MAX_JOBS="$MAX_JOBS" \
PYTORCH_VERSION="$TORCH_VERSION" \
PYTORCH_BUILD_VERSION="$TORCH_VERSION" \
PYTORCH_BUILD_NUMBER=0 \
python3 setup.py bdist_wheel

BUILD_END=$(date +%s)
BUILD_DURATION=$((BUILD_END - BUILD_START))
echo "$BUILD_DURATION" > /tmp/ppu_build_duration_sec
echo "[build_wheel] 编译耗时 ${BUILD_DURATION}s"

WHLS=("$REPO_DIR/dist"/torch-*.whl)
if [[ ! -e "${WHLS[0]}" ]]; then
    echo "[build_wheel] 错误: 未找到 dist/torch-*.whl" >&2
    exit 1
fi
echo "[build_wheel] 产物: $(ls "$REPO_DIR/dist"/torch-*.whl)"

if [[ "${PPU_CCACHE_ENABLED:-0}" == "1" ]]; then
    ccache -s 2>/dev/null | sed 's/^/[build_wheel] ccache /' || true
fi

echo "[build_wheel] 完成"
