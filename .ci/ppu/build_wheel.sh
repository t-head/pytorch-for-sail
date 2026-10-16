#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:?REPO_DIR 未设置}"
BUILD_ENVIRONMENT="${BUILD_ENVIRONMENT:-}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
BUILD_ADDITIONAL_PACKAGES="${BUILD_ADDITIONAL_PACKAGES:-}"
MAX_JOBS="${MAX_JOBS:-0}"

COMMIT=$(cat /tmp/ppu_source_commit 2>/dev/null || git -C "$REPO_DIR" rev-parse --short HEAD)
WHL_STAMP="$REPO_DIR/dist/.built_${COMMIT}_${TORCH_CUDA_ARCH_LIST// /_}"
if [[ -f "$WHL_STAMP" && "${FORCE_REBUILD:-0}" != "1" ]]; then
    echo "[build_wheel] 检测到已有编译产物（stamp $WHL_STAMP），跳过编译"
    ls "$REPO_DIR"/dist/torch-*.whl
    exit 0
fi

source "$(dirname "$0")/sdk_env.sh"
source "$(dirname "$0")/install_pccl.sh"
source "$(dirname "$0")/ccache_env.sh"

cd "$REPO_DIR"
echo "[build_wheel] BUILD_ENVIRONMENT=$BUILD_ENVIRONMENT"
echo "[build_wheel] TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

if [[ -n "$BUILD_ADDITIONAL_PACKAGES" ]]; then
    echo "[build_wheel] 警告: README 直编方式不支持 build-additional-packages: $BUILD_ADDITIONAL_PACKAGES（忽略）" >&2
fi

if [[ -n "${PIP_INDEX_URL:-}" ]]; then
    mkdir -p /root/.config/pip
    printf '[global]\nindex-url = %s\n' "$PIP_INDEX_URL" > /root/.config/pip/pip.conf
fi

apt remove -y cmake >/dev/null 2>&1 || true
pip uninstall -y cmake >/dev/null 2>&1 || true
pip cache purge >/dev/null 2>&1 || true
pip install cmake==3.31.10
ln -sf /usr/local/bin/cmake /usr/bin/cmake
pip install --upgrade pybind11
echo "[build_wheel] cmake=$(cmake --version | head -1)"
pip install -r requirements.txt

PPU_LINK_JOBS="${PPU_LINK_JOBS:-2}"
export CMAKE_JOB_POOLS="link=${PPU_LINK_JOBS}"
export CMAKE_JOB_POOL_LINK=link

TORCH_VERSION=$(sed -E 's/^([0-9]+\.[0-9]+\.[0-9]+).*/\1/' version.txt)
echo "[build_wheel] TORCH_VERSION=${TORCH_VERSION} MAX_JOBS=${MAX_JOBS} 链接并发=${PPU_LINK_JOBS}"

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

echo "${WHLS[*]##*/}" > "$WHL_STAMP"
echo "[build_wheel] 完成"
