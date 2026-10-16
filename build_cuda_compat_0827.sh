#!/usr/bin/env bash
# CUDA-compatible (cudafy) build for PPU — 2026-08-27
# Run inside docker pytorch_2-10_ppu202607212121_dh_0727
set -o pipefail

cd /mnt/ssd/huangdahai.hdh/project/pytorch-v2.10.0

# ── 环境准备 ─────────────────────────────────────────────────────────
[ -e /usr/local/cuda ] || ln -s /usr/local/PPU_SDK/CUDA_SDK /usr/local/cuda
source /usr/local/PPU_SDK/envsetup.sh

# 清理旧构建产物与 CMake 缓存（避免 SAIL 构建的缓存值残留）
# 增量重跑时设 SKIP_CLEAN=1 以复用已编译对象（版本切换时请勿使用）
if [ "${SKIP_CLEAN:-0}" != "1" ]; then
    rm -rf build/ dist/ *.egg-info torch.egg-info
    rm -f CMakeCache.txt
fi

# ccache（与 SAIL 构建隔离的缓存目录）
export USE_CCACHE=1
export CCACHE_DIR=/root/.ccache_cuda_compat
export CCACHE_MAXSIZE=10G

export MAX_JOBS=128

LOG_FILE=${LOG_FILE:-/mnt/ssd/huangdahai.hdh/project/pytorch-v2.10.0/cuda_compat_build_20260827.log}
echo "Build log: ${LOG_FILE}"

# ── 用户指定的构建参数 ───────────────────────────────────────────────
export TORCH_CUDA_ARCH_LIST="8.0"
# To build for SM89 only, comment out the line above and uncomment the following line.
# export TORCH_CUDA_ARCH_LIST="8.9"

# 3. Build the wheel package
NCCL_INCLUDE_DIR=/usr/local/PPU_SDK/CUDA_SDK/include \
NCCL_LIB_DIR=/usr/local/PPU_SDK/CUDA_SDK/lib64 \
PYTORCH_VERSION=2.13.0 \
PYTORCH_BUILD_VERSION=2.13.0 \
PYTORCH_BUILD_NUMBER=0 \
USE_FLASH_ATTENTION=True \
USE_MEM_EFF_ATTENTION=True \
USE_NCCL=True \
USE_DISTRIBUTED=True \
USE_SYSTEM_NCCL=1 \
BUILD_CAFFE2=False \
BUILD_TEST=True \
python3 setup.py bdist_wheel 2>&1 | tee "${LOG_FILE}"

echo "EXIT_CODE=${PIPESTATUS[0]}" | tee -a "${LOG_FILE}"
