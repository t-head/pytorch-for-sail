ln -s /usr/local/PPU_SDK/CUDA_SDK /usr/local/cuda
source /usr/local/PPU_SDK/envsetup.sh
#cd pytorch

pip install -r requirements.txt

rm -rf build/ dist/ *.egg-info torch.egg-info
# 清理 CMake 缓存，避免上次构建的 USE_CUDSS/USE_CUFILE 等值残留
rm -f CMakeCache.txt

# ── 5.8 ccache 编译缓存 ───────────────────────────────────────────────
export USE_CCACHE=1           # [—]     启用 ccache 加速重复编译
export CCACHE_DIR=/root/.ccache # [—]   缓存目录（建议挂载到宿主机持久化）
export CCACHE_MAXSIZE=10G     # [—]     缓存上限 10GB

# ── 5.9 verbose 日志 ─────────────────────────────────────────────────
export CUDA_VERBOSE_BUILD=1   # [—]     CMake 打印每个编译/链接命令
export CMAKE_VERBOSE_MAKEFILE=1

# ── 5.10 构建日志 ─────────────────────────────────────────────────
build_log="build_$(date +%Y%m%d_%H%M%S).log"
echo "Build log: ${build_log}"

# 编译sm80和sm89
export TORCH_CUDA_ARCH_LIST='8.0a'

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
python3 setup.py bdist_wheel 2>&1 | tee "${build_log}"
