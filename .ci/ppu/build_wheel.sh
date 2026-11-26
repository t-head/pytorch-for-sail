#!/usr/bin/env bash
set -euo pipefail
# Remember the checkout root for the final wheel output.
REPO_ROOT="$(git rev-parse --show-toplevel)"
export REPO_ROOT
# ${VAR:-default} keeps a non-empty value; otherwise it uses the default.
# Load the SDK toolchain and runtime paths into this shell.
export PPU_SDK="${PPU_SDK:-/usr/local/PPU_SDK}"
export PCCL_HOME="${PCCL_HOME:-/usr/local/pccl}"
source "$PPU_SDK/envsetup.sh"

source "$(dirname "$0")/sdk_env.sh"
source "$(dirname "$0")/install_pccl.sh"
source "$(dirname "$0")/ccache_env.sh"

pip install cmake==3.31.10
pip install numpy==2.1.0
source "$PCCL_HOME/envsetup.sh" release

# SDK driver settings used by the validated build.
export UMD_PLATFORM_TYPE=1 HGGC_DRIVER_CANDIDATE=UMD
# Enable the PPU build, PCCL distributed training, and both SDPA backends.
export USE_SAIL=1 USE_PCCL=1 USE_DISTRIBUTED=1
export USE_FLASH_ATTENTION=1 USE_MEM_EFF_ATTENTION=1
# Tell CMake where to find PCCL headers and libraries.
export PCCL_ROOT="$PCCL_HOME"
export PCCL_INCLUDE_DIR="$PCCL_HOME/include" PCCL_LIB_DIR="$PCCL_HOME/lib"
# Build both architectures by default; choose ppu_10 or ppu_15 for just one.
export PYTORCH_SAIL_ARCH="${PYTORCH_SAIL_ARCH:-ppu_10;ppu_15}"
# Set the release version used in the wheel metadata and filename.
export PYTORCH_VERSION=2.14.1 PYTORCH_BUILD_VERSION=2.14.1 PYTORCH_BUILD_NUMBER=0
# Disable Caffe2; set BUILD_TEST=0 to skip native C++ test binaries.
export BUILD_CAFFE2=0 BUILD_TEST="${BUILD_TEST:-1}"
# Limit concurrent compilation jobs to fit available CPU/RAM.
export MAX_JOBS="${MAX_JOBS:-128}" CMAKE_BUILD_PARALLEL_LEVEL="${MAX_JOBS:-128}"
# Clear inherited build/import overrides; unset USE_XNNPACK restores its default.
unset TORCH_CUDA_ARCH_LIST CUDA_HOME CUDA_PATH CUDACXX USE_XNNPACK LD_PRELOAD
unset SAILIFY_INJECT_COMPAT_INCLUDE SAILIFY_IN_PLACE PYTHONPATH CMAKE_ARGS
unset BUILD_PYTHON_ONLY PYTORCH_EXTRA_INSTALL_REQUIREMENTS
# Pass the selected architecture list to CMake.
export CMAKE_ARGS="-DPYTORCH_SAIL_ARCH=$PYTORCH_SAIL_ARCH"

# Use a fresh conversion/build directory to avoid stale CMake cache settings.
BUILD_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/pytorch-sail-wheel.XXXXXX")"
export BUILD_ROOT
export SAILIFY_OUTPUT_DIR="$BUILD_ROOT/sail"
# Save the wheel in the mounted checkout so it survives container removal.
export WHEELHOUSE="$REPO_ROOT/dist"

# Bypass existing editable-install redirects during code generation.
export SKBUILD_EDITABLE_SKIP="$(python3 -c 'import os,sys; print(os.pathsep.join(str(f.path) for f in sys.meta_path if getattr(f, "path", None)))')"
# Convert into a separate tree and include SAIL compatibility headers.
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
python3 -m tools.sailify --output-dir "$SAILIFY_OUTPUT_DIR" --inject-compat-include

# Compile the converted sources, not the original CUDA sources.
cd "$SAILIFY_OUTPUT_DIR"
export PYTHONPATH="$SAILIFY_OUTPUT_DIR"
# Fail if code generation imports torchgen from a different source tree.
python3 -c 'import pathlib,torchgen.aoti.fallback_ops as m; p=pathlib.Path(m.__file__).resolve(); print(p); p.relative_to(pathlib.Path.cwd().resolve())'
mkdir -p "$WHEELHOUSE"
# Build a wheel only; -v prints detailed logs, --wheel-dir selects dist/.
# --no-build-isolation uses existing build dependencies; --no-deps skips dependency wheels.
# --no-cache-dir disables the pip cache; it does not clear CMake's build cache.
python3 -m pip wheel . -v --no-build-isolation --no-deps --no-cache-dir \
  --wheel-dir "$WHEELHOUSE"

# Return to the checkout and clear temporary import overrides.
cd "$REPO_ROOT"
unset PYTHONPATH SKBUILD_EDITABLE_SKIP
echo "[build_wheel] Done"