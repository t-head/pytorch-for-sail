#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:?REPO_DIR is not set}"
BUILD_ENVIRONMENT="${BUILD_ENVIRONMENT:-}"
MAX_JOBS="${MAX_JOBS:-0}"

source "$(dirname "$0")/sdk_env.sh"
source "$(dirname "$0")/install_pccl.sh"
source "$(dirname "$0")/ccache_env.sh"

unset TORCH_CUDA_ARCH_LIST

cd "$REPO_DIR"
pip install cmake==3.31.10
pip install numpy==2.1.0

TORCH_VERSION=$(sed -E 's/^([0-9]+\.[0-9]+\.[0-9]+).*/\1/' version.txt)

echo "[build_wheel] Cleaning the old build directory before each build"
rm -rf "$REPO_DIR/build"

BUILD_START=$(date +%s)

pip install "setuptools>=77.0.0,<82" "scikit-build-core>=1.0" spin wheel \
    "packaging>=24.2" pyyaml numpy requests six "typing-extensions>=4.15.0" \
    "sympy>=1.13.3" "networkx>=2.5.1" jinja2 "fsspec>=0.8.5" filelock \
    "optree>=0.13.0"

export USE_SAIL=1
export PYTORCH_SAIL_ARCH="ppu_10;ppu_15"
export USE_PCCL=1
export USE_FLASH_ATTENTION=1
export USE_MEM_EFF_ATTENTION=1
export BUILD_TEST=1
export MAX_JOBS=4

# Release version. Without these, the wheel gets a development version tag.
export PYTORCH_VERSION=2.14.1
export PYTORCH_BUILD_VERSION=2.14.1
export PYTORCH_BUILD_NUMBER=0
export USE_DISTRIBUTED=1
export BUILD_CAFFE2=0

python3 -m tools.sailify
python3 -m pip install . --no-build-isolation -v

echo "[build_wheel] Done"