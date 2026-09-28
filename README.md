# PyTorch-for-SAIL

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/License-BSD-green)](LICENSE)

[English](README.md) | [简体中文](README.zh.md)

---

## Table of Contents

- [Introduction](#introduction)
- [Supported Hardware](#supported-hardware)
- [User Guide](#user-guide)
- [Build from Source](#build-from-source)
- [Resources](#resources)
- [Security](#security)
- [Disclaimer](#disclaimer)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Introduction

PyTorch-for-SAIL is developed based on the community open-source PyTorch project. It provides system-level adaptation and performance optimization for Zhenwu PPU hardware, aiming to offer developers an out-of-the-box experience for deep learning training and inference.

The project continuously tracks official PyTorch 2.x releases. While maintaining compatibility with upstream functionality, it adds backend adaptations and performance optimizations for Zhenwu PPU. The current version supports SDPA (Scaled Dot-Product Attention) backend adaptation for Zhenwu PPU, including Flash-Attention and Memory-Efficient Attention implementations.

---

## Supported Hardware

- Zhenwu M890
- Zhenwu 810
- Zhenwu 805
- Zhenwu 610
- Zhenwu 810E
- Zhenwu 610E

---

## User Guide

To use PyTorch-for-SAIL directly via Docker or install it from PyPI, please refer to the [PyTorch-for-SAIL User Guide](https://www.flytiger-eco.com/docs_center/doc_detail/index.html?projectId=6&documentId=99).

---

## Build from Source

The source tree supports two build modes:

- **SAIL build:** converts device code with Sailify, compiles it with the native SAIL SDK `hgcc` toolchain, and links against PCCL from the SDK.
- **CUDA-compatible build:** preserves CUDA programming interfaces, automatically converts the PPU Flash-Attention, ACTLIZE/CUTLASS, and xFormers sources with cudafy, and builds them with the CUDA-compatible environment in the SAIL SDK.

Build inside the [PyTorch-for-SAIL Docker image](https://www.flytiger-eco.com/download?businessType=DOCKER). The commands below target the current PyTorch 2.13.0 source tree. `PYTORCH_VERSION` and `PYTORCH_BUILD_VERSION` must match the source version.

### Prepare the Source Tree and Dependencies

```bash
# 1. Clone the source and initialize all submodules, including nested ones
git clone --recursive https://github.com/flytiger-eco/pytorch-for-sail.git -b v2.13.0
cd pytorch-for-sail

git submodule sync --recursive
git submodule update --init --recursive

# The nested ACTLIZE checkout in xFormers is required by the attention build.
git -C third_party/xformers submodule update --init third_party/actlize

# 2. Run the following commands inside the Docker container
source /usr/local/PPU_SDK/envsetup.sh
python3 -m pip install -r requirements.txt
python3 -m pip install wheel

# Optional verbose build output
# export CUDA_VERBOSE_BUILD=1
# export CMAKE_VERBOSE_MAKEFILE=1

# Adjust this value to the available memory; 128 has been verified on a
# large-memory build host.
export MAX_JOBS=128
```

### SAIL Build (Recommended: Out-of-place Conversion)

#### Communication Library Paths

SAIL builds default PCCL_INCLUDE_DIR and PCCL_LIB_DIR to $PPU_SDK/include and $PPU_SDK/lib when unset or empty; explicit values override them. SAIL uses native PCCL; do not set USE_NCCL or USE_SYSTEM_NCCL. Non-SAIL CUDA-compatible builds must still configure NCCL_INCLUDE_DIR and NCCL_LIB_DIR explicitly. To use non-default PCCL, configure its native header and library directories:

```bash
export PCCL_INCLUDE_DIR=/opt/pccl/include
export PCCL_LIB_DIR=/opt/pccl/lib

test -f "$PCCL_INCLUDE_DIR/pccl.h"
test -f "$PCCL_LIB_DIR/libpccl.so"
```

Out-of-place conversion preserves the original Git checkout. The build has two stages: Sailify conversion and wheel compilation.

```bash
cd /path/to/pytorch-for-sail
source /usr/local/PPU_SDK/envsetup.sh
export MAX_JOBS=128

export USE_SAIL=1
export USE_FLASH_ATTENTION=1
export USE_MEM_EFF_ATTENTION=1
export USE_DISTRIBUTED=1
export BUILD_CAFFE2=0
export BUILD_TEST=1
export PYTORCH_VERSION=2.13.0
export PYTORCH_BUILD_VERSION=2.13.0
export PYTORCH_BUILD_NUMBER=0
export SAILIFY_OUTPUT_DIR=/path/to/pytorch-sail-build

# This path in the current xFormers checkout is an optional example symlink
# that is not part of the build. Ignore it to prevent conversion failure.
export SAILIFY_IGNORES="**/third_party/cutlass2/examples/common/accutlass.h"

# SAIL uses native PPU architecture names and must not inherit this variable
# from a CUDA-compatible build environment.
unset TORCH_CUDA_ARCH_LIST

# Stage 1: copy and convert the sources. This does not produce a wheel.
python3 setup.py sailify

# Stage 2: build the wheel in the converted tree.
cd "$SAILIFY_OUTPUT_DIR"
python3 setup.py bdist_wheel
```

After a successful build, the wheel is available at `$SAILIFY_OUTPUT_DIR/dist/torch-*.whl`. The `.sailify_done` file records a source fingerprint and prevents redundant conversion. Source changes trigger regeneration automatically; set `SAILIFY_FORCE=1` to force regeneration.

`PYTORCH_SAIL_ARCH` accepts `ppu_10` and `ppu_15`; separate multiple architectures with semicolons. When it is unset or empty, SAIL builds default to `ppu_10;ppu_15`; set it explicitly to override that target set.

For CI or a disposable checkout, build in place instead:

```bash
unset SAILIFY_OUTPUT_DIR
export SAILIFY_IN_PLACE=1
python3 setup.py bdist_wheel
```

> **Warning:** In-place conversion rewrites convertible files in the current source tree and its `third_party` submodules. Do not use it in a development checkout with local changes that must be preserved. `SAILIFY_IN_PLACE` and `SAILIFY_OUTPUT_DIR` are mutually exclusive.

### CUDA-Compatible Build

During a CUDA-compatible build, `setup.py` automatically and idempotently converts Flash-Attention 2.7.2, ACTLIZE/CUTLASS 0.8.0, xFormers 0.0.27, and the nested xFormers ACTLIZE 0.5.0 checkout. Conversion stamps are stored under `.cudafy-for-sail/` in each target directory.

Because cudafy modifies the affected submodule worktrees, use a disposable checkout or preserve local submodule changes before building:

```bash
for module in third_party/cutlass third_party/flash-attention \
              third_party/xformers third_party/cudafy-for-sail; do
    git -C "$module" stash push -u -m "before-cuda-compatible-build"
done
```

Build the wheel:

```bash
cd /path/to/pytorch-for-sail
source /usr/local/PPU_SDK/envsetup.sh
export MAX_JOBS=128

# Some build tools still resolve the conventional CUDA path.
[ -e /usr/local/cuda ] || ln -s /usr/local/PPU_SDK/CUDA_SDK /usr/local/cuda

# Clean stale CMake state and build outputs after switching modes or versions.
rm -rf build/ dist/ *.egg-info torch.egg-info
rm -f CMakeCache.txt

# The PPU CUDA-compatible toolchain uses this value for its adapted targets.
export TORCH_CUDA_ARCH_LIST="8.0"
# Select another target only when supported by the installed SDK, for example:
# export TORCH_CUDA_ARCH_LIST="8.9"

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
python3 setup.py bdist_wheel
```

After a successful build, the wheel is available at `dist/torch-*.whl`.

### Install and Verify

Install the newest wheel from the selected build mode:

```bash
# CUDA-compatible build
python3 -m pip install --force-reinstall --no-deps dist/torch-*.whl

# Out-of-place SAIL build
# python3 -m pip install --force-reinstall --no-deps \
#     "$SAILIFY_OUTPUT_DIR"/dist/torch-*.whl
```

Confirm device visibility and run the Transformer/SDPA tests. Run the test from `test/` so that the source-tree `torch/` directory does not shadow the installed wheel:

```bash
python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())'
cd /path/to/pytorch-for-sail/test
python3 test_transformers.py -v
```

For the current 2.13.0 source and verified environment, both SAIL and CUDA-compatible builds complete successfully. The baseline result for `test/test_transformers.py` in both modes is `Ran 15795 tests` and `OK (skipped=3001)`.

---

## Resources

- [PyTorch Official Documentation](https://docs.pytorch.org/docs/stable/index.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

---

## Security

For security information, please see [SECURITY.md](SECURITY.md).

## Disclaimer

- This software is provided for development and debugging purposes. Users assume all risks associated with its use.
- Users are responsible for managing data generated during use and complying with applicable security and compliance requirements.

## License

For the PyTorch-for-SAIL license, please see the [LICENSE](LICENSE) file.

## Acknowledgments

PyTorch-for-SAIL is developed based on the community open-source PyTorch project. We thank the PyTorch team and the open-source community for their contributions, and welcome developers to contribute code, documentation, and tests to PyTorch-for-SAIL.
