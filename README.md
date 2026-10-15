<h1 align="center">PyTorch-for-SAIL</h1>

<p align="center">
  <strong>PyTorch for Zhenwu PPU with T-Head SAIL</strong>
</p>

[English](README.md) | [简体中文](README.zh.md)

[Overview](#overview) · [Key Features](#key-features) · [Hardware Support](#hardware-support) · [Installation](#installation) · [Quick Start](#quick-start)

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.13-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776ab?logo=python&logoColor=white" alt="Python">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-BSD--3--Clause-blue" alt="License"></a>
  <img src="https://img.shields.io/badge/Backend-Zhenwu%20PPU-6f42c1" alt="Zhenwu PPU">
</p>

---

## Overview

PyTorch-for-SAIL is developed based on the community open-source PyTorch project and brings PyTorch training and inference to Zhenwu PPU. It preserves familiar upstream Python APIs where possible while adapting the Zhenwu PPU toolchain, runtime, compute libraries, and distributed communication stack.

The SAIL build uses the native runtime and `hgcc` compiler from the T-Head SAIL SDK. For ecosystem compatibility, device APIs remain under the `torch.cuda` namespace.

> Download the T-Head SAIL SDK from the [T-Head Developer Center](https://developer.t-head.cn/download/index.html). For compatibility, this documentation retains references to `PPU_SDK`. `PPU_SDK` refers to the T-Head SAIL SDK; environment variable names and paths do not need to be changed.

This project is a derivative work of PyTorch. The original PyTorch copyright and license notices are retained in [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Key Features

- **PyTorch API compatibility:** Preserves familiar tensor, autograd, neural network, and optimizer workflows.
- **SAIL builds:** Converts device code with Sailify and compiles it with `hgcc`, then links against native T-Head SAIL SDK runtime and compute libraries.
- **Attention operators:** Provides Zhenwu PPU-adapted SDPA, Flash Attention, and Memory-Efficient Attention backends.
- **Distributed support:** Uses PCCL for collective communication and adapts the Gloo and TensorPipe build paths.
- **Compilation optimization:** Supports `torch.compile` with a matching T-Head SAIL-adapted Triton package.

## Hardware Support

Supported Zhenwu PPU models:

- Zhenwu M890
- Zhenwu 810E
- Zhenwu 810
- Zhenwu 610E
- Zhenwu 610

Capabilities may vary across SDK, driver, and hardware combinations. Use a SAIL SDK and driver release compatible with the target version.

---

## Installation

Build the T-Head SAIL wheel directly from source. If public binary packages or container images become available, follow the release notes for the corresponding version.

### Requirements

- Linux on x86-64
- Python 3.10 or later
- Git, CMake 3.27 or later, Ninja, and the Python build dependencies in `requirements.txt`
- All Git submodules initialized, including nested submodules
- [T-Head SAIL SDK](https://developer.t-head.cn/download/index.html) (`PPU_SDK`), installed at `/usr/local/PPU_SDK` by default
- For SDK v2.2, a separately installed PCCL package that matches the SDK/runtime, hggcrt generation, and operating system

Initialize the source tree and dependencies:

```bash
git submodule sync --recursive
git submodule update --init --recursive

# The nested ACTLIZE checkout is required by the xFormers attention build.
git -C third_party/xformers submodule update --init third_party/actlize

# Run the following command inside the build environment.
python3 -m pip install -r requirements.txt
```

### Build from Source

Build the T-Head SAIL wheel with `python3 setup.py bdist_wheel`. The primary switch for SAIL mode is `USE_SAIL=1`. Device code is compiled with `hgcc` and linked against native T-Head SAIL SDK libraries.

#### SDK v2.2 and PCCL

Starting with T-Head SAIL SDK v2.2, PCCL is distributed separately from the SDK. Install the SDK for the `hgcc` compiler and device runtime, and install a matching PCCL package for distributed communication.

Download PCCL from the [T-Head Developer Center](https://developer.t-head.cn/download/index.html). Select the package by all of the following attributes:

- **Artifact type:** select the PCCL library package required by the build.
- **Artifact version:** match the PCCL release required by the target SDK and project release.
- **hggcrt version:** match the `hggcrt` generation used by the installed SDK, such as `hggcrt3`.
- **Operating system:** match the target Linux distribution and version, such as Ubuntu 24.04.

Do not mix packages that are incompatible with the current environment. Install and initialize the independently distributed PCCL package as follows:

```bash
# Replace the file name with the package selected for your environment.
tar zxf pccl_hggcrt3_ubuntu2404-v2.2.0.tar.gz -C /usr/local

source /usr/local/PPU_SDK/envsetup.sh
source /usr/local/pccl/envsetup.sh release

export PCCL_INCLUDE_DIR="$PCCL_HOME/include"
export PCCL_LIB_DIR="$PCCL_HOME/lib"

test -f "$PCCL_INCLUDE_DIR/pccl.h"
test -f "$PCCL_LIB_DIR/libpccl.so"
```

PCCL has different configuration requirements at build time and runtime:

- **Build time:** When SDK v2.2 uses an external PCCL package, set `PCCL_INCLUDE_DIR` and `PCCL_LIB_DIR` after initializing the PCCL environment. The build system uses them to locate `pccl.h` and `libpccl.so`. These variables are not required for SDK v2.1 when using PCCL bundled with `PPU_SDK`.
- **Runtime:** PCCL header paths are not needed, and neither `PCCL_INCLUDE_DIR` nor `PCCL_LIB_DIR` is read. However, the dynamic linker must locate `libpccl.so`. Run `source /usr/local/pccl/envsetup.sh release` before importing torch or using distributed features unless an equivalent system library search path is already configured.

#### Build Option 1: In-place Conversion

In-place conversion is the default mode and suits CI or disposable checkouts. Without `SAILIFY_OUTPUT_DIR`, source conversion and wheel compilation complete in one invocation.

```bash
cd pytorch
source /usr/local/PPU_SDK/envsetup.sh

# SDK v2.2 is the default: initialize the independently installed PCCL package.
source /usr/local/pccl/envsetup.sh release
export PCCL_INCLUDE_DIR="$PCCL_HOME/include"
export PCCL_LIB_DIR="$PCCL_HOME/lib"
# For SDK v2.1 with PCCL bundled in PPU_SDK, skip the three lines above.

export USE_SAIL=1
export USE_PCCL=1
export USE_FLASH_ATTENTION=1
export USE_MEM_EFF_ATTENTION=1
export USE_DISTRIBUTED=1
export BUILD_CAFFE2=0
export BUILD_TEST=0
export MAX_JOBS=128

# Release version; omit these variables to use the development version derived
# from version.txt.
export PYTORCH_VERSION=2.13.0
export PYTORCH_BUILD_VERSION=2.13.0
export PYTORCH_BUILD_NUMBER=0

# Ignore an optional xFormers example symlink that is not part of the build.
export SAILIFY_IGNORES="**/third_party/cutlass2/examples/common/accutlass.h"

# SAIL uses native PPU architecture names.
unset TORCH_CUDA_ARCH_LIST

python3 setup.py bdist_wheel
```

> **Warning:** In-place conversion rewrites convertible sources in the current source tree and its `third_party` submodules. Do not use it in a development checkout with local changes that must be preserved.

#### Build Option 2: Out-of-place Conversion

Use out-of-place conversion for local development. Sailify writes converted sources to a separate directory, leaving the original Git checkout unchanged. The workflow has two stages: source conversion and wheel compilation.

```bash
cd pytorch
source /usr/local/PPU_SDK/envsetup.sh

# SDK v2.2 is the default: initialize the independently installed PCCL package.
source /usr/local/pccl/envsetup.sh release
export PCCL_INCLUDE_DIR="$PCCL_HOME/include"
export PCCL_LIB_DIR="$PCCL_HOME/lib"
# For SDK v2.1 with PCCL bundled in PPU_SDK, skip the three lines above.

export USE_SAIL=1
export USE_PCCL=1
export USE_FLASH_ATTENTION=1
export USE_MEM_EFF_ATTENTION=1
export USE_DISTRIBUTED=1
export BUILD_CAFFE2=0
export BUILD_TEST=0
export MAX_JOBS=128

export PYTORCH_VERSION=2.13.0
export PYTORCH_BUILD_VERSION=2.13.0
export PYTORCH_BUILD_NUMBER=0

# This path must be outside the source tree.
export SAILIFY_OUTPUT_DIR=/path/to/pytorch-sail-build
export SAILIFY_IGNORES="**/third_party/cutlass2/examples/common/accutlass.h"
unset TORCH_CUDA_ARCH_LIST

# Stage 1: copy and convert the sources; this stage does not build a wheel.
python3 setup.py sailify

# Stage 2: build the wheel in the converted source tree.
cd "$SAILIFY_OUTPUT_DIR"
python3 setup.py bdist_wheel
```

After a successful build, the wheel is available at:

```text
$SAILIFY_OUTPUT_DIR/dist/torch-*.whl
```

The `.sailify_done` marker prevents redundant conversion. Source updates trigger regeneration automatically; set `SAILIFY_FORCE=1` during Stage 1 to force regeneration. Sailify copies files into the output tree without removing unrelated files, so use a fresh output directory when stale artifacts may be present.

#### Common Build Variables

| Variable | Example | Description |
| --- | --- | --- |
| `USE_SAIL` | `1` | Enables SAIL mode; required |
| `SAILIFY_OUTPUT_DIR` | `/path/to/pytorch-sail-build` | Out-of-place conversion directory; must be outside the source tree |
| `SAILIFY_IN_PLACE` | `1` | Explicitly selects in-place conversion; normally not needed; must not be set with `SAILIFY_OUTPUT_DIR` |
| `SAILIFY_FORCE` | `1` | Forces source regeneration |
| `SAILIFY_IGNORES` | `**/third_party/cutlass2/examples/common/accutlass.h` | Ignores the optional broken xFormers example symlink during conversion |
| `PYTORCH_SAIL_ARCH` | `ppu_10;ppu_15` | PPU target architecture list; defaults to `ppu_10;ppu_15` |
| `USE_PCCL` | `1` | Enables native PCCL communication for SAIL builds |
| `PCCL_INCLUDE_DIR` / `PCCL_LIB_DIR` | `/usr/local/pccl/include` / `/usr/local/pccl/lib` | PCCL input directories for an external SDK v2.2 installation |
| `MAX_JOBS` | `128` | Maximum number of parallel compile jobs |
| `PYTORCH_VERSION` | `2.13.0` | Declares the wheel release version |
| `PYTORCH_BUILD_VERSION` | `2.13.0` | Release wheel version; set with `PYTORCH_BUILD_NUMBER` |
| `PYTORCH_BUILD_NUMBER` | `0` | Build number; removes the `+git<commit>` local identifier when set |
| `BUILD_TEST` | `0` | Set to `1` to build C++ test targets |
| `USE_FLASH_ATTENTION` / `USE_MEM_EFF_ATTENTION` | `1` | Builds the adapted attention operators |

#### Target Architectures

Set `PYTORCH_SAIL_ARCH` to select the Zhenwu PPU architectures supported by the wheel. Supported values are `ppu_10` and `ppu_15`; separate multiple architectures with semicolons and enclose the value in double quotes:

```bash
export PYTORCH_SAIL_ARCH="ppu_10"          # Build for ppu_10 only
export PYTORCH_SAIL_ARCH="ppu_10;ppu_15"   # Build for both targets
```

The build system passes the configured values directly to `hgcc`:

```text
-arch ppu_10 -arch ppu_15
```

`PYTORCH_SAIL_ARCH` defaults to `ppu_10;ppu_15` when unset or empty. Set it explicitly to build for one architecture or a different supported combination. Unsupported values fail during configuration.

### Install the Wheel

For an out-of-place build:

```bash
python3 -m pip uninstall -y torch
python3 -m pip install "$SAILIFY_OUTPUT_DIR"/dist/torch-*.whl
```

For an in-place build:

```bash
python3 -m pip uninstall -y torch
python3 -m pip install dist/torch-*.whl
```

### torch.compile and Triton

To use `torch.compile` with the Inductor backend, install a T-Head SAIL-adapted Triton package that matches this PyTorch release. For source code and build instructions, see [t-head/triton-for-sail](https://github.com/t-head/triton-for-sail).

---

## Quick Start

Initialize the SAIL SDK and PCCL runtime environments, then confirm that the driver can see the device:

```bash
source /usr/local/PPU_SDK/envsetup.sh
# Required for SDK v2.2 unless an equivalent library search path is configured.
source /usr/local/pccl/envsetup.sh release
ppu-smi
```

Run a minimal tensor example outside the source tree:

```python
import torch

assert torch.cuda.is_available(), "No available Zhenwu PPU was detected"

x = torch.randn(2, 2, device="cuda")
y = torch.randn(2, 2, device="cuda")
z = x @ y

print(f"PyTorch: {torch.__version__}")
print(f"SAIL: {getattr(torch.version, 'sail', None)}")
print(z)
```

PyTorch-for-SAIL retains the `torch.cuda` device interface, so existing PyTorch models can typically continue to use `model.to("cuda")` and `tensor.to("cuda")` to move workloads to Zhenwu PPU.

## Disclaimer

- This software is provided for development and debugging purposes. Users assume all risks associated with its use.
- Users are responsible for managing data generated during use and complying with applicable security and compliance requirements.

## License

PyTorch-for-SAIL is licensed under the BSD 3-Clause License. See [LICENSE](LICENSE).

## Acknowledgments

PyTorch-for-SAIL is developed based on the community open-source PyTorch project. We thank the PyTorch team and the open-source community for their contributions, and welcome developers to contribute code, documentation, and tests to PyTorch-for-SAIL.
