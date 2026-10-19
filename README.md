<h1 align="center">PyTorch-for-SAIL</h1>

<p align="center">
  <strong>PyTorch for Zhenwu PPU with T-Head SAIL</strong>
</p>

[English](README.md) | [简体中文](README.zh.md)

[Overview](#overview) · [Key Features](#key-features) · [Hardware Support](#hardware-support) · [Installation](#installation) · [Quick Start](#quick-start)

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.10-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
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
- **Compilation optimization:** Supports `torch.compile` (Inductor backend) with T-Head SAIL-adapted Triton 3.6.0.

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

Build the T-Head SAIL wheel from source. If public binary packages or container images become available, follow the release notes for the corresponding version.

### Requirements

- Linux on x86-64
- Python 3.10 or later
- Git, CMake 3.27 or later, Ninja, and the Python build dependencies in `requirements.txt`
- All Git submodules initialized
- [T-Head SAIL SDK](https://developer.t-head.cn/download/index.html) (`PPU_SDK`), installed at `/usr/local/PPU_SDK` by default
- For SDK v2.2, a separately installed PCCL package that matches the SDK/runtime and operating system

Initialize the source tree and dependencies:

```bash
git submodule sync --recursive
git submodule update --init --recursive

# Run the following commands inside the Docker container
python3 -m pip install -r requirements.txt
```

### Build from Source

Build the T-Head SAIL wheel directly with `python3 setup.py bdist_wheel`. The primary switch for SAIL mode is `USE_SAIL=1`. Device code is compiled with `hgcc` and linked against native T-Head SAIL SDK libraries.

#### SDK v2.2 and PCCL

Starting with T-Head SAIL SDK v2.2, PCCL is distributed separately from the SDK. Install the SDK for the `hgcc` compiler and device runtime, and install a matching PCCL package for distributed communication.

Download PCCL from the [T-Head Developer Center](https://developer.t-head.cn/download/index.html). On the download page, select the package by all of the following attributes:

- **Artifact type:** select the PCCL library package required by the build, rather than an unrelated tool package.
- **Artifact version:** match the PCCL release required by the target SDK and project release.
- **hggcrt version:** match the `hggcrt` generation used by the installed SDK, such as `hggcrt3`.
- **Operating system:** match the target Linux distribution and version, such as Ubuntu 24.04.

Select the appropriate PCCL package for your build requirements based on the artifact type, artifact version, `hggcrt` version, and operating system, and do not mix packages that are incompatible with the current environment.

Install and initialize the independently distributed PCCL package as follows:

```bash
# Replace the file name with the package selected for your environment.
tar zxf pccl_hggcrt3_ubuntu2404-v2.2.0.tar.gz -C /usr/local

source /usr/local/PPU_SDK/envsetup.sh
source /usr/local/pccl/envsetup.sh release

export PCCL_INCLUDE_DIR="$PCCL_HOME/include"
export PCCL_LIB_DIR="$PCCL_HOME/lib"
```

PCCL has different configuration requirements at build time and runtime:

- **Build time:** When SDK v2.2 uses an external PCCL package, set `PCCL_INCLUDE_DIR` and `PCCL_LIB_DIR` after initializing the PCCL environment. The build system uses them to locate `pccl.h` and `libpccl.so`. These variables are not required for SDK v2.1 when using PCCL bundled with `PPU_SDK`.
- **Runtime:** PCCL header paths are not needed, and neither `PCCL_INCLUDE_DIR` nor `PCCL_LIB_DIR` is read. However, the dynamic linker must be able to locate `libpccl.so`. Run `source /usr/local/pccl/envsetup.sh release` before using torch to configure the required `LD_LIBRARY_PATH`. This step may be omitted when an equivalent system library search path is already configured; otherwise, importing torch or using distributed features may fail because `libpccl.so` cannot be found.

#### Build Option 1: In-place Conversion

In-place conversion is the default mode and suits CI or disposable checkouts; without `SAILIFY_OUTPUT_DIR`, source conversion and wheel compilation complete in one invocation.

```bash
cd pytorch
source /usr/local/PPU_SDK/envsetup.sh

# SDK v2.2 is the default: PCCL is distributed separately, so initialize it
# and specify the external PCCL paths.
source /usr/local/pccl/envsetup.sh release
export PCCL_INCLUDE_DIR="$PCCL_HOME/include"
export PCCL_LIB_DIR="$PCCL_HOME/lib"
# For SDK v2.1 with PCCL bundled in PPU_SDK, skip the three lines above.

export USE_SAIL=1
export PYTORCH_SAIL_ARCH="ppu_10;ppu_15"
export USE_PCCL=1
export USE_FLASH_ATTENTION=1
export USE_MEM_EFF_ATTENTION=1
export BUILD_TEST=1
export MAX_JOBS=128

# Release version; without these the wheel defaults to the development
# version derived from version.txt (with a +git identifier).
export PYTORCH_VERSION=2.10.0
export PYTORCH_BUILD_VERSION=2.10.0
export PYTORCH_BUILD_NUMBER=0

python3 setup.py bdist_wheel
```

> **Warning:** In-place conversion rewrites convertible sources in the current source tree and its `third_party` submodules. Do not use it in a development checkout with local changes that must be preserved.

#### Build Option 2: Out-of-place Conversion

Use out-of-place conversion for local development. Sailify writes converted sources to a separate directory, leaving the original Git checkout unchanged. The workflow has two stages: source conversion and wheel compilation.

Use the following out-of-place two-stage source build workflow:

```bash
cd pytorch
source /usr/local/PPU_SDK/envsetup.sh

# SDK v2.2 is the default: PCCL is distributed separately, so initialize it
# and specify the external PCCL paths.
source /usr/local/pccl/envsetup.sh release
export PCCL_INCLUDE_DIR="$PCCL_HOME/include"
export PCCL_LIB_DIR="$PCCL_HOME/lib"
# For SDK v2.1 with PCCL bundled in PPU_SDK, skip the three lines above.

export USE_SAIL=1
export PYTORCH_SAIL_ARCH="ppu_10;ppu_15"
export USE_PCCL=1
export USE_FLASH_ATTENTION=1
export USE_MEM_EFF_ATTENTION=1
export BUILD_TEST=1
export MAX_JOBS=128

# Release version; without these the wheel defaults to the development
# version derived from version.txt (with a +git identifier).
export PYTORCH_VERSION=2.10.0
export PYTORCH_BUILD_VERSION=2.10.0
export PYTORCH_BUILD_NUMBER=0

# This path must be outside the source tree, for example:
export SAILIFY_OUTPUT_DIR=/path/to/pytorch-sail-build

# Stage 1: copy and convert the sources into a converted tree; this stage does not build a wheel.
python3 setup.py sailify

# Stage 2: build the wheel in the converted source tree.
cd "$SAILIFY_OUTPUT_DIR"
python3 setup.py bdist_wheel
```

After a successful build, the wheel is available at:

```text
$SAILIFY_OUTPUT_DIR/dist/torch-*.whl
```

The `.sailify_done` marker in the converted tree prevents redundant conversion. After updating the original sources, set `SAILIFY_FORCE=1` during Stage 1 to regenerate the converted tree. Regeneration overwrites corresponding source files in that tree.

#### Build Option 3: Docker Build Environment (SDK v2.2)

`Dockerfile.sail-sdk2.2` builds a reusable environment image on top of the T-Head SAIL SDK v2.2 (`hggcrt3`) base image. It bundles the toolchain (CMake, Ninja, `hgcc`), the Python build dependencies and a matching PCCL, but not the source tree, so the image stays small and the repository is bind-mounted at build time. The base image is pinned by the `BASE_IMAGE` argument and can be overridden with `--build-arg BASE_IMAGE=...`.

Prerequisites:

- Docker installed, with network access to pull the base image `download.t-head.cn/docker_release/ppu:sdk2.2.0-hggcrt3-ubuntu24.04-py312-20261005`.
- A source checkout with all Git submodules initialized (`git submodule update --init --recursive`).
- A scratch directory on a large disk (tens of GB free) to hold the out-of-place build tree and the wheel.
- Building the wheel needs no PPU device; only running it does.

**Step 1 - Build the environment image.** The build context can be empty (the Dockerfile copies nothing), so point Docker at an empty directory to avoid streaming the source tree to the daemon:

```bash
docker build -f Dockerfile.sail-sdk2.2 -t pytorch-sail-buildenv:2.10-sdk2.2 <empty-dir>
```

**Step 2 - Compile the wheel** out-of-place by bind-mounting the source tree and the scratch directory. The source tree stays unchanged:

```bash
docker run --rm --name sail-build-v210 \
  -v "$PWD":/workspace/pytorch \
  -v /path/on/big/disk/sail-build:/workspace/sail-build \
  -e SAILIFY_OUTPUT_DIR=/workspace/sail-build \
  -e MAX_JOBS=128 \
  pytorch-sail-buildenv:2.10-sdk2.2 \
  bash -lc 'source "$PPU_SDK/envsetup.sh" && source "$PCCL_HOME/envsetup.sh" release && \
            cd /workspace/pytorch && python3 setup.py sailify && \
            cd "$SAILIFY_OUTPUT_DIR" && python3 setup.py bdist_wheel'
```

The wheel is written to `/path/on/big/disk/sail-build/dist/torch-2.10.0-cp312-cp312-linux_x86_64.whl` on the host (about 360 MB). On a many-core host the full build takes roughly 15-20 minutes and finishes with `0 FAILED`. The `dist/` directory may also contain an older wheel copied from the source tree during conversion, so confirm freshness by the file timestamp.

**Step 3 - Install and smoke-test** (on a host with PPU devices attached):

```bash
python3 -m pip install /path/on/big/disk/sail-build/dist/torch-*.whl
source /usr/local/pccl/envsetup.sh release
python3 -c "import torch; print(torch.__version__, torch.version.sail, torch.cuda.is_available())"
```

Notes:

- `SAILIFY_OUTPUT_DIR` must point outside the mounted source tree; omitting it selects in-place conversion, which rewrites the bind-mounted repository.
- PCCL: the image uses the PCCL bundled in the base image when present. If the base image ships none, download the matching package from the [T-Head Developer Center](https://developer.t-head.cn/download/index.html) and pass the link at image build time with `--build-arg PCCL_URL=<link>`. No download link is hard-coded in the Dockerfile.
- `MAX_JOBS` is not baked into the image; pass it per run and tune it to the host CPU and memory. Add `-e BUILD_TEST=1` to also build the C++ test targets.
- Building the wheel needs no PPU device. Running it does: start the runtime container with the PPU device nodes (for example `--device=/dev/alixpu_ctl --device=/dev/alixpu`).

#### Common Build Variables

| Variable | Example | Description |
| --- | --- | --- |
| `USE_SAIL` | `1` | Enables SAIL mode; required |
| `SAILIFY_OUTPUT_DIR` | `/path/to/pytorch-sail-build` | Out-of-place conversion directory; must be outside the source tree; setting it selects out-of-place conversion |
| `SAILIFY_IN_PLACE` | `1` | Explicitly selects in-place conversion (the default behavior); normally not needed; must not be set together with `SAILIFY_OUTPUT_DIR` |
| `SAILIFY_FORCE` | `1` | Forces source regeneration; set only when needed |
| `PYTORCH_SAIL_ARCH` | `ppu_10;ppu_15` | PPU target architecture list; defaults to `ppu_10;ppu_15`; see below |
| `USE_PCCL` | `1` | Enables native PCCL communication for SAIL builds |
| `PCCL_INCLUDE_DIR` / `PCCL_LIB_DIR` | `/usr/local/pccl/include` / `/usr/local/pccl/lib` | PCCL input directories; examples default to SDK v2.2 and point to the independent PCCL installation; SDK v2.1 needs no override when using PCCL bundled with `PPU_SDK` |
| `MAX_JOBS` | `128` | Maximum number of parallel compile jobs |
| `PYTORCH_VERSION` | `2.10.0` | Declares the wheel release version |
| `PYTORCH_BUILD_VERSION` | `2.10.0` | Version used when building a release wheel; must be set together with `PYTORCH_BUILD_NUMBER` |
| `PYTORCH_BUILD_NUMBER` | `0` | Build number; when set, the version no longer carries a `+git<commit>` local identifier |
| `BUILD_TEST` | `1` | Builds C++ test targets |
| `USE_FLASH_ATTENTION` / `USE_MEM_EFF_ATTENTION` | `1` | Builds the adapted attention operators |

#### Target Architectures

Set `PYTORCH_SAIL_ARCH` to select the Zhenwu PPU architectures supported by the wheel. Supported values are `ppu_10` and `ppu_15`; separate multiple architectures with semicolons and enclose the value in double quotes:

```bash
export PYTORCH_SAIL_ARCH="ppu_10"          # Build for ppu_10 only
export PYTORCH_SAIL_ARCH="ppu_10;ppu_15"   # Build for both ppu_10 and ppu_15
```

The build system passes the configured values directly to `hgcc` as architecture flags:

```text
-arch ppu_10 -arch ppu_15
```

`PYTORCH_SAIL_ARCH` defaults to `ppu_10;ppu_15` when unset or empty. Set it explicitly to build for one architecture or a different architecture combination. Unsupported architecture values fail during configuration.

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

To use `torch.compile` (Inductor backend), you can install the matching T-Head SAIL-adapted Triton 3.6.0 wheel as an alternative to building it from source. For the source code and build instructions, see [t-head/triton-for-sail](https://github.com/t-head/triton-for-sail).

---

## Quick Start

Initialize the SAIL SDK environment and confirm that the driver can see the device:

```bash
source /usr/local/PPU_SDK/envsetup.sh
ppu-smi
```

Run a minimal tensor example:

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

We thank the PyTorch team and all upstream contributors, as well as everyone who contributes code, tests, and documentation to PyTorch-for-SAIL.
