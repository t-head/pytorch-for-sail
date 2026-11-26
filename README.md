# PyTorch-for-SAIL

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/License-BSD-green)](LICENSE)

---

PyTorch-for-SAIL is a PyTorch distribution for Zhenwu PPU, built on the upstream PyTorch project and the T-HEAD SAIL software stack. The toolchain and runtime drive Zhenwu PPU directly.

The upstream PyTorch API is unchanged. Code written for `torch.cuda` runs on Zhenwu PPU without modification. The current version supports the SDPA Flash-Attention and Memory-Efficient Attention backends.

This project is a derivative work of PyTorch. The original PyTorch copyright and license notices are retained in [LICENSE](LICENSE) and [NOTICE](NOTICE).

<!-- toc -->

- [How It Works](#how-it-works)
- [Zhenwu PPU Support and Optimizations](#zhenwu-ppu-support-and-optimizations)
- [Installation](#installation)
  - [From Source](#from-source)
    - [Prepare the environment](#prepare-the-environment)
    - [Build a wheel without installing](#build-a-wheel-without-installing)
    - [Install a built wheel manually](#install-a-built-wheel-manually)
- [Getting Started](#getting-started)
- [Resources](#resources)
- [Security](#security)
- [Disclaimer](#disclaimer)
- [License](#license)
- [Acknowledgments](#acknowledgments)

<!-- tocstop -->

---

## How It Works

PyTorch-for-SAIL connects PyTorch to Zhenwu PPU with three pieces of the T-HEAD SAIL stack:

| Component | What it does |
| --------- | ------------ |
| SAIL SDK | HGCC compiler and runtime for Zhenwu PPU; no CUDA runtime is required. |
| sailify | Converts CUDA-facing source code to SAIL native interfaces. |
| Adapted libraries | actlize, flash-attention-for-sail, xformers-for-sail, and gloo-for-sail provide PPU kernels and communication support. |

Distributed training uses PCCL, the collective communication library for the SAIL stack.

---

## Zhenwu PPU Support and Optimizations

The current version runs on the SAIL SDK v2.2.x with the matching PCCL release.

| Feature | Notes |
| ------- | ----- |
| SDPA backends | Flash-Attention and Memory-Efficient Attention |
| Precision | FP16 / BF16 for Flash-Attention; FP16 / BF16 / FP32 for Memory-Efficient Attention (forward and backward) |
| Distributed training | Over PCCL |
| Hardware | Zhenwu M890 / 810 / 810E |

---

## Installation

### From Source

Prepare the environment, build a wheel into `dist/`, then install it manually when needed. The build commands do not install Torch or rewrite the original checkout.

Use the commands below for wheel-only builds. [`docker/sail/build-sail.sh`](docker/sail/build-sail.sh) still runs `pip install .` and is not a wheel-only helper.

> Download the T-Head SAIL SDK from the [T-Head Developer Center](https://developer.t-head.cn/download/index.html).
> For compatibility, this documentation retains references to `PPU_SDK`. `PPU_SDK` refers to
> the T-Head SAIL SDK; environment variable names and paths do not need to be changed.

#### Prepare the environment

The example uses Linux x86_64, Python 3.12, SAIL SDK v2.2.1, and PCCL v2.2.2 (hggcrt3, ubuntu2404). Start with the branch you want to build:

```bash
git clone --recursive -b v2.14.1 https://github.com/t-head/pytorch-for-sail.git
cd pytorch-for-sail
# For an existing checkout, initialize any missing submodules:
# git submodule update --init --recursive
```

Use a compatible SDK image from the download portal, or build one with the optional Dockerfile instructions below. Replace quoted placeholders before running commands.

<details>
<summary>Optional: build an SDK environment image</summary>

[`Dockerfile.sail-sdk2.2`](docker/sail/Dockerfile.sail-sdk2.2) installs SDK v2.2.1, PCCL v2.2.2, and Python build dependencies. Supply a compatible SDK v2.2.0/hggcrt3 base image and official package URLs:

```bash
docker build -f docker/sail/Dockerfile.sail-sdk2.2 \
  -t pytorch-sail-buildenv:2.14.1-sdk2.2.1 \
  --build-arg BASE_IMAGE="<compatible-sdk-v2.2.0-hggcrt3-image>" \
  --build-arg SDK_RUN_URL="<sdk-v2.2.1-run-url>" \
  --build-arg PCCL_URL="<pccl-v2.2.2-package-url>" \
  "<empty-context-directory>"
```

Use `pytorch-sail-buildenv:2.14.1-sdk2.2.1` as `PPU_IMAGE` below. See the Dockerfile for download and package-index options.

</details>

From the checkout root, start the container. Compilation does not need accelerator device mapping.

```bash
# If the SDK image was downloaded as an archive:
# docker load -i "<sdk-image-archive.tar>"
export PPU_IMAGE="<local-sail-sdk-image>"
docker run -it --rm -v "$PWD":/workspace/pytorch -w /workspace/pytorch \
  --entrypoint /bin/bash "$PPU_IMAGE"
```

Run the remaining build commands inside this container. If Python build dependencies are missing, install them first; skip this when using the prepared environment image:

```bash
python3 -m pip install "setuptools>=77.0.3,<82" "scikit-build-core>=1.0" wheel \
  "packaging>=24.2" pyyaml numpy requests six "typing-extensions>=4.15.0" \
  "sympy>=1.13.3" "networkx>=2.5.1" jinja2 "fsspec>=0.8.5" filelock \
  "optree>=0.13.0"
```

Also ensure CMake >=3.27, Ninja, a C/C++ compiler, Git, and `patch` are available. This prepares the environment; it does not install the Torch wheel.

#### Build a wheel without installing

Run these steps in the same Bash shell, starting at the checkout root. Source conversion and compilation use a fresh directory outside the checkout; the wheel is written directly to the checkout's `dist/`.

Local storage is recommended for the checkout and its initialized submodules. If copying from network storage is slow, prepare a complete checkout on local storage before starting these steps.

**1. Configure the build.** Keep `TMPDIR` on local storage outside the checkout, with enough space for the converted tree and native build.

```bash
# Stop on command failures, including failures inside pipelines.
set -eo pipefail
# Remember the checkout root for the final wheel output.
REPO_ROOT="$(git rev-parse --show-toplevel)"
export REPO_ROOT
# ${VAR:-default} keeps a non-empty value; otherwise it uses the default.
# Load the SDK toolchain and runtime paths into this shell.
export PPU_SDK="${PPU_SDK:-/usr/local/PPU_SDK}"
export PCCL_HOME="${PCCL_HOME:-/usr/local/pccl}"
source "$PPU_SDK/envsetup.sh"

# Download matching PCCL from the Developer Center if it is not installed.
# SDK v2.2.1 example: pccl_hggcrt3_ubuntu2404-v2.2.2.tar.gz (contains pccl/).
# Extract manually into the existing writable parent of PCCL_HOME; otherwise skip:
# tar -xzf "/path/to/pccl_hggcrt3_ubuntu2404-v2.2.2.tar.gz" -C "$(dirname "$PCCL_HOME")"
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
```

Set `MAX_JOBS`, `PYTORCH_SAIL_ARCH`, or `BUILD_TEST` before step 1 to override defaults. Adjust the `unset` lines if you intentionally use custom settings. Do not reuse an old `build/` after changing architecture or SDK.

**2. Convert and build.** Build from the converted tree, leaving the original checkout unchanged.

```bash
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
```

The wheel remains in `dist/` after the container exits. A same-name wheel can be overwritten, so preserve previous artifacts before rebuilding. No Torch installation occurs. CMake install messages refer to wheel staging, not the existing `site-packages`.

The validated package used SDK `2.2.1-184ee6`, PCCL 2.2.2, FlashAttention 2.7.2, and actlize 0.8.0. Wheel integrity and unpacked import/CPU smoke passed without changing the existing installation; hardware regression tests were not run.

#### Install a built wheel manually

Use the target environment's Python and choose its matching wheel. The example below is for CPython 3.12/Linux x86_64; it is not a `manylinux` portability guarantee. Runtime needs compatible SDK/PCCL and linked native libraries, such as MKL/OpenMPI for the validated build; pip does not install these native dependencies.

From the checkout root, check the wheel and install it only when wanted:

```bash
# Optional: check archive integrity without installing the wheel.
python3 -m zipfile -t ./dist/torch-2.14.1-cp312-cp312-linux_x86_64.whl
# Install only when needed, using the target environment's Python.
python3 -m pip install ./dist/torch-2.14.1-cp312-cp312-linux_x86_64.whl
```

To replace a same-version Torch installation, add `--force-reinstall`. Use `--no-deps` only when Python dependencies are already satisfied. Source the SDK/PCCL environment before importing Torch. Hardware execution also requires mapping all accelerator device nodes into the container.

---

## Getting Started

After installation, verify the build and run a simple computation on Zhenwu PPU:

```python
import torch

# Verify the SAIL build
print(torch.__version__, torch.version.sail)
print(torch.cuda.device_count())

# Tensors run on Zhenwu PPU through the familiar CUDA-facing API
x = torch.randn(1024, 1024, device="cuda")
y = torch.randn(1024, 1024, device="cuda")
print(torch.mm(x, y))
```

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
