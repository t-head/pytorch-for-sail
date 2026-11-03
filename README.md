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
| SAIL SDK | The HGCC compiler toolchain and the device runtime. Kernels compile directly against the Zhenwu PPU instruction set, so the build has no CUDA runtime dependency. |
| sailify | A source conversion tool that rewires CUDA-facing entry points to the SAIL native interfaces. It keeps the upstream code structure intact, which keeps the branch easy to follow across upstream releases. |
| Adapted libraries | Zhenwu PPU-adapted forks of cutlass (actlize), flash-attention (flash-attention-for-sail), xformers (xformers-for-sail), and gloo (gloo-for-sail), integrated as submodules and wired into the build. The SDPA attention backends build on actlize and flash-attention-for-sail. |

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

**SDPA on Zhenwu PPU.** The Flash-Attention and Memory-Efficient Attention backends run through the adapted cutlass and flash-attention libraries. actlize maps matrix multiply-accumulate onto Zhenwu PPU Tensor Cell instructions.

**Runtime routing.** Inductor and the Python runtime are routed to the SAIL device (`torch.version.sail`, `torch._smi`), and the Inductor CUDA codegen compiles with the SAIL toolchain.

---

## Installation

### From Source

Two Docker-based workflows are provided. Method 1 is the one verified during development. A ready-to-run script, [`docker/sail/build-sail.sh`](docker/sail/build-sail.sh), wraps the Method 1 build steps: `./docker/sail/build-sail.sh` builds in place, and `./docker/sail/build-sail.sh <output-dir>` converts out of place.

#### Method 1: `Dockerfile.sail-sdk2.2`

[`Dockerfile.sail-sdk2.2`](docker/sail/Dockerfile.sail-sdk2.2) builds a compact environment image for the T-Head SAIL SDK v2.2.1 (hggcrt3). It upgrades the SDK bundled in the base image from v2.2.0 to v2.2.1, upgrades PCCL to v2.2.2, and pre-installs the Python packages needed to build PyTorch 2.14.1 (`scikit-build-core`, `spin`, and a matching `setuptools`). The source code is mounted from the host at build time instead of being copied into the image, keeping the image small. The image check, the two-stage build, and the hardware checks below are the ones verified during development.

**Step 1 — build the environment image.** The build context can be an empty directory; the Dockerfile copies nothing. `SDK_RUN_URL` and `PCCL_URL` default to placeholders and must be supplied. Get the SDK v2.2.1 (hggcrt3, ubuntu2404) `.run` installer and the matching PCCL v2.2.2 package from the [T-Head Developer Center download page](https://developer.t-head.cn/download/index.html):

```bash
docker build -f docker/sail/Dockerfile.sail-sdk2.2 \
  -t pytorch-sail-buildenv:2.14.1-sdk2.2.1 \
  --build-arg SDK_RUN_URL=<sdk-v2.2.1-run-url> \
  --build-arg PCCL_URL=<pccl-v2.2.2-package-url> \
  <empty-context-directory>
```

The image checks itself at the end of the build (SDK v2.2.1, PCCL v2.2.2, and the Python packages), so a broken environment fails here instead of in the middle of a long compile. `PIP_INDEX_URL` defaults to the TUNA mirror; if the build host cannot reach it, pass `--build-arg PIP_INDEX_URL=<index>` to use another mirror. If the container cannot download the SDK or PCCL package (a `403` error, for example), add `--add-host <download-host>:<public-ip>` to force the public address — see the Dockerfile header for details.

**Step 2 — run the two-stage build.** Stage one is `python3 -m tools.sailify` (in this version, `sailify` is no longer part of `setup.py`). By default, it converts the source tree in place, rewriting the mounted checkout. To keep the checkout untouched, convert to a separate directory instead: add `-e SAILIFY_OUTPUT_DIR=<outside-the-source-tree>`, pass `--output-dir "$SAILIFY_OUTPUT_DIR"` to sailify, and build from that directory. Stage two compiles and installs the wheel from the converted tree:

```bash
docker run --rm --name sail-build-v2141 \
  -v "$PWD":/workspace/pytorch \
  -e MAX_JOBS=128 \
  pytorch-sail-buildenv:2.14.1-sdk2.2.1 \
  bash -lc 'source "$PPU_SDK/envsetup.sh" && source "$PCCL_HOME/envsetup.sh" release && \
            cd /workspace/pytorch && USE_SAIL=1 python3 -m tools.sailify && \
            python3 -m pip install . --no-build-isolation -v'
```

When the build finishes, the wheel is in `dist/` (the source tree itself, since the conversion happens in place). The image already sets the build options you need: `USE_SAIL=1`, `USE_PCCL=1`, both attention backends, both target architectures (`PYTORCH_SAIL_ARCH="ppu_10;ppu_15"`), the release version numbers, and `MAX_JOBS=128`. To build for a single architecture, add `-e PYTORCH_SAIL_ARCH=ppu_10` (or `ppu_15`) — the compile takes roughly half the time. `BUILD_TEST` defaults to `1` (the C++ test binaries are compiled); set `-e BUILD_TEST=0` for a lean build.

**Step 3 — verify on hardware.** Start the image with the accelerator devices mapped (every device the host exposes, e.g. all `/dev/alixpu*` entries) and run the checks in [Getting Started](#getting-started):

```bash
docker run -it --rm \
  --device /dev/alixpu --device /dev/alixpu_ctl --device /dev/alixpu_sep \
  --device /dev/alixpu-caps \
  -v "$PWD":/workspace/pytorch \
  pytorch-sail-buildenv:2.14.1-sdk2.2.1 \
  bash -lc 'source "$PPU_SDK/envsetup.sh" && source "$PCCL_HOME/envsetup.sh" release && \
            pip install /workspace/pytorch/dist/torch-2.14.1-*.whl && \
            python3 -c "import torch; print(torch.__version__, torch.version.sail); print(torch.cuda.device_count())"'
```

#### Method 2: Use a downloaded SAIL SDK image

Download a compatible SAIL SDK Docker image from the [T-Head Developer Center download page](https://developer.t-head.cn/download/index.html) and set `PPU_IMAGE` to its local name or tag. The source checkout stays on the host and the build runs inside the container. Import the archive first if the download is an image file:

```bash
docker load -i <downloaded-image-archive.tar>
```

```bash
# 1. Clone the PyTorch source code and initialize submodules
git clone --recursive <pytorch-for-sail-repository-url> -b v2.14.1
cd pytorch-for-sail

# If --recursive was not used or submodules are incomplete, run:
git submodule sync
git submodule update --init --recursive

# 2. Start the downloaded SAIL SDK image with the source checkout mounted.
# Replace the placeholder with the local image name or tag obtained above.
export PPU_IMAGE=<downloaded-sail-sdk-image>
docker run -it --rm \
  -v "$(pwd):/workspace/pytorch-for-sail" \
  -w /workspace/pytorch-for-sail \
  --entrypoint /bin/bash \
  "${PPU_IMAGE}"

# Run the following commands inside the PyTorch-for-SAIL Docker container
# 3. Configure the build environment
source /usr/local/PPU_SDK/envsetup.sh

# Starting from artifact version v2.2, PCCL is distributed separately from the
# SDK. Download a package matching the artifact version and operating system
# from https://developer.t-head.cn/download/index.html and extract it (only
# once per container):
tar zxf <matching-pccl-package>.tar.gz -C /usr/local
source /usr/local/pccl/envsetup.sh release

# Artifact version v2.1 ships PCCL in the SDK; skip the two lines above and
# source the SDK environment only in that case.

# 4. Install build dependencies. The 2.14.1 build chain requires the
# scikit-build-core PEP 517 backend, spin, and setuptools >=77,<82 with
# packaging >=24.2 for the PEP 639 license metadata:
pip install "setuptools>=77.0.0,<82" "scikit-build-core>=1.0" spin wheel \
    "packaging>=24.2" pyyaml numpy requests six "typing-extensions>=4.15.0" \
    "sympy>=1.13.3" "networkx>=2.5.1" jinja2 "fsspec>=0.8.5" filelock \
    "optree>=0.13.0"

# Optional: uncomment these variables to print verbose build logs for troubleshooting
# export CUDA_VERBOSE_BUILD=1      # Print full commands used to compile CUDA sources
# export CMAKE_VERBOSE_MAKEFILE=1  # Print every compile and link command generated by CMake

# 5. Build configuration. PYTORCH_SAIL_ARCH defaults to "ppu_10;ppu_15" and
# covers the 810/810E and Zhenwu M890 boards; set it to ppu_10 or ppu_15 to
# build for a single board only.
export USE_SAIL=1
export PYTORCH_SAIL_ARCH="ppu_10;ppu_15"
export USE_PCCL=1
export USE_FLASH_ATTENTION=1
export USE_MEM_EFF_ATTENTION=1
export BUILD_TEST=1
export MAX_JOBS=128

# Release version. Without these, the wheel gets a development version tag.
export PYTORCH_VERSION=2.14.1
export PYTORCH_BUILD_VERSION=2.14.1
export PYTORCH_BUILD_NUMBER=0
export USE_DISTRIBUTED=1
export BUILD_CAFFE2=0

# 6. Convert the tree and build. By default, sailify converts the source
# tree in place, so the checkout is rewritten. To keep the checkout
# untouched, convert out of place instead: export SAILIFY_OUTPUT_DIR to a
# directory outside the source tree, add
# --output-dir "${SAILIFY_OUTPUT_DIR}" to the sailify command, and cd
# "${SAILIFY_OUTPUT_DIR}" before the pip install.
python3 -m tools.sailify
python3 -m pip install . --no-build-isolation -v
# The wheel is written to ./dist/.
```

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
