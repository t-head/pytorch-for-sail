# PyTorch-for-SAIL

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/License-BSD-green)](LICENSE)

[English](README.md) | [简体中文](README.zh.md)

---

## 目录

- [简介](#简介)
- [支持的硬件型号](#支持的硬件型号)
- [用户指南](#用户指南)
- [源码编译](#源码编译)
- [资源链接](#资源链接)
- [安全声明](#安全声明)
- [免责声明](#免责声明)
- [许可证](#许可证)
- [致谢](#致谢)

---

## 简介

PyTorch-for-SAIL 基于社区开源 PyTorch 项目开发，面向真武 PPU 硬件进行系统级适配与性能优化，旨在为开发者提供开箱即用的深度学习训练与推理能力。

项目持续跟进 PyTorch 2.x 官方版本，在保持上游功能兼容的基础上，补充真武 PPU 相关后端适配和性能优化。当前版本支持 SDPA（Scaled Dot-Product Attention）模块的真武 PPU 后端适配，包括 Flash-Attention 与 Memory-Efficient Attention 实现。

---

## 支持的硬件型号

- 真武 M890
- 真武 810
- 真武 805
- 真武 610
- 真武 810E
- 真武 610E

---

## 用户指南

如需直接通过 Docker 使用 PyTorch-for-SAIL 或从 PyPI 安装，请参阅 [PyTorch-for-SAIL 用户指南](https://www.flytiger-eco.com/docs_center/doc_detail/index.html?projectId=6&documentId=99)。

---

## 源码编译

源码支持以下两种构建模式：

- **SAIL 构建**：通过 Sailify 转换设备代码，使用 SAIL SDK 原生 `hgcc` 工具链编译，并链接 SDK 提供的 PCCL。
- **CUDA 兼容构建**：保留 CUDA 编程接口，通过 cudafy 自动转换 PPU 版 Flash-Attention、ACTLIZE/CUTLASS 和 xFormers 源码，再使用 SAIL SDK 的 CUDA 兼容环境编译。

请在 [PyTorch-for-SAIL Docker 镜像](https://www.flytiger-eco.com/download?businessType=DOCKER)内执行编译。以下命令针对当前 PyTorch 2.13.0 源码，`PYTORCH_VERSION` 与 `PYTORCH_BUILD_VERSION` 必须和源码版本保持一致。

### 准备源码与依赖

```bash
# 1. 下载源码并初始化所有子模块（包括嵌套子模块）
git clone --recursive https://github.com/flytiger-eco/pytorch-for-sail.git -b v2.13.0
cd pytorch-for-sail

git submodule sync --recursive
git submodule update --init --recursive

# xFormers 的嵌套 ACTLIZE 是注意力算子构建的必要依赖；建议显式确认
# 该目录已初始化。
git -C third_party/xformers submodule update --init third_party/actlize

# 2. 以下命令需在 Docker 容器内执行
source /usr/local/PPU_SDK/envsetup.sh
python3 -m pip install -r requirements.txt
python3 -m pip install wheel

# 可选：输出详细编译命令
# export CUDA_VERBOSE_BUILD=1
# export CMAKE_VERBOSE_MAKEFILE=1

# 根据机器内存调整并行度；128 已在大内存构建环境中验证
export MAX_JOBS=128
```

### SAIL 构建（推荐：源树外转换）

源树外转换不会改写原始 Git 工作树。构建分为 Sailify 转换和 wheel 编译两个阶段。

```bash
cd /path/to/pytorch-for-sail
source /usr/local/PPU_SDK/envsetup.sh
export MAX_JOBS=128

export USE_SAIL=1
export USE_FLASH_ATTENTION=1
export USE_MEM_EFF_ATTENTION=1
export USE_NCCL=1
export USE_DISTRIBUTED=1
export USE_SYSTEM_NCCL=1
export BUILD_CAFFE2=0
export BUILD_TEST=1
export PYTORCH_SAIL_ARCH="ppu_10;ppu_15"
export PYTORCH_VERSION=2.13.0
export PYTORCH_BUILD_VERSION=2.13.0
export PYTORCH_BUILD_NUMBER=0
export SAILIFY_OUTPUT_DIR=/path/to/pytorch-sail-build

# 当前 xFormers 中该路径是非构建必需的可选示例链接；忽略它可避免转换中断。
export SAILIFY_IGNORES="**/third_party/cutlass2/examples/common/accutlass.h"

# SAIL 使用原生 PPU 架构名，不能继承 CUDA 兼容模式的架构变量。
unset TORCH_CUDA_ARCH_LIST

# 阶段 1：复制并转换源码。此阶段不生成 wheel。
python3 setup.py sailify

# 阶段 2：在转换后的源码树中编译 wheel。
cd "$SAILIFY_OUTPUT_DIR"
python3 setup.py bdist_wheel
```

构建成功后，wheel 位于 `$SAILIFY_OUTPUT_DIR/dist/torch-*.whl`。转换树中的 `.sailify_done` 会记录源码指纹并避免重复转换；原始源码更新后会自动重新转换，也可设置 `SAILIFY_FORCE=1` 强制重新生成转换树。

`PYTORCH_SAIL_ARCH` 支持 `ppu_10`、`ppu_15`，多个架构使用英文分号分隔。SAIL 模式下 `setup.py` 会自动将 NCCL 构建路径映射到 SAIL SDK 中的 PCCL，无需手工设置 `NCCL_INCLUDE_DIR` 和 `NCCL_LIB_DIR`。

如在 CI 或可丢弃的临时工作树中构建，也可改用原地构建：

```bash
unset SAILIFY_OUTPUT_DIR
export SAILIFY_IN_PLACE=1
python3 setup.py bdist_wheel
```

> **警告：** 原地转换会改写当前源码树及 `third_party` 子模块中的可转换源码，不要在需要保留本地修改的开发工作树中使用。`SAILIFY_IN_PLACE` 和 `SAILIFY_OUTPUT_DIR` 不得同时设置。

### CUDA 兼容构建

CUDA 兼容构建会在 `setup.py` 中自动、幂等地执行以下转换：Flash-Attention 2.7.2、ACTLIZE/CUTLASS 0.8.0、xFormers 0.0.27，以及 xFormers 嵌套 ACTLIZE 0.5.0。转换标记保存在各目标目录的 `.cudafy-for-sail/` 中。

由于 cudafy 会修改相关子模块工作树，建议使用可丢弃的检出目录，或在构建前保存子模块中的本地修改：

```bash
for module in third_party/cutlass third_party/flash-attention \
              third_party/xformers third_party/cudafy-for-sail; do
    git -C "$module" stash push -u -m "before-cuda-compatible-build"
done
```

执行构建：

```bash
cd /path/to/pytorch-for-sail
source /usr/local/PPU_SDK/envsetup.sh
export MAX_JOBS=128

# 部分工具仍会查找标准 CUDA 路径。
[ -e /usr/local/cuda ] || ln -s /usr/local/PPU_SDK/CUDA_SDK /usr/local/cuda

# 切换构建模式或版本后应清理旧 CMake 缓存和产物。
rm -rf build/ dist/ *.egg-info torch.egg-info
rm -f CMakeCache.txt

# PPU CUDA 兼容工具链特有行为：8.0 会启用已适配的目标架构编译。
export TORCH_CUDA_ARCH_LIST="8.0"
# 如需使用另一目标架构，请根据 SDK 支持情况调整，例如：
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

构建成功后，wheel 位于 `dist/torch-*.whl`。

### 安装与验证

安装对应模式生成的最新 wheel：

```bash
# CUDA 兼容构建
python3 -m pip install --force-reinstall --no-deps dist/torch-*.whl

# SAIL 源树外构建
# python3 -m pip install --force-reinstall --no-deps \
#     "$SAILIFY_OUTPUT_DIR"/dist/torch-*.whl
```

验证设备可见并运行 Transformer/SDPA 测试。必须进入 `test/` 目录执行，避免源码树中的 `torch/` 遮蔽已安装的 wheel：

```bash
python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())'
cd /path/to/pytorch-for-sail/test
python3 test_transformers.py -v
```

当前 2.13.0 代码在已验证环境中，SAIL 与 CUDA 兼容两种构建均成功，`test/test_transformers.py` 的基线结果均为 `Ran 15795 tests`、`OK (skipped=3001)`。

---

## 资源链接

- [PyTorch 官方文档](https://docs.pytorch.org/docs/stable/index.html)
- [PyTorch 教程](https://pytorch.org/tutorials/)

---

## 安全声明

安全相关说明请参见 [SECURITY.md](SECURITY.md)。

## 免责声明

- 本软件仅供开发和调试使用，使用者需自行承担使用风险。
- 用户需自行管理运行过程中产生的数据，并遵守相关安全和合规要求。

## 许可证

PyTorch-for-SAIL 的使用许可证，请参见 [LICENSE](LICENSE) 文件。

## 致谢

PyTorch-for-SAIL 基于社区开源 PyTorch 项目开发。感谢 PyTorch 团队和开源社区的贡献，欢迎开发者参与 PyTorch-for-SAIL 的代码、文档和测试贡献。
