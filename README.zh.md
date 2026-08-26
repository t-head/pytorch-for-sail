<h1 align="center">PyTorch-for-SAIL</h1>

<p align="center">
  <strong>基于 T-Head SAIL 的真武 PPU 版 PyTorch</strong>
</p>

[English](README.md) | [简体中文](README.zh.md)

[概述](#概述) · [核心特性](#核心特性) · [硬件支持](#硬件支持) · [安装](#安装) · [快速开始](#快速开始)

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.10-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776ab?logo=python&logoColor=white" alt="Python">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-BSD--3--Clause-blue" alt="License"></a>
  <img src="https://img.shields.io/badge/Backend-Zhenwu%20PPU-6f42c1" alt="真武 PPU">
</p>

---

## 概述

PyTorch-for-SAIL 基于社区开源 PyTorch 项目开发，为真武 PPU 提供 PyTorch 训练与推理能力。项目尽量保持上游 PyTorch 的 Python API 和使用习惯，并针对真武 PPU 工具链、运行时、计算库及分布式通信进行适配。

SAIL 构建使用 T-Head SAIL SDK 提供的原生运行时和 `hgcc` 编译器。为兼容 PyTorch 生态，设备 API 仍沿用 `torch.cuda` 命名空间。

> T-Head SAIL SDK 可从 [T-Head 开发者中心下载页面](https://developer.t-head.cn/download/index.html) 获取。出于兼容性考虑，文档中保留了对 `PPU_SDK` 的引用。请注意，`PPU_SDK` 即指代 T-Head SAIL SDK，您在配置环境变量或引用路径时无需修改。

本项目是 PyTorch 的衍生作品。PyTorch 原始版权及许可证声明保留在 [LICENSE](LICENSE) 和 [NOTICE](NOTICE) 中。

## 核心特性

- **PyTorch API 兼容：** 保持主流张量、自动求导、神经网络和优化器接口的使用方式。
- **SAIL 构建：** 设备代码经 Sailify 转换后由 `hgcc` 编译，链接 T-Head SAIL SDK 原生运行时与计算库。
- **注意力算子：** 支持面向真武 PPU 适配的 SDPA、Flash Attention 和 Memory-Efficient Attention 后端。
- **分布式能力：** 通过 SAIL SDK 提供的 PCCL 支持集合通信，并适配 Gloo 和 TensorPipe 构建路径。
- **编译优化：** 支持配合 T-Head SAIL 适配版 Triton 3.6.0 使用 `torch.compile`（Inductor 后端）。

## 硬件支持

支持的真武 PPU 型号：

- 真武 M890
- 真武 810E
- 真武 810
- 真武 610E
- 真武 610

不同 SDK、驱动和硬件组合的可用能力可能存在差异，请使用与目标版本匹配的 SAIL SDK 和驱动。

---

## 安装

目前请从源码构建 T-Head SAIL wheel。公开二进制包或容器镜像发布后，应以对应版本的发布说明为准。

### 环境要求

- x86-64 Linux
- Python 3.10 或更高版本
- Git、CMake 3.27 或更高版本、Ninja 以及 `requirements.txt` 中的 Python 构建依赖
- 完整初始化的 Git 子模块
- [T-Head SAIL SDK](https://developer.t-head.cn/download/index.html)（`PPU_SDK`），默认安装路径为 `/usr/local/PPU_SDK`
- 通过 SDK 环境提供的 `hgcc` 编译器和 PCCL 库

初始化源码和依赖：

```bash
git submodule sync --recursive
git submodule update --init --recursive

# 以下命令需在 Docker 容器内执行
python3 -m pip install -r requirements.txt
python3 -m pip install wheel
```

### 源码构建

使用 `python3 setup.py bdist_wheel` 直接构建 T-Head SAIL wheel。SAIL 模式的核心开关是 `USE_SAIL=1`。设备代码由 `hgcc` 编译，底层链接 T-Head SAIL SDK 原生库。

#### 方式一：原地转换

原地转换是默认转换方式，适用于 CI 或可丢弃的临时工作树；未设置 `SAILIFY_OUTPUT_DIR` 时，源码转换和 wheel 编译会在一次调用中完成。

```bash
cd pytorch
source /usr/local/PPU_SDK/envsetup.sh

export USE_SAIL=1
export USE_FLASH_ATTENTION=1
export USE_MEM_EFF_ATTENTION=1
export USE_NCCL=1
export USE_SYSTEM_NCCL=1
export BUILD_TEST=1
export PYTORCH_SAIL_ARCH="ppu_10;ppu_15"
export MAX_JOBS=32

# 发布版本号；不设置时默认为 version.txt 派生的开发版本（带 +git 标识）
export PYTORCH_VERSION=2.10.0
export PYTORCH_BUILD_VERSION=2.10.0
export PYTORCH_BUILD_NUMBER=0

python3 setup.py bdist_wheel
```

> **警告：** 原地转换会改写当前源码树及 `third_party` 子模块中的可转换源码。请勿在需要保留本地修改的开发工作树中使用。

#### 方式二：源树外转换

源树外转换适用于本地开发。Sailify 将转换结果写入独立目录，原始 Git 工作树保持不变。构建分为“源码转换”和“wheel 编译”两个阶段。

```bash
cd pytorch
source /usr/local/PPU_SDK/envsetup.sh

export USE_SAIL=1
export USE_FLASH_ATTENTION=1
export USE_MEM_EFF_ATTENTION=1
export USE_NCCL=1
export USE_SYSTEM_NCCL=1
export BUILD_TEST=1
export PYTORCH_SAIL_ARCH="ppu_10;ppu_15"
export MAX_JOBS=32

# 发布版本号；不设置时默认为 version.txt 派生的开发版本（带 +git 标识）
export PYTORCH_VERSION=2.10.0
export PYTORCH_BUILD_VERSION=2.10.0
export PYTORCH_BUILD_NUMBER=0

# 必须指向源码树之外的目录，例如：
export SAILIFY_OUTPUT_DIR=/path/to/pytorch-sail-build

# 阶段 1：复制并转换源码，生成转换树；此阶段不编译 wheel。
python3 setup.py sailify

# 阶段 2：在转换后的源码树中编译 wheel。
cd "$SAILIFY_OUTPUT_DIR"
python3 setup.py bdist_wheel
```

构建成功后，wheel 位于：

```text
$SAILIFY_OUTPUT_DIR/dist/torch-*.whl
```

转换树中的 `.sailify_done` 用于避免重复转换。原始源码更新后，可在阶段 1 设置 `SAILIFY_FORCE=1` 重新生成转换树；该操作会覆盖转换树中的对应源码文件。

#### 常用构建变量

| 变量 | 示例值 | 说明 |
| --- | --- | --- |
| `USE_SAIL` | `1` | 启用 SAIL 模式，必须设置 |
| `SAILIFY_OUTPUT_DIR` | `/path/to/pytorch-sail-build` | 源树外转换目录，必须位于源码树之外；设置后切换为源树外转换 |
| `SAILIFY_IN_PLACE` | `1` | 显式声明原地转换（默认行为），通常无需设置；不得与 `SAILIFY_OUTPUT_DIR` 同时设置 |
| `SAILIFY_FORCE` | `1` | 强制重新转换源码，仅在需要时设置 |
| `PYTORCH_SAIL_ARCH` | `ppu_10;ppu_15` | 真武 PPU 目标架构列表，详见下文 |
| `MAX_JOBS` | `32` | 最大并行任务数；根据可用内存调整 |
| `PYTORCH_VERSION` | `2.10.0` | wheel 发布版本号声明 |
| `PYTORCH_BUILD_VERSION` | `2.10.0` | 构建发布 wheel 时使用的版本号；须与 `PYTORCH_BUILD_NUMBER` 同时设置 |
| `PYTORCH_BUILD_NUMBER` | `0` | 构建编号；设置后版本号不再携带 `+git<提交号>` 本地标识 |
| `BUILD_TEST` | `1` | 构建 C++ 测试目标 |
| `USE_FLASH_ATTENTION` / `USE_MEM_EFF_ATTENTION` | `1` | 构建已适配的注意力算子 |
| `USE_NCCL` / `USE_SYSTEM_NCCL` | `1` | 使用 SAIL SDK 提供的 PCCL |

#### 目标架构

请通过 `PYTORCH_SAIL_ARCH` 指定 wheel 支持的真武 PPU 目标架构，可配置 `ppu_10`、`ppu_15`，多个架构使用英文分号分隔并用双引号包裹：

```bash
export PYTORCH_SAIL_ARCH="ppu_10"          # 仅编译 ppu_10
export PYTORCH_SAIL_ARCH="ppu_10;ppu_15"   # 同时编译 ppu_10 和 ppu_15
```

构建系统会将配置值直接传递为多个 `hgcc` 架构参数：

```text
-arch ppu_10 -arch ppu_15
```

必须显式设置 `PYTORCH_SAIL_ARCH`。未设置该变量或指定不支持的架构时，构建会报错并提示正确配置。

### 安装 wheel

源树外构建：

```bash
python3 -m pip uninstall -y torch
python3 -m pip install "$SAILIFY_OUTPUT_DIR"/dist/torch-*.whl
```

原地构建：

```bash
python3 -m pip uninstall -y torch
python3 -m pip install dist/torch-*.whl
```

### torch.compile 与 Triton

如需使用 `torch.compile`（Inductor 后端），可安装与本仓库匹配的 T-Head SAIL 适配版 Triton 3.6.0 wheel，作为源码构建安装之外的另一种安装方式。源码及构建说明请参见 [t-head/triton-for-sail](https://github.com/t-head/triton-for-sail)。

---

## 快速开始

初始化 SAIL SDK 环境并确认驱动可见：

```bash
source /usr/local/PPU_SDK/envsetup.sh
ppu-smi
```

运行一个最小张量计算示例：

```python
import torch

assert torch.cuda.is_available(), "未检测到可用的真武 PPU"

x = torch.randn(2, 2, device="cuda")
y = torch.randn(2, 2, device="cuda")
z = x @ y

print(f"PyTorch: {torch.__version__}")
print(f"SAIL: {getattr(torch.version, 'sail', None)}")
print(z)
```

PyTorch-for-SAIL 保留 `torch.cuda` 设备接口，因此现有 PyTorch 模型通常可以继续使用 `model.to("cuda")` 和 `tensor.to("cuda")` 迁移到真武 PPU。

## 免责声明

- 本软件仅供开发和调试使用，使用者需自行承担使用风险。
- 用户需自行管理运行过程中产生的数据，并遵守相关安全和合规要求。

## 许可证

PyTorch-for-SAIL 采用 BSD 3-Clause 许可证，详见 [LICENSE](LICENSE)。

## 致谢

感谢 PyTorch 团队及所有上游贡献者，并感谢每一位参与 PyTorch-for-SAIL 开发、测试和文档建设的社区贡献者。
