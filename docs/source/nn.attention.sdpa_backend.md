# SDPA 调用链与 Backend 选择机制

本文梳理 `torch.nn.functional.scaled_dot_product_attention` 在 PyTorch 中的调用链、自动选择 backend 的机制，以及不同 backend 的适用场景。代码依据主要来自 `aten/src/ATen/native/transformers/cuda/sdp_utils.cpp`、`aten/src/ATen/native/transformers/sdp_utils_cpp.cpp`、`aten/src/ATen/native/transformers/attention.cpp`、`aten/src/ATen/native/transformers/cuda/attention.cu`、`torch/nn/attention/__init__.py` 和 `torch/backends/cuda/__init__.py`。

## 总体结论

SDPA 的自动选择是一个运行时 dispatch 过程：先根据设备 dispatch 到 CPU/CUDA/XPU 的 `_fused_sdp_choice`，再构造 `sdp_params`，最后按优先级遍历候选 backend。每个 backend 都有一组约束检查函数，只有编译期可用、运行期开关打开、输入形状和硬件条件全部满足时，才会被选中。

默认 CUDA 优先级通常是 `Flash Attention -> Memory Efficient Attention -> Math`。在满足特定 cuDNN 条件时，优先级会变为 `cuDNN Attention -> Flash Attention -> Memory Efficient Attention -> Math`。`Math` 是通用 fallback；如果用户关闭了 `Math` 且所有 fused backend 都不满足条件，最终会报错并打印各 backend 不可用原因。

## 调用链

### Python 与 ATen 入口

用户侧入口是：

```python
import torch
from torch.nn.functional import scaled_dot_product_attention

out = scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)
```

对应 native schema 定义在 `aten/src/ATen/native/native_functions.yaml`：

```text
scaled_dot_product_attention(Tensor query, Tensor key, Tensor value, Tensor? attn_mask=None, float dropout_p=0.0, bool is_causal=False, *, float? scale=None, bool enable_gqa=False) -> Tensor
```

为了测试和暴露选择结果，还定义了 `_fused_sdp_choice(...) -> int`。该函数按设备 dispatch：CPU/NestedTensorCPU 到 `_fused_sdp_choice_cpp`，CUDA/NestedTensorCUDA 到 `_fused_sdp_choice_cuda`，XPU 到 `_fused_sdp_choice_xpu`。

### CUDA 调用链

CUDA 侧的核心链路如下：

```text
scaled_dot_product_attention
  -> _fused_sdp_choice_cuda
    -> 构造 sdp::sdp_params{query, key, value, attn_mask, dropout_p, is_causal, enable_gqa}
    -> select_sdp_backend(kernel_params)
      -> priority_order(kernel_params)
      -> can_use_cudnn_attention / can_use_flash_attention / can_use_mem_efficient_attention / Math fallback
  -> 根据返回的 SDPBackend 调用对应实现
    -> _scaled_dot_product_flash_attention_cuda
    -> _scaled_dot_product_efficient_attention_cuda
    -> _scaled_dot_product_cudnn_attention_cuda
    -> _scaled_dot_product_attention_math
```

`_fused_sdp_choice_cuda` 位于 `aten/src/ATen/native/transformers/cuda/attention.cu`，其逻辑是构造 `sdp_params` 并调用 `select_sdp_backend`。具体 backend 的 CUDA 前向入口也在同一文件中：Flash 调用 `_flash_attention_forward`，Efficient 调用 `_efficient_attention_forward`，cuDNN 调用 `_cudnn_attention_forward`。

### CPU 调用链

CPU 侧选择函数是 `_fused_sdp_choice_cpp`，它调用 `select_sdp_backend_cpp`。CPU 侧当前只在 `Flash Attention for CPU` 和 `Math` 之间选择，优先级固定为 `Flash Attention -> Math`。CPU 侧 fused 路径要求无 dropout、非 NestedTensor、4D Q/K/V、dtype 在 `float/double/bfloat16/half` 中，且 mask shape、head dim、stride 等满足约束；否则回退到 `Math`。

### Native MHA 快路径

`native_multi_head_attention_cuda` 内部也会构造 Q/K/V 并调用 `select_sdp_backend`。如果选择到 Flash、Efficient 或 cuDNN 且没有 mask、NestedTensor 序列长度等额外限制满足，则会走 `scaled_dot_product_attention` 快路径；如果返回 `Math` 或不可用，则不使用该 fused 快路径。

## 自动选择机制

### `sdp_params`

所有 backend 的选择都基于同一个参数结构：

```cpp
struct sdp_params {
  at::Tensor query;
  at::Tensor key;
  at::Tensor value;
  std::optional<at::Tensor> attn_mask;
  double dropout;
  bool is_causal;
  bool enable_gqa;
};
```

这些字段覆盖了 backend 选择需要判断的主要维度：设备、dtype、形状、head 数、head_dim、mask、dropout、causal、GQA 和梯度状态。

### 优先级

`select_sdp_backend` 的基本流程是：

```text
1. 检查 Math / Flash / Efficient / cuDNN 是否至少有一个 runtime 开关开启。
2. 调用 priority_order 获取候选 backend 顺序。
3. 按顺序遍历 backend。
4. 对 cuDNN / Flash / Efficient 分别调用 can_use_*_attention。
5. Math 只检查 runtime 开关，开启即可作为 fallback。
6. 第一个满足条件的 backend 立即返回。
7. 若没有 backend 可用，则用 debug=true 重新运行各检查函数并报错。
```

默认优先级来自全局 context。CUDA 中 `priority_order` 会在首次初始化时检查 `check_prefer_cudnn_attention()`：如果环境变量 `TORCH_CUDNN_SDPA_DEPRIORITIZED` 没有设置为 true，运行时 cuDNN 版本大于 9.15.0，并且当前 GPU 为 SM90/SM100/SM103，则把 cuDNN 放到首位。否则维持默认顺序。

### 用户可控开关

用户可以通过 `torch.nn.attention.sdpa_kernel` 控制可选 backend，也可以通过 `torch.backends.cuda` 中的低层开关单独启停：

```python
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention

with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH], set_priority=True):
    out = scaled_dot_product_attention(q, k, v)
```

常用开关与探测函数包括：

| API | 作用 |
| --- | --- |
| `torch.nn.attention.sdpa_kernel(backends, set_priority=False)` | 临时启用指定 backend；`set_priority=True` 时列表顺序也作为优先级 |
| `torch.backends.cuda.enable_flash_sdp(bool)` | 启停 Flash Attention |
| `torch.backends.cuda.enable_mem_efficient_sdp(bool)` | 启停 Memory Efficient Attention |
| `torch.backends.cuda.enable_math_sdp(bool)` | 启停 Math fallback |
| `torch.backends.cuda.enable_cudnn_sdp(bool)` | 启停 cuDNN Attention |
| `torch.backends.cuda.can_use_flash_attention(params, debug=False)` | 判断 Flash 是否可用于一组 `SDPAParams` |
| `torch.backends.cuda.can_use_efficient_attention(params, debug=False)` | 判断 Efficient 是否可用于一组 `SDPAParams` |
| `torch.backends.cuda.can_use_cudnn_attention(params, debug=False)` | 判断 cuDNN 是否可用于一组 `SDPAParams` |
| `torch.backends.cuda.is_flash_attention_available()` | 判断当前 build 是否编译了 Flash Attention |

如果想知道某个输入为什么没有走 fused kernel，可以构造 `SDPAParams` 并调用 `can_use_*_attention(..., debug=True)`，或者在不可用且 `Math` 被禁用时由 `select_sdp_backend` 打印原因。

## Backend 适用场景

### Flash Attention

Flash 是 CUDA fused 路径的首选，适合高吞吐、低显存访问的标准 attention 场景。典型输入是 CUDA 上的 4D Q/K/V，形状语义为 `[batch, num_heads, seq_len, head_dim]`。

主要条件：

| 维度 | 条件 |
| --- | --- |
| 编译期 | 需要 `USE_FLASH_ATTENTION` |
| 运行期开关 | `userEnabledFlashSDP()` 为 true |
| 设备 | CUDA 可用；非 ROCm 路径要求 SM80 到 SM121 |
| dtype | SM80+ 支持 `float16`、`bfloat16`；低于 SM80 仅支持 `float16` |
| Q/K/V 维度 | 必须都是 4D；dense 输入 batch size 一致；非 GQA 时 head 数一致 |
| head_dim | Q/K/V 最后一维相同；CUDA 通常要求 `head_dim <= 256`；NestedTensor 还要求是 8 的倍数 |
| mask | 不支持非空 `attn_mask` |
| causal | `is_causal=True` 时不支持 `seq_len_q != seq_len_k` 的非方阵情况 |
| dropout | 支持，但部分 GPU/head_dim/训练组合有限制 |
| GQA | 支持，要求 key/value head 数能整除 query head 数 |
| stride | dense 输入最后一维 stride 必须为 1，head_dim 为 1 时可忽略 |
| NestedTensor | 支持有限，要求 batch、head_dim、非零 seq_len 等约束满足 |

额外注意：在 SM86、SM89、SM120、SM121 上，训练且 `head_dim` 位于 `(192, 224]`，或者 `head_dim > 224` 且 `dropout > 0` 时，Flash 会被禁用以避免 backward 不支持。

推荐场景：无显式 mask、半精度或 BF16、标准 causal/non-causal attention、GQA、对性能最敏感的 Transformer 训练或推理。

不推荐或不可用场景：需要任意 `attn_mask`、CPU、float32 CUDA 输入、非方阵 causal attention、head_dim 超过 256，或者特殊 GPU 上的大 head_dim 训练。

### Memory Efficient Attention

Memory Efficient Attention 也是 CUDA fused 路径，目标是减少 attention 中间矩阵显存占用。它通常在 Flash 不满足条件但仍可使用 fused CUDA kernel 时作为第二选择。

主要条件：

| 维度 | 条件 |
| --- | --- |
| 编译期 | 需要 `USE_MEM_EFF_ATTENTION` |
| 运行期开关 | `userEnabledMemEfficientSDP()` 为 true |
| 设备 | CUDA 可用；非 ROCm 路径要求 SM50 到 SM121 |
| dtype | SM80+ 支持 `float16`、`float32`、`bfloat16`；SM80 以下支持 `float16`、`float32` |
| Q/K/V 维度 | 必须都是 4D；dense 输入 batch size 一致；head 数要求更严格 |
| head_dim | Q/K/V head_dim 需要满足 GEMM 对齐；SM80+ 半精度通常要求按 8 对齐，float32 通常按 4 对齐 |
| mask | CUDA 非 ROCm 路径选择阶段没有通用 mask shape 检查；ROCm 要求 mask 为 bool 或与 Q/K/V 同 dtype |
| dropout | 支持；超大 batch 且 dropout 非 0 时会被内部逻辑限制 |
| GQA | 不支持 GQA |
| NestedTensor | 非 ROCm 下不支持训练态 NestedTensor |
| stride | dense 输入 Q/K/V 与 GPU mask 最后一维 stride 必须为 1 |

推荐场景：Flash 因 dtype、硬件或 head_dim 对齐等原因不可用，但仍希望使用 fused CUDA kernel；长序列场景中希望降低显存占用。

不推荐或不可用场景：GQA、复杂广播、训练态 NestedTensor、不满足 head_dim 对齐、CPU，以及需要 Flash 支持语义但 Efficient 不支持的场景。

### cuDNN Attention

cuDNN Attention 依赖 cuDNN SDPA 实现。它可在部分新 GPU 与新版 cuDNN 上被提升为最高优先级，并且相较 Flash 支持更多 mask 形态。

主要条件：

| 维度 | 条件 |
| --- | --- |
| 编译期 | 需要启用 cuDNN，且编译期 `CUDNN_VERSION >= 90000` |
| 运行期开关 | `userEnabledCuDNNSDP()` 为 true |
| 设备 | CUDA 可用；要求 SM80 到 SM121 |
| dtype | SM80+ 支持 `float16`、`bfloat16`；低于 SM80 仅支持 `float16` |
| Q/K/V 维度 | 必须都是 4D；dense 输入 batch size 一致 |
| head_dim | 通常 `head_dim <= 128` 且是 8 的倍数；cuDNN >= 9.1 且 SM90 时可放宽到 256 |
| seq_len | 不支持 key/value sequence length 为 1；query length 为 1 且 dropout 非 0 也不支持 |
| mask | 支持 2D 和 4D mask，shape 可按 batch/head/query/key 维度广播；mask 不能 requires_grad |
| dropout | 支持但受 cuDNN 版本限制；cuDNN 9.11 到 9.13 的 dropout 会被禁用 |
| GQA | 支持；cuDNN 检查允许 query heads 是 key/value heads 的整数倍 |
| deterministic | `torch.use_deterministic_algorithms(True)` 且不是 warn-only 时禁用 |
| NestedTensor | 实验性支持，需 `TORCH_CUDNN_SDPA_NESTED_TENSOR_ENABLED=1`，且仅 SM90/SM100 类设备 |

推荐场景：新 GPU 和新版 cuDNN，输入为半精度/BF16，使用 2D/4D mask 或希望让 cuDNN 实现优先处理 SDPA。对 SM90/SM100 且 cuDNN 版本足够新的环境，cuDNN 可能自动排到 Flash 前面。

不推荐或不可用场景：需要 deterministic 强保证、head_dim 超出限制、cuDNN 版本较旧、seq_len 特殊值、float32 输入，或者未启用 cuDNN SDPA 的构建。

### Math Backend

Math 是最通用的 fallback，不依赖 Flash/Efficient/cuDNN 编译宏。只要 `userEnabledMathSDP()` 为 true，且其它 fused backend 不可用，选择逻辑就会回退到 Math。

主要特点：

| 维度 | 条件 |
| --- | --- |
| 编译期 | 无特殊 fused 编译要求 |
| 运行期开关 | `userEnabledMathSDP()` 为 true |
| 设备 | CPU/CUDA/MPS 等通用路径均可有 math 实现 |
| dtype | 支持范围最广，通常覆盖 float32/float64/float16/bfloat16 等 |
| mask | 支持通用 mask 语义 |
| dropout | 支持 |
| GQA | 支持 |
| 性能 | 通常慢于 fused backend，且中间张量显存占用更高 |

推荐场景：调试、数值对齐、fused backend 条件不满足、需要最大语义兼容性，或在 CPU/不支持 fused kernel 的硬件上运行。

不推荐场景：性能敏感的大规模训练或推理，尤其是长序列 attention。

### Overrideable Backend

`SDPBackend::OVERRIDEABLE` 是为扩展预留的 backend 类型。当前 CUDA `select_sdp_backend` 中如果检测到它被启用会直接触发 `Invalid backend`，因此不参与常规自动选择。实际扩展能力更多通过 `torch.nn.attention` 中的 Flash Attention registry 机制暴露。

## 选择决策速查

| 场景 | 最可能选择 | 原因 |
| --- | --- | --- |
| CUDA SM80+、fp16/bf16、无 mask、head_dim <= 256 | Flash | 默认最高优先级，约束满足时性能优先 |
| CUDA SM90/SM100、新版 cuDNN、满足 cuDNN 条件 | cuDNN | `check_prefer_cudnn_attention()` 可能把 cuDNN 提到首位 |
| CUDA、float32、Flash 不支持但 Efficient 支持 | Efficient | Efficient 在 SM80+ 支持 float32 |
| 需要 2D/4D mask 且 dtype/hardware 满足 cuDNN | cuDNN | Flash 不支持 mask，cuDNN 支持更通用 mask shape |
| GQA 且无 mask、半精度 | Flash 或 cuDNN | Efficient 不支持 GQA |
| CPU 或 fused 条件不满足 | Math | 通用 fallback |
| 用户只启用 Flash 但输入带 mask | 报错或不可用 | Flash 不支持非空 `attn_mask`，Math 若关闭则无 fallback |
| deterministic algorithms 开启且非 warn-only | 非 cuDNN | cuDNN SDPA 被禁用，继续尝试其它 backend |

## 常见排查方法

1. 使用 `torch.nn.attention.sdpa_kernel([...], set_priority=True)` 固定候选 backend 和优先级，验证是否是选择策略导致的差异。
2. 使用 `torch.backends.cuda.can_use_flash_attention(params, debug=True)`、`can_use_efficient_attention(params, debug=True)`、`can_use_cudnn_attention(params, debug=True)` 查看具体失败约束。
3. 如果禁用了 `Math`，确保至少一个 fused backend 满足编译期、运行期、设备、dtype、shape、mask 和 dropout 条件，否则会报 `No available kernel`。
4. 对性能问题优先检查是否意外落到 `Math`；常见原因包括输入 dtype 为 float32、带 mask、head_dim 不对齐、stride 不是 1、GQA 使用 Efficient、或当前 GPU/cuDNN 版本不满足要求。
5. 对数值或确定性问题优先检查是否启用了 cuDNN、dropout、deterministic algorithms，以及 `sdpa_kernel` 是否改变了优先级。

## 关键源码位置

| 文件 | 作用 |
| --- | --- |
| `aten/src/ATen/native/native_functions.yaml` | `scaled_dot_product_attention`、`_fused_sdp_choice` 和各 backend op 的 schema 与 dispatch 定义 |
| `aten/src/ATen/native/transformers/cuda/sdp_utils.cpp` | CUDA SDPA backend 优先级、约束检查和 `select_sdp_backend` 核心逻辑 |
| `aten/src/ATen/native/transformers/sdp_utils_cpp.h` | `sdp_params` 与通用约束检查函数，如 dtype、mask、shape、GQA、stride |
| `aten/src/ATen/native/transformers/sdp_utils_cpp.cpp` | CPU 侧 `select_sdp_backend_cpp` 逻辑 |
| `aten/src/ATen/native/transformers/attention.cpp` | CPU `_fused_sdp_choice_cpp`、math backend 和 dispatcher stub |
| `aten/src/ATen/native/transformers/cuda/attention.cu` | CUDA `_fused_sdp_choice_cuda` 与 Flash/Efficient/cuDNN 前向入口 |
| `torch/nn/attention/__init__.py` | `SDPBackend`、`sdpa_kernel`、优先级上下文管理器和 Flash registry |
| `torch/backends/cuda/__init__.py` | backend runtime 开关、可用性检查和调试 API |
