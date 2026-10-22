#!/usr/bin/env bash
set -euo pipefail

export SDK_INSTALL_DIR="${SDK_INSTALL_DIR:-/usr/local}"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source .ci/ppu/sdk_env.sh
source .ci/ppu/install_pccl.sh
bash .ci/ppu/install_wheel.sh
bash .ci/ppu/install_triton.sh

bash .ci/ppu/install_test_deps.sh
source .ci/ppu/cuda_only_filter.sh

echo "=== CUDA inductor accuracy tests (run_test.py --include allowlist filter, CUDA-related only) ==="

K_FP8="not fp8 and not FP8 and not Fp8 \
and not float8 and not Float8 \
and not e4m3 and not E4M3 \
and not e5m2 and not E5M2"

K_SKIP_CASES="not test_not_disabling_ftz_yields_zero \
and not test_triton_interpret \
and not test_graph_partition_user_defined_triton_kernel_reuse \
and not test_graph_partition_reorder_cpu_and_gpu_interleave \
and not test_avg_pool3d_backward2_cuda  \
and not test_consecutive_split_ \
and not test_linalg_eig_stride_consistency_cuda \
and not test_sort_stable_cuda \
and not test_copy_non_blocking_is_pinned_use_cat_True_cuda \
and not test_split_ \
and not RNN \
and not LSTM \
and not GRU \
and not test_put_cuda_float16 \
and not test_corrcoef_cuda_complex \
and not test_cov_cuda_complex"

echo "=== CUDA inductor accuracy tests (single card, full set) ==="
python test/run_test.py --inductor \
    --include \
        inductor/test_cuda_repro \
        inductor/test_cudagraph_trees \
        inductor/test_gpu_select_algorithm \
        inductor/test_torchinductor \
        test_modules \
        test_torch \
    -k "$(ppu_cuda_only_k_expr "$K_FP8" "$K_SKIP_CASES")" \
    --verbose

echo "[accuracy] Done"
