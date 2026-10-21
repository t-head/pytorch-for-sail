#!/usr/bin/env bash
set -euo pipefail

export SDK_INSTALL_DIR="${SDK_INSTALL_DIR:-/usr/local}"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
echo "[smoke] Source directory: $(pwd)"
source .ci/ppu/sdk_env.sh
source .ci/ppu/install_pccl.sh
bash .ci/ppu/install_wheel.sh
bash .ci/ppu/install_triton.sh

echo "=== Environment self-check ==="
echo "pr=${PR_NUMBER:-none} hostname=$(hostname) node=${NODE_NAME:-unknown}"

bash .ci/ppu/install_test_deps.sh
source .ci/ppu/cuda_only_filter.sh
echo "=== CUDA smoke cases (run_test.py --include exact filter, CUDA-related only) ==="

K_SKIP_CASES="not test_benchmark_choice_fail_in_subproc \
and not test_lazy_template_fusion_multiple_candidates_use_async_compile \
and not test_template_epilogue_fusion_static_analysis_test_case_spills_reject_use_async_compile \
and not test_template_epilogue_fusion_static_analysis_test_case_timing_reject_use_async_compile \
and not test_async_autotuner_cache_same_inputs \
and not test_bmm_out_dtype \
and not test_cat_max_autotune_extern \
and not test_compilation_after_inactivity \
and not test_linear_and_cel \
and not test_max_autotune_mm_plus_mm_zero_size_input_dynamic_False_search_space \
and not test_max_autotune_regular_mm_zero_size_input_dynamic \
and not test_mutation_rename \
and not test_cublas_baddbmm_large_input \
and not test_mm_bmm_dtype_overload_float16_M \
and not test_mm_with_mH_args_backend_cublas_cuda \
and not test_matmul_dropout_device_cpu \
and not test_float8_error_messages_cuda \
and not test_float8_rowwise_scaling_sanity_use_fast_accum_False_cuda \
and not test_float8_rowwise_scaling_sanity_use_fast_accum_True_cuda \
and not test_float8_scale_fast_accum_cuda \
and not test_scaled_mm_vs_emulated_row_wise_bfloat16_shapes0_cuda \
and not test_float8_basics_cuda \
and not test_cublas_addmm_reduced_precision_fp16 \
and not test_mm_with_mH_args_backend_cublaslt_cuda \
and not test_template_epilogue_fusion_extra_reads_fuse_epilogue \
and not test_cublas_and_lt_reduced_precision_fp16_accumulate_cuda \
and not test_mixed_dtypes_linear_cuda_float16 \
and not test_fake_crossref_backward_no_amp_nn_functional_bilinear_cuda_float32 \
and not test_numpy_ref_nn_functional_pdist_cuda_float64 \
and not test_noncontiguous_samples_nn_functional_grid_sample_cuda_float32 \
and not test_triton_template_generated_code_caching \
and not test_triton_template_generated_code_caching_bmm \
and not test_triton_template_generated_code_caching_mm_plus_mm \
and not test_cublas_addmm_size_10000_backend_cublas_cuda_float32 \
and not test_float8_basics_layout_permutations_cpu \
and not test_max_autotune_remote_caching_dynamic_False \
and not test_addmm_baddmm_dtype_overload_bfloat16 \
and not test_addmm_baddmm_dtype_overload_float16 \
and not test_mm_bmm_dtype_overload_bfloat16 \
and not test_mixed_dtypes_linear_cuda_bfloat16"

python test/run_test.py \
    --include \
        test_matmul_cuda \
        test_scaled_matmul_cuda \
        nn/attention/test_open_registry \
        inductor/test_flex_flash \
    -k "$K_SKIP_CASES" \
    --keep-going \
    --verbose


echo "[smoke] Done"
