#!/usr/bin/env bash
set -euo pipefail

export SDK_INSTALL_DIR="${SDK_INSTALL_DIR:-/usr/local}"

if [[ -z "${SHARD_NUMBER:-}" || -z "${NUM_TEST_SHARDS:-}" ]]; then
    echo "[full-ut][error] SHARD_NUMBER and NUM_TEST_SHARDS must be provided (by the workflow matrix)" >&2
    exit 1
fi
if (( NUM_TEST_SHARDS != 2 )); then
    echo "[full-ut][error] This script is hand-orchestrated for 2 shards, NUM_TEST_SHARDS must be 2, got: ${NUM_TEST_SHARDS}" >&2
    exit 1
fi
if (( SHARD_NUMBER < 1 || SHARD_NUMBER > 2 )); then
    echo "[full-ut][error] SHARD_NUMBER must be 1 or 2, got: ${SHARD_NUMBER}" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
source .ci/ppu/sdk_env.sh
source .ci/ppu/install_pccl.sh
bash .ci/ppu/install_wheel.sh
bash .ci/ppu/install_triton.sh

echo "=== Environment self-check (CUDA) ==="
echo "pr=${PR_NUMBER:-none} hostname=$(hostname) node=${NODE_NAME:-unknown}"

bash .ci/ppu/install_test_deps.sh
source .ci/ppu/cuda_only_filter.sh

CRASH_LIST="${CRASH_LIST:-.ci/ppu/badcase_list/full_ut.txt}"

FAILED_CASES=()
run_case() {
    local test_name="$1"
    local k_expr="${2:-}"
    local crash_args=()
    if [[ "$test_name" == "test_nn" ]]; then
        crash_args=(--crash-list "$CRASH_LIST")
    fi
    echo "=== [case] ${test_name}${k_expr:+  -k \"${k_expr}\"} ==="
    if python .ci/ppu/multi_run_case.py "test/${test_name}.py" ${crash_args[@]+"${crash_args[@]}"} \
        -k "$(ppu_cuda_only_k_expr "${k_expr:+( ${k_expr} )}")"; then
        echo "=== [pass] ${test_name} ==="
    else
        echo "::error::[fail] ${test_name}"
        FAILED_CASES+=("${test_name}")
    fi
}

shard_heavy() {
    run_case inductor/test_aot_inductor \
        "cuda and not test_aoti_load_package_in_fresh_subprocess_cuda and not test_repeated_calling_cuda and not test_simple_multi_arch_embed_kernel_binary_False_cuda"

    run_case test_transformers "cuda"

    run_case test_foreach "cuda"

    run_case test_sparse \
        "cuda and not test_mm_cuda_complex and not test_sparse_addmm and not test_sparse_matmul_cuda_bfloat16 and not test_sparse_matmul_cuda_complex and not test_sparse_matmul_cuda_float16 and not test_constructor_autograd_SparseBS and not test_gradcheck_mm_SparseCOO and not test_gradcheck_mm_SparseCSC and not test_gradcheck_mm_SparseCSR"
}

shard_rest() {
    run_case test_torch \
        "cuda and not test_corrcoef_cuda_complex64 and not test_terminate_handler_on_crash and not test_cov_cuda_complex and not test_put_cuda_float16"

    run_case test_nn "cuda"

    run_case test_autograd "cuda and not test_thread_shutdown"

    run_case test_binary_ufuncs "TestBinaryUfuncsCUDA"

    run_case test_unary_ufuncs "cuda and not TestUnaryUfuncsCPU"

    run_case test_reductions "cuda and not test_ref_extremal_values_hash_tensor_cuda_float32"

    run_case test_shape_ops "TestShapeOpsCUDA"

    run_case test_indexing "cuda"

    run_case test_tensor_creation_ops "cuda"

    run_case test_dataloader \
        "cuda and not test_fd_limit_exceeded and not test_multiprocessing_contexts and not test_sparse_tensor_multiprocessing and not test_early_exit and not test_segfault"

    run_case test_optim "cuda"

    run_case test_serialization "cuda"

    run_case test_fx "cuda"

    run_case test_autocast "cuda"

    run_case test_type_promotion "cuda"

    run_case test_complex "cuda"

    run_case test_sort_and_select "cuda"

    run_case nn/test_convolution \
        "cuda and not test_conv_transposed_large_cuda and not test_cudnn_convolution_add_relu_cuda and not test_cudnn_convolution_relu_cuda and not test_depthwise_conv_64bit_indexing_cuda"

    run_case nn/test_embedding "cuda"

    run_case nn/test_pooling "cuda"

    run_case dynamo/test_misc \
        "cuda and not test_outside_linear_module_free and not test_packaging_version_parse and not test_parameter_free and not test_pytree_tree_ and not test_tracing_nested and not test_tracing_pytree_cxx"

    run_case dynamo/test_compile "cuda"

    run_case dynamo/test_repros "cuda"

    run_case dynamo/test_export "cuda"

    run_case distributed/test_c10d_common "cuda"

    run_case test_dynamic_shapes "cuda and not test_do_not_guard_unbacked_inputs"
}

if (( SHARD_NUMBER == 1 )); then
    echo "=== CUDA full UT (shard 1/2: the three heavy files) ==="
    shard_heavy
else
    echo "=== CUDA full UT (shard 2/2: all remaining files) ==="
    shard_rest
fi

if [ "${#FAILED_CASES[@]}" -ne 0 ]; then
    echo "=== Failure list (${#FAILED_CASES[@]} files) ==="
    for case_label in "${FAILED_CASES[@]}"; do
        echo "  - ${case_label}"
    done
    echo "::error::[full-ut] ${#FAILED_CASES[@]} test files failed, see the logs of each [case] section above"
    exit 1
fi

echo "[full-ut] Done, all passed (shard=${SHARD_NUMBER}/${NUM_TEST_SHARDS})"
