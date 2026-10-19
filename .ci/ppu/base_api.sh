#!/usr/bin/env bash
set -euo pipefail

export SDK_INSTALL_DIR="${SDK_INSTALL_DIR:-/usr/local}"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
echo "[base_api] 源码目录: $(pwd)"
source .ci/ppu/sdk_env.sh
source .ci/ppu/install_pccl.sh
bash .ci/ppu/install_wheel.sh
bash .ci/ppu/install_triton.sh

echo "=== 环境自检 ==="
echo "pr=${PR_NUMBER:-none} hostname=$(hostname) node=${NODE_NAME:-unknown}"

bash .ci/ppu/install_test_deps.sh
source .ci/ppu/cuda_only_filter.sh

CRASH_LIST="${CRASH_LIST:-.ci/ppu/badcase_list/base_api.txt}"
echo "[base_api] crash list: ${CRASH_LIST}"

FAILED_CASES=()
run_pytest() {
    local test_file="$1"
    local k_expr="${2:-}"
    echo "=== [case] ${test_file}${k_expr:+  -k ${k_expr}} ==="
    if python .ci/ppu/multi_run_case.py "$test_file" --crash-list "$CRASH_LIST" \
        -k "$(ppu_cuda_only_k_expr "${k_expr:+( ${k_expr} )}")"; then
        echo "=== [pass] ${test_file} ==="
    else
        echo "::error::[fail] ${test_file}"
        FAILED_CASES+=("${test_file}")
    fi
}

echo "=== CUDA base api 用例（test_ops / test_ops_gradients ，仅 CUDA 相关）==="

run_pytest test/test_ops.py "cuda"
run_pytest test/test_ops_gradients.py "cuda"

if [ "${#FAILED_CASES[@]}" -ne 0 ]; then
    echo "=== 失败清单（${#FAILED_CASES[@]} 个文件）==="
    for case_label in "${FAILED_CASES[@]}"; do
        echo "  - ${case_label}"
    done
    echo "::error::[base_api] ${#FAILED_CASES[@]} 个测试文件失败，详见上方各 [case] 段落的日志"
    exit 1
fi

echo "[base_api] 完成，全部通过"