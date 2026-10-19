#!/usr/bin/env bash
set -euo pipefail

export SDK_INSTALL_DIR="${SDK_INSTALL_DIR:-/usr/local}"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
echo "[base_api] Source directory: $(pwd)"
source .ci/ppu/sdk_env.sh
source .ci/ppu/install_pccl.sh
bash .ci/ppu/install_wheel.sh
bash .ci/ppu/install_triton.sh

echo "=== Environment self-check ==="
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

echo "=== CUDA base api cases (test_ops / test_ops_gradients, CUDA-related only) ==="

run_pytest test/test_ops.py "cuda"
run_pytest test/test_ops_gradients.py "cuda"

if [ "${#FAILED_CASES[@]}" -ne 0 ]; then
    echo "=== Failure list (${#FAILED_CASES[@]} files) ==="
    for case_label in "${FAILED_CASES[@]}"; do
        echo "  - ${case_label}"
    done
    echo "::error::[base_api] ${#FAILED_CASES[@]} test files failed, see the logs of each [case] section above"
    exit 1
fi

echo "[base_api] Done, all passed"
