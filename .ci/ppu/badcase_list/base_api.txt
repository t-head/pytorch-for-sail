#!/usr/bin/env bash
set -euo pipefail

export SDK_INSTALL_DIR="${SDK_INSTALL_DIR:-/usr/local}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
echo "[perf] Source directory: $(pwd)"
source .ci/ppu/sdk_env.sh
source .ci/ppu/install_pccl.sh
bash .ci/ppu/install_wheel.sh
bash .ci/ppu/install_triton.sh
bash .ci/ppu/install_test_deps.sh
source .ci/ppu/cuda_only_filter.sh

TEST_REPORTS_DIR="$REPO_ROOT/test/test-reports"
mkdir -p "$TEST_REPORTS_DIR"

echo "=== [1/2] CUDA inductor performance tests (run_test.py --include allowlist + -k) ==="

K_SKIP_CASES="not test_fusion_choice4_cpu \
and not test_cat \
and not test_cat_pointwise_many_complex_inputs \
and not test_cat_pointwise_many_simple_inputs \
and not test_cat_pointwise_config_option \
and not test_cat_pointwise \
and not test_partitioning_with_view \
and not test_equivalent_template_code \
and not test_tield_kernel_fusion_cuda \
and not test_split_scan"
python test/run_test.py \
    --include \
        inductor/test_perf \
        inductor/test_benchmark_fusion \
        inductor/test_benchmarking \
        inductor/test_analysis \
    -k "${K_SKIP_CASES}" \
    --verbose

if [[ "${RUN_GPT_FAST:-1}" != "1" ]]; then
    echo "=== [2/2] gpt_fast micro-benchmark: explicitly disabled by RUN_GPT_FAST=0, skipping ==="
    echo "[perf] Done"
    exit 0
fi

echo "=== [2/2] gpt_fast micro-benchmark (torch.compile end-to-end timing) ==="
if ! (cd /tmp && python -c "import torchao" >/dev/null 2>&1); then
    source "$REPO_ROOT/.ci/ppu/pip_sources.sh"
    ppu_build_pip_candidates
    for index in "${PIP_CANDIDATES[@]}" __pip_default__; do
        pip_args=(--disable-pip-version-check --retries 1 --timeout 20 --no-deps)
        if [[ "${index}" == "__pip_default__" ]]; then
            label="pip default index"
        else
            label="${index}"
            pip_args+=(-i "${index}")
        fi
        if python -m pip install "${pip_args[@]}" torchao; then
            echo "[perf] torchao install succeeded (index: ${label})"
            break
        fi
        echo "[perf][warn] Index ${label} failed, trying the next one"
    done
fi

if ! (cd /tmp && python -c "import torchao; print('torchao', torchao.__version__)"); then
    echo "[perf][warn] torchao unavailable; skipping gpt_fast."
    echo "[perf] Done (stage 1 only)"
    exit 0
fi

GPT_FAST_CSV="$TEST_REPORTS_DIR/gpt_fast_benchmark.csv"
rm -f "$GPT_FAST_CSV" "${GPT_FAST_CSV%.csv}.json"
for experiment in mlp_layer_norm_gelu layer_norm gather_gemv gemv; do
    echo "--- gpt_fast: ${experiment} ---"
    python benchmarks/gpt_fast/benchmark.py \
        --only "${experiment}" \
        --output "$GPT_FAST_CSV"
done

echo "=== gpt_fast micro-benchmark results ==="
if [[ -s "$GPT_FAST_CSV" ]]; then
    cat "$GPT_FAST_CSV"
else
    echo "[perf][warn] No results found in ${GPT_FAST_CSV}: all 4 calls above exited 0 but no csv was written."
    echo "[perf][warn] First check whether the --output / output_csv semantics of benchmarks/gpt_fast/benchmark.py were changed upstream."
fi

echo "[perf] Done"
