#!/usr/bin/env bash

set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/local_ut_results}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="${RESULT_ROOT}/${RUN_ID}"
LOG_DIR="${RESULT_DIR}/logs"
XML_DIR="${RESULT_DIR}/xml"
NORMAL_TIMEOUT="${UT_TEST_TIMEOUT:-600}"
EXCLUSIVE_TIMEOUT="${UT_EXCLUSIVE_TEST_TIMEOUT:-1200}"

readarray -t TEST_CASES <<'EOF'
test/test_autograd.py::TestAutograd::test_grad_mode_class_decoration
test/test_autograd.py::TestAutograd::test_tensor_grad_warnings
test/test_autograd.py::TestAutograd::test_grad
test/test_autograd.py::TestAutograd::test_backward_to_node
test/test_autograd.py::TestAutograd::test_grad_nonleaf
test/test_autograd.py::TestAutogradForwardModeBatchedGrad::test_out_of_place_basic
test/test_autograd_fallback.py
test/test_nn.py::TestNN::test_parse_to
test/test_nn.py::TestNN::test_requires_grad_
test/test_nn.py::TestNN::test_to
test/test_nn.py::TestNN::test_zero_grad
test/test_nn.py::TestNN::test_no_grad
test/test_nn.py::TestNN::test_parameters_and_named_parameters
test/test_nn.py::TestNN::test_buffers_and_named_buffers
test/test_nn.py::TestNN::test_children
test/test_torch.py::TestBasicVitalSigns::test_basic_vitals
test/test_ops.py::TestSelfKwarg::test_self_kwargs
test/test_autocast.py::TestAutocastGPU::test_cast_cache_is_global
test/test_autocast.py::TestAutocastGPU::test_cache_disabled
test/test_autocast.py::TestAutocastGPU::test_autocast_prioritize
test/test_autocast.py::TestTorchAutocast::test_autocast_fast_dtype
test/test_serialization.py::TestSerialization::test_serialization_zipfile_utils
test/test_serialization.py::TestSerialization::test_serialization
test/test_serialization.py::TestSerialization::test_serialization_safe
test/test_serialization.py::TestSerialization::test_serialization_filelike
test/test_accelerator.py
test/test_cuda_multigpu.py
test/test_jiterator.py
test/test_segment_reductions.py
test/test_transformers.py
test/test_legacy_vmap.py
test/test_fx_passes.py
test/test_fx_reinplace_pass.py
test/test_functionalization_of_rng_ops.py
test/test_native_functions.py
test/test_type_info.py
test/nn/test_dropout.py
test/nn/test_module_hooks.py
test/nn/test_parametrization.py
test/nn/test_load_state_dict.py
test/nn/test_packed_sequence.py
test/optim/test_optim.py
test/optim/test_lrscheduler.py
test/optim/test_swa_utils.py
test/dynamo/test_compile.py
test/dynamo/test_decorators.py
test/dynamo/test_exceptions.py
test/dynamo/test_guard_manager.py
test/dynamo/test_modules.py
test/dynamo/test_recompiles.py
test/dynamo/test_sdpa.py
test/dynamo/test_python_autograd.py
test/dynamo/test_export_mutations.py
test/dynamo/test_modes.py
test/dynamo/test_subgraphs.py
test/dynamo/test_view.py
test/dynamo/test_aot_compile.py
test/dynamo/test_backward_higher_order_ops.py
test/dynamo/test_verify_correctness.py
test/inductor/test_compile.py
test/inductor/test_config.py
test/inductor/test_indexing.py
test/inductor/test_fx_fusion.py
test/inductor/test_async_compile.py
test/inductor/test_scatter_optimization.py
test/inductor/test_kernel_optimization.py
test/inductor/test_online_softmax.py
test/inductor/test_inplacing_pass.py
test/inductor/test_split_cat_fx_passes.py
test/inductor/test_block_analysis.py
test/inductor/test_utils.py
test/export/test_converter.py
test/export/test_verifier.py
test/export/test_schema.py
test/export/test_unflatten.py
test/export/test_package.py
test/export/test_pass_infra.py
test/export/test_lift_unlift.py
test/export/test_draft_export.py
test/distributed/test_c10d_common.py
test/distributed/test_c10d_spawn_nccl.py
test/distributed/test_collective_utils.py
test/distributed/test_control_collectives.py
test/distributed/algorithms/ddp_comm_hooks/test_ddp_hooks.py
test/distributed/fsdp/test_fsdp_comm_hooks.py
test/distributed/_composable/fsdp/test_fully_shard_mixed_precision.py
test/distributed/checkpoint/test_file_system_checkpoint.py
test/distributed/tensor/test_op_schema.py
test/distributed/tensor/test_common_rules.py
test/distributed/test_fake_pg.py
test/distributed/test_serialization.py
test/functorch/test_ac.py
test/functorch/test_aot_joint_with_descriptors.py
test/higher_order_ops/test_invoke_quant.py
test/test_function_schema.py
EOF

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[ERROR] Python executable not found: ${PYTHON_BIN}" >&2
  exit 2
fi
if ! command -v timeout >/dev/null 2>&1; then
  echo "[ERROR] GNU timeout is required." >&2
  exit 2
fi
if ! "${PYTHON_BIN}" -c 'import pytest' >/dev/null 2>&1; then
  echo "[ERROR] pytest is not installed for ${PYTHON_BIN}." >&2
  exit 2
fi

PYTEST_MAJOR="$("${PYTHON_BIN}" -c 'import pytest; print(pytest.__version__.split(".")[0])')"
if [[ "${PYTEST_MAJOR}" -ge 9 ]]; then
  echo "[ERROR] Install the compatible pytest: ${PYTHON_BIN} -m pip install --force-reinstall pytest==8.1.2" >&2
  exit 2
fi

MISSING=()
PYTEST_CASES=()
for test_case in "${TEST_CASES[@]}"; do
  file_path="${test_case%%::*}"
  if [[ -f "${file_path}" ]]; then
    PYTEST_CASES+=("${test_case#test/}")
  else
    MISSING+=("${file_path}")
  fi
done
if [[ ${#MISSING[@]} -gt 0 ]]; then
  printf '[ERROR] Missing test file: %s\n' "${MISSING[@]}" >&2
  exit 2
fi

if [[ -f /usr/local/PPU_SDK/envsetup.sh ]]; then
  # shellcheck disable=SC1091
  source /usr/local/PPU_SDK/envsetup.sh
fi
if ! "${PYTHON_BIN}" -c 'import importlib.util; raise SystemExit(0 if importlib.util.find_spec("zstandard") or importlib.util.find_spec("pyzstd") else 1)' >/dev/null 2>&1; then
  echo "[INFO] Installing required Python package: zstandard"
  "${PYTHON_BIN}" -m pip install zstandard || exit 2
fi

mkdir -p "${LOG_DIR}" "${XML_DIR}"
cd "${REPO_ROOT}/test"

FILTERED_PYTHONPATH=""
IFS=':' read -r -a PYTHONPATH_ENTRIES <<< "${PYTHONPATH:-}"
for path_entry in "${PYTHONPATH_ENTRIES[@]}"; do
  [[ -z "${path_entry}" || "${path_entry}" == "${REPO_ROOT}" ]] && continue
  FILTERED_PYTHONPATH="${FILTERED_PYTHONPATH:+${FILTERED_PYTHONPATH}:}${path_entry}"
done
export PYTHONPATH="${REPO_ROOT}/test${FILTERED_PYTHONPATH:+:${FILTERED_PYTHONPATH}}"
export PYTORCH_TESTING_DEVICE_ONLY_FOR="${PYTORCH_TESTING_DEVICE_ONLY_FOR:-cuda}"

TORCH_ORIGIN="$("${PYTHON_BIN}" -c 'import torch; print(torch.__file__)' 2>/dev/null)" || {
  echo "[ERROR] Install the built PyTorch wheel before running this script." >&2
  exit 2
}
if [[ "${TORCH_ORIGIN}" == "${REPO_ROOT}/torch/"* ]]; then
  echo "[ERROR] PyTorch resolved to the source tree: ${TORCH_ORIGIN}" >&2
  exit 2
fi

VISIBLE_DEVICE_COUNT="$("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())')"
if [[ -n "${GPU_IDS:-}" ]]; then
  IFS=',' read -r -a DEVICE_IDS <<< "${GPU_IDS}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a DEVICE_IDS <<< "${CUDA_VISIBLE_DEVICES}"
else
  DEVICE_IDS=()
  for ((device_index = 0; device_index < VISIBLE_DEVICE_COUNT; device_index++)); do
    DEVICE_IDS+=("${device_index}")
  done
fi
if [[ ${#DEVICE_IDS[@]} -eq 0 ]]; then
  echo "[ERROR] No visible PPU devices found." >&2
  exit 2
fi

MAX_PARALLEL="${MAX_PARALLEL:-${#DEVICE_IDS[@]}}"
if ! [[ "${MAX_PARALLEL}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] MAX_PARALLEL must be a positive integer." >&2
  exit 2
fi
if [[ ${MAX_PARALLEL} -gt ${#DEVICE_IDS[@]} ]]; then
  MAX_PARALLEL=${#DEVICE_IDS[@]}
fi
ALL_DEVICE_IDS="$(IFS=,; echo "${DEVICE_IDS[*]}")"
PYTEST_EXTRA_ARGS=("$@")

requires_all_devices() {
  case "${1%%::*}" in
    distributed/*|test_cuda_multigpu.py|test_multiprocessing_spawn.py) return 0 ;;
    *) return 1 ;;
  esac
}

safe_case_id() {
  printf '%s' "$1" | sed 's/[^A-Za-z0-9_.-]/_/g'
}

run_test_case() {
  local test_case="$1" index="$2" visible_devices="$3" timeout_seconds="$4"
  local case_id log_file junit_xml master_port
  case_id=$(safe_case_id "${test_case}")
  log_file="${LOG_DIR}/${case_id}.log"
  junit_xml="${XML_DIR}/${case_id}.xml"
  master_port=$((29500 + index))
  (
    export CUDA_VISIBLE_DEVICES="${visible_devices}"
    export MASTER_PORT="${master_port}"
    unset LOCAL_RANK RANK WORLD_SIZE MASTER_ADDR
    timeout --signal=TERM --kill-after=30 "${timeout_seconds}" \
      "${PYTHON_BIN}" -m pytest "${test_case}" -v --tb=short \
      --junitxml="${junit_xml}" "${PYTEST_EXTRA_ARGS[@]}" >"${log_file}" 2>&1
  )
}

printf '[INFO] PyTorch: %s\n' "${TORCH_ORIGIN}"
printf '[INFO] Test entries: %d\n' "${#PYTEST_CASES[@]}"
printf '[INFO] PPU devices: %s; parallel workers: %d\n' "${ALL_DEVICE_IDS}" "${MAX_PARALLEL}"
printf '[INFO] Timeouts: normal=%ss exclusive=%ss\n' "${NORMAL_TIMEOUT}" "${EXCLUSIVE_TIMEOUT}"
printf '[INFO] Results: %s\n' "${RESULT_DIR}"

PARALLEL_CASES=()
PARALLEL_INDEXES=()
EXCLUSIVE_CASES=()
EXCLUSIVE_INDEXES=()
for index in "${!PYTEST_CASES[@]}"; do
  if requires_all_devices "${PYTEST_CASES[$index]}"; then
    EXCLUSIVE_CASES+=("${PYTEST_CASES[$index]}")
    EXCLUSIVE_INDEXES+=("${index}")
  else
    PARALLEL_CASES+=("${PYTEST_CASES[$index]}")
    PARALLEL_INDEXES+=("${index}")
  fi
done

START_TIME=$(date +%s)
FAIL_COUNT=0
for ((batch_start = 0; batch_start < ${#PARALLEL_CASES[@]}; batch_start += MAX_PARALLEL)); do
  PIDS=()
  BATCH_CASES=()
  for ((slot = 0; slot < MAX_PARALLEL && batch_start + slot < ${#PARALLEL_CASES[@]}; slot++)); do
    batch_index=$((batch_start + slot))
    test_case="${PARALLEL_CASES[$batch_index]}"
    test_index="${PARALLEL_INDEXES[$batch_index]}"
    device_id="${DEVICE_IDS[$slot]}"
    printf '[RUN] %s (PPU=%s)\n' "${test_case}" "${device_id}"
    run_test_case "${test_case}" "${test_index}" "${device_id}" "${NORMAL_TIMEOUT}" &
    PIDS+=("$!")
    BATCH_CASES+=("${test_case}")
  done
  for slot in "${!PIDS[@]}"; do
    test_case="${BATCH_CASES[$slot]}"
    if wait "${PIDS[$slot]}"; then
      printf '[PASS] %s\n' "${test_case}"
    else
      rc=$?
      case_id=$(safe_case_id "${test_case}")
      printf '[FAIL] %s (RC=%d, log=%s/%s.log)\n' "${test_case}" "${rc}" "${LOG_DIR}" "${case_id}"
      tail -n 20 "${LOG_DIR}/${case_id}.log"
      FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
  done
done

for slot in "${!EXCLUSIVE_CASES[@]}"; do
  test_case="${EXCLUSIVE_CASES[$slot]}"
  test_index="${EXCLUSIVE_INDEXES[$slot]}"
  printf '[RUN] %s (all PPUs=%s)\n' "${test_case}" "${ALL_DEVICE_IDS}"
  if run_test_case "${test_case}" "${test_index}" "${ALL_DEVICE_IDS}" "${EXCLUSIVE_TIMEOUT}"; then
    printf '[PASS] %s\n' "${test_case}"
  else
    rc=$?
    case_id=$(safe_case_id "${test_case}")
    printf '[FAIL] %s (RC=%d, log=%s/%s.log)\n' "${test_case}" "${rc}" "${LOG_DIR}" "${case_id}"
    tail -n 20 "${LOG_DIR}/${case_id}.log"
    FAIL_COUNT=$((FAIL_COUNT + 1))
  fi
done

TOTAL_DURATION=$(($(date +%s) - START_TIME))
printf '[SUMMARY] Total=%d Passed=%d Failed=%d Duration=%ss\n' \
  "${#PYTEST_CASES[@]}" "$(( ${#PYTEST_CASES[@]} - FAIL_COUNT ))" "${FAIL_COUNT}" "${TOTAL_DURATION}"

[[ ${FAIL_COUNT} -eq 0 ]]
