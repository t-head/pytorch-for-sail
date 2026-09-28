#!/usr/bin/env bash
# Run an individual SAIL test with the SDK runtime inherited by spawned workers.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
test_root=${SAIL_TEST_ROOT:-"$repo_root/test"}
cuda_sdk=${CUDA_SDK:-}
runtime_lib="$cuda_sdk/targets/x86_64-linux/lib"
if [[ ! -d "$runtime_lib" ]]; then
    cuda_sdk=/usr/local/PPU_SDK-ori/CUDA_SDK
    runtime_lib="$cuda_sdk/targets/x86_64-linux/lib"
fi

if [[ ! -d "$runtime_lib" ]]; then
    echo "SAIL CUDA-compatible runtime not found: $runtime_lib" >&2
    exit 2
fi
if [[ $# -lt 1 ]]; then
    echo "usage: $0 <test-path-relative-to-test-root> [test-name ...]" >&2
    exit 2
fi

export CUDA_SDK="$cuda_sdk"
export LD_LIBRARY_PATH="$runtime_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1

cd "$test_root"
python3 -c 'import torch; assert torch.version.sail is not None, torch.version.sail; print(f"torch={torch.__version__} sail={torch.version.sail}")'
python3 "$@"
