#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WHEEL_DIR="${WHEEL_DIR:-$REPO_ROOT/.ci/ppu/wheelhouse}"
shopt -s nullglob
WHLS=("$WHEEL_DIR"/torch-*.whl)
if [[ ${#WHLS[@]} -eq 0 ]]; then
    cat >&2 <<EOF
[wheel][error] torch-*.whl not found under $WHEEL_DIR.
Possible causes:
  - the workflow step "download the whl produced by the ppu-ci-zhenwu-810e/ppu-ci-zhenwu-m890 build job" did not run (or downloaded it to another directory);
  - t-head/ppu-scheduler-action@main dropped this whl when packaging the source; the directory must be inside the source tree
    and must not be matched by .gitignore (this is also why .ci/ppu/wheelhouse is used here instead of dist/:
    dist/ is in .gitignore, and packaging that filters by gitignore would drop it entirely).
Current directory contents:
EOF
    ls -la "$WHEEL_DIR" >&2 2>/dev/null || echo "  (directory does not exist)" >&2
    exit 1
fi
if [[ ${#WHLS[@]} -gt 1 ]]; then
    echo "[wheel][error] Multiple torch whls under $WHEEL_DIR, cannot decide which one to install:" >&2
    printf '  %s\n' "${WHLS[@]}" >&2
    exit 1
fi
WHEEL="${WHLS[0]}"
echo "=== Installing Zhenwu PPU torch whl ==="
echo "[wheel] Package: $(basename "$WHEEL") ($(du -h "$WHEEL" | cut -f1))"

if python -m pip uninstall -y torch >/dev/null 2>&1; then
    echo "[wheel] Uninstalled the torch previously present in the environment"
else
    echo "[wheel] No torch in the environment originally (expected state of the Zhenwu PPU base image)"
fi

source "$REPO_ROOT/.ci/ppu/pip_sources.sh"
ppu_build_pip_candidates

installed=0
for index in "${PIP_CANDIDATES[@]}" __pip_default__; do
    pip_args=(--disable-pip-version-check --no-cache-dir --retries 1 --timeout 20)
    if [[ "${index}" == "__pip_default__" ]]; then
        label="pip default index"
    else
        label="${index}"
        pip_args+=(-i "${index}")
    fi
    echo "[wheel] Trying index: ${label}"
    if python -m pip install "${pip_args[@]}" "$WHEEL"; then
        echo "[wheel] Install succeeded (index: ${label})"
        installed=1
        break
    fi
    echo "[wheel][warn] Index ${label} failed, trying the next one"
done

if [[ "${installed}" -ne 1 ]]; then
    cat >&2 <<'EOF'
[wheel][error] Failed to install the whl. torch itself is a local file and needs no network, so failure almost certainly means its
runtime dependencies (filelock / typing-extensions / setuptools / sympy / networkx / jinja2 /
fsspec) could not be fetched, or the disk is full. For how to diagnose pip index connectivity, see the
"pip index connectivity diagnostics" section of install_test_deps.sh.
EOF
    exit 1
fi

(
    cd /tmp
    python - "$REPO_ROOT" <<'PY'
import sys

import torch

repo_root = sys.argv[1]
print("torch", torch.__version__, torch.__file__)
print("git_version", getattr(torch.version, "git_version", "<unknown>"))
print("built_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())

if torch.__file__.startswith(repo_root):
    sys.exit(f"[wheel] Imported torch from the source tree ({torch.__file__}), the whl did not take effect")
PY
)

echo "[wheel] Done"
