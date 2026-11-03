#!/bin/bash
# Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
#
# Quick PyTorch-for-SAIL build. Runs the two-stage sailify build from
# README.md inside the environment image built by Dockerfile.sail-sdk2.2
# (README "From Source", Step 1).
#
# Usage:
#   ./docker/sail/build-sail.sh              # in-place build; rewrites the source tree
#   ./docker/sail/build-sail.sh <output-dir> # out-of-place; converted tree at <output-dir>
#
# Overrides:
#   IMAGE=...    build image (default pytorch-sail-buildenv:2.14.1-sdk2.2.1)
#   MAX_JOBS=... parallel compile jobs (default 128)
#
# The wheel is written to dist/ in the converted tree.
set -euo pipefail

IMAGE="${IMAGE:-pytorch-sail-buildenv:2.14.1-sdk2.2.1}"
MAX_JOBS="${MAX_JOBS:-128}"
SRC_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_DIR="${1:-}"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Error: image $IMAGE not found. Build it first (README 'From Source', Step 1)." >&2
  exit 1
fi

MOUNT_ARGS=(-v "$SRC_DIR":/workspace/pytorch)
BUILD_CMD="cd /workspace/pytorch && USE_SAIL=1 python3 -m tools.sailify && \
python3 -m pip install . --no-build-isolation -v"

if [[ -n "$OUT_DIR" ]]; then
  mkdir -p "$OUT_DIR"
  OUT_DIR="$(cd "$OUT_DIR" && pwd)"
  MOUNT_ARGS+=(-v "$OUT_DIR":/workspace/sail-build)
  BUILD_CMD="cd /workspace/pytorch && USE_SAIL=1 python3 -m tools.sailify --output-dir /workspace/sail-build && \
cd /workspace/sail-build && python3 -m pip install . --no-build-isolation -v"
fi

docker run --rm --name sail-build-v2141 \
  "${MOUNT_ARGS[@]}" \
  -e MAX_JOBS="$MAX_JOBS" \
  "$IMAGE" \
  bash -lc 'source "$PPU_SDK/envsetup.sh" && source "$PCCL_HOME/envsetup.sh" release && '"$BUILD_CMD"

if [[ -n "${OUT_DIR:-}" ]]; then
  echo "Wheel: $OUT_DIR/dist/torch-2.14.1-*.whl"
else
  echo "Wheel: $SRC_DIR/dist/torch-2.14.1-*.whl"
fi
