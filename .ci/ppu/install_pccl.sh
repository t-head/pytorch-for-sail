#!/usr/bin/env bash
set -euo pipefail

DEFAULT_PCCL_URL="https://download.t-head.cn/artifactory/ptgai-generic-release/COMM/v2.2.0/pccl_hggcrt3_ubuntu2404-v2.2.0.tar.gz"
PCCL_URL="${PCCL_URL:-$DEFAULT_PCCL_URL}"
SDK_INSTALL_DIR="${SDK_INSTALL_DIR:-/usr/local}"
ENVSETUP="$SDK_INSTALL_DIR/pccl/envsetup.sh"

TARBALL="/tmp/$(basename "$PCCL_URL")"
echo "[install_pccl] Downloading: $PCCL_URL"
echo "[install_pccl] Saving to: $TARBALL"
if command -v wget >/dev/null 2>&1; then
    DOWNLOADER=wget
    echo "[install_pccl] Downloader: $(wget --version 2>&1 | head -1)"
elif command -v curl >/dev/null 2>&1; then
    DOWNLOADER=curl
    echo "[install_pccl] Downloader: $(curl --version 2>&1 | head -1)"
else
    echo "[install_pccl] Neither wget nor curl is available in the image, cannot download PCCL" >&2
    exit 1
fi
echo "[install_pccl] Available disk space:"
df -h /tmp "$SDK_INSTALL_DIR" 2>&1 || true

if [[ "$DOWNLOADER" == wget ]]; then
    wget -nv -O "$TARBALL" "$PCCL_URL" 2>&1
else
    curl -fSL --retry 3 -o "$TARBALL" "$PCCL_URL" 2>&1
fi
echo "[install_pccl] Download complete: $(ls -lh "$TARBALL" | awk '{print $5}')"

echo "[install_pccl] Extracting to: $SDK_INSTALL_DIR"
tar -zxf "$TARBALL" -C "$SDK_INSTALL_DIR"
rm -f "$TARBALL"
[[ -f "$ENVSETUP" ]] || {
    echo "[install_pccl] $ENVSETUP not found after extraction" >&2; exit 1; }

set +u
source "$ENVSETUP" release
set -u
echo "[install_pccl] envsetup.sh sourced (release)"

if [[ -z "${PCCL_HOME:-}" ]]; then
    echo "[install_pccl] PCCL environment error: PCCL_HOME is not set" >&2
    exit 1
fi

export PCCL_HOME
export PCCL_INCLUDE_DIR="$PCCL_HOME/include"
export PCCL_LIB_DIR="$PCCL_HOME/lib"

echo "[install_pccl] Done: PCCL_HOME=$PCCL_HOME"
