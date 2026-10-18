#!/usr/bin/env bash
set -euo pipefail

DEFAULT_PCCL_URL="https://download.t-head.cn/artifactory/ptgai-generic-release/COMM/v2.2.0/pccl_hggcrt3_ubuntu2404-v2.2.0.tar.gz"
PCCL_URL="${PCCL_URL:-$DEFAULT_PCCL_URL}"
SDK_INSTALL_DIR="${SDK_INSTALL_DIR:-/usr/local}"
ENVSETUP="$SDK_INSTALL_DIR/pccl/envsetup.sh"

TARBALL="/tmp/$(basename "$PCCL_URL")"
echo "[install_pccl] 下载: $PCCL_URL"
echo "[install_pccl] 落盘: $TARBALL"
if command -v wget >/dev/null 2>&1; then
    DOWNLOADER=wget
    echo "[install_pccl] 下载器: $(wget --version 2>&1 | head -1)"
elif command -v curl >/dev/null 2>&1; then
    DOWNLOADER=curl
    echo "[install_pccl] 下载器: $(curl --version 2>&1 | head -1)"
else
    echo "[install_pccl] 镜像内既无 wget 也无 curl，无法下载 PCCL" >&2
    exit 1
fi
echo "[install_pccl] 磁盘可用量:"
df -h /tmp "$SDK_INSTALL_DIR" 2>&1 || true

if [[ "$DOWNLOADER" == wget ]]; then
    wget -nv -O "$TARBALL" "$PCCL_URL" 2>&1
else
    curl -fSL --retry 3 -o "$TARBALL" "$PCCL_URL" 2>&1
fi
echo "[install_pccl] 下载完成: $(ls -lh "$TARBALL" | awk '{print $5}')"

echo "[install_pccl] 解压到: $SDK_INSTALL_DIR"
tar -zxf "$TARBALL" -C "$SDK_INSTALL_DIR"
rm -f "$TARBALL"
[[ -f "$ENVSETUP" ]] || {
    echo "[install_pccl] 解压后未找到 $ENVSETUP" >&2; exit 1; }

set +u
source "$ENVSETUP" release
set -u
echo "[install_pccl] envsetup.sh 已 source (release)"

if [[ -z "${PCCL_HOME:-}" ]]; then
    echo "[install_pccl] PCCL 环境异常: PCCL_HOME 未设置" >&2
    exit 1
fi

export PCCL_HOME
export PCCL_INCLUDE_DIR="$PCCL_HOME/include"
export PCCL_LIB_DIR="$PCCL_HOME/lib"

echo "[install_pccl] 完成: PCCL_HOME=$PCCL_HOME"
