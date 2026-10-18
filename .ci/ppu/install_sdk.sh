#!/usr/bin/env bash
set -euo pipefail

DEFAULT_SDK_URL="https://download.t-head.cn/artifactory/ptgai-generic-release/PPU_SDK/v2.2.0/ppu_hggcrt3_ubuntu2404-2.2.0-663c22.run"
SDK_URL="${SDK_URL:-$DEFAULT_SDK_URL}"
SDK_INSTALL_DIR="${SDK_INSTALL_DIR:-/usr/local}"
SDK_LICENSE_ANSWER="${SDK_LICENSE_ANSWER:-accept}"
SDK_INSTALL_TIMEOUT="${SDK_INSTALL_TIMEOUT:-300}"
ENVSETUP="$SDK_INSTALL_DIR/PPU_SDK/envsetup.sh"

log() { echo "[install_sdk] $*"; }
die() { echo "[install_sdk] $*" >&2; exit 1; }

download_sdk() {
    local dest="$1" downloader
    log "下载: $SDK_URL"
    log "落盘: $dest"
    if command -v wget >/dev/null 2>&1; then
        downloader=wget
        log "下载器: $(wget --version 2>&1 | head -1)"
    elif command -v curl >/dev/null 2>&1; then
        downloader=curl
        log "下载器: $(curl --version 2>&1 | head -1)"
    else
        die "镜像内既无 wget 也无 curl，无法下载 SDK"
    fi
    log "磁盘可用量（SDK 包为 GB 级，/tmp 若是小 tmpfs 会写满）:"
    df -h /tmp "$SDK_INSTALL_DIR" 2>&1 || true

    if [[ "$downloader" == wget ]]; then
        wget -nv -O "$dest" "$SDK_URL" 2>&1
    else
        curl -fSL --retry 3 -o "$dest" "$SDK_URL" 2>&1
    fi
    log "下载完成: $(ls -lh "$dest" | awk '{print $5}')"
}

try_silent_install() {
    local runfile="$1" help flags=() f
    help="$(timeout 120 "$runfile" --help </dev/null 2>&1 || true)"
    for f in --silent --unattended --no-questions --accept-license --quiet; do
        if grep -q -- "$f" <<<"$help"; then
            flags+=("$f")
        fi
    done
    [[ ${#flags[@]} -eq 0 ]] && return 1
    log "检测到免交互参数，尝试静默安装: ${flags[*]}"
    timeout "$SDK_INSTALL_TIMEOUT" "$runfile" "${flags[@]}" </dev/null
}

run_with_expect() {
    local runfile="$1"
    log "使用 expect 自动化交互（翻页 + 输入 '$SDK_LICENSE_ANSWER'）"
    expect -f - "$runfile" "$SDK_LICENSE_ANSWER" "$SDK_INSTALL_TIMEOUT" <<'EXPECT_EOF'
set runfile [lindex $argv 0]
set answer  [lindex $argv 1]
set timeout [lindex $argv 2]
log_user 1
spawn $runfile
expect {
    -re {--More|\(END\)|\[Press [^\]]*\]|Press (space|SPACE|ENTER|Enter)|press (space|enter)} {
        send " "
        exp_continue
    }
    -nocase -re {accept|agree|y/n|yes/no|\[y/n\]|do you (accept|agree)} {
        send "$answer\r"
        exp_continue
    }
    -nocase -re {continue|proceed|install|prefix|director|destination|press enter} {
        send "\r"
        exp_continue
    }
    timeout {
        puts "\n\[install_sdk\] expect 交互超时"
        exit 124
    }
    eof
}
catch wait result
exit [lindex $result 3]
EXPECT_EOF
}

run_with_pipe() {
    local runfile="$1" i
    log "未检测到 expect，改用管道喂入按键序列（翻页 + '$SDK_LICENSE_ANSWER'）"
    {
        for ((i = 0; i < 200; i++)); do printf ' \n'; done
        printf '%s\n' "$SDK_LICENSE_ANSWER"
        for ((i = 0; i < 50; i++)); do printf '\n'; done
    } | timeout "$SDK_INSTALL_TIMEOUT" "$runfile"
}

install_from_runfile() {
    local runfile="$1"
    chmod +x "$runfile"

    if try_silent_install "$runfile" && [[ -f "$ENVSETUP" ]]; then
        log "免交互静默安装成功"
        return 0
    fi

    log "回退到交互式自动化安装: $runfile"
    if command -v expect >/dev/null 2>&1; then
        run_with_expect "$runfile" || true
    else
        run_with_pipe "$runfile" || true
    fi
}

if [[ -f "$ENVSETUP" ]]; then
    log "检测到已安装 SDK，跳过下载: $SDK_INSTALL_DIR/PPU_SDK"
else
    [[ "$SDK_INSTALL_DIR" == /usr/local ]] || \
        log "警告: SDK_INSTALL_DIR=$SDK_INSTALL_DIR 非默认 /usr/local，若安装程序不支持自定义前缀，可能仍装到默认路径"

    RUNFILE="/tmp/$(basename "$SDK_URL")"
    download_sdk "$RUNFILE"
    [[ -f "$RUNFILE" ]] || die "下载失败: $RUNFILE"

    log "开始安装（自解压 .run，将自动翻页并接受许可）: $RUNFILE"
    install_from_runfile "$RUNFILE"
    rm -f "$RUNFILE"

    [[ -f "$ENVSETUP" ]] || die "安装后未找到 $ENVSETUP"
fi

if [[ ! -e /usr/local/cuda ]]; then
    ln -s "$SDK_INSTALL_DIR/PPU_SDK/CUDA_SDK" /usr/local/cuda
    log "已创建软链: /usr/local/cuda -> PPU_SDK/CUDA_SDK"
else
    log "/usr/local/cuda 已存在，跳过软链"
fi

set +u
source "$ENVSETUP"
set -u
log "envsetup.sh 已 source"
command -v nvcc && nvcc --version | tail -2 || { echo "[install_sdk] 警告: nvcc 不在 PATH" >&2; }

log "完成: $SDK_INSTALL_DIR/PPU_SDK"
