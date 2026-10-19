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
    log "Downloading: $SDK_URL"
    log "Saving to: $dest"
    if command -v wget >/dev/null 2>&1; then
        downloader=wget
        log "Downloader: $(wget --version 2>&1 | head -1)"
    elif command -v curl >/dev/null 2>&1; then
        downloader=curl
        log "Downloader: $(curl --version 2>&1 | head -1)"
    else
        die "Neither wget nor curl is available in the image, cannot download the SDK"
    fi
    log "Available disk space (the SDK package is GB-sized; a small tmpfs /tmp will fill up):"
    df -h /tmp "$SDK_INSTALL_DIR" 2>&1 || true

    if [[ "$downloader" == wget ]]; then
        wget -nv -O "$dest" "$SDK_URL" 2>&1
    else
        curl -fSL --retry 3 -o "$dest" "$SDK_URL" 2>&1
    fi
    log "Download complete: $(ls -lh "$dest" | awk '{print $5}')"
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
    log "Non-interactive flags detected, trying silent install: ${flags[*]}"
    timeout "$SDK_INSTALL_TIMEOUT" "$runfile" "${flags[@]}" </dev/null
}

run_with_expect() {
    local runfile="$1"
    log "Using expect to automate the interaction (page through + answer '$SDK_LICENSE_ANSWER')"
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
        puts "\n\[install_sdk\] expect interaction timed out"
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
    log "expect not found, piping a key sequence instead (page through + '$SDK_LICENSE_ANSWER')"
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
        log "Non-interactive silent install succeeded"
        return 0
    fi

    log "Falling back to automated interactive install: $runfile"
    if command -v expect >/dev/null 2>&1; then
        run_with_expect "$runfile" || true
    else
        run_with_pipe "$runfile" || true
    fi
}

if [[ -f "$ENVSETUP" ]]; then
    log "SDK already installed, skipping download: $SDK_INSTALL_DIR/PPU_SDK"
else
    [[ "$SDK_INSTALL_DIR" == /usr/local ]] || \
        log "Warning: SDK_INSTALL_DIR=$SDK_INSTALL_DIR is not the default /usr/local; if the installer does not support a custom prefix, it may still install to the default path"

    RUNFILE="/tmp/$(basename "$SDK_URL")"
    download_sdk "$RUNFILE"
    [[ -f "$RUNFILE" ]] || die "Download failed: $RUNFILE"

    log "Starting install (self-extracting .run, will page through and accept the license automatically): $RUNFILE"
    install_from_runfile "$RUNFILE"
    rm -f "$RUNFILE"

    [[ -f "$ENVSETUP" ]] || die "$ENVSETUP not found after install"
fi

if [[ ! -e /usr/local/cuda ]]; then
    ln -s "$SDK_INSTALL_DIR/PPU_SDK/CUDA_SDK" /usr/local/cuda
    log "Created symlink: /usr/local/cuda -> PPU_SDK/CUDA_SDK"
else
    log "/usr/local/cuda already exists, skipping symlink"
fi

set +u
source "$ENVSETUP"
set -u
log "envsetup.sh sourced"
command -v nvcc && nvcc --version | tail -2 || { echo "[install_sdk] Warning: nvcc is not in PATH" >&2; }

log "Done: $SDK_INSTALL_DIR/PPU_SDK"
