# Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
# mypy: allow-untyped-defs
r"""Centralized SMI (System Management Interface) helpers.

Single source of truth for selecting between NVIDIA's ``nvidia-smi``
(CUDA-compatible mode) and T-Head PPU's ``ppu-smi`` (SAIL mode), and for
querying values that differ between the two tools.

Design notes (why this module exists / how to maintain the SAIL patch):

* This module depends ONLY on the Python standard library -- it must NOT
  ``import torch``.  That keeps it free of circular-import issues, usable very
  early during ``import torch``, and cheap to call from anywhere.
* It is a brand-new file (upstream PyTorch has no equivalent), so it never
  conflicts when the SAIL patch is rebased onto a new PyTorch version.
  All PPU/nvidia-smi divergence lives here; call sites change by only one or
  two lines, minimizing the patch's contact surface with upstream code.
* ``torch/utils/collect_env.py`` deliberately keeps its OWN self-contained
  copy of ``is_sail_mode``/command selection, because that script must
  stay runnable even when ``import torch`` fails (its whole purpose is
  diagnostics).  Keep the detection semantics here and there in sync.
"""

import re
import shutil
import subprocess
import sys


# Safe fallback for the max SM/CU clock (MHz) on current PPU devices
# (e.g. PPU-ZW810E) when ppu-smi cannot be queried.
_PPU_DEFAULT_MAX_SM_CLOCK_MHZ = 1700


def _build_time_sail_marker():
    """Return ``torch.version.sail`` when torch is ALREADY imported, else None.

    This deliberately does NOT ``import torch`` -- it only inspects
    ``sys.modules`` and then imports the generated, dependency-free
    ``torch.version`` submodule.  That preserves this module's contract (never
    force-import the full ``torch`` package, stay usable very early during
    ``import torch`` and from diagnostics), while still letting the build-time
    marker win once ``torch`` is available.

    ``torch.version.sail`` is non-None only in USE_SAIL builds, mirroring
    how ``torch.version.hip`` marks ROCm builds.
    """
    if "torch" not in sys.modules:
        return None
    try:
        from torch.version import sail as _sail
    except Exception:
        return None
    return _sail


def is_sail_mode() -> bool:
    """Return True when running in SAIL (PPU) mode.

    SAIL is determined SOLELY by the build-time marker
    ``torch.version.sail`` (non-None only for USE_SAIL builds), exactly
    mirroring how ROCm is detected via ``torch.version.hip is not None``.  The
    marker is read via :func:`_build_time_sail_marker`, which inspects
    ``sys.modules`` instead of importing torch, so this stays faithful to the
    module's 'never force-import torch' contract and is safe to call very early
    (it simply reports False until ``torch.version`` is available).
    """
    return _build_time_sail_marker() is not None


def get_smi_command() -> str:
    """Return the GPU management CLI name for the current mode.

    ``"ppu-smi"`` in SAIL mode, otherwise ``"nvidia-smi"``.  Callers that
    build a full command line (e.g. ``f"{get_smi_command()} topo -m"``) get
    automatic dual-mode behavior with no other changes.
    """
    return "ppu-smi" if is_sail_mode() else "nvidia-smi"


def get_ppu_max_sm_clock_mhz() -> int:
    """Return the max SM/CU clock rate (MHz) for a PPU device.

    Tries, in order:

    1. ``ppu-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits``
       (the nvidia-smi-compatible query; works when ppu-smi mirrors the CLI).
    2. Parse ``ppu-smi -q -d CLOCK`` for the first "CU" line under "Max Clocks".
    3. A hardcoded safe default (:data:`_PPU_DEFAULT_MAX_SM_CLOCK_MHZ`).
    """
    ppu_smi = shutil.which("ppu-smi") or "ppu-smi"

    # 1) nvidia-smi compatible query (most robust when supported).
    try:
        out = subprocess.check_output(
            [ppu_smi, "--query-gpu=clocks.max.sm", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
        ).strip()
        val = int(float(out.splitlines()[0].strip()))
        if val > 0:
            return val
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError, IndexError):
        pass

    # 2) Verbose output parse: first "CU" under the "Max Clocks" block.
    try:
        out = subprocess.check_output(
            [ppu_smi, "-q", "-d", "CLOCK"],
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
        )
        in_max_block = False
        for line in out.splitlines():
            if "Max Clocks" in line:
                in_max_block = True
                continue
            if in_max_block and line.strip().startswith("CU"):
                m = re.search(r"(\d+)\s*MHz", line)
                if m:
                    return int(m.group(1))
                break
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # 3) Hardcoded safe default.
    return _PPU_DEFAULT_MAX_SM_CLOCK_MHZ
