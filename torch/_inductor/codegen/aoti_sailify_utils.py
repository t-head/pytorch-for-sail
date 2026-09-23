# Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
import functools
import os
import re

# It is not a good idea to directly apply sailify to codegen, which will be vulnerable to cases like:
#   "...
#    from ..codecache import CudaKernelParamCache
#   ..."
# In such cases, we do not need to sailify the original class/file name in codegen/codecache
#
# This mirrors the approach in aoti_hipify_utils.py but uses PPU mappings instead of HIP.
#
# IMPORTANT: sailify converts CUDA_VERSION → COMPATIBLE_VERSION, __CUDA_ARCH__ →
# COMPATIBLE_ARCH, cudaDriverGetVersion → compatibleDriverGetVersion, etc.  These
# COMPATIBLE_* macros and compatible* functions are defined in
# compatible_wrapper.h.
#
# Version consistency: compatible_wrapper.h is pre-built during PyTorch
# build and shipped in torch/.ppu_compat/ via torch_package_data.  At runtime,
# _ensure_compat_header() reuses it if present; otherwise it regenerates with
# default config via write_compat_wrapper_header().

# Detect PPU platform via torch.version.sail (build-time marker, non-None
# only for USE_SAIL builds), mirroring torch.version.hip for ROCm.
def _detect_ppu() -> bool:
    try:
        import torch
        return getattr(torch.version, 'sail', None) is not None
    except Exception:
        return False

_IS_PPU = _detect_ppu()


@functools.lru_cache(1)
def _get_compat_header_dir() -> str:
    """Return the directory where compatible_wrapper.h is cached for Inductor."""
    import torch
    base = os.path.join(os.path.dirname(torch.__file__), '.ppu_compat')
    os.makedirs(base, exist_ok=True)
    return base


@functools.lru_cache(1)
def _ensure_compat_header() -> str:
    """Ensure compatible_wrapper.h exists in the cache dir and return the dir.

    If the header is already shipped in torch/.ppu_compat/ (via wheel
    package_data), it is reused directly.  Otherwise, it is regenerated
    via write_compat_wrapper_header(compat_dir) which writes headers
    directly to compat_dir.

    Uses lru_cache so the header is generated only once per process.
    """
    compat_dir = _get_compat_header_dir()
    header_path = os.path.join(compat_dir, 'compatible_wrapper.h')
    if not os.path.isfile(header_path):
        try:
            from torch.sailify.sailify_python import write_compat_wrapper_header
            write_compat_wrapper_header(compat_dir)
        except ImportError:
            pass  # sailify not available; header won't exist, but that's OK
    return compat_dir


def maybe_sailify_code_wrapper(source_codes: str, force_sailify: bool = False) -> str:
    if not _IS_PPU and not force_sailify:
        return source_codes

    try:
        from torch.sailify.sailify_python import _PPU_MAP, _PPU_TRIE
    except ImportError:
        # sailify not available for non-PPU builds
        return source_codes

    # Ensure compatible_wrapper.h exists before any CUDA identifier is converted
    # (e.g., CUDA_VERSION → COMPATIBLE_VERSION).  The header defines COMPATIBLE_*
    # macros that the converted code will reference.
    _ensure_compat_header()

    def c2_repl(m: re.Match[str]) -> object:
        return _PPU_MAP[m.group(0)]

    # We need to redefine RE_PPU_PREPROCESSOR here since in sailify_python,
    # it will apply positive lookbehind (?<=\W) to the pattern to avoid matching
    # keyword at the beginning of code line. However, this can happen in codegen,
    # which will cause the pattern to not match.

    # Note that lookahead (?=\W) is still needed to keep sailification idempotent, for example
    # we need to skip replacing "getStreamFromExternal" in "getStreamFromExternalMasqueradingAsCUDA"
    RE_PPU_PREPROCESSOR = re.compile(rf"({_PPU_TRIE.export_to_regex()})(?=\W)")

    source_codes = RE_PPU_PREPROCESSOR.sub(c2_repl, source_codes)  # type: ignore[arg-type]
    return source_codes
