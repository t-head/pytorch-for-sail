# Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
"""Sailify source-tree conversion and third-party patch application.

Extracted from PyTorch v2.13 ``setup.py`` and adapted for the v2.14+
scikit-build-core layout. This module is imported by ``__main__.py``
(CLI) and by ``.spin/cmds.py`` (``spin sailify``).
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def str2bool(value: str | None) -> bool:
    if value is None:
        return False
    return value.lower() in ("true", "1", "yes", "on")


def check_path_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"The path doesn't exist: '{path}', please check PPU_SDK environment."
        )


def _missing_submodule_error(
    torch_root: str, submodule_path: str
) -> FileNotFoundError:
    return FileNotFoundError(
        "Required third-party source is missing: "
        f"{os.path.join(torch_root, submodule_path)}\n"
        "Please fetch it with git submodule, for example:\n"
        f"  git submodule update --init --recursive {submodule_path}\n"
        "Or initialize all required submodules with:\n"
        "  git submodule update --init --recursive"
    )


# ---------------------------------------------------------------------------
# Patch application
# ---------------------------------------------------------------------------

def apply_patch(patch_file: str, target_root: str) -> None:
    if not os.path.isfile(patch_file):
        raise FileNotFoundError(f"The patch file doesn't exist: {patch_file}")

    command = ["patch", "--batch", "-p1", "-i", patch_file]
    dry_run = subprocess.run(
        [*command, "--dry-run"],
        cwd=target_root,
        capture_output=True,
        text=True,
    )
    combined_output = (dry_run.stdout or "") + (dry_run.stderr or "")
    if re.search(
        "already exists|Reversed|previously applied",
        combined_output,
        re.IGNORECASE,
    ):
        print("Patch already applied, skipping.")
        return
    if dry_run.returncode != 0:
        print(f"error_output: {combined_output}")
        raise RuntimeError(
            f"Failed to validate patch before applying: {patch_file}"
        )
    subprocess.run(command, cwd=target_root, check=True)
    print(f"Apply patch: {patch_file}")


def apply_cutlass_sail_patch(patch_file: str, target_root: str) -> None:
    """Apply the CUTLASS SAIL compatibility patch as one logical unit."""
    markers = {
        "include/cutlass/bfloat16.h": (
            "struct numeric_limits<cutlass::bfloat16_t>",
            "namespace platform",
        ),
        "include/cutlass/half.h": (
            "struct numeric_limits<cutlass::half_t>",
            "namespace cutlass {\nnamespace platform {",
        ),
        "include/cutlass/arch/arch.h": ("using Sm80 = PPU0010;",),
        "include/cutlass/epilogue/threadblock/fusion/visitor_2x.hpp": (
            "using Sm80EVT = PPU0010EVT<NodeOp, ChildOps...>;",
        ),
    }
    already_applied = True
    for relative_path, required_markers in markers.items():
        content = Path(target_root, relative_path).read_text(
            encoding="utf-8", errors="replace"
        )
        if not all(marker in content for marker in required_markers):
            already_applied = False
            break
    if already_applied:
        print("CUTLASS SAIL compatibility patch already applied, skipping.")
        return
    apply_patch(patch_file, target_root)


def apply_fa_patches_with_markers(
    namespace_patch: str, num_splits_patch: str, fa_root: str
) -> None:
    """Apply the flash-attention patches idempotently."""
    namespace_header = Path(
        fa_root, "csrc", "flash_attn", "src", "namespace_config.h"
    )
    if namespace_header.is_file() and "FLASH_NAMESPACE_CONFIG_H" in namespace_header.read_text(
        encoding="utf-8", errors="replace"
    ):
        print("flash-attention namespace patch already applied, skipping.")
    else:
        apply_patch(namespace_patch, fa_root)

    flash_api = Path(fa_root, "csrc", "flash_attn", "flash_api.cpp")
    if flash_api.is_file() and "num_splits is not supported on PPU" in flash_api.read_text(
        encoding="utf-8", errors="replace"
    ):
        print("flash-attention num_splits guard already applied, skipping.")
    else:
        apply_patch(num_splits_patch, fa_root)


def apply_kineto_sail_patch(patch_file: str, kineto_root: str) -> None:
    """Allow SAIL builds to reuse pre-created CUDA::cupti/cudart targets.

    In a T-Head SAIL build container, hg_native.cmake creates CMake imported
    targets ``CUDA::cupti`` and ``CUDA::cudart`` that map to ``libhgpti.so``
    and ``libhggcrt``.  Skip the mandatory ``find_package(CUDAToolkit)`` probe
    when those targets are already present, otherwise configure fails for lack
    of nvcc/CUDA toolkit layout.
    """
    cmake_lists = Path(kineto_root, "libkineto", "CMakeLists.txt")
    if cmake_lists.is_file() and "PPU SAIL: skip CUDAToolkit probe" in cmake_lists.read_text(
        encoding="utf-8", errors="replace"
    ):
        print("kineto SAIL CMake patch already applied, skipping.")
        return
    apply_patch(patch_file, kineto_root)


# ---------------------------------------------------------------------------
# Sailify conversion
# ---------------------------------------------------------------------------

# Marker written into the converted tree.
SAILIFY_MARKER = ".sailify_done"

_SAILIFY_EXTS = (".cu", ".cuh", ".c", ".cc", ".cpp", ".h", ".hpp", ".inl")
_SAILIFY_FINGERPRINT_EXTS = _SAILIFY_EXTS + (
    ".py", ".cmake", ".txt", ".json", ".patch", ".in",
)
_SAILIFY_SKIP_DIRS = {".git", "build", "dist", "__pycache__", ".ppu_compat"}


def apply_gloo_sailify_cmake_patch(source_root: Path) -> None:
    """Apply the torch-level Sailify.cmake patch to gloo-for-sail.

    The patch is maintained in the PyTorch tree (not inside the
    gloo-for-sail submodule) so the submodule remains unmodified.
    It makes the gloo conversion use CMake-managed PYTHONPATH to
    invoke sailify_cli directly, avoiding the .sailify-bin wrapper.
    """
    use_sail = str2bool(os.getenv("USE_SAIL"))
    if not use_sail:
        return

    target = source_root / "third_party" / "gloo-for-sail" / "cmake" / "Sailify.cmake"
    if not target.is_file():
        return

    # Idempotency: the patched file defines SAILIFY_RUN_COMMAND.
    content = target.read_text(encoding="utf-8", errors="replace")
    if "SAILIFY_RUN_COMMAND" in content:
        print("gloo-for-sail Sailify.cmake patch already applied, skipping.")
        return

    patch_file = source_root / "tools" / "sailify" / "patches" / "gloo-for-sail-cmake-sailify.patch"
    if not patch_file.is_file():
        raise FileNotFoundError(
            f"gloo Sailify.cmake patch not found at {patch_file}. "
            "Run qoder_scripts/setup_torch_level_patch.sh to generate it."
        )
    apply_patch(str(patch_file), str(source_root))


def _sailify_fingerprint(source_root: Path) -> str:
    """Return a stable metadata fingerprint for all convertible sources."""
    digest = hashlib.sha256()
    count = 0
    for dirpath, dirnames, filenames in os.walk(source_root):
        dirnames[:] = sorted(
            d for d in dirnames if d not in _SAILIFY_SKIP_DIRS
        )
        for name in sorted(filenames):
            if not name.endswith(_SAILIFY_FINGERPRINT_EXTS):
                continue
            path = os.path.join(dirpath, name)
            try:
                stat = os.lstat(path)
            except OSError:
                continue
            relative_path = os.path.relpath(path, source_root)
            digest.update(
                relative_path.encode("utf-8", errors="surrogateescape")
            )
            digest.update(
                f"\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode()
            )
            count += 1
    return f"sources={count} metadata_sha256={digest.hexdigest()}"


def _print_sailify_handover(out_dir: str) -> None:
    print(
        "\n"
        "=========================================================\n"
        "This tree was NOT modified and is still CUDA code, so the\n"
        "build has to run in the converted tree:\n"
        f"    cd {out_dir}\n"
        "    pip install . --no-build-isolation -v\n"
        "========================================================="
    )


def run_sailify_conversion(
    source_root: Path,
    *,
    output_dir: str | None = None,
    force: bool = False,
    in_place_explicit: bool = False,
) -> bool:
    """Convert this tree's CUDA sources to PPU.

    Returns True when the conversion was in place (so the caller should keep
    building this tree), False when it went to a separate output tree.

    Two modes:

    * In-place (default). sailify() with an empty output_directory falls
      back to output_directory = project_directory and rewrites every
      convertible file under the tree -- including third_party. This is only
      safe on a throwaway checkout (CI).

    * Out-of-place (output_dir set). The result is written elsewhere and this
      tree is left untouched.
    """
    if in_place_explicit and output_dir:
        raise RuntimeError(
            "SAILIFY_IN_PLACE and --output-dir are mutually exclusive; "
            "choose one."
        )
    in_place = in_place_explicit or not output_dir
    if in_place:
        out_dir = str(source_root.resolve())
    else:
        out_dir = os.path.abspath(output_dir)
        if out_dir == str(source_root.resolve()):
            raise RuntimeError(
                f"--output-dir must differ from the source tree ({source_root}); "
                "omit it to rewrite this tree in place instead."
            )

    marker = Path(out_dir, SAILIFY_MARKER)
    fingerprint = _sailify_fingerprint(source_root)
    if marker.is_file() and not force:
        previous = marker.read_text(encoding="utf-8")
        if f"fingerprint: {fingerprint}\n" in previous:
            print(
                f"sailify: {out_dir} is already up to date ({fingerprint}); "
                "skipping conversion. Use --force to convert anyway."
            )
            if not in_place:
                _print_sailify_handover(out_dir)
            return in_place
        print(
            "sailify: sources changed since the last conversion, "
            "re-converting."
        )
        print(f"  recorded: {previous.strip()}")
        print(f"  current : fingerprint: {fingerprint}")
        print(
            "  NOTE: this overwrites source files in the target tree; "
            "any edits made there will be lost."
        )

    sailify_src = source_root / "third_party" / "sailify"
    if not sailify_src.is_dir():
        raise FileNotFoundError(
            f"sailify source not found at {sailify_src}. "
            "Ensure the submodule is checked out."
        )
    # Make sailify importable without `pip install -e`
    if str(sailify_src) not in sys.path:
        sys.path.insert(0, str(sailify_src))
    from sailify.sailify_python import sailify  # type: ignore[import]

    extra_mapping = str(
        source_root / "third_party" / "sailify_torch_extra_mappings.json"
    )
    ignores = [
        "**/third_party/cutlass/include/cutlass/bfloat16.h",
        "**/third_party/cutlass/include/cutlass/half.h",
        "**/third_party/cutlass/include/cutlass/arch/arch.h",
        "**/third_party/cutlass/include/cutlass/epilogue/threadblock/fusion/visitor_2x.hpp",
        *(
            pattern
            for pattern in os.getenv("SAILIFY_IGNORES", "").split(os.pathsep)
            if pattern
        ),
    ]
    print(
        f"=== sailify: {source_root} -> "
        f"{'(in place)' if in_place else out_dir} ==="
    )
    if marker.is_file():
        marker.unlink()

    results = sailify(
        project_directory=str(source_root),
        output_directory="" if in_place else out_dir,
        extra_mapping=extra_mapping,
        ignores=ignores,
        verbose=True,
    )
    n_ok = sum(1 for r in results.values() if r.status == "ok")
    n_skip = sum(1 for r in results.values() if r.status == "skipped")
    failures = {
        path: result.status
        for path, result in results.items()
        if result.status not in {"ok", "skipped"}
    }
    print(f"sailify: {n_ok} files converted, {n_skip} files skipped.")
    if failures:
        details = "\n".join(
            f"  {path}: {status}"
            for path, status in list(failures.items())[:20]
        )
        raise RuntimeError(
            f"sailify failed to convert {len(failures)} file(s):\n{details}"
        )

    marker.write_text(
        f"source: {source_root}\nin_place: {in_place}\n"
        f"fingerprint: {fingerprint}\n",
        encoding="utf-8",
    )
    if not in_place:
        _print_sailify_handover(out_dir)
    return in_place


def vendor_sailify_package(target_root: Path) -> None:
    """Copy sailify modules into torch/sailify/ for wheel packaging.

    In v2.13, setup.py used ``package_dir`` to map ``torch.sailify`` to
    ``third_party/sailify/sailify/``. scikit-build-core (v2.14+) does not
    support ``package_dir``, so we physically copy the modules before the
    wheel build. The ``torch/sailify/__init__.py`` in the source tree is
    preserved; only missing modules are copied.
    """
    sailify_src = target_root / "third_party" / "sailify" / "sailify"
    torch_sailify = target_root / "torch" / "sailify"
    if not sailify_src.is_dir():
        print(
            f"Warning: sailify source not found at {sailify_src}; "
            "skipping vendor step. JIT/CUDAExtension sailify features "
            "may not work in the installed wheel."
        )
        return
    torch_sailify.mkdir(parents=True, exist_ok=True)
    # Copy modules, preserving the custom __init__.py if present
    for item in sailify_src.iterdir():
        if item.name == "__pycache__":
            continue
        dest = torch_sailify / item.name
        if item.name == "__init__.py" and dest.exists():
            # Keep the custom bridge __init__.py
            continue
        if item.is_dir():
            if dest.exists():
                shutil.rmtree(str(dest))
            shutil.copytree(str(item), str(dest))
        else:
            shutil.copy2(str(item), str(dest))
    print(f"Vendored sailify package: {sailify_src} -> {torch_sailify}")


# ---------------------------------------------------------------------------
# SAIL build-environment initialization (pre-build step)
# ---------------------------------------------------------------------------

def init_ppu_build_env(source_root: Path) -> None:
    """Set up PCCL paths and PPU architecture before CMake configure.

    This replaces the former ``Init_ppu_build_env()`` in setup.py.
    In v2.14+ the env vars are forwarded to CMake automatically by
    ``cmake/EnvVarForwarding.cmake`` (USE_*/BUILD_* prefixes) plus
    explicit passthrough for PYTORCH_SAIL_ARCH, PCCL_INCLUDE_DIR, etc.
    """
    use_sail = str2bool(os.getenv("USE_SAIL"))
    if use_sail:
        ppu_sdk = os.environ.get("PPU_SDK", "/usr/local/PPU_SDK")

        # PCCL paths: explicit values take precedence.
        pccl_inc = os.getenv("PCCL_INCLUDE_DIR")
        pccl_lib = os.getenv("PCCL_LIB_DIR")
        if not pccl_inc:
            pccl_inc = os.path.join(ppu_sdk, "include")
            os.environ["PCCL_INCLUDE_DIR"] = pccl_inc
            print(f"Set PCCL_INCLUDE_DIR to: {pccl_inc}")
        else:
            print(f"Using existing PCCL_INCLUDE_DIR: {pccl_inc}")
        if not pccl_lib:
            pccl_lib = os.path.join(ppu_sdk, "lib")
            os.environ["PCCL_LIB_DIR"] = pccl_lib
            print(f"Set PCCL_LIB_DIR to: {pccl_lib}")
        else:
            print(f"Using existing PCCL_LIB_DIR: {pccl_lib}")

        print("PPU build mode: SAIL (USE_SAIL=1), NCCL mapped to PCCL")
    else:
        raise RuntimeError(
            "SAIL-only branch: CUDA-compatible builds are disabled. "
            "Set USE_SAIL=1 to build."
        )

    # Flash/mem-eff attention defaults
    if os.getenv("USE_FLASH_ATTENTION") is None:
        os.environ["USE_FLASH_ATTENTION"] = "True"
    if os.getenv("USE_MEM_EFF_ATTENTION") is None:
        os.environ["USE_MEM_EFF_ATTENTION"] = "True"

    # Architecture validation
    if os.getenv("TORCH_CUDA_ARCH_LIST"):
        raise RuntimeError(
            "SAIL builds do not support TORCH_CUDA_ARCH_LIST. "
            "Unset it and set PYTORCH_SAIL_ARCH to ppu_10, ppu_15, "
            "or a semicolon-separated list of both."
        )
    sail_arch = os.getenv("PYTORCH_SAIL_ARCH")
    if not sail_arch:
        # Default to dual architecture
        os.environ["PYTORCH_SAIL_ARCH"] = "ppu_10;ppu_15"
        print("PYTORCH_SAIL_ARCH not set, defaulting to ppu_10;ppu_15")
    else:
        sail_arches = [
            arch
            for arch in sail_arch.replace(",", ";").replace(" ", ";").split(";")
            if arch
        ]
        unsupported = sorted(set(sail_arches) - {"ppu_10", "ppu_15"})
        if not sail_arches or unsupported:
            invalid = ", ".join(unsupported) or "empty value"
            raise RuntimeError(
                f"Unsupported PYTORCH_SAIL_ARCH value(s): {invalid}. "
                "Supported values are ppu_10 and ppu_15."
            )

    # PCCL is the native collective library in SAIL mode. USE_NCCL remains
    # enabled only as the upstream C++ compile guard; bundled/system NCCL is off.
    if os.getenv("USE_PCCL") is None:
        os.environ["USE_PCCL"] = "True"
    if os.getenv("USE_NCCL") is None:
        os.environ["USE_NCCL"] = "True"
    if os.getenv("USE_SYSTEM_NCCL") is None:
        os.environ["USE_SYSTEM_NCCL"] = "False"
    if os.getenv("BUILD_TEST") is None:
        os.environ["BUILD_TEST"] = "False"

    print("Init ppu build env done ...")


# ---------------------------------------------------------------------------
# Patch orchestration
# ---------------------------------------------------------------------------

def apply_all_patches(source_root: Path) -> None:
    """Apply all third-party patches required for SAIL builds."""
    torch_root = str(source_root)
    patch_root = os.path.join(torch_root, "third_party")
    cutlass_root = os.path.join(patch_root, "actlize")

    if not os.path.exists(cutlass_root):
        raise _missing_submodule_error(torch_root, "third_party/actlize")

    # CUTLASS platform numeric_limits and arch aliases
    cutlass_patch_file = os.path.join(
        patch_root,
        "cutlass_platform_numeric_limits_and_arch_aliases.patch",
    )
    if os.path.isfile(cutlass_patch_file):
        apply_cutlass_sail_patch(cutlass_patch_file, cutlass_root)

    # Flash-attention patches
    use_fa = os.environ.get("USE_FLASH_ATTENTION")
    if use_fa in ("True", "1", "TRUE"):
        fa_root = os.path.join(patch_root, "flash-attention-for-sail")
        if not os.path.exists(fa_root):
            raise _missing_submodule_error(
                torch_root, "third_party/flash-attention-for-sail"
            )
        fa_namespace_patch = os.path.join(
            patch_root, "flash_attention_namespace_config.patch"
        )
        fa_num_splits_patch = os.path.join(
            patch_root, "flash_attention_num_splits_guard.patch"
        )
        apply_fa_patches_with_markers(
            fa_namespace_patch, fa_num_splits_patch, fa_root
        )

    # Kineto hgpti (CUPTI) backend patch
    kineto_root = os.path.join(patch_root, "kineto")
    if os.path.exists(kineto_root):
        kineto_patch_file = os.path.join(
            patch_root, "kineto_libkineto_cmake_sail.patch"
        )
        apply_kineto_sail_patch(kineto_patch_file, kineto_root)

    # Gloo sailify CMake patch (torch-level, only touches submodule at build time)
    apply_gloo_sailify_cmake_patch(source_root)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_full_sailify(
    source_root: Path,
    *,
    output_dir: str | None = None,
    force: bool = False,
) -> bool:
    """Run the full SAIL pre-build pipeline.

    1. Initialize PPU build environment
    2. Run sailify source conversion
    3. Create a project-local ``sailify`` CLI for Gloo's CMake
    4. Apply third-party patches

    Returns True if in-place, False if out-of-place.
    """
    use_sail = str2bool(os.getenv("USE_SAIL"))
    if not use_sail:
        raise RuntimeError("SAIL pre-build requires USE_SAIL=1.")

    # Step 1: env setup
    init_ppu_build_env(source_root)

    # Step 2: source conversion. The conversion function owns marker and
    # fingerprint handling so source changes are never hidden by a stale marker.
    in_place = run_sailify_conversion(
        source_root,
        output_dir=output_dir,
        force=force,
    )

    target = source_root if in_place else Path(os.path.abspath(output_dir))  # type: ignore[arg-type]

    # Step 3: persist the command where CMake can find it in a later process.

    # Step 4: patches (applied to the target tree)
    apply_all_patches(target)

    # Step 5: vendor sailify into torch/sailify/ for wheel packaging
    vendor_sailify_package(target)

    return in_place
