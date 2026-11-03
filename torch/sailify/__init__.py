# Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
"""Vendored sailify package -- runtime bridge.

This package provides ``sailify_python`` and ``nvcc_hgcc_options`` for
``torch.utils.cpp_extension`` (CUDAExtension and JIT compilation) in SAIL
(PPU) builds.

In a **source-tree** development setup, the modules are imported from the
``third_party/sailify/sailify/`` submodule; this ``__init__.py`` makes
them visible under ``torch.sailify`` by adding the submodule to
``sys.path``.

In an **installed wheel**, ``spin sailify`` (the pre-build step) copies the
sailify modules into ``torch/sailify/`` before ``pip install .`` so that
they are physically present in the wheel.
"""
from __future__ import annotations

import importlib
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------
# Source-tree development bridge: make the submodule importable as torch.sailify.*
# --------------------------------------------------------------------------
_vendor = os.path.join(
    os.path.dirname(os.path.dirname(_here)),  # ..../pytorch-2-14/
    "third_party", "sailify",
)

_has_vendored_modules = os.path.isfile(os.path.join(_here, "sailify_python.py"))

if not _has_vendored_modules and os.path.isdir(_vendor):
    # Running from a source tree without pre-build vendoring: splice the
    # submodule's package into this namespace so ``from torch.sailify.X``
    # resolves to ``third_party/sailify/sailify/X``.
    _sailify_pkg = os.path.join(_vendor, "sailify")
    if os.path.isdir(_sailify_pkg):
        if _vendor not in sys.path:
            sys.path.insert(0, _vendor)
        # Import the submodule's package and re-export its path so that
        # sub-modules like sailify_python are discoverable.
        try:
            _sub = importlib.import_module("sailify")
            __path__ = _sub.__path__  # type: ignore[attr-defined]
        except ImportError:
            pass
