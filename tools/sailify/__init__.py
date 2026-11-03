# Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
"""SAIL pre-build conversion and patching utilities.

This package extracts the sailify source-tree transformation and third-party
patch application from the former ``setup.py`` into a standalone tool that
runs *before* the PEP 517 build (scikit-build-core).

Usage:
    python -m tools.sailify [--output-dir DIR] [--force]
    spin sailify [--output-dir DIR] [--force]
"""
