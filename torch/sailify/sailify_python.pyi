# Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
# Static interfaces for modules provided by the source bridge or wheel vendoring.

class SailifyResult:
    ppuified_path: str
    status: str

class Trie:
    def export_to_regex(self) -> str: ...

_PPU_MAP: dict[str, str]
_PPU_TRIE: Trie

def write_compat_wrapper_header(
    compat_dir: str,
    version_config: dict[str, str | int] | None = ...,
    ppu_sdk_version: tuple[int, ...] | None = ...,
    runtime_api_version: int | None = ...,
    stub_probe: bool = ...,
) -> None: ...
def sailify_extra_files_recursive(
    output_directory: str,
    extra_files: list[str],
    header_include_dirs: list[str] | None = ...,
    show_detailed: bool = ...,
    backup: bool = ...,
    install_ppu_compat: bool = ...,
    inject_compat_include: bool = ...,
    inject_compat_include_ignores: list[str] | None = ...,
    stub_probe: bool = ...,
) -> dict[str, SailifyResult]: ...
