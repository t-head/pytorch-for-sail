import importlib.util
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SDK_VERSION = "2.2.1-184ee6"


class TestGenerateTorchVersion(TestCase):
    def setUp(self):
        super().setUp()
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.root = Path(temp_dir.name)
        (self.root / "tools").mkdir()
        (self.root / "torch").mkdir()
        (self.root / "tools" / "__init__.py").touch()
        for name in ("generate_torch_version.py", "strtobool.py"):
            shutil.copyfile(REPO_ROOT / "tools" / name, self.root / "tools" / name)
        (self.root / "version.txt").write_text("2.14.1\n")
        self.version_path = self.root / "torch" / "version.py"
        self.env = {
            **os.environ,
            "PYTHONPATH": str(self.root),
            "PYTORCH_BUILD_VERSION": "2.14.1",
            "PYTORCH_BUILD_NUMBER": "1",
        }

    def _generate(self, args):
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.generate_torch_version",
                "--is-debug=false",
                *args,
            ],
            cwd=self.root,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=30,
        )

    @parametrize("separator", ["-", "_"])
    @parametrize(
        "markers",
        [
            (None, None),
            ("", None),
            (None, ""),
            ("", ""),
            (SDK_VERSION, None),
            (SDK_VERSION, ""),
            (None, SDK_VERSION),
            ("", SDK_VERSION),
        ],
    )
    def test_marker_matrix(self, separator, markers):
        ppu, sail = markers
        fields = {"cuda": "13.0", "hip": "6.4", "rocm": "6.4.1", "xpu": "2026.0"}
        args = [f"--{name}{separator}version={value}" for name, value in fields.items()]
        args.extend(
            f"--{name}{separator}version={value}"
            for name, value in (("ppu", ppu), ("sail", sail))
            if value is not None
        )
        result = self._generate(args)
        self.assertEqual(result.returncode, 0, result.stderr)
        version = runpy.run_path(str(self.version_path))
        self.assertEqual(version["ppu"], ppu or None)
        self.assertEqual(version["sail"], sail or None)
        for name, value in fields.items():
            self.assertEqual(version[name], value)
        self.assertEqual(version["__version__"], "2.14.1")
        self.assertEqual(version["git_version"], "Unknown")
        self.assertFalse(version["debug"])
        self.assertIn("ppu", version["__all__"])
        self.assertIn("sail", version["__all__"])

    @parametrize("separator", ["-", "_"])
    @parametrize("sail", [SDK_VERSION, "different-version"])
    @parametrize("existing_output", [False, True])
    def test_conflicting_markers_do_not_write(self, separator, sail, existing_output):
        sentinel = "# Existing version metadata must not be truncated.\n"
        if existing_output:
            self.version_path.write_text(sentinel)
        result = self._generate(
            [
                f"--ppu{separator}version={SDK_VERSION}",
                f"--sail{separator}version={sail}",
            ]
        )
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("mutually exclusive", result.stderr)
        if existing_output:
            self.assertEqual(self.version_path.read_text(), sentinel)
        else:
            self.assertFalse(self.version_path.exists())

    def test_marker_switch_clears_previous_value(self):
        for args, expected in (
            ([f"--ppu-version={SDK_VERSION}"], (SDK_VERSION, None)),
            ([f"--sail-version={SDK_VERSION}"], (None, SDK_VERSION)),
            ([], (None, None)),
        ):
            result = self._generate(args)
            self.assertEqual(result.returncode, 0, result.stderr)
            version = runpy.run_path(str(self.version_path))
            self.assertEqual((version["ppu"], version["sail"]), expected)

    @unittest.skipUnless(
        importlib.util.find_spec("spin") and importlib.util.find_spec("click"),
        "spin and click are required for the regeneration entry point",
    )
    @parametrize("ppu", ["", SDK_VERSION])
    @parametrize("sail", ["", SDK_VERSION])
    def test_spin_regeneration_forwards_markers(self, ppu, sail):
        with mock.patch.object(sys, "path", sys.path.copy()):
            commands = runpy.run_path(str(REPO_ROOT / ".spin" / "cmds.py"))
        with (
            mock.patch.dict(os.environ, {"PPU_VERSION": ppu, "SAIL_VERSION": sail}),
            mock.patch("spin.util.run") as run,
        ):
            commands["regenerate_version"].callback()
        expected_args = ["--is-debug=false"]
        if ppu:
            expected_args.append(f"--ppu-version={ppu}")
        if sail:
            expected_args.append(f"--sail-version={sail}")
        run.assert_called_once_with(
            [sys.executable, "-m", "tools.generate_torch_version", *expected_args]
        )
        result = self._generate(run.call_args.args[0][3:])
        if ppu and sail:
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("mutually exclusive", result.stderr)
            self.assertFalse(self.version_path.exists())
        else:
            self.assertEqual(result.returncode, 0, result.stderr)
            version = runpy.run_path(str(self.version_path))
            self.assertEqual(
                (version["ppu"], version["sail"]), (ppu or None, sail or None)
            )

    def _cmake_markers(self, modes, release):
        sdk_root = self.root / "sdk"
        sdk_root.mkdir()
        if release is not None:
            (sdk_root / "release.yaml").write_text(release)
        # Exercise the actual metadata block without configuring or building torch.
        source = (REPO_ROOT / "torch" / "CMakeLists.txt").read_text()
        block = source[source.index('set(PPU_VERSION "")') :]
        block = block[: block.index("# Compute CUDA/SAIL architecture flags")]
        fragment = self.root / "metadata.cmake"
        fragment.write_text(block)
        output = self.root / "markers.txt"
        script = ['set(PPU_VERSION "stale-ppu")', 'set(SAIL_VERSION "stale-sail")']
        for use_ppu, use_sail in modes:
            script.extend(
                [
                    f"set(USE_PPU {use_ppu})",
                    f"set(USE_SAIL {use_sail})",
                    f'include("{fragment.as_posix()}")',
                    f'file(APPEND "{output.as_posix()}" "${{PPU_VERSION}}|${{SAIL_VERSION}}\\n")',
                ]
            )
        script_path = self.root / "check_metadata.cmake"
        script_path.write_text("\n".join(script))
        result = subprocess.run(
            ["cmake", "-P", str(script_path)],
            env={**os.environ, "PPU_SDK": str(sdk_root), "PPU_HOME": ""},
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result, output

    @unittest.skipUnless(shutil.which("cmake"), "cmake is required for metadata tests")
    def test_cmake_metadata_matrix_and_switch(self):
        result, output = self._cmake_markers(
            [("ON", "OFF"), ("ON", "ON"), ("OFF", "ON"), ("OFF", "OFF")],
            f"version: {SDK_VERSION}\n",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            output.read_text().splitlines(),
            [f"{SDK_VERSION}|", f"|{SDK_VERSION}", f"|{SDK_VERSION}", "|"],
        )

    @unittest.skipUnless(shutil.which("cmake"), "cmake is required for metadata tests")
    @parametrize("mode", [("ON", "OFF"), ("ON", "ON"), ("OFF", "OFF")])
    @parametrize("release", [None, "name: SDK\n", "version:   \n"])
    def test_cmake_missing_sdk_version(self, mode, release):
        result, output = self._cmake_markers([mode], release)
        if mode == ("OFF", "OFF"):
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(output.read_text(), "|\n")
        else:
            self.assertNotEqual(result.returncode, 0)
            marker = "SAIL_VERSION" if mode[1] == "ON" else "PPU_VERSION"
            self.assertIn(f"{marker} is empty", result.stderr)
            self.assertFalse(output.exists())


instantiate_parametrized_tests(TestGenerateTorchVersion)

if __name__ == "__main__":
    run_tests()
