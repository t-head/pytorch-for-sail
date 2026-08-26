# Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
import ctypes
import os
import sys
from typing import Any, Optional, Union

import torch

# The _get_device_index has been moved to torch.utils._get_device_index
from torch._utils import _get_device_index as _torch_get_device_index


def _is_sail_mode() -> bool:
    # SAIL (PPU) mode is fixed at build time and recorded in
    # torch.version.sail (non-None only for USE_SAIL builds), mirroring how
    # torch.version.hip marks ROCm builds. Read the marker directly from the
    # generated, dependency-free torch/version.py (which never imports torch);
    # fall back to False when it is absent (e.g. an uncompiled source tree).
    try:
        from torch.version import sail as _sail
    except Exception:
        return False
    return _sail is not None


def _load_ppu_sdk_library(*names: str) -> ctypes.CDLL:
    # `source $PPU_SDK/envsetup.sh ppu` puts the PPU SDK lib directory on
    # LD_LIBRARY_PATH at runtime, so prefer bare-soname resolution via the
    # dynamic loader. Only fall back to an explicit $PPU_SDK path if the loader
    # cannot find the library (e.g. envsetup was not sourced). Never hardcode a
    # fixed install prefix.
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    ppu_sdk = os.environ.get("PPU_SDK")
    if ppu_sdk:
        lib_dir = os.path.join(ppu_sdk, "targets", "x86_64-linux", "lib")
        for name in names:
            path = os.path.join(lib_dir, name)
            if os.path.exists(path):
                return ctypes.CDLL(path)
    raise OSError(f"Could not find any PPU SDK library from {names}")


def _get_hip_runtime_library() -> ctypes.CDLL:
    if sys.platform == "win32":
        lib = ctypes.CDLL(f"amdhip64_{torch.version.hip[0]}.dll")
    else:  # Unix-based systems
        lib = ctypes.CDLL("libamdhip64.so")
    lib.cuGetErrorString = lib.hipGetErrorString  # type: ignore[attr-defined]
    lib.cuModuleLoadData = lib.hipModuleLoadData  # type: ignore[attr-defined]
    lib.cuModuleGetFunction = lib.hipModuleGetFunction  # type: ignore[attr-defined]
    lib.cuLaunchKernel = lib.hipModuleLaunchKernel  # type: ignore[attr-defined]
    lib.cuFuncSetAttribute = lib.hipFuncSetAttribute  # type: ignore[attr-defined]
    return lib


def _get_cuda_library() -> ctypes.CDLL:
    if sys.platform == "win32":
        return ctypes.CDLL("nvcuda.dll")
    else:  # Unix-based systems
        # sail only replaces the Linux .so path; the Windows DLL branch
        # above stays exactly as upstream to minimize rebase conflicts.
        if _is_sail_mode():
            lib = _load_ppu_sdk_library("libhggc.so")
            # Provide aliases for PPU driver functions to match CUDA driver API
            # (same pattern as _get_hip_runtime_library for AMD/HIP)
            _ppu_driver_aliases = {
                'cuGetErrorString':    'hgGetErrorString',
                'cuModuleLoadData':    'hgModuleLoadData',
                'cuModuleGetFunction': 'hgModuleGetFunction',
                'cuLaunchKernel':      'hgLaunchKernel',
                'cuFuncSetAttribute':  'hgFuncSetAttribute',
            }
            for cu_name, hg_name in _ppu_driver_aliases.items():
                if hasattr(lib, hg_name):
                    setattr(lib, cu_name, getattr(lib, hg_name))
            return lib
        return ctypes.CDLL("libcuda.so.1")


# Load GPU driver runtime
def _get_gpu_runtime_library() -> ctypes.CDLL:
    if torch.version.hip:
        return _get_hip_runtime_library()
    else:
        return _get_cuda_library()


# Helper: check CUDA errors
def _check_cuda(result: int) -> None:
    if result == 0:
        return
    err_str = ctypes.c_char_p()
    libcuda = _get_gpu_runtime_library()  # Get reference to CUDA library
    libcuda.cuGetErrorString(result, ctypes.byref(err_str))
    error_message = (
        err_str.value.decode() if err_str.value is not None else "Unknown CUDA error"
    )
    raise RuntimeError(f"CUDA error: {error_message}")


def _get_hiprtc_library() -> ctypes.CDLL:
    if sys.platform == "win32":
        version_str = "".join(["0", torch.version.hip[0], "0", torch.version.hip[2]])
        lib = ctypes.CDLL(f"hiprtc{version_str}.dll")
    else:
        lib = ctypes.CDLL("libhiprtc.so")

    # Provide aliases for HIP RTC functions to match NVRTC API
    lib.nvrtcGetErrorString = lib.hiprtcGetErrorString  # type: ignore[attr-defined]
    lib.nvrtcCreateProgram = lib.hiprtcCreateProgram  # type: ignore[attr-defined]
    lib.nvrtcDestroyProgram = lib.hiprtcDestroyProgram  # type: ignore[attr-defined]
    lib.nvrtcCompileProgram = lib.hiprtcCompileProgram  # type: ignore[attr-defined]
    lib.nvrtcGetPTXSize = lib.hiprtcGetCodeSize  # type: ignore[attr-defined]
    lib.nvrtcGetPTX = lib.hiprtcGetCode  # type: ignore[attr-defined]
    lib.nvrtcGetProgramLogSize = lib.hiprtcGetProgramLogSize  # type: ignore[attr-defined]
    lib.nvrtcGetProgramLog = lib.hiprtcGetProgramLog  # type: ignore[attr-defined]
    lib.nvrtcAddNameExpression = lib.hiprtcAddNameExpression  # type: ignore[attr-defined]
    lib.nvrtcGetLoweredName = lib.hiprtcGetLoweredName  # type: ignore[attr-defined]
    return lib


def _get_nvrtc_library() -> ctypes.CDLL:
    # sail only replaces the Linux .so path; Windows DLLs are untouched.
    if sys.platform != "win32" and _is_sail_mode():
        lib = _load_ppu_sdk_library("libhgrtc.so")
        aliases = {
            "nvrtcGetErrorString": "hgrtcGetErrorString",
            "nvrtcCreateProgram": "hgrtcCreateProgram",
            "nvrtcDestroyProgram": "hgrtcDestroyProgram",
            "nvrtcCompileProgram": "hgrtcCompileProgram",
            "nvrtcGetPTXSize": "hgrtcGetCodeSize",
            "nvrtcGetPTX": "hgrtcGetCode",
            "nvrtcGetProgramLogSize": "hgrtcGetProgramLogSize",
            "nvrtcGetProgramLog": "hgrtcGetProgramLog",
            "nvrtcAddNameExpression": "hgrtcAddNameExpression",
            "nvrtcGetLoweredName": "hgrtcGetLoweredName",
        }
        for nvrtc_name, hgrtc_name in aliases.items():
            if hasattr(lib, hgrtc_name):
                setattr(lib, nvrtc_name, getattr(lib, hgrtc_name))
        return lib

    major_version = int(torch.version.cuda.split(".")[0])  # type: ignore[union-attr]
    if sys.platform == "win32":
        nvrtc_libs = [
            f"nvrtc64_{major_version}0_0.dll",
        ]
    else:
        nvrtc_libs = [
            "libhgrtc.so",
        ]
    for lib_name in nvrtc_libs:
        try:
            return ctypes.CDLL(lib_name)
        except OSError:
            continue
    raise OSError("Could not find any NVRTC library")


def _get_gpu_rtc_library() -> ctypes.CDLL:
    # Since PyTorch already loads the GPU RTC library, we can use the system library
    # which should be compatible with PyTorch's version
    if torch.version.hip:
        return _get_hiprtc_library()
    else:
        return _get_nvrtc_library()


def _get_gpu_rtc_compatible_flags() -> list[str]:
    """
    Get HIPCC/NVCC flags that are compatible with NVRTC compilation.

    Returns:
        List of HIPCC/NVCC flags that can be safely used with NVRTC.
    """
    from torch.utils.cpp_extension import COMMON_HIPCC_FLAGS, COMMON_NVCC_FLAGS

    nvrtc_unsupported_flags = {
        "--expt-relaxed-constexpr",
    }

    # Filter out unsupported flags
    compatible_flags = [
        flag for flag in COMMON_NVCC_FLAGS if flag not in nvrtc_unsupported_flags
    ]

    if torch.version.hip:
        compatible_flags.extend(COMMON_HIPCC_FLAGS)

    return compatible_flags


def _nvrtc_compile(
    kernel_source: str,
    kernel_name: str,
    compute_capability: Optional[str] = None,
    cuda_include_dirs: Optional[list] = None,
    nvcc_options: Optional[list] = None,
    auto_pch: bool = False,
) -> tuple[bytes, str]:
    """
    Compiles a CUDA kernel using NVRTC and returns the PTX code.

    Args:
        kernel_source (str): The CUDA kernel source code as a string
        kernel_name (str): The name of the kernel function to compile
        compute_capability (str, None): The compute capability to target (e.g., "86").
                                           If None, will detect from current device.
        cuda_include_dirs (list, None): List of directories containing CUDA headers
        nvcc_options (list, None): Additional options to pass to NVRTC
        auto_pch (bool): Enable automatic precompiled headers (CUDA 12.8+)

    Returns:
        Tuple[bytes, str]: The compiled PTX code and mangled kernel name
    """
    # Ensure CUDA is initialized
    import torch.cuda

    # Load NVRTC library
    libnvrtc = _get_gpu_rtc_library()

    # NVRTC constants
    NVRTC_SUCCESS = 0

    # Helper: check NVRTC errors
    def check_nvrtc(result: int) -> None:
        if result != NVRTC_SUCCESS:
            err_str = ctypes.c_char_p()
            libnvrtc.nvrtcGetErrorString(result, ctypes.byref(err_str))
            error_message = (
                err_str.value.decode()
                if err_str.value is not None
                else "Unknown CUDA error"
            )
            raise RuntimeError(f"CUDA error: {error_message}")

    # Convert source to bytes
    source_bytes = kernel_source.encode("utf-8")

    # Get compute capability if not provided
    if compute_capability is None:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        if torch.version.hip:
            compute_capability = f"{props.gcnArchName}"
        else:
            compute_capability = f"{props.major}{props.minor}"

    # Prepare compilation options
    options = []
    if torch.version.hip:
        options.append(f"--offload-arch={compute_capability}".encode())
    elif _is_sail_mode():
        # PPU: hgrtc expects --gpu-architecture=ppu_XX, not sm_XX
        from torch.utils.cpp_extension import _cuda_arch_to_ppu
        # compute_capability is like "80" or "90a"; strip 'a' suffix
        # to parse major/minor, _cuda_arch_to_ppu also handles it defensively.
        cc = compute_capability.rstrip('a')
        if len(cc) >= 2:
            major, minor = int(cc[:-1]), int(cc[-1])
        else:
            major, minor = 8, 0  # default fallback
        ppu_arch = _cuda_arch_to_ppu(f"{major}.{minor}")
        options.append(f"--gpu-architecture={ppu_arch}".encode())
    else:
        options.append(f"--gpu-architecture=sm_{compute_capability}".encode())

    # Auto-detect and add CUDA include paths
    from torch.utils.cpp_extension import include_paths

    cuda_include_paths = include_paths("cuda")
    for cuda_path in cuda_include_paths:
        options.append(f"-I{cuda_path}".encode())

    # Add custom include directories
    if cuda_include_dirs:
        for directory in cuda_include_dirs:
            options.append(f"-I{directory}".encode())

    # Enable automatic precompiled headers (CUDA 12.8+)
    if auto_pch:
        assert str(torch.version.cuda) >= "12.8", "PCH requires CUDA 12.8+"
        if nvcc_options is None:
            nvcc_options = []
        nvcc_options.append("--pch")

    # Add custom NVCC options (sailify for PPU to convert arch flags)
    if nvcc_options:
        if _is_sail_mode():
            from torch.utils.cpp_extension import _sailify_nvcc_flags
            nvcc_options = _sailify_nvcc_flags(nvcc_options)
        for option in nvcc_options:
            options.append(option.encode("utf-8"))

    nvrtc_compatible_flags = _get_gpu_rtc_compatible_flags()
    options.extend([flag.encode("utf-8") for flag in nvrtc_compatible_flags])

    # Convert options to C array
    num_options = len(options)
    options_array = (ctypes.c_char_p * num_options)(*options)

    # Create program
    prog = ctypes.c_void_p()
    check_nvrtc(
        libnvrtc.nvrtcCreateProgram(
            ctypes.byref(prog),
            source_bytes,
            f"{kernel_name}.cu".encode(),
            0,
            None,
            None,
        )
    )

    # Add kernel name, which can be a template expression
    c_kernel_name = kernel_name.encode("utf-8")
    check_nvrtc(libnvrtc.nvrtcAddNameExpression(prog, c_kernel_name))

    # Compile program
    res = libnvrtc.nvrtcCompileProgram(prog, num_options, options_array)

    # Handle compilation errors
    if res != NVRTC_SUCCESS:
        # Get log
        log_size = ctypes.c_size_t()
        libnvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(log_size))
        log = ctypes.create_string_buffer(log_size.value)
        libnvrtc.nvrtcGetProgramLog(prog, log)
        raise RuntimeError(f"Kernel compilation failed:\n{log.value.decode()}")

    # Get PTX
    ptx_size = ctypes.c_size_t()
    check_nvrtc(libnvrtc.nvrtcGetPTXSize(prog, ctypes.byref(ptx_size)))
    ptx = ctypes.create_string_buffer(ptx_size.value)
    check_nvrtc(libnvrtc.nvrtcGetPTX(prog, ptx))

    # Get mangled name
    c_mangled_name = ctypes.c_char_p()
    check_nvrtc(
        libnvrtc.nvrtcGetLoweredName(prog, c_kernel_name, ctypes.byref(c_mangled_name))
    )
    if c_mangled_name.value is not None:
        mangled_name = c_mangled_name.value.decode()  # make a copy
    else:
        mangled_name = ""

    libnvrtc.nvrtcDestroyProgram(ctypes.byref(prog))

    # For HIP, hipRTC generates raw CO binaries instead of PTX,
    # and for some reason, ".value" causes the string to be truncated,
    # likely due to the presence of '\0' in the string. So we use .raw instead.
    # Likewise, in SAIL mode hgrtcGetCode returns ELF binaries
    # containing embedded '\0' bytes; .value would truncate them as well.
    ptx_bytes = ptx.raw if (torch.version.hip or _is_sail_mode()) else ptx.value
    return ptx_bytes, mangled_name


class _CudaModule:
    def __init__(self, module: ctypes.c_void_p) -> None:
        self._module = module
        self._kernels: dict[str, _CudaKernel] = {}

    def __getattr__(self, name: str) -> "_CudaKernel":
        if name in self._kernels:
            return self._kernels[name]

        # Import the CUDA library inside the method
        # pyrefly: ignore [missing-module-attribute]
        from torch.cuda._utils import _get_gpu_runtime_library

        libcuda = _get_gpu_runtime_library()

        func = ctypes.c_void_p()
        try:
            _check_cuda(
                libcuda.cuModuleGetFunction(
                    ctypes.byref(func), self._module, name.encode("utf-8")
                )
            )
            kernel = _CudaKernel(func, self._module)
            self._kernels[name] = kernel
            return kernel

        except RuntimeError as err:
            raise AttributeError(f"No kernel named '{name}' in this module") from err


class _CudaKernel:
    """
    Represents a compiled CUDA kernel that can be called with PyTorch tensors.
    """

    def __init__(self, func: ctypes.c_void_p, module: ctypes.c_void_p) -> None:
        self.func = func
        self.module = module
        self._max_shared_mem_bytes = 0

    def __call__(
        self,
        grid: tuple[int, int, int] = (1, 1, 1),
        block: tuple[int, int, int] = (1, 1, 1),
        args: Optional[list] = None,
        shared_mem: int = 0,
        stream: Optional[Any] = None,
    ) -> None:
        """
        Call the compiled CUDA kernel

        Args:
            grid (tuple): Grid dimensions (grid_x, grid_y, grid_z)
            block (tuple): Block dimensions (block_x, block_y, block_z)
            args (list): List of arguments to pass to the kernel.
                         PyTorch tensor arguments will be automatically converted to pointers.
            shared_mem (int): Shared memory size in bytes
            stream (torch.cuda.Stream): CUDA stream to use. If None, uses current stream.
        """
        import torch

        libcuda = torch.cuda._utils._get_gpu_runtime_library()

        if not args:
            args = []

        # Process arguments and convert tensors to pointers
        processed_args: list[ctypes.c_void_p] = []
        c_args = []

        for arg in args:
            if isinstance(arg, torch.Tensor):
                if not arg.is_cuda and not (arg.is_cpu and arg.is_pinned()):
                    raise ValueError(
                        "All tensor arguments must be CUDA tensors or pinned CPU tensors"
                    )
                # Get pointer to tensor data
                ptr = ctypes.c_void_p(arg.data_ptr())
                processed_args.append(ptr)
                c_args.append(ctypes.byref(ptr))
            elif isinstance(arg, int):
                # Convert integers to C int
                c_int = ctypes.c_int(arg)
                # Store the C int for reference keeping, not in processed_args
                c_args.append(ctypes.byref(c_int))
            elif isinstance(arg, float):
                # Python floats are doubles - use double by default
                c_double = ctypes.c_double(arg)
                # Store the C double for reference keeping, not in processed_args
                c_args.append(ctypes.byref(c_double))
            else:
                raise TypeError(f"Unsupported argument type: {type(arg)}")

        # Convert to array of void pointers
        c_args_array = (ctypes.c_void_p * len(c_args))()
        for i, arg in enumerate(c_args):
            c_args_array[i] = ctypes.cast(arg, ctypes.c_void_p)

        # Get the stream
        if stream is None:
            # Defer import to avoid circular imports
            import torch.cuda

            stream = torch.cuda.current_stream()

        # Check if kernel requires large shared memory but hasn't been configured
        if shared_mem >= 48 * 1024 and (
            self._max_shared_mem_bytes == 0 or shared_mem > self._max_shared_mem_bytes
        ):
            configured_msg = (
                "not configured"
                if self._max_shared_mem_bytes == 0
                else f"only {self._max_shared_mem_bytes} bytes configured"
            )
            raise RuntimeError(
                f"Kernel requires {shared_mem} bytes of shared memory (>= 48KB), "
                f"but {configured_msg}. "
                "Call kernel.set_shared_memory_config(shared_mem) after compilation "
                "and before launching the kernel."
            )

        _check_cuda(
            libcuda.cuLaunchKernel(
                self.func,
                grid[0],
                grid[1],
                grid[2],
                block[0],
                block[1],
                block[2],
                shared_mem,
                stream._as_parameter_,
                c_args_array,
                None,
            )
        )

    def set_shared_memory_config(self, shared_mem_bytes: int) -> None:
        if shared_mem_bytes < 48 * 1024:
            # No configuration needed for <= 48KB, just update the value
            self._max_shared_mem_bytes = shared_mem_bytes
            return

        libcuda = _get_gpu_runtime_library()

        # Get device properties to validate against limits
        device_props = torch.cuda.get_device_properties()
        # HIP doesn't have shared_memory_per_block_optin in device properties, so we hard-code it here
        if torch.version.hip:
            # navi, CDNA1-CDNA3 allows a max of 64KB shared memory
            # CDNA4 allows a max of 160KB shared memory
            max_shared_mem = (
                65536 if device_props.gcnArchName != "gfx950" else 160 * 1024
            )
        else:
            max_shared_mem = getattr(
                device_props, "shared_memory_per_block_optin", 49152
            )

        if shared_mem_bytes > max_shared_mem:
            raise RuntimeError(
                f"Requested shared memory ({shared_mem_bytes} bytes) exceeds "
                f"device limit ({max_shared_mem} bytes). "
                "Consider reducing block size or shared memory usage."
            )

        # Set the function attribute once
        # https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html
        cudaFuncAttributeMaxDynamicSharedMemorySize = 8
        _check_cuda(
            libcuda.cuFuncSetAttribute(
                self.func,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                shared_mem_bytes,
            )
        )

        self._max_shared_mem_bytes = shared_mem_bytes


def _cuda_load_module(
    ptx: Union[str, bytes], kernel_names: Optional[list[str]] = None
) -> Union[_CudaModule, dict[str, "_CudaKernel"]]:
    """
    Loads a CUDA module from PTX code and returns a module object that can access kernels.

    Args:
        ptx (bytes or str): The PTX code to load
        kernel_names (list, optional): List of kernel names to extract from the module.
                                      If None, will return a module object with __getattr__.

    Returns:
        object: If kernel_names is None, returns a module object with __getattr__ to access kernels.
               If kernel_names is provided, returns a dict mapping kernel names to _CudaKernel objects.
    """
    # Ensure CUDA is initialized
    import torch.cuda

    # Load CUDA driver library
    libcuda = _get_gpu_runtime_library()

    # Convert PTX to bytes if it's a string
    if isinstance(ptx, str):
        ptx = ptx.encode("utf-8")

    # Load PTX module
    module = ctypes.c_void_p()
    # Get the current stream without directly importing torch.cuda at module level
    stream = torch.cuda.current_stream()
    with stream:
        _check_cuda(libcuda.cuModuleLoadData(ctypes.byref(module), ptx))

    if not kernel_names:
        return _CudaModule(module)

    # Return specific kernels
    kernels = {}
    for name in kernel_names:
        func = ctypes.c_void_p()
        _check_cuda(
            libcuda.cuModuleGetFunction(
                ctypes.byref(func), module, name.encode("utf-8")
            )
        )
        kernels[name] = _CudaKernel(func, module)
    return kernels


def _get_device_index(
    device: Any, optional: bool = False, allow_cpu: bool = False
) -> int:
    r"""Get the device index from :attr:`device`, which can be a torch.device object, a Python integer, or ``None``.

    If :attr:`device` is a torch.device object, returns the device index if it
    is a CUDA device. Note that for a CUDA device without a specified index,
    i.e., ``torch.device('cuda')``, this will return the current default CUDA
    device if :attr:`optional` is ``True``. If :attr:`allow_cpu` is ``True``,
    CPU devices will be accepted and ``-1`` will be returned in this case.

    If :attr:`device` is a Python integer, it is returned as is.

    If :attr:`device` is ``None``, this will return the current default CUDA
    device if :attr:`optional` is ``True``.
    """
    if isinstance(device, int):
        return device
    if isinstance(device, str):
        device = torch.device(device)
    if isinstance(device, torch.device):
        if allow_cpu:
            if device.type not in ["cuda", "cpu"]:
                raise ValueError(f"Expected a cuda or cpu device, but got: {device}")
        elif device.type != "cuda":
            raise ValueError(f"Expected a cuda device, but got: {device}")
    if not torch.jit.is_scripting():
        if isinstance(device, torch.cuda.device):
            return device.idx
    return _torch_get_device_index(device, optional, allow_cpu)
