# Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
# ---[ hg_native
#
# PPU SAIL mode SDK wiring, built on the standard cmake-hgcc HG language
# package (third_party/cmake-hgcc submodule).
#
# This file is the SAIL counterpart of cmake/public/cuda.cmake. It is
# included from cmake/Dependencies.cmake when USE_SAIL (which implies USE_PPU).
#
# Design:
#   * Use the standard cmake-hgcc HG language (`enable_language(HG)`), NOT CUDA.
#     The HG language modules live in third_party/cmake-hgcc/cmake/Modules and
#     are discovered via CMAKE_MODULE_PATH set below.
#   * Create HGGC:: namespace imported targets via FindHGGCToolkit.cmake.
#   * Define compatibility torch::/caffe2::/CUDA:: targets that redirect to
#     HGGC:: targets, so downstream target_link_libraries(... torch::cudart ...)
#     keeps working unchanged (this glue layer is torch-specific and NOT part
#     of the upstream cmake-hgcc package - it is preserved here).
#   * Provide ppu_select_hgcc_arch_flags()/cuda_select_nvcc_arch_flags() (also
#     torch-specific, preserved here) consumed by cmake/public/utils.cmake.

# ---------------------------------------------------------------------------
# 0. Make the cmake-hgcc HG language modules discoverable.
#    Two layouts have to work, because this file is used both while building
#    torch and by downstream projects that only have the installed tree:
#      in-tree    <root>/cmake/public/hg_native.cmake
#                 -> <root>/third_party/cmake-hgcc/cmake/Modules
#      installed  <prefix>/share/cmake/Caffe2/public/hg_native.cmake
#                 -> <prefix>/share/cmake/Caffe2/cmake-hgcc/Modules
#                 (see the install(DIRECTORY ...) rule in the top-level
#                  CMakeLists.txt, which flattens cmake/Modules to Modules/)
# ---------------------------------------------------------------------------
set(_HG_MODULE_CANDIDATES
  "${CMAKE_CURRENT_LIST_DIR}/../../third_party/cmake-hgcc/cmake/Modules"
  "${CMAKE_CURRENT_LIST_DIR}/../cmake-hgcc/Modules")
set(_HG_CMAKE_HGCC_MODULES "")
foreach(_hg_cand IN LISTS _HG_MODULE_CANDIDATES)
  get_filename_component(_hg_cand_abs "${_hg_cand}" ABSOLUTE)
  if(IS_DIRECTORY "${_hg_cand_abs}")
    set(_HG_CMAKE_HGCC_MODULES "${_hg_cand_abs}")
    break()
  endif()
endforeach()
if(NOT _HG_CMAKE_HGCC_MODULES)
  message(FATAL_ERROR
    "cmake-hgcc HG language modules not found. Looked in:\n"
    "  ${_HG_MODULE_CANDIDATES}\n"
    "In a source tree, initialize the submodule: "
    "git submodule update --init third_party/cmake-hgcc")
endif()
if(NOT "${_HG_CMAKE_HGCC_MODULES}" IN_LIST CMAKE_MODULE_PATH)
  list(PREPEND CMAKE_MODULE_PATH "${_HG_CMAKE_HGCC_MODULES}")
endif()

# Third-party subprojects locate FindHGGCToolkit.cmake through CMAKE_PREFIX_PATH
# rather than CMAKE_MODULE_PATH: gloo's cmake/Hggc.cmake does
#   find_path(HGCC_MODULE_DIR NAMES FindHGGCToolkit.cmake PATH_SUFFIXES cmake/Modules)
# which only searches prefixes (upstream cmake-hgcc expects its envsetup.sh to
# export CMAKE_PREFIX_PATH). Since cmake-hgcc is a submodule here, add its root
# so such subprojects resolve the module without any external setup.
#
# Only the source-tree layout is handled: the installed layout flattens
# cmake/Modules to Modules/, so the cmake/Modules suffix would not match anyway,
# and no third-party subproject gets built against an installed torch.
get_filename_component(_HG_CMAKE_HGCC_ROOT
  "${CMAKE_CURRENT_LIST_DIR}/../../third_party/cmake-hgcc" ABSOLUTE)
if(IS_DIRECTORY "${_HG_CMAKE_HGCC_ROOT}"
   AND NOT "${_HG_CMAKE_HGCC_ROOT}" IN_LIST CMAKE_PREFIX_PATH)
  list(PREPEND CMAKE_PREFIX_PATH "${_HG_CMAKE_HGCC_ROOT}")
endif()

# ---------------------------------------------------------------------------
# 0a. Architecture flag helpers (torch-specific; preserved from the retired
#     PPUUtilities.cmake). cmake/public/utils.cmake's
#     torch_ppu_get_nvcc_gencode_flag() calls ppu_select_hgcc_arch_flags(),
#     and third-party FindCUDA-style code may call cuda_select_nvcc_arch_flags().
#     HGGCUtilities does not provide these, so we keep them here.
# ---------------------------------------------------------------------------

# Validate native PPU architecture names accepted by hgcc.
macro(_ppu_validate_arch _arch_var)
  set(_arch_val "${${_arch_var}}")
  if(NOT _arch_val STREQUAL "ppu_10" AND NOT _arch_val STREQUAL "ppu_15")
    message(FATAL_ERROR
      "PPU: unsupported PYTORCH_SAIL_ARCH value '${_arch_val}'. "
      "Supported values are ppu_10 and ppu_15.")
  endif()
endmacro()

# ppu_select_hgcc_arch_flags(<output_variable> [extra_arch ...])
function(ppu_select_hgcc_arch_flags _outvar)
  if((DEFINED TORCH_CUDA_ARCH_LIST AND NOT TORCH_CUDA_ARCH_LIST STREQUAL "")
     OR (DEFINED ENV{TORCH_CUDA_ARCH_LIST} AND NOT "$ENV{TORCH_CUDA_ARCH_LIST}" STREQUAL ""))
    message(FATAL_ERROR
      "PPU: SAIL builds do not support TORCH_CUDA_ARCH_LIST. "
      "Unset it and configure PYTORCH_SAIL_ARCH with ppu_10, ppu_15, "
      "or a semicolon-separated list of both.")
  endif()

  if(DEFINED PYTORCH_SAIL_ARCH AND NOT PYTORCH_SAIL_ARCH STREQUAL "")
    set(_arch_list "${PYTORCH_SAIL_ARCH}")
  elseif(DEFINED ENV{PYTORCH_SAIL_ARCH} AND NOT "$ENV{PYTORCH_SAIL_ARCH}" STREQUAL "")
    set(_arch_list "$ENV{PYTORCH_SAIL_ARCH}")
  else()
    message(FATAL_ERROR
      "PPU: SAIL builds require PYTORCH_SAIL_ARCH. Set it to ppu_10, "
      "ppu_15, or a semicolon-separated list of both.")
  endif()

  string(REPLACE " " ";" _arch_list "${_arch_list}")
  set(_arch_values "")
  foreach(_arch IN LISTS _arch_list)
    if(_arch)
      _ppu_validate_arch(_arch)
      list(APPEND _arch_values "${_arch}")
    endif()
  endforeach()
  if(NOT _arch_values)
    message(FATAL_ERROR
      "PPU: PYTORCH_SAIL_ARCH must include ppu_10, ppu_15, or both.")
  endif()

  foreach(_extra_arch IN LISTS ARGN)
    if(_extra_arch)
      _ppu_validate_arch(_extra_arch)
      list(APPEND _arch_values "${_extra_arch}")
    endif()
  endforeach()

  list(REMOVE_DUPLICATES _arch_values)
  set(_arch_flags "")
  foreach(_arch IN LISTS _arch_values)
    list(APPEND _arch_flags "-arch" "${_arch}")
  endforeach()
  set(${_outvar} "${_arch_flags}" PARENT_SCOPE)
endfunction()

# cuda_select_nvcc_arch_flags stub -> redirect to ppu_select_hgcc_arch_flags.
if(NOT COMMAND cuda_select_nvcc_arch_flags)
  function(cuda_select_nvcc_arch_flags _outvar)
    ppu_select_hgcc_arch_flags(${_outvar} ${ARGN})
    set(${_outvar} "${${_outvar}}" PARENT_SCOPE)
  endfunction()
endif()

# -- Pre-set pthread variables for cross-compilation --
# hgcc is a PPU device compiler and cannot compile host-side C code that
# FindThreads uses for pthread detection.
set(CMAKE_THREAD_LIBS_INIT "-lpthread" CACHE STRING "Thread library")
set(CMAKE_HAVE_LIBC_PTHREAD ON CACHE BOOL "pthread in libc")
set(CMAKE_USE_PTHREADS_INIT ON CACHE BOOL "Use pthreads")
set(THREADS_FOUND TRUE CACHE BOOL "Threads found")
set(THREADS_PREFER_PTHREAD_FLAG OFF CACHE BOOL "Prefer pthread flag")

# Poor man's include guard (mirrors cuda.cmake's `if(TARGET torch::cudart)`).
if(TARGET torch::hggcrt)
  return()
endif()

# ---------------------------------------------------------------------------
# 1. HG compiler selection + enable_language(HG)
# ---------------------------------------------------------------------------
# Auto-locate hgcc if not provided (CMakeDetermineHGCompiler also does this,
# but pre-setting keeps parity with prior behaviour and honours PPU_SDK).
if(NOT CMAKE_HG_COMPILER AND NOT DEFINED ENV{HGCC})
  if(DEFINED PPU_SDK_ROOT)
    set(_hg_sdk_bin "${PPU_SDK_ROOT}/bin")
  elseif(DEFINED ENV{PPU_SDK})
    set(_hg_sdk_bin "$ENV{PPU_SDK}/bin")
  elseif(DEFINED ENV{PPU_HOME})
    set(_hg_sdk_bin "$ENV{PPU_HOME}/bin")
  else()
    set(_hg_sdk_bin "/usr/local/PPU_SDK/bin")
  endif()
  find_program(HG_HGCC_COMPILER NAMES hgcc PATHS "${_hg_sdk_bin}" NO_DEFAULT_PATH)
  if(HG_HGCC_COMPILER)
    set(CMAKE_HG_COMPILER "${HG_HGCC_COMPILER}"
        CACHE FILEPATH "PPU hgcc device compiler" FORCE)
  endif()
endif()

# Pass host compiler (mirrors CMAKE_CUDA_HOST_COMPILER)
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang|GNU")
  set(CMAKE_HG_HOST_COMPILER "${CMAKE_CXX_COMPILER}" CACHE FILEPATH "" FORCE)
endif()

# Compile existing .cu/.cuh source files (collected by aten/caffe2) as HG.
# Must be set BEFORE enable_language(HG). The vendored CMakeHGInformation.cmake
# registers `hg cu cuh` when this is ON.
set(CMAKE_HG_ENABLE_CU_EXTENSION ON CACHE BOOL "Treat .cu/.cuh files as HG sources" FORCE)

enable_language(HG)

if("X${CMAKE_HG_STANDARD}" STREQUAL "X")
  set(CMAKE_HG_STANDARD ${CMAKE_CXX_STANDARD})
endif()
set(CMAKE_HG_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# 2. find_package(HGGCToolkit) - creates HGGC:: namespace targets
# ---------------------------------------------------------------------------
cmake_policy(PUSH)
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.12.0)
  cmake_policy(SET CMP0074 NEW)
endif()

find_package(HGGCToolkit REQUIRED)

cmake_policy(POP)

# Set convenience variables for downstream (mirrors FindCUDA variables)
set(PPU_TOOLKIT_ROOT_DIR "${HGGCToolkit_ROOT}" CACHE PATH "" FORCE)
set(PPU_INCLUDE_DIRS     "${HGGCToolkit_INCLUDE_DIR}" CACHE PATH "" FORCE)
set(PPU_NVCC_EXECUTABLE  "${CMAKE_HG_COMPILER}" CACHE FILEPATH "" FORCE)
set(HGGC_VERSION         "${HGGCToolkit_VERSION}" CACHE STRING "" FORCE)
set(PPU_VERSION_STRING   "${HGGCToolkit_VERSION}" CACHE STRING "" FORCE)

# CUDA-compat version, decoupled from the hgcc release version. cmake-hgcc's
# FindHGGCToolkit reports the hgcc version (e.g. 2.1.1), but downstream CUDA
# version gates / generate_torch_version.py / tensorpipe expect the CUDA API
# level the PPU SDK emulates (CUDA 13.0). Overridable via cache.
set(HGGC_CUDA_COMPAT_VERSION "13.0" CACHE STRING "CUDA API level emulated by the PPU SDK")
set(CUDA_VERSION         "${HGGC_CUDA_COMPAT_VERSION}" CACHE STRING "" FORCE)

# CUDA-compat variables needed by downstream version checks
# (CMakeLists.txt checks CMAKE_CUDA_COMPILER_VERSION, torch/CMakeLists.txt uses
# CUDAToolkit_VERSION_MAJOR/MINOR for generate_torch_version.py)
set(CMAKE_CUDA_COMPILER_VERSION "${HGGC_CUDA_COMPAT_VERSION}" CACHE STRING "PPU mapped as CUDA compiler version" FORCE)
if(HGGC_CUDA_COMPAT_VERSION MATCHES "([0-9]+)\\.([0-9]+)")
  set(CUDAToolkit_VERSION_MAJOR "${CMAKE_MATCH_1}" CACHE STRING "" FORCE)
  set(CUDAToolkit_VERSION_MINOR "${CMAKE_MATCH_2}" CACHE STRING "" FORCE)
else()
  set(CUDAToolkit_VERSION_MAJOR "13" CACHE STRING "" FORCE)
  set(CUDAToolkit_VERSION_MINOR "0" CACHE STRING "" FORCE)
endif()
# Also set CUDA_TOOLKIT_ROOT_DIR and CUDA_INCLUDE_DIRS (some code references these)
set(CUDA_TOOLKIT_ROOT_DIR "${HGGCToolkit_ROOT}" CACHE PATH "" FORCE)
set(CUDA_INCLUDE_DIRS     "${HGGCToolkit_INCLUDE_DIR}" CACHE PATH "" FORCE)
set(CUDA_NVCC_EXECUTABLE  "${CMAKE_HG_COMPILER}" CACHE FILEPATH "" FORCE)

message(STATUS "PyTorch: PPU (HG) detected: ${HGGC_VERSION}")
message(STATUS "PyTorch: PPU hgcc is: ${PPU_NVCC_EXECUTABLE}")
message(STATUS "PyTorch: PPU toolkit directory: ${PPU_TOOLKIT_ROOT_DIR}")

# ---------------------------------------------------------------------------
# 2b. CUDA compatibility wrapper injection
#     Keep generated/source headers untouched. Host C++ files under c10/cuda
#     and device files compiled by the HG language both need COMPATIBLE_VERSION
#     and compatible* APIs provided by .ppu_compat/cuda_compat_wrapper.h.
# ---------------------------------------------------------------------------
# sailify version naming differs: v1.0.3 emits compatible_wrapper.h; the older
# .tpl-based sailify emits cuda_compat_wrapper.h. Accept either.
#
# The directory itself has two locations for the same reason the HG modules do:
# while building torch, sailify writes .ppu_compat into the source root, which
# is CMAKE_CURRENT_SOURCE_DIR here. A downstream find_package(Torch) instead
# includes this file from <prefix>/share/cmake/Caffe2/public/, where
# CMAKE_CURRENT_SOURCE_DIR is the *consumer's* directory and has no
# .ppu_compat -- fall back to the copy shipped at the install prefix.
get_filename_component(_PPU_COMPAT_INSTALLED
  "${CMAKE_CURRENT_LIST_DIR}/../../../../.ppu_compat" ABSOLUTE)
set(_PPU_COMPAT_DIRS
  "${CMAKE_CURRENT_SOURCE_DIR}/.ppu_compat"
  "${_PPU_COMPAT_INSTALLED}")
set(_PPU_CUDA_COMPAT_WRAPPER "")
foreach(_dir IN LISTS _PPU_COMPAT_DIRS)
  foreach(_cand compatible_wrapper.h cuda_compat_wrapper.h)
    if(EXISTS "${_dir}/${_cand}")
      set(_PPU_CUDA_COMPAT_WRAPPER "${_dir}/${_cand}")
      break()
    endif()
  endforeach()
  if(_PPU_CUDA_COMPAT_WRAPPER)
    break()
  endif()
endforeach()
if(_PPU_CUDA_COMPAT_WRAPPER)
  # Host-side .cpp compilation (for example c10/cuda/*.cpp) needs the wrapper
  # even though those files are compiled by the host C++ compiler, not hgcc.
  add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-include>
    $<$<COMPILE_LANGUAGE:CXX>:${_PPU_CUDA_COMPAT_WRAPPER}>)

  # Device-side HG compilation also needs the same wrapper. De-duplicate here
  # because build scripts may already seed CMAKE_HG_FLAGS.
  string(FIND "${CMAKE_HG_FLAGS}" "${_PPU_CUDA_COMPAT_WRAPPER}" _ppu_compat_wrapper_in_hg_flags)
  if(_ppu_compat_wrapper_in_hg_flags EQUAL -1)
    string(APPEND CMAKE_HG_FLAGS " -include ${_PPU_CUDA_COMPAT_WRAPPER}")
  endif()
  message(STATUS "PPU SAIL compat wrapper: ${_PPU_CUDA_COMPAT_WRAPPER}")
else()
  message(WARNING "PPU SAIL compat wrapper not found. Looked for "
    "compatible_wrapper.h / cuda_compat_wrapper.h under: ${_PPU_COMPAT_DIRS}")
endif()

# Add .ppu_compat/ as a system include directory so that PPU compat headers
# (hgperf_host.h, hgperf_common.h, ppu_sdk_fixups.h, etc.) are discoverable
# by all targets, including third-party (kineto, gloo, etc.). Same two
# locations as the wrapper above: source root while building torch, install
# prefix when a downstream find_package(Torch) pulls this file in.
set(_PPU_COMPAT_DIR "")
foreach(_dir IN LISTS _PPU_COMPAT_DIRS)
  if(IS_DIRECTORY "${_dir}")
    set(_PPU_COMPAT_DIR "${_dir}")
    break()
  endif()
endforeach()
if(_PPU_COMPAT_DIR)
  include_directories(SYSTEM "${_PPU_COMPAT_DIR}")
  message(STATUS "PPU SAIL compat include dir: ${_PPU_COMPAT_DIR}")
endif()

# The force-included compat wrapper pulls <hggc.h>, <hggc_library_types.h>,
# etc. from the PPU SDK. Put the SDK include dir on the GLOBAL include path so
# host C++ translation units (protobuf, etc.), not just HG device TUs, can
# resolve them.
if(IS_DIRECTORY "${HGGCToolkit_INCLUDE_DIR}")
  include_directories(SYSTEM "${HGGCToolkit_INCLUDE_DIR}")
  message(STATUS "PPU SAIL SDK include dir: ${HGGCToolkit_INCLUDE_DIR}")
endif()

if(HGGC_CUDA_COMPAT_VERSION VERSION_LESS 12.0)
  message(FATAL_ERROR "PyTorch requires the PPU SDK to emulate CUDA 12.0 or above.")
endif()

# ---------------------------------------------------------------------------
# 3. Compatibility targets: HGGC:: -> torch::/caffe2:: (torch-specific glue)
#    This ensures every downstream target_link_libraries(... torch::cudart ...)
#    works unchanged.
# ---------------------------------------------------------------------------

# caffe2::cuda  (HGGC::hggc - driver library)
add_library(caffe2::cuda INTERFACE IMPORTED)
if(TARGET HGGC::hggc)
  set_property(TARGET caffe2::cuda PROPERTY INTERFACE_LINK_LIBRARIES HGGC::hggc)
endif()

# torch::hggcrt (HGGC::hggcrt)
add_library(torch::hggcrt INTERFACE IMPORTED)
set_property(TARGET torch::hggcrt PROPERTY INTERFACE_LINK_LIBRARIES
    HGGC::hggcrt)

# torch::cudart -> INTERFACE IMPORTED (linking to torch::hggcrt)
# c10/cuda/CMakeLists.txt and caffe2/CMakeLists.txt link torch::cudart.
# NOTE: Cannot use ALIAS here - downstream code may call
# target_include_directories() / set_property() on torch::cudart, which
# CMake disallows on ALIAS targets.
if(NOT TARGET torch::cudart)
  add_library(torch::cudart INTERFACE IMPORTED)
  set_property(TARGET torch::cudart PROPERTY INTERFACE_LINK_LIBRARIES torch::hggcrt)
endif()

# caffe2::cublas  (HGGC::acblas + HGGC::acblasLt)
add_library(caffe2::cublas INTERFACE IMPORTED)
set(_ppu_cublas_deps "")
if(TARGET HGGC::acblas)
  list(APPEND _ppu_cublas_deps HGGC::acblas)
endif()
if(TARGET HGGC::acblasLt)
  list(APPEND _ppu_cublas_deps HGGC::acblasLt)
endif()
set_property(TARGET caffe2::cublas PROPERTY INTERFACE_LINK_LIBRARIES
    ${_ppu_cublas_deps})

# torch::cudnn  (HGGC::acdnn)
if(CAFFE2_USE_CUDNN)
  if(TARGET HGGC::acdnn)
    add_library(torch::acdnn INTERFACE IMPORTED)
    set_property(TARGET torch::acdnn PROPERTY INTERFACE_LINK_LIBRARIES HGGC::acdnn)
    target_include_directories(torch::acdnn INTERFACE ${HGGCToolkit_INCLUDE_DIR})
    # Compatibility: torch::cudnn -> torch::acdnn (INTERFACE IMPORTED, not ALIAS)
    if(NOT TARGET torch::cudnn)
      add_library(torch::cudnn INTERFACE IMPORTED)
      set_property(TARGET torch::cudnn PROPERTY INTERFACE_LINK_LIBRARIES torch::acdnn)
    endif()
    # cuDNN compat version: PPU acdnn version is independent of NVIDIA cuDNN;
    # fixed to 8.9.0 only to satisfy the downstream CUDNN_VERSION >= 8.5 gate.
    set(CUDNN_FOUND TRUE CACHE BOOL "cuDNN found (PPU acdnn)" FORCE)
    set(CUDNN_VERSION "8.9.0" CACHE STRING "cuDNN version (PPU acdnn fixed compat value)" FORCE)
    set(CUDNN_INCLUDE_PATH "${HGGCToolkit_INCLUDE_DIR}" CACHE PATH "" FORCE)
    message(STATUS "PPU acdnn (cuDNN compat) version: ${CUDNN_VERSION}")
  else()
    message(WARNING "USE_CUDNN=ON but HGGC::acdnn not found; disabling.")
    set(CAFFE2_USE_CUDNN OFF)
  endif()
else()
  message(STATUS "USE_CUDNN is set to 0. Compiling without cuDNN (acdnn) support")
endif()

# torch::cusparselt  (HGGC::acsparseLt - not provided by FindHGGCToolkit;
# disable if absent. USE_CUSPARSELT is 0 in SAIL builds.)
if(CAFFE2_USE_CUSPARSELT)
  if(TARGET HGGC::acsparseLt)
    add_library(torch::cusparselt INTERFACE IMPORTED)
    set_property(TARGET torch::cusparselt PROPERTY INTERFACE_LINK_LIBRARIES
        HGGC::acsparseLt)
    target_include_directories(torch::cusparselt INTERFACE ${HGGCToolkit_INCLUDE_DIR})
  else()
    message(STATUS "cuSPARSELt (acsparseLt) not found; disabling.")
    set(CAFFE2_USE_CUSPARSELT OFF)
  endif()
else()
  message(STATUS "USE_CUSPARSELT is set to 0. Compiling without cuSPARSELt support")
endif()

# cuDSS - PPU SDK has no <cudss.h> header. Downstream guards the include with
# `#if defined(USE_CUDSS)`, so force-disabling prevents the include.
set(USE_CUDSS OFF CACHE BOOL
    "Use cuDSS (disabled: no PPU native <cudss.h>)" FORCE)
message(STATUS "SAIL mode: cuDSS has no PPU native equivalent; USE_CUDSS forced OFF.")

# cuFile - PPU native has no cuFile equivalent.
set(USE_CUFILE OFF CACHE BOOL "Use cuFile (disabled: no PPU native equivalent)" FORCE)
set(CAFFE2_USE_CUFILE OFF CACHE BOOL "Use cuFile (disabled: no PPU native equivalent)" FORCE)
message(STATUS "SAIL mode: cuFile has no PPU native equivalent; USE_CUFILE forced OFF.")

# ---------------------------------------------------------------------------
# 3b. CUDA:: namespace compatibility targets
#     aten/src/ATen/CMakeLists.txt and caffe2/CMakeLists.txt hardcode
#     references to CUDA::cusparse / CUDA::cufft / CUDA::cusolver (and
#     their _static / _nocallback variants) under USE_CUDA branches.
#     Provide stub targets that map to the HGGC ac* SDK libraries.
# ---------------------------------------------------------------------------
macro(_ppu_make_cuda_compat_target _cuda_name _hggc_name)
  if(NOT TARGET ${_cuda_name})
    if(TARGET HGGC::${_hggc_name})
      add_library(${_cuda_name} INTERFACE IMPORTED GLOBAL)
      set_property(TARGET ${_cuda_name} PROPERTY INTERFACE_LINK_LIBRARIES HGGC::${_hggc_name})
    else()
      # Empty stub: PPU SDK doesn't ship the library; downstream link succeeds.
      add_library(${_cuda_name} INTERFACE IMPORTED GLOBAL)
    endif()
  endif()
endmacro()

# cusparse family
_ppu_make_cuda_compat_target(CUDA::cusparse        acsparse)
_ppu_make_cuda_compat_target(CUDA::cusparse_static acsparse)

# cusolver family
_ppu_make_cuda_compat_target(CUDA::cusolver        acsolver)
_ppu_make_cuda_compat_target(CUDA::cusolver_static acsolver)

# cufft family
_ppu_make_cuda_compat_target(CUDA::cufft                 acfft)
_ppu_make_cuda_compat_target(CUDA::cufft_static          acfft)
_ppu_make_cuda_compat_target(CUDA::cufft_static_nocallback acfft)

# cublas family (also provide CUDA:: namespace)
_ppu_make_cuda_compat_target(CUDA::cublas   acblas)
_ppu_make_cuda_compat_target(CUDA::cublasLt acblasLt)

# curand / cudart (CUDA:: namespace)
_ppu_make_cuda_compat_target(CUDA::curand  acrand)
_ppu_make_cuda_compat_target(CUDA::cudart  hggcrt)

# cupti (CUDA:: namespace) -> libhgpti.so
if(NOT TARGET CUDA::cupti)
  find_library(PPU_HGPTI_LIBRARY
    NAMES hgpti
    PATHS "${PPU_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib"
          "${PPU_TOOLKIT_ROOT_DIR}/asight/lib"
    NO_DEFAULT_PATH)
  if(PPU_HGPTI_LIBRARY)
    add_library(CUDA::cupti SHARED IMPORTED GLOBAL)
    set_target_properties(CUDA::cupti PROPERTIES IMPORTED_LOCATION "${PPU_HGPTI_LIBRARY}")
    message(STATUS "CUDA::cupti -> ${PPU_HGPTI_LIBRARY}")
  else()
    add_library(CUDA::cupti INTERFACE IMPORTED GLOBAL)
    message(STATUS "CUDA::cupti stub (libhgpti.so not found)")
  endif()
endif()

# caffe2::curand  (HGGC::acrand)
add_library(caffe2::curand INTERFACE IMPORTED)
if(TARGET HGGC::acrand)
  set_property(TARGET caffe2::curand PROPERTY INTERFACE_LINK_LIBRARIES HGGC::acrand)
endif()

# caffe2::cufft  (HGGC::acfft)
add_library(caffe2::cufft INTERFACE IMPORTED)
if(TARGET HGGC::acfft)
  set_property(TARGET caffe2::cufft PROPERTY INTERFACE_LINK_LIBRARIES HGGC::acfft)
endif()

# caffe2::nvrtc  (HGGC::hgrtc)
add_library(caffe2::hgrtc INTERFACE IMPORTED)
set_property(TARGET caffe2::hgrtc PROPERTY INTERFACE_LINK_LIBRARIES
    HGGC::hgrtc caffe2::cuda)
# Compatibility: caffe2::nvrtc -> caffe2::hgrtc (INTERFACE IMPORTED, not ALIAS)
if(NOT TARGET caffe2::nvrtc)
  add_library(caffe2::nvrtc INTERFACE IMPORTED)
  set_property(TARGET caffe2::nvrtc PROPERTY INTERFACE_LINK_LIBRARIES caffe2::hgrtc)
endif()

# CUDA_NVRTC_LIB and CUDA_NVRTC_SHORTHASH (used by caffe2_nvrtc target)
if(TARGET HGGC::hgrtc)
  get_target_property(_hgrtc_loc HGGC::hgrtc IMPORTED_LOCATION)
  set(CUDA_NVRTC_LIB "${_hgrtc_loc}" CACHE FILEPATH "")
  if(CUDA_NVRTC_LIB AND NOT CUDA_NVRTC_SHORTHASH)
    find_package(Python COMPONENTS Interpreter)
    execute_process(
      COMMAND ${Python_EXECUTABLE} -c
      "import hashlib;hash=hashlib.sha256();hash.update(open('${CUDA_NVRTC_LIB}','rb').read());print(hash.hexdigest()[:8])"
      RESULT_VARIABLE _retval
      OUTPUT_VARIABLE CUDA_NVRTC_SHORTHASH)
    if(NOT _retval EQUAL 0)
      message(WARNING "Failed to compute shorthash for libhgrtc.so")
      set(CUDA_NVRTC_SHORTHASH "XXXXXXXX")
    else()
      string(STRIP "${CUDA_NVRTC_SHORTHASH}" CUDA_NVRTC_SHORTHASH)
      message(STATUS "${CUDA_NVRTC_LIB} shorthash is ${CUDA_NVRTC_SHORTHASH}")
    endif()
  endif()
endif()

# ---------------------------------------------------------------------------
# 4. HG device compile flags
# ---------------------------------------------------------------------------
set(PPU_NVCC_FLAGS "" CACHE STRING "" FORCE)

# ONNX namespace
if(ONNX_NAMESPACE)
  list(APPEND PPU_NVCC_FLAGS "-DONNX_NAMESPACE=${ONNX_NAMESPACE}")
else()
  list(APPEND PPU_NVCC_FLAGS "-DONNX_NAMESPACE=onnx_c2")
endif()

# Architecture flags via torch_cuda_get_nvcc_gencode_flag (utils.cmake), which
# under USE_SAIL redirects to torch_ppu_get_nvcc_gencode_flag ->
# ppu_select_hgcc_arch_flags (defined above in section 0a).
torch_cuda_get_nvcc_gencode_flag(NVCC_FLAGS_EXTRA)

if(DEFINED CMAKE_HG_ARCHITECTURES AND NOT CMAKE_HG_ARCHITECTURES STREQUAL "")
  message(WARNING
    "pytorch is not compatible with `CMAKE_HG_ARCHITECTURES` and will ignore its value. "
    "Please configure `PYTORCH_SAIL_ARCH` instead.")
  set(CMAKE_HG_ARCHITECTURES "")
endif()

list(APPEND PPU_NVCC_FLAGS ${NVCC_FLAGS_EXTRA})
message(STATUS "Added PPU NVCC flags for: ${NVCC_FLAGS_EXTRA}")

# Warning suppression (hgcc -Xppufe)
foreach(diag cc_clobber_ignored
             field_without_dll_interface
             base_class_has_different_dll_interface
             dll_interface_conflict_none_assumed
             dll_interface_conflict_dllexport_assumed
             bad_friend_decl)
  list(APPEND SUPPRESS_WARNING_FLAGS --diag_suppress=${diag})
endforeach()
string(REPLACE ";" "," SUPPRESS_WARNING_FLAGS "${SUPPRESS_WARNING_FLAGS}")
include(CheckCXXCompilerFlag)
set(_ppu_test_flag "-Xppufe,--diag_suppress=cc_clobber_ignored")
check_cxx_compiler_flag("${_ppu_test_flag}" _PPU_HAS_XPPUFE)
if(_PPU_HAS_XPPUFE)
  list(APPEND PPU_NVCC_FLAGS -Xppufe ${SUPPRESS_WARNING_FLAGS})
endif()

set(PPU_PROPAGATE_HOST_FLAGS_BLOCKLIST "-Werror")

# Debug/Release
if(PPU_DEVICE_DEBUG)
  list(APPEND PPU_NVCC_FLAGS "-g" "-G")
endif()

# hgcc-supported CUDA-compatible flags observed from known-good builds.
list(APPEND PPU_NVCC_FLAGS
  "--forward-unknown-to-host-compiler"
  "--forward-unknown-to-host-linker"
  "-DLIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_OPERATIONS"
  "-std=c++17"
  "--hgfatbinary-options"
  "-compress-all")

# Constexpr and lambda support.
list(APPEND PPU_NVCC_FLAGS
  "--expt-relaxed-constexpr"
  "--extended-lambda"
  "-D__CUDACC_EXTENDED_LAMBDA__")

# CUDA compatibility macros that must reach hgcc via CMAKE_HG_FLAGS.
list(APPEND PPU_NVCC_FLAGS
  "-DCUB_WRAPPED_NAMESPACE=at_cuda_detail"
  "-DDISABLE_CUSPARSE_DEPRECATED"
  "-DCUDA_HAS_FP16=1"
  "-D__HGGC_NO_HALF_OPERATORS__"
  "-D__HGGC_NO_HALF_CONVERSIONS__"
  "-D__HGGC_NO_HALF2_OPERATORS__"
  "-D__HGGC_NO_BFLOAT16_CONVERSIONS__")

# Fold into CMAKE_HG_FLAGS
foreach(FLAG ${PPU_NVCC_FLAGS})
  string(FIND "${FLAG}" " " flag_space_position)
  if(NOT flag_space_position EQUAL -1)
    message(FATAL_ERROR "Found spaces in PPU_NVCC_FLAGS entry '${FLAG}'")
  endif()
  string(APPEND CMAKE_HG_FLAGS " ${FLAG}")
endforeach()

# Also set CUDA_NVCC_FLAGS for compatibility (some third_party code reads it)
set(CUDA_NVCC_FLAGS "${PPU_NVCC_FLAGS}" CACHE STRING "" FORCE)

message(STATUS "Final CMAKE_HG_FLAGS: ${CMAKE_HG_FLAGS}")

# ---------------------------------------------------------------------------
# 5. ppu_override_cuda_targets()
#    Called from Dependencies.cmake AFTER third-party subdirectories (kineto)
#    that may invoke find_package(CUDAToolkit) and overwrite the CUDA:: targets
#    created in section 3b. Re-set IMPORTED_LOCATION back to PPU native libs.
# ---------------------------------------------------------------------------
function(ppu_override_cuda_targets)
  message(STATUS "PPU SAIL: Overriding CUDA:: targets to PPU libraries")

  macro(_ppu_override_one _cuda_name _ppu_lib_name)
    if(TARGET ${_cuda_name})
      find_library(_PPU_${_ppu_lib_name}_LIBRARY
        NAMES ${_ppu_lib_name}
        PATHS "${PPU_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib"
              "${PPU_TOOLKIT_ROOT_DIR}/lib64"
        NO_DEFAULT_PATH)
      if(_PPU_${_ppu_lib_name}_LIBRARY)
        set_target_properties(${_cuda_name} PROPERTIES
          IMPORTED_LOCATION "${_PPU_${_ppu_lib_name}_LIBRARY}")
        message(STATUS "  ${_cuda_name} -> ${_PPU_${_ppu_lib_name}_LIBRARY}")
      else()
        message(WARNING "  ${_cuda_name}: PPU library ${_ppu_lib_name} not found")
      endif()
    endif()
  endmacro()

  _ppu_override_one(CUDA::cublas    acblas)
  _ppu_override_one(CUDA::cublasLt  acblasLt)
  _ppu_override_one(CUDA::cufft     acfft)
  _ppu_override_one(CUDA::cusparse  acsparse)
  _ppu_override_one(CUDA::cusolver  acsolver)
  _ppu_override_one(CUDA::curand    acrand)
  _ppu_override_one(CUDA::cudnn     acdnn)
  _ppu_override_one(CUDA::nvrtc     hgrtc)
  _ppu_override_one(CUDA::cupti     hgpti)
endfunction()

# ---------------------------------------------------------------------------
# 6. CUDA compile rules - intentionally NOT provided
# ---------------------------------------------------------------------------
# enable_language(CUDA) is intentionally NOT called in SAIL mode - hgcc is the
# device compiler, registered via enable_language(HG). The HG language module
# registers .cu/.cuh via CMAKE_HG_SOURCE_FILE_EXTENSIONS; CMake gives custom-
# language extensions priority over built-in mappings, so .cu files are
# classified as HG and compiled through CMAKE_HG_COMPILE_OBJECT.
