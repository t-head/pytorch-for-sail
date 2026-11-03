# Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
# Flex flash attention (FA3) SDPA backend integration stub.
#
# The full flex-flash-attention submodule and its build path live in the
# CUDA-compat tree only; this tree keeps the include site in
# caffe2/CMakeLists.txt compatible while always disabling the backend.
# See the CUDA-compat tree's cmake/flex_flash_attention.cmake for the real
# implementation.
option(USE_FLEX_FLASH_ATTENTION "Whether to build flex attention backend" ON)
if(NOT USE_FLEX_FLASH_ATTENTION)
  return()
endif()
message(STATUS "Flex flash attention is not available in this build configuration; disabling it.")
set(USE_FLEX_FLASH_ATTENTION OFF CACHE BOOL "" FORCE)
return()
