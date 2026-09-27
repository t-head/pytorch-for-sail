# Find the PCCL libraries used by SAIL builds.
#
# The following variables optionally override the SDK default directories:
#   PCCL_INCLUDE_DIR: Directory where pccl.h is found.
#   PCCL_LIB_DIR: Directory where libpccl is found.
#
# The following variables are set after configuration is done:
#   PCCL_FOUND
#   PCCL_INCLUDE_DIRS
#   PCCL_LIBRARIES

set(PCCL_INCLUDE_DIR $ENV{PCCL_INCLUDE_DIR} CACHE PATH "Folder contains PCCL headers")
set(PCCL_LIB_DIR $ENV{PCCL_LIB_DIR} CACHE PATH "Folder contains PCCL libraries")

# setup.py supplies SDK defaults for wheel builds. Keep direct CMake
# configuration equivalent. NO_DEFAULT_PATH below ensures an explicit bad
# directory cannot silently fall back to another installation.
if(NOT PCCL_INCLUDE_DIR AND DEFINED ENV{PPU_SDK})
  if(EXISTS "$ENV{PPU_SDK}/include/pccl.h" AND EXISTS "$ENV{PPU_SDK}/lib/libpccl.so")
    set(PCCL_INCLUDE_DIR "$ENV{PPU_SDK}/include" CACHE PATH "Folder contains PCCL headers" FORCE)
  else()
    set(PCCL_INCLUDE_DIR "$ENV{PPU_SDK}/targets/x86_64-linux/include" CACHE PATH "Folder contains PCCL headers" FORCE)
  endif()
endif()
if(NOT PCCL_LIB_DIR AND DEFINED ENV{PPU_SDK})
  if(EXISTS "$ENV{PPU_SDK}/include/pccl.h" AND EXISTS "$ENV{PPU_SDK}/lib/libpccl.so")
    set(PCCL_LIB_DIR "$ENV{PPU_SDK}/lib" CACHE PATH "Folder contains PCCL libraries" FORCE)
  else()
    set(PCCL_LIB_DIR "$ENV{PPU_SDK}/targets/x86_64-linux/lib" CACHE PATH "Folder contains PCCL libraries" FORCE)
  endif()
endif()

message(STATUS "SAIL: searching for PCCL (pccl.h, libpccl)")
find_path(PCCL_INCLUDE_DIRS
  NAMES pccl.h
  HINTS ${PCCL_INCLUDE_DIR}
  NO_DEFAULT_PATH)
find_library(PCCL_LIBRARIES
  NAMES pccl
  HINTS ${PCCL_LIB_DIR}
  NO_DEFAULT_PATH)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PCCL DEFAULT_MSG PCCL_INCLUDE_DIRS PCCL_LIBRARIES)

if(PCCL_FOUND)
  set(PCCL_HEADER_FILE "${PCCL_INCLUDE_DIRS}/pccl.h")
  message(STATUS "Determining PCCL version from ${PCCL_HEADER_FILE}...")
  file(STRINGS "${PCCL_HEADER_FILE}" _PCCL_MAJOR_LINE REGEX "^#define[ \t]+PCCL_MAJOR[ \t]+[0-9]+")
  file(STRINGS "${PCCL_HEADER_FILE}" _PCCL_MINOR_LINE REGEX "^#define[ \t]+PCCL_MINOR[ \t]+[0-9]+")
  file(STRINGS "${PCCL_HEADER_FILE}" _PCCL_PATCH_LINE REGEX "^#define[ \t]+PCCL_PATCH[ \t]+[0-9]+")
  string(REGEX REPLACE ".*PCCL_MAJOR[ \t]+([0-9]+).*" "\\1" _PCCL_MAJOR "${_PCCL_MAJOR_LINE}")
  string(REGEX REPLACE ".*PCCL_MINOR[ \t]+([0-9]+).*" "\\1" _PCCL_MINOR "${_PCCL_MINOR_LINE}")
  string(REGEX REPLACE ".*PCCL_PATCH[ \t]+([0-9]+).*" "\\1" _PCCL_PATCH "${_PCCL_PATCH_LINE}")
  if(_PCCL_MAJOR AND _PCCL_MINOR AND _PCCL_PATCH)
    set(PCCL_VERSION_FROM_HEADER "${_PCCL_MAJOR}.${_PCCL_MINOR}.${_PCCL_PATCH}")
    message(STATUS "PCCL version: ${PCCL_VERSION_FROM_HEADER}")
  else()
    message(STATUS "PCCL version: unknown")
  endif()
  mark_as_advanced(PCCL_INCLUDE_DIR PCCL_LIB_DIR PCCL_INCLUDE_DIRS PCCL_LIBRARIES)
endif()
