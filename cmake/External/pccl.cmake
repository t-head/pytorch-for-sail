if(NOT __PCCL_INCLUDED)
  set(__PCCL_INCLUDED TRUE)

  find_package(PCCL REQUIRED)
  if(PCCL_FOUND)
    add_library(__caffe2_pccl INTERFACE)
    target_link_libraries(__caffe2_pccl INTERFACE ${PCCL_LIBRARIES})
    target_include_directories(__caffe2_pccl INTERFACE ${PCCL_INCLUDE_DIRS})

    # Distributed C++ sources retain the upstream NCCL API and target name.
    # PCCL provides that ABI in SAIL builds, so preserve this alias only at
    # the compatibility boundary.
    add_library(__caffe2_nccl ALIAS __caffe2_pccl)
  endif()
endif()
