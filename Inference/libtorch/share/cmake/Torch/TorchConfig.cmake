# FindTorch
# -------
#
# Finds the Torch library
#
# This will define the following variables:
#
#   TORCH_FOUND        -- True if the system has the Torch library
#   TORCH_INCLUDE_DIRS -- The include directories for torch
#   TORCH_LIBRARIES    -- Libraries to link against
#   TORCH_CXX_FLAGS    -- Additional (required) compiler flags
#
# and the following imported targets:
#
#   torch

include(FindPackageHandleStandardArgs)

if (DEFINED ENV{TORCH_INSTALL_PREFIX})
  set(TORCH_INSTALL_PREFIX $ENV{TORCH_INSTALL_PREFIX})
else()
  # Assume we are in <install-prefix>/share/cmake/Torch/TorchConfig.cmake
  get_filename_component(CMAKE_CURRENT_LIST_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
  get_filename_component(TORCH_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)
endif()

# Include directories.
if (EXISTS "${TORCH_INSTALL_PREFIX}/include")
  set(TORCH_INCLUDE_DIRS
    ${TORCH_INSTALL_PREFIX}/include
    ${TORCH_INSTALL_PREFIX}/include/torch/csrc/api/include)
else()
  set(TORCH_INCLUDE_DIRS
    ${TORCH_INSTALL_PREFIX}/include
    ${TORCH_INSTALL_PREFIX}/include/torch/csrc/api/include)
endif()

# Library dependencies.
if (ON)
  find_package(Caffe2 REQUIRED PATHS ${CMAKE_CURRENT_LIST_DIR}/../Caffe2)
endif()

if (NOT ANDROID)
  find_library(TORCH_LIBRARY torch PATHS "${TORCH_INSTALL_PREFIX}/lib")
else()
  find_library(TORCH_LIBRARY NO_CMAKE_FIND_ROOT_PATH torch PATHS "${TORCH_INSTALL_PREFIX}/lib")
endif()

set(TORCH_LIBRARIES torch ${Caffe2_MAIN_LIBS})

if (NOT ANDROID)
  find_library(C10_LIBRARY c10 PATHS "${TORCH_INSTALL_PREFIX}/lib")
else()
  find_library(C10_LIBRARY c10 NO_CMAKE_FIND_ROOT_PATH PATHS "${TORCH_INSTALL_PREFIX}/lib")
endif()
list(APPEND TORCH_LIBRARIES ${C10_LIBRARY})

if (True)
  if(MSVC)
    set(NVTOOLEXT_HOME "C:/Program Files/NVIDIA Corporation/NvToolsExt")
    if ($ENV{NVTOOLEXT_HOME})
      set(NVTOOLEXT_HOME $ENV{NVTOOLEXT_HOME})
    endif()
    set(TORCH_CUDA_LIBRARIES
      ${NVTOOLEXT_HOME}/lib/x64/nvToolsExt64_1.lib
      ${CUDA_LIBRARIES})
    list(APPEND TORCH_INCLUDE_DIRS ${NVTOOLEXT_HOME}/include)
    find_library(CAFFE2_NVRTC_LIBRARY caffe2_nvrtc PATHS "${TORCH_INSTALL_PREFIX}/lib")
    list(APPEND TORCH_CUDA_LIBRARIES ${CAFFE2_NVRTC_LIBRARY})
  elseif(APPLE)
    set(TORCH_CUDA_LIBRARIES
      ${CUDA_TOOLKIT_ROOT_DIR}/lib/libcudart.dylib
      ${CUDA_TOOLKIT_ROOT_DIR}/lib/libnvrtc.dylib
      ${CUDA_TOOLKIT_ROOT_DIR}/lib/libnvToolsExt.dylib
      ${CUDA_LIBRARIES})
  else()
    find_library(LIBNVTOOLSEXT libnvToolsExt.so PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64/)
    set(TORCH_CUDA_LIBRARIES
      ${CUDA_CUDA_LIB}
      ${CUDA_NVRTC_LIB}
      ${LIBNVTOOLSEXT}
      ${CUDA_LIBRARIES})
  endif()
  find_library(C10_CUDA_LIBRARY c10_cuda PATHS "${TORCH_INSTALL_PREFIX}/lib")
  list(APPEND TORCH_CUDA_LIBRARIES ${C10_CUDA_LIBRARY})
  list(APPEND TORCH_LIBRARIES ${TORCH_CUDA_LIBRARIES})
endif()

# When we build libtorch with the old GCC ABI, dependent libraries must too.
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=")
endif()

set_target_properties(torch PROPERTIES
    IMPORTED_LOCATION "${TORCH_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${TORCH_INCLUDE_DIRS}"
    CXX_STANDARD 11
)
if (TORCH_CXX_FLAGS)
  set_property(TARGET torch PROPERTY INTERFACE_COMPILE_OPTIONS "${TORCH_CXX_FLAGS}")
endif()

find_package_handle_standard_args(torch DEFAULT_MSG TORCH_LIBRARY TORCH_INCLUDE_DIRS)
