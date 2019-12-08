# ---[ cuda

# Poor man's include guard
if(TARGET torch::cudart)
  return()
endif()

# sccache is only supported in CMake master and not in the newest official
# release (3.11.3) yet. Hence we need our own Modules_CUDA_fix to enable sccache.
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/../Modules_CUDA_fix)

# We don't want to statically link cudart, because we rely on it's dynamic linkage in
# python (follow along torch/cuda/__init__.py and usage of cudaGetErrorName).
# Technically, we can link cudart here statically, and link libtorch_python.so
# to a dynamic libcudart.so, but that's just wasteful.
# However, on Windows, if this one gets switched off, the error "cuda: unknown error"
# will be raised when running the following code:
# >>> import torch
# >>> torch.cuda.is_available()
# >>> torch.cuda.current_device()
# More details can be found in the following links.
# https://github.com/pytorch/pytorch/issues/20635
# https://github.com/pytorch/pytorch/issues/17108
if (NOT MSVC)
  set(CUDA_USE_STATIC_CUDA_RUNTIME OFF CACHE INTERNAL "")
endif()

# Find CUDA.
find_package(CUDA)
if(NOT CUDA_FOUND)
  message(WARNING
    "Caffe2: CUDA cannot be found. Depending on whether you are building "
    "Caffe2 or a Caffe2 dependent library, the next warning / error will "
    "give you more info.")
  set(CAFFE2_USE_CUDA OFF)
  return()
endif()
message(STATUS "Caffe2: CUDA detected: " ${CUDA_VERSION})
message(STATUS "Caffe2: CUDA nvcc is: " ${CUDA_NVCC_EXECUTABLE})
message(STATUS "Caffe2: CUDA toolkit directory: " ${CUDA_TOOLKIT_ROOT_DIR})
if(CUDA_VERSION VERSION_LESS 9.0)
  message(FATAL_ERROR "PyTorch requires CUDA 9.0 and above.")
endif()

if(CUDA_FOUND)
  # Sometimes, we may mismatch nvcc with the CUDA headers we are
  # compiling with, e.g., if a ccache nvcc is fed to us by CUDA_NVCC_EXECUTABLE
  # but the PATH is not consistent with CUDA_HOME.  It's better safe
  # than sorry: make sure everything is consistent.
  if(MSVC AND CMAKE_GENERATOR MATCHES "Visual Studio")
    # When using Visual Studio, it attempts to lock the whole binary dir when
    # `try_run` is called, which will cause the build to fail.
    string(RANDOM BUILD_SUFFIX)
    set(PROJECT_RANDOM_BINARY_DIR "${PROJECT_BINARY_DIR}/${BUILD_SUFFIX}")
  else()
    set(PROJECT_RANDOM_BINARY_DIR "${PROJECT_BINARY_DIR}")
  endif()
  set(file "${PROJECT_BINARY_DIR}/detect_cuda_version.cc")
  file(WRITE ${file} ""
    "#include <cuda.h>\n"
    "#include <cstdio>\n"
    "int main() {\n"
    "  printf(\"%d.%d\", CUDA_VERSION / 1000, (CUDA_VERSION / 10) % 100);\n"
    "  return 0;\n"
    "}\n"
    )
  try_run(run_result compile_result ${PROJECT_RANDOM_BINARY_DIR} ${file}
    CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${CUDA_INCLUDE_DIRS}"
    LINK_LIBRARIES ${CUDA_LIBRARIES}
    RUN_OUTPUT_VARIABLE cuda_version_from_header
    COMPILE_OUTPUT_VARIABLE output_var
    )
  if(NOT compile_result)
    message(FATAL_ERROR "Caffe2: Couldn't determine version from header: " ${output_var})
  endif()
  message(STATUS "Caffe2: Header version is: " ${cuda_version_from_header})
  if(NOT ${cuda_version_from_header} STREQUAL ${CUDA_VERSION_STRING})
    # Force CUDA to be processed for again next time
    # TODO: I'm not sure if this counts as an implementation detail of
    # FindCUDA
    set(${cuda_version_from_findcuda} ${CUDA_VERSION_STRING})
    unset(CUDA_TOOLKIT_ROOT_DIR_INTERNAL CACHE)
    # Not strictly necessary, but for good luck.
    unset(CUDA_VERSION CACHE)
    # Error out
    message(FATAL_ERROR "FindCUDA says CUDA version is ${cuda_version_from_findcuda} (usually determined by nvcc), "
      "but the CUDA headers say the version is ${cuda_version_from_header}.  This often occurs "
      "when you set both CUDA_HOME and CUDA_NVCC_EXECUTABLE to "
      "non-standard locations, without also setting PATH to point to the correct nvcc.  "
      "Perhaps, try re-running this command again with PATH=${CUDA_TOOLKIT_ROOT_DIR}/bin:$PATH.  "
      "See above log messages for more diagnostics, and see https://github.com/pytorch/pytorch/issues/8092 for more details.")
  endif()
endif()

# Find cuDNN.
if(CAFFE2_STATIC_LINK_CUDA AND NOT USE_STATIC_CUDNN)
  message(WARNING "cuDNN will be linked statically because CAFFE2_STATIC_LINK_CUDA is ON. "
    "Set USE_STATIC_CUDNN to ON to suppress this warning.")
endif()
if(CAFFE2_STATIC_LINK_CUDA OR USE_STATIC_CUDNN)
  set(CUDNN_STATIC ON CACHE BOOL "")
else()
  set(CUDNN_STATIC OFF CACHE BOOL "")
endif()

find_package(CUDNN)

if(NOT CUDNN_FOUND)
  message(WARNING
    "Caffe2: Cannot find cuDNN library. Turning the option off")
  set(CAFFE2_USE_CUDNN OFF)
else()
  set(CAFFE2_USE_CUDNN ON)
endif()

# Optionally, find TensorRT
if(CAFFE2_USE_TENSORRT)
  find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES include)
  find_library(TENSORRT_LIBRARY nvinfer
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)
  find_package_handle_standard_args(
    TENSORRT DEFAULT_MSG TENSORRT_INCLUDE_DIR TENSORRT_LIBRARY)
  if(NOT TENSORRT_FOUND)
    message(WARNING
      "Caffe2: Cannot find TensorRT library. Turning the option off")
    set(CAFFE2_USE_TENSORRT OFF)
  endif()
endif()

# ---[ Extract versions
if(CAFFE2_USE_CUDNN)
  # Get cuDNN version
  file(READ ${CUDNN_INCLUDE_PATH}/cudnn.h CUDNN_HEADER_CONTENTS)
  string(REGEX MATCH "define CUDNN_MAJOR * +([0-9]+)"
               CUDNN_VERSION_MAJOR "${CUDNN_HEADER_CONTENTS}")
  string(REGEX REPLACE "define CUDNN_MAJOR * +([0-9]+)" "\\1"
               CUDNN_VERSION_MAJOR "${CUDNN_VERSION_MAJOR}")
  string(REGEX MATCH "define CUDNN_MINOR * +([0-9]+)"
               CUDNN_VERSION_MINOR "${CUDNN_HEADER_CONTENTS}")
  string(REGEX REPLACE "define CUDNN_MINOR * +([0-9]+)" "\\1"
               CUDNN_VERSION_MINOR "${CUDNN_VERSION_MINOR}")
  string(REGEX MATCH "define CUDNN_PATCHLEVEL * +([0-9]+)"
               CUDNN_VERSION_PATCH "${CUDNN_HEADER_CONTENTS}")
  string(REGEX REPLACE "define CUDNN_PATCHLEVEL * +([0-9]+)" "\\1"
               CUDNN_VERSION_PATCH "${CUDNN_VERSION_PATCH}")
  # Assemble cuDNN version
  if(NOT CUDNN_VERSION_MAJOR)
    set(CUDNN_VERSION "?")
  else()
    set(CUDNN_VERSION
        "${CUDNN_VERSION_MAJOR}.${CUDNN_VERSION_MINOR}.${CUDNN_VERSION_PATCH}")
  endif()
  message(STATUS "Found cuDNN: v${CUDNN_VERSION}  (include: ${CUDNN_INCLUDE_PATH}, library: ${CUDNN_LIBRARY_PATH})")
  if(CUDNN_VERSION VERSION_LESS "7.0.0")
    message(FATAL_ERROR "PyTorch requires cuDNN 7 and above.")
  endif()
endif()

# ---[ CUDA libraries wrapper

# find libcuda.so and lbnvrtc.so
# For libcuda.so, we will find it under lib, lib64, and then the
# stubs folder, in case we are building on a system that does not
# have cuda driver installed. On windows, we also search under the
# folder lib/x64.
find_library(CUDA_CUDA_LIB cuda
    PATHS ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/stubs lib64/stubs lib/x64)
find_library(CUDA_NVRTC_LIB nvrtc
    PATHS ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)

# Create new style imported libraries.
# Several of these libraries have a hardcoded path if CAFFE2_STATIC_LINK_CUDA
# is set. This path is where sane CUDA installations have their static
# libraries installed. This flag should only be used for binary builds, so
# end-users should never have this flag set.

# cuda
add_library(caffe2::cuda UNKNOWN IMPORTED)
set_property(
    TARGET caffe2::cuda PROPERTY IMPORTED_LOCATION
    ${CUDA_CUDA_LIB})
set_property(
    TARGET caffe2::cuda PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    ${CUDA_INCLUDE_DIRS})

# cudart. CUDA_LIBRARIES is actually a list, so we will make an interface
# library.
add_library(torch::cudart INTERFACE IMPORTED)
if(CAFFE2_STATIC_LINK_CUDA)
    set_property(
        TARGET torch::cudart PROPERTY INTERFACE_LINK_LIBRARIES
        "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudart_static.a" rt dl)
else()
    set_property(
        TARGET torch::cudart PROPERTY INTERFACE_LINK_LIBRARIES
        ${CUDA_LIBRARIES})
endif()
set_property(
    TARGET torch::cudart PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    ${CUDA_INCLUDE_DIRS})

# cudnn
# static linking is handled by USE_STATIC_CUDNN environment variable
if(CAFFE2_USE_CUDNN)
  add_library(caffe2::cudnn UNKNOWN IMPORTED)
  set_property(
      TARGET caffe2::cudnn PROPERTY IMPORTED_LOCATION
      ${CUDNN_LIBRARY_PATH})
  set_property(
      TARGET caffe2::cudnn PROPERTY INTERFACE_INCLUDE_DIRECTORIES
      ${CUDNN_INCLUDE_PATH})
endif()

# curand
add_library(caffe2::curand UNKNOWN IMPORTED)
if(CAFFE2_STATIC_LINK_CUDA)
    set_property(
        TARGET caffe2::curand PROPERTY IMPORTED_LOCATION
        "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcurand_static.a")
else()
    set_property(
        TARGET caffe2::curand PROPERTY IMPORTED_LOCATION
        ${CUDA_curand_LIBRARY})
endif()
set_property(
    TARGET caffe2::curand PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    ${CUDA_INCLUDE_DIRS})

# cufft. CUDA_CUFFT_LIBRARIES is actually a list, so we will make an
# interface library similar to cudart.
add_library(caffe2::cufft INTERFACE IMPORTED)
if(CAFFE2_STATIC_LINK_CUDA)
    set_property(
        TARGET caffe2::cufft PROPERTY INTERFACE_LINK_LIBRARIES
        "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcufft_static.a")
else()
    set_property(
        TARGET caffe2::cufft PROPERTY INTERFACE_LINK_LIBRARIES
        ${CUDA_CUFFT_LIBRARIES})
endif()
set_property(
    TARGET caffe2::cufft PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    ${CUDA_INCLUDE_DIRS})

# TensorRT
if(CAFFE2_USE_TENSORRT)
  add_library(caffe2::tensorrt UNKNOWN IMPORTED)
  set_property(
      TARGET caffe2::tensorrt PROPERTY IMPORTED_LOCATION
      ${TENSORRT_LIBRARY})
  set_property(
      TARGET caffe2::tensorrt PROPERTY INTERFACE_INCLUDE_DIRECTORIES
      ${TENSORRT_INCLUDE_DIR})
endif()

# cublas. CUDA_CUBLAS_LIBRARIES is actually a list, so we will make an
# interface library similar to cudart.
add_library(caffe2::cublas INTERFACE IMPORTED)
if(CAFFE2_STATIC_LINK_CUDA)
    set_property(
        TARGET caffe2::cublas PROPERTY INTERFACE_LINK_LIBRARIES
        "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcublas_static.a")
    if (CUDA_VERSION VERSION_EQUAL 10.1)
      set_property(
        TARGET caffe2::cublas APPEND PROPERTY INTERFACE_LINK_LIBRARIES
        "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcublasLt_static.a")
    endif()
else()
    set_property(
        TARGET caffe2::cublas PROPERTY INTERFACE_LINK_LIBRARIES
        ${CUDA_CUBLAS_LIBRARIES})
endif()
set_property(
    TARGET caffe2::cublas PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    ${CUDA_INCLUDE_DIRS})

# nvrtc
add_library(caffe2::nvrtc UNKNOWN IMPORTED)
set_property(
    TARGET caffe2::nvrtc PROPERTY IMPORTED_LOCATION
    ${CUDA_NVRTC_LIB})
set_property(
    TARGET caffe2::nvrtc PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    ${CUDA_INCLUDE_DIRS})

# Note: in theory, we can add similar dependent library wrappers. For
# now, Caffe2 only uses the above libraries, so we will only wrap
# these.

# Special care for windows platform: we know that 32-bit windows does not
# support cuda.
if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
  if(NOT (CMAKE_SIZEOF_VOID_P EQUAL 8))
    message(FATAL_ERROR
            "CUDA support not available with 32-bit windows. Did you "
            "forget to set Win64 in the generator target?")
    return()
  endif()
endif()

# Add onnx namepsace definition to nvcc
if (ONNX_NAMESPACE)
  list(APPEND CUDA_NVCC_FLAGS "-DONNX_NAMESPACE=${ONNX_NAMESPACE}")
else()
  list(APPEND CUDA_NVCC_FLAGS "-DONNX_NAMESPACE=onnx_c2")
endif()

# CUDA 9.0 & 9.1 require GCC version <= 5
# Although they support GCC 6, but a bug that wasn't fixed until 9.2 prevents
# them from compiling the std::tuple header of GCC 6.
# See Sec. 2.2.1 of
# https://developer.download.nvidia.com/compute/cuda/9.2/Prod/docs/sidebar/CUDA_Toolkit_Release_Notes.pdf
if ((CUDA_VERSION VERSION_EQUAL   9.0) OR
    (CUDA_VERSION VERSION_GREATER 9.0  AND CUDA_VERSION VERSION_LESS 9.2))
  if (CMAKE_C_COMPILER_ID STREQUAL "GNU" AND
      NOT CMAKE_C_COMPILER_VERSION VERSION_LESS 6.0 AND
      CUDA_HOST_COMPILER STREQUAL CMAKE_C_COMPILER)
    message(FATAL_ERROR
      "CUDA ${CUDA_VERSION} is not compatible with std::tuple from GCC version "
      ">= 6. Please upgrade to CUDA 9.2 or set the following environment "
      "variable to use another version (for example): \n"
      "  export CUDAHOSTCXX='/usr/bin/gcc-5'\n")
  endif()
endif()

# CUDA 9.0 / 9.1 require MSVC version < 19.12
# CUDA 9.2 require MSVC version < 19.13
# CUDA 10.0 require MSVC version < 19.20
if ((CUDA_VERSION VERSION_EQUAL   9.0) OR
    (CUDA_VERSION VERSION_GREATER 9.0  AND CUDA_VERSION VERSION_LESS 9.2))
  if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" AND
      NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19.12 AND
      NOT DEFINED ENV{CUDAHOSTCXX})
        message(FATAL_ERROR
          "CUDA ${CUDA_VERSION} is not compatible with MSVC toolchain version "
          ">= 19.12. (a.k.a Visual Studio 2017 Update 5, VS 15.5) "
          "Please upgrade to CUDA >= 9.2 or set the following environment "
          "variable to use another version (for example): \n"
          "  set \"CUDAHOSTCXX=C:\\Program Files (x86)\\Microsoft Visual Studio"
          "\\2017\\Enterprise\\VC\\Tools\\MSVC\\14.11.25503\\bin\\HostX64\\x64\\cl.exe\"\n")
  endif()
elseif(CUDA_VERSION VERSION_EQUAL   9.2)
  if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" AND
      NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19.13 AND
      NOT DEFINED ENV{CUDAHOSTCXX})
    message(FATAL_ERROR
      "CUDA ${CUDA_VERSION} is not compatible with MSVC toolchain version "
      ">= 19.13. (a.k.a Visual Studio 2017 Update 7, VS 15.7) "
      "Please upgrade to CUDA >= 10.0 or set the following environment "
      "variable to use another version (for example): \n"
      "  set \"CUDAHOSTCXX=C:\\Program Files (x86)\\Microsoft Visual Studio"
      "\\2017\\Enterprise\\VC\\Tools\\MSVC\\14.12.25827\\bin\\HostX64\\x64\\cl.exe\"\n")
  endif()
elseif(CUDA_VERSION VERSION_EQUAL   10.0)
  if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" AND
      NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19.20 AND
      NOT DEFINED ENV{CUDAHOSTCXX})
    message(FATAL_ERROR
      "CUDA ${CUDA_VERSION} is not compatible with MSVC toolchain version "
      ">= 19.20. (a.k.a Visual Studio 2019, VS 16.0) "
      "Please upgrade to CUDA >= 10.1 or set the following environment "
      "variable to use another version (for example): \n"
      "  set \"CUDAHOSTCXX=C:\\Program Files (x86)\\Microsoft Visual Studio"
      "\\2017\\Enterprise\\VC\\Tools\\MSVC\\14.16.27023\\bin\\HostX64\\x64\\cl.exe\"\n")
  endif()
endif()

# setting nvcc arch flags
torch_cuda_get_nvcc_gencode_flag(NVCC_FLAGS_EXTRA)
list(APPEND CUDA_NVCC_FLAGS ${NVCC_FLAGS_EXTRA})
message(STATUS "Added CUDA NVCC flags for: ${NVCC_FLAGS_EXTRA}")

# disable some nvcc diagnostic that apears in boost, glog, glags, opencv, etc.
foreach(diag cc_clobber_ignored integer_sign_change useless_using_declaration set_but_not_used)
  list(APPEND CUDA_NVCC_FLAGS -Xcudafe --diag_suppress=${diag})
endforeach()

# Set C++11 support
set(CUDA_PROPAGATE_HOST_FLAGS_BLACKLIST "-Werror")
if (MSVC)
  list(APPEND CUDA_PROPAGATE_HOST_FLAGS_BLACKLIST "/EHa")
else()
  list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
  list(APPEND CUDA_NVCC_FLAGS "-Xcompiler" "-fPIC")
endif()

# Debug and Release symbol support
if (MSVC)
  if (${CAFFE2_USE_MSVC_STATIC_RUNTIME})
    list(APPEND CUDA_NVCC_FLAGS "-Xcompiler" "-MT$<$<CONFIG:Debug>:d>")
  else()
    list(APPEND CUDA_NVCC_FLAGS "-Xcompiler" "-MD$<$<CONFIG:Debug>:d>")
  endif()
elseif (CUDA_DEVICE_DEBUG)
  list(APPEND CUDA_NVCC_FLAGS "-g" "-G")  # -G enables device code debugging symbols
endif()

# Set expt-relaxed-constexpr to suppress Eigen warnings
list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")

# Set expt-extended-lambda to support lambda on device
list(APPEND CUDA_NVCC_FLAGS "--expt-extended-lambda")
