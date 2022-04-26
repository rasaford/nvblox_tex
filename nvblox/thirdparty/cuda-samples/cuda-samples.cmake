include(FetchContent)
FetchContent_Declare(
  ext_cuda-samples
  PREFIX cuda-samples
  GIT_REPOSITORY https://github.com/NVIDIA/cuda-samples.git
  GIT_TAG        v11.6
  UPDATE_COMMAND ""
)

# Download the files
FetchContent_MakeAvailable(ext_cuda-samples)
add_library(cuda-samples INTERFACE)
target_include_directories(cuda-samples INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/_deps/ext_cuda-samples-src/Common>
    $<INSTALL_INTERFACE:Common>
)