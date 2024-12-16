cmake_minimum_required(VERSION 3.16..3.27)
include(FetchContent)

project(cuda-xpbd VERSION 0.1
    DESCRIPTION "A CUDA-accelerated Differential XPBD Framework"
    LANGUAGES CUDA CXX)

option(USE_CUDA "Use CUDA implementation instead of CPU" ON)
option(WRITE "Write the state of the simulation to stdout" OFF)

# Relative to GPU, compute 8.6
set(CMAKE_CUDA_ARCHITECTURES 86)

# Make sure to --expt-relaxed-constexpr, otherwise bad things will happen
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --generate-line-info --expt-relaxed-constexpr")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  # to nicely support folders in IDEs
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)

  include(CTest)
endif()

# Uncomment on first build to fetch Eigen
FetchContent_Declare(
  Eigen-git
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        25270e35dbfb9d407175a321707a3b51a079588d
)
FetchContent_MakeAvailable(Eigen-git)
find_package (Eigen3 3.4 REQUIRED NO_MODULE)

include_directories(
    "/mnt/c/Users/andre/anaconda3/envs/coal_env/Library/include"
    # "/mnt/c/Users/andre/anaconda3/envs/coal_env/Library/include/eigen3"
)

link_directories(
    "/mnt/c/Users/andre/anaconda3/envs/coal_env/Library/lib"
)

file(DOWNLOAD https://github.com/ericjang/svd3/raw/master/svd3_cuda/svd3_cuda.h ${CMAKE_CURRENT_BINARY_DIR}/svd3_cuda.h)
add_library(svd3_cuda INTERFACE)
target_include_directories(svd3_cuda INTERFACE ${CMAKE_CURRENT_BINARY_DIR})

add_subdirectory(src)

if((CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR MODERN_CMAKE_BUILD_TESTING)
   AND BUILD_TESTING)
  add_subdirectory(tests)
endif()

