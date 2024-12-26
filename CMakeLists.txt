cmake_minimum_required(VERSION 3.16)
include(FetchContent)

project(cuda-xpbd VERSION 0.1
    LANGUAGES CUDA CXX)

option(USE_CUDA "Use CUDA implementation instead of CPU" OFF)
option(WRITE "Write the state of the simulation to stdout" OFF)

# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CUDA_STANDARD 17)
# set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Relative to GPU, compute 8.6
set(CMAKE_CUDA_ARCHITECTURES 75 86)

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

file(DOWNLOAD https://github.com/ericjang/svd3/raw/master/svd3_cuda/svd3_cuda.h ${CMAKE_CURRENT_BINARY_DIR}/svd3_cuda.h)
add_library(svd3_cuda INTERFACE)
target_include_directories(svd3_cuda INTERFACE ${CMAKE_CURRENT_BINARY_DIR})

list(APPEND CMAKE_PREFIX_PATH "/home/andy/pkgs/coal/install")
find_package(coal CONFIG REQUIRED)
include_directories("/home/andy/pkgs/coal/install/include")
link_directories("/home/andy/pkgs/coal/install/lib")

add_subdirectory(src)

if((CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR MODERN_CMAKE_BUILD_TESTING)
   AND BUILD_TESTING)
  add_subdirectory(tests)
endif()

