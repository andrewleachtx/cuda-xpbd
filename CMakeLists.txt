cmake_minimum_required(VERSION 3.16...3.28)

# include(FetchContent)

project(cuda-xpbd VERSION 0.1
    LANGUAGES CUDA CXX)

option(USE_CUDA "Use CUDA implementation instead of CPU" OFF)
option(WRITE "Write the state of the simulation to stdout" OFF)

# Relative to GPU, compute 8.6
set(CMAKE_CUDA_ARCHITECTURES 89)

# Make sure to --expt-relaxed-constexpr, otherwise bad things will happen
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --generate-line-info --expt-relaxed-constexpr")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  # to nicely support folders in IDEs
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)

  include(CTest)
endif()

# Optionally use ccache
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
else()
    message(STATUS "Feel free to sudo apt install ccache if you want faster compilation times.")
endif()

# Uncomment on first build to fetch Eigen
# FetchContent_Declare(
#   Eigen-git
#   GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
#   GIT_TAG        25270e35dbfb9d407175a321707a3b51a079588d
# )
# FetchContent_MakeAvailable(Eigen-git)
add_subdirectory(include/eigen)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

file(DOWNLOAD https://github.com/ericjang/svd3/raw/master/svd3_cuda/svd3_cuda.h ${CMAKE_CURRENT_BINARY_DIR}/svd3_cuda.h)
add_library(svd3_cuda INTERFACE)
target_include_directories(svd3_cuda INTERFACE ${CMAKE_CURRENT_BINARY_DIR})

# Set to your coal path
set(COAL_PATH "/home/ugrads/a/andrewleachtx/packages/coal/install")
if (NOT EXISTS ${COAL_PATH})
    message(FATAL_ERROR "${COAL_PATH} not found, did you update COAL_PATH in CMakeLists.txt?")
endif()

list(APPEND CMAKE_PREFIX_PATH ${COAL_PATH})
find_package(coal CONFIG REQUIRED)
include_directories(${COAL_PATH}/include)
link_directories(${COAL_PATH}/lib)

# CMake gets upset when nvcc doesnt get explicitly passed to nvcc - this will take some management on your end if it happens
set(BOOST_INCLUDE_PATH "/home/ugrads/a/andrewleachtx/packages/coal/.pixi/envs/default/")
if (NOT EXISTS ${BOOST_INCLUDE_PATH})
    message(FATAL_ERROR "${BOOST_INCLUDE_PATH} not found, did you update BOOST_INCLUDE_PATH in CMakeLists.txt?")
endif()

include_directories("${BOOST_INCLUDE_PATH}/include")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I/home/ugrads/a/andrewleachtx/packages/coal/.pixi/envs/default/include")

add_subdirectory(src)

if((CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR MODERN_CMAKE_BUILD_TESTING)
   AND BUILD_TESTING)
  add_subdirectory(tests)
endif()
