cmake_minimum_required(VERSION 3.16...3.28)
# simple program, just build and link libhop.cu
project (cmaes-stack LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11 REQUIRED)

# default to release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# link libcmaes installed to ~/.local, and there is a libcmaesConfig.cmake.in at /home/andyleach/pkgs/libcmaes
find_package(libcmaes CONFIG REQUIRED)

add_executable(cmaes-stack src/cmaes.cpp)
target_link_libraries(cmaes-stack PRIVATE libcmaes::cmaes)