# giant library glob for avoiding separable compilation
file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/**/*.h")
file(GLOB SOURCE_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/src/**/*.cu")

add_library(apbd-lib STATIC util.cu ${SOURCE_LIST} ${HEADER_LIST})
target_compile_options(apbd-lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       >)
set_property(TARGET apbd-lib PROPERTY CUDA_SEPARABLE_COMPILATION ON)
# set_property(TARGET apbd-lib PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_include_directories(apbd-lib PUBLIC ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(apbd-lib PUBLIC coal svd3_cuda Eigen3::Eigen Pagmo::pagmo)

source_group(
  TREE "${PROJECT_SOURCE_DIR}/include"
  PREFIX "Header Files"
  FILES ${HEADER_LIST})

if (USE_CUDA)
  target_compile_definitions(apbd-lib PUBLIC USE_CUDA)
endif()
if (WRITE)
  target_compile_definitions(apbd-lib PUBLIC WRITE)
endif()

