add_executable(integration integration.cu)
add_executable(performance performance.cu)
add_executable(hop hop.cu)

if (MSVC)
    add_compile_options(/W4 /WX)
else()
    add_compile_options(-Wall -Wextra -pedantic -Werror)
endif()

target_compile_options(performance PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       >)

add_compile_options(-use_fast_math)

set_property(TARGET integration PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET performance PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET hop         PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET integration PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_property(TARGET performance PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_property(TARGET hop         PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_link_libraries(integration PRIVATE apbd-lib)
target_link_libraries(performance PRIVATE apbd-lib)
target_link_libraries(hop PRIVATE apbd-lib)

add_test(NAME integration-test COMMAND integration)
add_test(NAME performance-test COMMAND performance)
add_test(NAME hop-test COMMAND hop)

