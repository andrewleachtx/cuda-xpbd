cmake --build build/debug_cuda -t performance

cuda-gdb --args build/debug_cuda/tests/performance -m 1 -s 1 -t 20