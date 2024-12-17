#!/bin/bash

if [ -d "build" ]; then
    rm -rf build
fi

cmake -S . -B build/release_cuda -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON

cmake --build build/release_cuda --parallel -t performance