mkdir -p build
mkdir -p build/release_cuda
if [ ! -d "build/release_cuda" ]; then
    cmake -S . -B build/release_cuda -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON
fi

cmake --build build/release_cuda --parallel -t performance