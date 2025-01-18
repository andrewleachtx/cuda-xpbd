# "version" is probably "release_cuda" or "release_cpu"; it is the subdirectory inside build - build/<version>/
version=$1
# model ID, scene_ct, substeps
m=$2
s=$3
t=$4

# cmake -S . -B build/$version -G Ninja -DCMAKE_BUILD_TYPE=$version -DUSE_CUDA=OFF -DWRITE=ON 

time cmake --build build/$version --parallel -t performance

# if that cmake failed do nothing
if [ $? -ne 0 ]; then
  exit 1
fi


if [[ $version == *"release"* ]]; then
    ./build/$version/tests/performance -m $m -s $s -t $t
else
   gdb --args ./build/$version/tests/performance -m $m -s $s -t $t
#    ./build/$version/tests/performance -m $m -s $s -t $t
fi