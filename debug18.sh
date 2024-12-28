# "version" is probably "release_cuda" or "release_cpu"; it is the subdirectory inside build - build/<version>/
version=$1
# model ID and scene ct
m=$2
s=$3

# cmake --build build/$version --parallel -t performance
cmake --build build/$version --parallel -t performance

# if that cmake failed do nothing
if [ $? -ne 0 ]; then
  exit 1
fi

./build/$version/tests/performance -m $m -s $s
# gdb --args ./build/$version/tests/performance -m $m -s $s