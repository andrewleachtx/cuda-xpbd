#!/usr/bin/env bash

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# "version" is probably "release_cuda" or "release_cpu"; it is the subdirectory inside build - build/<version>/
version=$1
# model ID, scene_ct, substeps
m=$2
s=$3
t=$4
w=$5

BUILD_PATH="$SCRIPT_PATH/build/$version"
EXEC_PATH="$BUILD_PATH/tests/hop"

# this script is run from ./cmaes/src/cmaes.cpp and should run
# ./build/release_cuda/tests/hop -m 99 -s <scene count, should be given> -t <substeps>
echo "### BUILDING ###"
# if [[ $version != *"release"* ]]; then
    time cmake --build $BUILD_PATH --parallel -t hop
# fi

echo "### RUNNING ###"
if [[ $version == *"release"* ]]; then
    time $EXEC_PATH -m $m -s $s -t $t -w $w
else
    # cuda-gdb --args $EXEC_PATH -m $m -s $s -t $t
    # valgrind --tool=memcheck --leak-check=full --track-origins=yes --show-leak-kinds=all $EXEC_PATH -m $m -s $s -t $t
    $EXEC_PATH -m $m -s $s -t $t
fi