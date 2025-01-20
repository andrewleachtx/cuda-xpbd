time cmake --build build/ --parallel

# if that cmake failed do nothing
if [ $? -ne 0 ]; then
  exit 1
fi

time ./build/cmaes-stack
# gdb ./build/cmaes-stack