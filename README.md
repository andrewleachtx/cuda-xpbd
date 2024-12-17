# XPBD CUDA

A CUDA-accelerated XPBD-based physics simulation framework.

## Installation / Setup
1. Working in a WSL / Linux environment, make sure the following are installed (in order):
   1. CMake v3.16 or greater
      1. If you have a version < 3.16, you can try changing the top line in `./CMakeLists.txt`. If you are using a package manager and it is maxed out at an older version, see answer 1 [here](https://askubuntu.com/questions/829310/how-to-upgrade-cmake-in-ubuntu).
      2.  `sudo apt install cmake` 
   2. CUDA v12.6 or greater
      1. Earlier versions may (probably will) work. This document has commands to run at the bottom.
      2. [NVIDIA WSL CUDA Download](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local)
   3. NVCC
      1. This should come with the CUDA install, if **nvcc --version doesn't work, make sure /usr/local/CUDA/bin is added to your path**
      2. Add `PATH=/usr/local/cuda/bin:$PATH` to your `~/.bashrc` to get CMake to recognize NVCC.
2. Once you have these downloads (it will take some time), install `Eigen` by uncommenting the fetch content lines in `./CMakeLists.txt`:
   1. Line 2
   2. Lines 20-25
   3. Comment these after building cmake for the first time, or after deleting `build/`.

There may be hardcoded paths to `coal` and `octomap` in the project root's `CMakeLists.txt`.

You would need to clone and install [coal](https://github.com/coal-library/coal/blob/devel/development/build.md) as necessary, as well as its dependencies. Also,
   
## First Build / Clean Resets
Run
```bash
cmake -B build
# Same as cmake -S . -B build, assumes you are in ./
```
This will take some time, including downloading and building Eigen with the uncommented lines from instruction 2 [above](#installation--setup).

After it finishes, recomment the `FetchContent` lines in `CMakeLists.txt`, as it will error out later.

Now that the `build/` has been populated, you should use

```
cmake --build build
```

to build after any changes - or use any of the additional targets described below. You can add `--parallel` to speed this up, or use `-G Ninja` if you have that generator.

## Building

```bash
cmake -S . -B build
# or with options
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON -DWRITE=OFF
cmake --build build
```

## Running Tests

```bash
cmake --build build --target <test>
```

There are `performance` and `integration` builds, you can use `cmake --build build --target help` to find them.

## Profiling or Benchmarking

See `run_perf_test.sh` and `run_profiler.sh`.


#### release_cuda
```sh
# for setup
mkdir build/release_cuda
cmake -S . -B build/release_cuda -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON

# for building
cmake --build build/release_cuda --parallel -t performance

# NOTE: you might want to remove the parallel argument (and in the script), just to make sure the compile step is deterministic
```

Use this with `./run_perf_test.sh release_cuda`



## Formatting

```bash
git ls-files -- '*.cu' '*.h' | xargs clang-format -i -style=file
```

## Acknowledgments

Code in the JGT-float modules is public domain and accessed from this URL:

[https://web.archive.org/web/20070715170639/jgt.akpeters.com/papers/MahovskyWyvill04/](https://web.archive.org/web/20070715170639/jgt.akpeters.com/papers/MahovskyWyvill04/)

It has been edited to be CUDA-compatible.

