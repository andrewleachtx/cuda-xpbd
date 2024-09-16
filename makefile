CMAKE_FLAGS = -S . -B build
BUILD_DIR = build
EIGEN_REPO = https://gitlab.com/libeigen/eigen.git
EIGEN_VERSION = 25270e35dbfb9d407175a321707a3b51a079588d
CUDA_FLAGS = -DUSE_CUDA=ON
BUILD_TYPE = -DCMAKE_BUILD_TYPE=Release
CMAKE_MIN_VERSION = 3.16

cmake_version:
	@cmake_user_version=$(shell cmake --version | head -n 1 | cut -d ' ' -f 3)

	if [ $(cmake_user_version) \< $(CMAKE_MIN_VERSION) ]; then \
		echo "CMake requires version $(CMAKE_MIN_VERSION), you have $(cmake_user_version)"; \
		exit 1; \
	fi