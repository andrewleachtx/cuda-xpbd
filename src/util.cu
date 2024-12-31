#include "util.h"

thread_local size_t _thread_scene_id = 0;
size_t _global_scene_count = 0;

void *alloc_device_bytes(size_t bytes)
{
#ifdef USE_CUDA
    void *device_ptr;
    CUDA_CHECK(cudaMalloc(&device_ptr, bytes));
    return device_ptr;
#else
    return malloc(bytes);
#endif
}

void memcpy_device_bytes(void *__restrict dest, void *__restrict src,
                         size_t bytes)
{
#ifdef __CUDA_ARCH__
    // do manual memcpy because the kernel doesn't have a memcpy function
    for (size_t i = 0; i < bytes; i++)
        reinterpret_cast<char *>(dest)[i] = reinterpret_cast<char *>(src)[i];
#else
#ifdef USE_CUDA
    CUDA_CHECK(cudaMemcpy(dest, src, bytes, cudaMemcpyDeviceToDevice));
#else
    memcpy(dest, src, bytes);
#endif
#endif
}

void memcpy_host_device(void *__restrict dest, void *__restrict src,
                        size_t bytes)
{
#ifdef USE_CUDA
    CUDA_CHECK(cudaMemcpy(dest, src, bytes, cudaMemcpyHostToDevice));
#else
    memcpy(dest, src, bytes);
#endif
}
