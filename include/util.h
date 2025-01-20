#pragma once
#include <iostream>

#include "config.h"

/// if M_PI not defined (Windows) define it
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/// Ensures the success of a given CUDA API call, aborting the program on a
/// failure.
#define CUDA_CHECK(expr)                                                      \
    {                                                                         \
        cudaError_t err = (expr);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr,                                                   \
                    __FILE__                                                  \
                    "(%d):\n CUDA Error Code  : %d\n     Error String: %s\n", \
                    __LINE__, err, cudaGetErrorString(err));                  \
            exit(err);                                                        \
        }                                                                     \
    }

/// Switch for quickly disabling debug macros
#define DEBUG_PRINTS

#ifdef DEBUG_PRINTS
/// Prints an Eigen::VectorNf where N is given
#define DEBUG_VEC(vec, n)             \
    {                                 \
        printf(#vec " = ");           \
        for (int i = 0; i < n; i++) { \
            printf("%f ", vec(i));    \
        }                             \
        printf("\n");                 \
    }
/// Prints an Eigen::MatrixNf where m and n are the matrix dimensions
#define DEBUG_MAT(vec, m, n)              \
    {                                     \
        printf(#vec " = ");               \
        for (int i = 0; i < n; i++) {     \
            for (int j = 0; j < m; j++) { \
                printf("%f ", vec(i, j)); \
            }                             \
            printf("\n");                 \
        }                                 \
        printf("\n");                     \
    }
/// Prints the current function, line, and file
#define TRACE(val) \
    { printf(__FILE__ ":%d - %s: " #val "\n", __LINE__, __PRETTY_FUNCTION__); }
/// Prints a single floating point value
#define DEBUG_FLOAT(float) \
    { printf(#float " = %f\n", float); }
#else
#define DEBUG_VEC(vec, n) \
    {}
#define DEBUG_MAT(vec, m, n) \
    {}
#define TRACE(val) \
    {}
#define DEBUG_FLOAT(float) \
    {}
#endif

#ifndef NDEBUG
#define DEBUG_ASSERT(val, message)                                          \
    {                                                                       \
        bool res = val;                                                     \
        if (!res) {                                                         \
            printf("%s:%d - %s: Assert failed! '%s': `%s` was false\n",     \
                   __FILE__, __LINE__, __PRETTY_FUNCTION__, message, #val); \
        }                                                                   \
    }
#else
#define DEBUG_ASSERT(val, message) \
    {}
#endif

__host__ __device__ [[noreturn]] inline void unreachable() {
#ifdef __CUDA_ARCH__
    TRACE();
    __trap();
#else
    std::abort();
#endif
}

// global objects to imitate the threadIdx and threadDim from CUDA on CPU
extern thread_local size_t _thread_scene_id;
extern size_t _global_scene_count;

using byte = unsigned char;

/**
 * Allocates a buffer with the given size in bytes from the appropriate device
 * (GPU if USE_CUDA is defined, CPU if not)
 */
void *alloc_device_bytes(size_t bytes);
template <typename T>
T *alloc_device(size_t count) {
    size_t bytes = count * sizeof(T);
    return reinterpret_cast<T *>(alloc_device_bytes(bytes));
}

/**
 * Copies bytes from a device ptr to another device ptr. Uses the appropriate
 * device based on USE_CUDA.
 */
__host__ __device__ void memcpy_device_bytes(void *__restrict dest,
                                             void *__restrict src,
                                             size_t bytes);
/**
 * Copies an array of items from a device ptr to a device ptr.
 */
template <typename T>
__host__ __device__ void memcpy_device(T *__restrict dest, T *__restrict src,
                                       size_t count) {
    memcpy_device_bytes(dest, src, count * sizeof(T));
}
/**
 * Copies bytes from a host buffer to a device buffer.
 */
void memcpy_host_device(void *__restrict dest, void *__restrict src,
                        size_t bytes);

/**
 * Allocates a new devie buffer and copies the data in host_ptr to new new
 * array.
 */
template <typename T>
T *move_array_to_device(T *host_ptr, size_t count) {
    auto new_ptr = alloc_device<T>(count);
    memcpy_host_device(new_ptr, host_ptr, count * sizeof(T));
    delete[] host_ptr;
    return new_ptr;
}
