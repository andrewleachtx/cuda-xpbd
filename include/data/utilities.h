#pragma once
#include "util.h"
#ifndef EIGEN_DEFAULT_DENSE_INDEX_TYPE
    #define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#endif
#include <stddef.h>

#include <Eigen/Dense>

/**
 * Implements only the getter for an SOAStore stored attribute.
 */
#define IMPLEMENT_READONLY_ACCESS_FUNCTIONS(AttributeType, RefType, Type, \
                                            attribute)                    \
    inline AttributeType RefType::attribute() const {                     \
        return data::global_store.Type.attribute.get(index);              \
    }
/**
 * Implements the getter and setter for an SOAStore stored attribute.
 */
#define IMPLEMENT_ACCESS_FUNCTIONS(AttributeType, RefType, Type, attribute) \
    inline AttributeType RefType::attribute() const {                       \
        return data::global_store.Type.attribute.get(index);                \
    }                                                                       \
    inline void RefType::attribute(AttributeType const new_val) {           \
        data::global_store.Type.attribute.set(index, new_val);              \
    }

using vec7 = Eigen::Matrix<float, 7, 1>;
using vec12 = Eigen::Matrix<float, 12, 1>;
namespace data {

/// Calculates the index for accessing SOA data for the current thread.
__host__ __device__ inline size_t soa_index(unsigned int index) {
#ifdef __CUDA_ARCH__
    const size_t scene_id = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t scene_count = blockDim.x * gridDim.x;
    return scene_id + index * scene_count;
#else
    return _thread_scene_id + index * _global_scene_count;
#endif
}

/// Determines the optimal buffer alignment for a given type.
template <typename T>
constexpr __host__ __device__ size_t get_aligned_size(size_t count) {
    const size_t alignment_count = 32;
    const size_t byte_size = count * sizeof(T);
    if (byte_size % alignment_count == 0)
        return byte_size;
    else
        return (byte_size / alignment_count + 1) * alignment_count;
}

/// Extracts a segment of a larger allocation based on a given type and number
/// of elements
template <typename T>
T *get_aligned_buffer_segment(byte *data_store, size_t &offset, size_t count) {
    DEBUG_ASSERT(offset % 32 == 0, "Offset not aligned in get aligned buffer!");
    T *buffer_segment = reinterpret_cast<T *>(data_store + offset);
    offset += get_aligned_size<T>(count);
    return buffer_segment;
}

}  // namespace data
