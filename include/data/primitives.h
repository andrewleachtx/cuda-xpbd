#pragma once
#include "data/utilities.h"

namespace data {
/**
 * An SOA store that handles any generic data type. This may not use the most
 * optimal buffer sizes or alignments.
 */
template <typename T> struct _SOAStoreGeneric {
  T *data;

  /// A default uninitialized constructor. Accessing data without using the full
  /// constructor is undefined behavior.
  __host__ __device__ _SOAStoreGeneric() {}
  _SOAStoreGeneric(byte *data_store, size_t &offset, size_t count);
  /// Calculates the size necessary to store the data in this buffer with count
  /// elements.
  static constexpr size_t size(size_t count) {
    return get_aligned_size<T>(count);
  }

  __host__ __device__ T get(unsigned int index) const;
  __host__ __device__ void set(unsigned int index, const T &new_val);
};

/**
 * SOA Store for an Eigen::Quaternionf. Handles conversion from 4 floats to a
 * Quaternionf object.
 */
struct _SOAStoreQuaterion {
  float4 *data;

  /// A default uninitialized constructor. Accessing data without using the full
  /// constructor is undefined behavior.
  __host__ __device__ _SOAStoreQuaterion() {}
  _SOAStoreQuaterion(byte *data_store, size_t &offset, size_t count);
  /// Calculates the size necessary to store the data in this buffer with count
  /// elements.
  static constexpr size_t size(size_t count) {
    return get_aligned_size<float4>(count);
  }

  __host__ __device__ Eigen::Quaternionf get(unsigned int index) const;
  __host__ __device__ void set(unsigned int index, Eigen::Quaternionf new_val);
};

/**
 * SOA Store for an Eigen::Vector3f.
 */
struct _SOAStoreVec3 {
  float2 *x01;
  float *x2;

  /// A default uninitialized constructor. Accessing data without using the full
  /// constructor is undefined behavior.
  __host__ __device__ _SOAStoreVec3() {}
  _SOAStoreVec3(byte *data_store, size_t &offset, size_t count);
  /// Calculates the size necessary to store the data in this buffer with count
  /// elements.
  static constexpr size_t size(size_t count) {
    return get_aligned_size<float2>(count) + get_aligned_size<float>(count);
  }

  __host__ __device__ Eigen::Vector3f get(unsigned int index) const;
  __host__ __device__ void set(unsigned int index, Eigen::Vector3f new_val);
};

/**
 * SOA Store for an Eigen::Vector3f.
 */
struct _SOAStoreVec4 {
  float4 *x;

  /// A default uninitialized constructor. Accessing data without using the full
  /// constructor is undefined behavior.
  __host__ __device__ _SOAStoreVec4() {}
  _SOAStoreVec4(byte *data_store, size_t &offset, size_t count);
  /// Calculates the size necessary to store the data in this buffer with count
  /// elements.
  static constexpr size_t size(size_t count) {
    return get_aligned_size<float4>(count);
  }

  __host__ __device__ Eigen::Vector4f get(unsigned int index) const;
  __host__ __device__ void set(unsigned int index, Eigen::Vector4f new_val);
};

/**
 * SOA Store for an Eigen::Vector7f.
 */
struct _SOAStoreVec7 {
  // see
  // https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/
  // for motivation for the sizes of these buffers
  float4 *x03;
  float2 *x45;
  float *x6;

  /// A default uninitialized constructor. Accessing data without using the full
  /// constructor is undefined behavior.
  __host__ __device__ _SOAStoreVec7() {}
  _SOAStoreVec7(byte *data_store, size_t &offset, size_t count);
  /// Calculates the size necessary to store the data in this buffer with count
  /// elements.
  static constexpr size_t size(size_t count) {
    return get_aligned_size<float4>(count) + get_aligned_size<float2>(count) +
           get_aligned_size<float>(count);
  }

  __host__ __device__ vec7 get(unsigned int index) const;
  __host__ __device__ void set(unsigned int index, vec7 new_val);
};

/**
 * SOA Store for an Eigen::Matrix4f.
 */
struct _SOAStoreMat4 {
  // see
  // https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/
  // for motivation for the sizes of these buffers
  float4 *x1;
  float4 *x2;
  float4 *x3;
  float4 *x4;

  /// A default uninitialized constructor. Accessing data without using the full
  /// constructor is undefined behavior.
  __host__ __device__ _SOAStoreMat4() {}
  _SOAStoreMat4(byte *data_store, size_t &offset, size_t count);
  /// Calculates the size necessary to store the data in this buffer with count
  /// elements.
  static constexpr size_t size(size_t count) {
    return get_aligned_size<float4>(count) * 4;
  }

  __host__ __device__ Eigen::Matrix4f get(unsigned int index) const;
  __host__ __device__ void set(unsigned int index, Eigen::Matrix4f new_val);
};

inline _SOAStoreQuaterion::_SOAStoreQuaterion(byte *data_store, size_t &offset,
                                              size_t count) {
  this->data = get_aligned_buffer_segment<float4>(data_store, offset, count);
}

inline Eigen::Quaternionf _SOAStoreQuaterion::get(unsigned int index) const {
  const float4 data_val = data[index];
  return Eigen::Quaternionf(
      Eigen::Vector4f(data_val.x, data_val.y, data_val.z, data_val.w));
}

inline void _SOAStoreQuaterion::set(unsigned int index,
                                    Eigen::Quaternionf new_val) {
  const auto coeffs = new_val.coeffs();
  data[index] = make_float4(coeffs(0), coeffs(1), coeffs(2), coeffs(3));
}

inline _SOAStoreVec3::_SOAStoreVec3(byte *data_store, size_t &offset,
                                    size_t count) {
  this->x01 = get_aligned_buffer_segment<float2>(data_store, offset, count);
  this->x2 = get_aligned_buffer_segment<float>(data_store, offset, count);
}

inline Eigen::Vector3f _SOAStoreVec3::get(unsigned int index) const {
  const float2 el01 = x01[index];
  const float el2 = x2[index];
  return Eigen::Vector3f(el01.x, el01.y, el2);
}

inline void _SOAStoreVec3::set(unsigned int index, Eigen::Vector3f new_val) {
  x01[index] = make_float2(new_val(0), new_val(1));
  x2[index] = new_val(2);
}

inline Eigen::Vector4f _SOAStoreVec4::get(unsigned int index) const {
  const float4 el = x[index];
  return Eigen::Vector4f(el.x, el.y, el.z, el.w);
}

inline void _SOAStoreVec4::set(unsigned int index, Eigen::Vector4f new_val) {
  x[index] = make_float4(new_val(0), new_val(1), new_val(2), new_val(3));
}

inline _SOAStoreVec7::_SOAStoreVec7(byte *data_store, size_t &offset,
                                    size_t count) {
  this->x03 = get_aligned_buffer_segment<float4>(data_store, offset, count);
  this->x45 = get_aligned_buffer_segment<float2>(data_store, offset, count);
  this->x6 = get_aligned_buffer_segment<float>(data_store, offset, count);
}

inline vec7 _SOAStoreVec7::get(unsigned int index) const {
  const float4 el03 = x03[index];
  const float2 el45 = x45[index];
  const float el6 = x6[index];
  return vec7(el03.x, el03.y, el03.z, el03.w, el45.x, el45.y, el6);
}

inline void _SOAStoreVec7::set(unsigned int index, vec7 new_val) {
  x03[index] = make_float4(new_val(0), new_val(1), new_val(2), new_val(3));
  x45[index] = make_float2(new_val(4), new_val(5));
  x6[index] = new_val(6);
}

inline _SOAStoreMat4::_SOAStoreMat4(byte *data_store, size_t &offset,
                                    size_t count) {
  this->x1 = get_aligned_buffer_segment<float4>(data_store, offset, count);
  this->x2 = get_aligned_buffer_segment<float4>(data_store, offset, count);
  this->x3 = get_aligned_buffer_segment<float4>(data_store, offset, count);
  this->x4 = get_aligned_buffer_segment<float4>(data_store, offset, count);
}

inline Eigen::Matrix4f _SOAStoreMat4::get(unsigned int index) const {
  Eigen::Matrix4f output;
  float4 *const data_ptr = reinterpret_cast<float4 *>(output.data());
  data_ptr[0] = x1[index];
  data_ptr[1] = x2[index];
  data_ptr[2] = x3[index];
  data_ptr[3] = x4[index];
  return output;
}

inline void _SOAStoreMat4::set(unsigned int index, Eigen::Matrix4f new_val) {
  x1[index] =
      make_float4(new_val(0, 0), new_val(0, 1), new_val(0, 2), new_val(0, 3));
  x2[index] =
      make_float4(new_val(1, 0), new_val(1, 1), new_val(1, 2), new_val(1, 3));
  x3[index] =
      make_float4(new_val(2, 0), new_val(2, 1), new_val(2, 2), new_val(2, 3));
  x4[index] =
      make_float4(new_val(3, 0), new_val(3, 1), new_val(3, 2), new_val(3, 3));
}

template <typename T>
_SOAStoreGeneric<T>::_SOAStoreGeneric(byte *data_store, size_t &offset,
                                      size_t count) {
  this->data = get_aligned_buffer_segment<T>(data_store, offset, count);
}

template <typename T> T _SOAStoreGeneric<T>::get(unsigned int index) const {
  return data[index];
}

template <typename T>
void _SOAStoreGeneric<T>::set(unsigned int index, const T &new_val) {
  data[index] = new_val;
}

} // namespace data
