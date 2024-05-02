#pragma once
#include <stddef.h>
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#include "apbd/Body.h"
#include "apbd/Shape.h"
#include "util.h"
#include <Eigen/Dense>

namespace data {

/// Calculates the index for accessing SOA data for the current thread.
__host__ __device__ size_t soa_index(unsigned int index);

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

typedef Eigen::Matrix<float, 7, 1> vec7;
typedef Eigen::Matrix<float, 12, 1> vec12;

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
 * SOA Store for a BodyRigid object. Provides access to lower-level SOA types
 * for each element.
 */
struct _SOAStoreBodyRigid {
  _SOAStoreVec7 xdotInit;
  _SOAStoreVec3 position;
  _SOAStoreQuaterion rotation;
  _SOAStoreVec7 x0;
  _SOAStoreVec7 x1;
  _SOAStoreQuaterion x1_0_rot;
  _SOAStoreVec7 dxJacobi;
  _SOAStoreVec7 dxJacobiShock;
  _SOAStoreGeneric<bool> collide;
  _SOAStoreGeneric<float> mu;
  _SOAStoreGeneric<apbd::Shape> shape;
  _SOAStoreGeneric<float> density;
  _SOAStoreVec3 Mr;
  _SOAStoreGeneric<float> Mp;
  _SOAStoreGeneric<unsigned int> layer;

  __host__ __device__ _SOAStoreBodyRigid() {}
  _SOAStoreBodyRigid(byte *data_store, size_t &offset, size_t count);
  /// Calculates the size necessary to store the data in this buffer with count
  /// elements.
  static constexpr size_t size(size_t count) {
    return _SOAStoreVec7::size(count) * 5 + _SOAStoreVec3::size(count) * 2 +
           _SOAStoreQuaterion::size(count) * 2 +
           _SOAStoreGeneric<float>::size(count) * 3 +
           _SOAStoreGeneric<bool>::size(count) +
           _SOAStoreGeneric<apbd::Shape>::size(count) +
           _SOAStoreGeneric<unsigned int>::size(count);
  }

  __host__ __device__ void set(unsigned int index, const apbd::BodyRigid &data);
};

/**
 * Struct-of-Arrays data store. Manually implements data accesses in a single
 * giant global buffer with the optimal alignments, sizes, and data divisions.
 */
class SOAStore {
public:
  struct _SOAStoreBodyRigid BodyRigid;

  __host__ __device__ SOAStore() {}
  SOAStore(size_t body_rigid_count, size_t scene_count);

  void deallocate();
};

#ifdef __CUDA_ARCH__
#define global_store device_global_store
#else
#define global_store host_global_store
#endif
extern __device__ SOAStore device_global_store;
// dummy for host/device functions
extern SOAStore host_global_store;

inline size_t soa_index(unsigned int index) {
#ifdef __CUDA_ARCH__
  const size_t scene_id = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t scene_count = blockDim.x * gridDim.x;
  return scene_id + index * scene_count;
#else
  return _thread_scene_id + index * _global_scene_count;
#endif
}

inline SOAStore::SOAStore(size_t body_rigid_count, size_t scene_count) {
  const size_t alignment_count = BLOCK_SIZE;
  size_t aligned_scene_count = 0;
  if (scene_count % alignment_count == 0)
    aligned_scene_count = scene_count;
  else
    aligned_scene_count = (scene_count / alignment_count + 1) * alignment_count;

  const size_t object_count = body_rigid_count * aligned_scene_count;
  // we need three aligned buffers of floats
  const size_t total_buffer_size = _SOAStoreBodyRigid::size(object_count);
  byte *const data_store = alloc_device<byte>(total_buffer_size);

  size_t offset = 0;
  this->BodyRigid = _SOAStoreBodyRigid(data_store, offset, object_count);
}

inline void SOAStore::deallocate() {}

inline _SOAStoreBodyRigid::_SOAStoreBodyRigid(byte *data_store, size_t &offset,
                                              size_t count)
    : xdotInit(data_store, offset, count), position(data_store, offset, count),
      rotation(data_store, offset, count), x0(data_store, offset, count),
      x1(data_store, offset, count), x1_0_rot(data_store, offset, count),
      dxJacobi(data_store, offset, count),
      dxJacobiShock(data_store, offset, count),
      collide(data_store, offset, count), mu(data_store, offset, count),
      shape(data_store, offset, count), density(data_store, offset, count),
      Mr(data_store, offset, count), Mp(data_store, offset, count),
      layer(data_store, offset, count) {}

inline void _SOAStoreBodyRigid::set(unsigned int index,
                                    const apbd::BodyRigid &data) {
  xdotInit.set(index, data.xdotInit);
  position.set(index, data.x.block<3, 1>(4, 0));
  rotation.set(index, Eigen::Quaternionf(data.x.block<4, 1>(0, 0)));
  x0.set(index, data.x0);
  x1.set(index, data.x1);
  x1_0_rot.set(index, Eigen::Quaternionf(data.x1_0.block<4, 1>(0, 0)));
  dxJacobi.set(index, data.dxJacobi);
  dxJacobiShock.set(index, data.dxJacobiShock);
  collide.set(index, data.collide);
  mu.set(index, data.mu);
  shape.set(index, data.shape);
  density.set(index, data.density);
  Mr.set(index, data.Mr);
  Mp.set(index, data.Mp);
  layer.set(index, data.layer);
}

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

} // namespace data
