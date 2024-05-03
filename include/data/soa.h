#pragma once
#include "data/primitives.h"
#include "data/utilities.h"

namespace data {

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

#ifdef __CUDA_ARCH__
#define global_store device_global_store
#else
#define global_store host_global_store
#endif
extern __device__ SOAStore device_global_store;
// dummy for host/device functions
extern SOAStore host_global_store;

} // namespace data
