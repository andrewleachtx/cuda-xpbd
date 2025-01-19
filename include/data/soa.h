#pragma once
#include "apbd/Body.h"
#include "apbd/BodyReference.h"
#include "apbd/CollisionReference.h"
#include "apbd/Shape.h"
#include "data/primitives.h"
#include "data/utilities.h"

namespace apbd {
class Collision;
}

namespace data {

// TODO: For new GPQP matrices, should add _SOAStoreVec24, etc

struct _SOAStoreConstraintGround {
    _SOAStoreVec3 lambda;
    _SOAStoreVec3 nw;
    _SOAStoreVec3 xl;
    _SOAStoreVec3 xw;
    _SOAStoreVec3 d;
    _SOAStoreGeneric<apbd::BodyRigidReference> body;

    _SOAStoreMat3 contactFrame;

    _SOAStoreVec3 w1;
    _SOAStoreMat3 delLinVel1;
    _SOAStoreMat3 angDelta1;
    _SOAStoreMat3 raXnI1;
    _SOAStoreMat3 raXn;

    _SOAStoreGeneric<apbd::CollisionReference> collision;

    __host__ __device__ _SOAStoreConstraintGround() {}
    _SOAStoreConstraintGround(byte *data_store, size_t &offset, size_t count);
    /// Calculates the size necessary to store the data in this buffer with
    /// count elements.
    static constexpr size_t size(size_t count) {
        return _SOAStoreMat3::size(count) * 5 + _SOAStoreVec3::size(count) * 6 +
               _SOAStoreGeneric<apbd::CollisionReference>::size(count) +
               _SOAStoreGeneric<apbd::BodyRigidReference>::size(count);
    }
};

struct _SOAStoreConstraintRigid {
    _SOAStoreVec3 lambda;
    _SOAStoreVec3 nw;
    _SOAStoreVec3 x1;
    _SOAStoreVec3 x2;
    _SOAStoreVec3 dlambdaSP;
    _SOAStoreVec3 d;
    _SOAStoreGeneric<apbd::BodyRigidReference> body1;
    _SOAStoreGeneric<apbd::BodyRigidReference> body2;

    _SOAStoreMat3 contactFrame;

    _SOAStoreVec3 w1;
    _SOAStoreMat3 delLinVel1;
    _SOAStoreMat3 angDelta1;
    _SOAStoreMat3 raXnI1;

    _SOAStoreVec3 w2;
    _SOAStoreMat3 delLinVel2;
    _SOAStoreMat3 angDelta2;
    _SOAStoreMat3 raXnI2;

    _SOAStoreGeneric<apbd::CollisionReference> collision;

    __host__ __device__ _SOAStoreConstraintRigid() {}
    _SOAStoreConstraintRigid(byte *data_store, size_t &offset, size_t count);
    /// Calculates the size necessary to store the data in this buffer with
    /// count elements.
    static constexpr size_t size(size_t count) {
        return _SOAStoreMat3::size(count) * 7 + _SOAStoreVec3::size(count) * 8 +
               _SOAStoreGeneric<apbd::CollisionReference>::size(count) +
               _SOAStoreGeneric<apbd::BodyRigidReference>::size(count) * 2;
    }
};

/**
 * SOA Store for a BodyRigid object. Provides access to lower-level SOA types
 * for each element.
 */
struct _SOAStoreBodyRigid {
    _SOAStoreVec3 position;
    _SOAStoreQuaterion rotation;
    _SOAStoreVec7 x0;
    _SOAStoreGeneric<bool> collide;
    _SOAStoreGeneric<float> mu;
    _SOAStoreGeneric<apbd::Shape> shape;
    _SOAStoreGeneric<float> density;
    _SOAStoreVec3 Mr;
    _SOAStoreGeneric<float> Mp;
    _SOAStoreGeneric<unsigned int> layer;
    _SOAStoreVec3 v;
    _SOAStoreVec3 w;
    _SOAStoreVec3 deltaBody2Worldp;
    _SOAStoreQuaterion deltaBody2Worldq;
    _SOAStoreVec3 deltaAngDt;
    _SOAStoreVec3 deltaLinDt;

    // GPQP
    _SOAStoreVec7 dxJacobi;
    _SOAStoreVec7 dphiJacobi;
    _SOAStoreGeneric<vec6f> LTx;

    __host__ __device__ _SOAStoreBodyRigid() {}
    _SOAStoreBodyRigid(byte *data_store, size_t &offset, size_t count);
    /// Calculates the size necessary to store the data in this buffer with
    /// count elements.
    // TODO: For readability added variables, can remove
    static constexpr size_t size(size_t count) {
        size_t init_sz = _SOAStoreVec7::size(count) +
                         _SOAStoreVec3::size(count) * 7 +
                         _SOAStoreQuaterion::size(count) * 2 +
                         _SOAStoreGeneric<float>::size(count) * 3 +
                         _SOAStoreGeneric<bool>::size(count) +
                         _SOAStoreGeneric<apbd::Shape>::size(count) +
                         _SOAStoreGeneric<unsigned int>::size(count);

        size_t gpqp_sz = _SOAStoreVec7::size(count) * 2 +
                         _SOAStoreGeneric<vec6f>::size(count);

        return init_sz + gpqp_sz;
    }

    __host__ __device__ void set(unsigned int index,
                                 const apbd::BodyRigid &data);
};

struct _SOAStoreCollision {
    _SOAStoreGeneric<unsigned int> contactNum;
    _SOAStoreGeneric<bool> broken;
    _SOAStoreGeneric<apbd::BodyReference> body1;
    _SOAStoreGeneric<apbd::BodyReference> body2;

    // GPQP
    _SOAStoreGeneric<unsigned int> mIndices;
    // May not need this
    _SOAStoreGeneric<int> layer;

    _SOAStoreGeneric<mat24x6f> J1I;
    _SOAStoreGeneric<mat24x6f> J2I;
    _SOAStoreGeneric<vec24f> b;
    _SOAStoreGeneric<float> mu;

    _SOAStoreGeneric<vec24f> lambda;
    _SOAStoreGeneric<vec24f> lambdac;
    _SOAStoreGeneric<vec24f> lambdad;
    _SOAStoreGeneric<vec24f> t_bar;
    _SOAStoreGeneric<vec24f> g;
    _SOAStoreGeneric<vec24f> p;
    _SOAStoreGeneric<vec24f> Ax;
    _SOAStoreGeneric<vec24b> freeIndex;

    // CG
    _SOAStoreGeneric<vec24f> r_cg;
    _SOAStoreGeneric<vec24f> b_cg;
    _SOAStoreGeneric<mat24x6f> J1I_cg;
    _SOAStoreGeneric<mat24x6f> J2I_cg;
    _SOAStoreGeneric<vec24f> Minv_cg;
    _SOAStoreGeneric<vec24f> g_cg;
    _SOAStoreGeneric<vec24f> d_cg;

    __host__ __device__ _SOAStoreCollision() {}
    _SOAStoreCollision(byte *data_store, size_t &offset, size_t count);
    /// Calculates the size necessary to store the data in this buffer with
    /// count elements.
    // TODO: For readability added variables, can remove
    static constexpr size_t size(size_t count) {
        size_t init_sz = _SOAStoreGeneric<unsigned int>::size(count) +
                         _SOAStoreGeneric<bool>::size(count) +
                         _SOAStoreGeneric<apbd::BodyReference>::size(count) * 2;

        size_t new_sz = _SOAStoreGeneric<unsigned int>::size(count) +
                        _SOAStoreGeneric<int>::size(count) +
                        _SOAStoreGeneric<mat24x6f>::size(count) * 4 +
                        _SOAStoreGeneric<vec24f>::size(count) * 13 +
                        _SOAStoreGeneric<float>::size(count) * 1 +
                        _SOAStoreGeneric<vec24b>::size(count);

        return init_sz + new_sz;
    }
};

/**
 * Struct-of-Arrays data store. Manually implements data accesses in a single
 * giant global buffer with the optimal alignments, sizes, and data divisions.
 */
class SOAStore {
   public:
    struct _SOAStoreBodyRigid BodyRigid;
    struct _SOAStoreConstraintGround ConstraintGround;
    struct _SOAStoreConstraintRigid ConstraintRigid;
    struct _SOAStoreCollision Collision;

    __host__ __device__ SOAStore() {}
    SOAStore(size_t body_rigid_count, size_t constraint_ground_count,
             size_t constraint_rigid_count, size_t collision_rigid_count,
             size_t scene_count);

    void deallocate();
};

inline SOAStore::SOAStore(size_t body_rigid_count,
                          size_t constraint_ground_count,
                          size_t constraint_rigid_count,
                          size_t collision_rigid_count, size_t scene_count) {
    // Note: this is block size instead of 32 because we are aligning to the
    // number of actual kernel threads, not for memory performance
    const size_t alignment_count = BLOCK_SIZE;
    size_t aligned_scene_count = 0;
    if (scene_count % alignment_count == 0)
        aligned_scene_count = scene_count;
    else
        aligned_scene_count =
            (scene_count / alignment_count + 1) * alignment_count;

    const size_t aligned_body_rigid_count =
        body_rigid_count * aligned_scene_count;
    const size_t aligned_constraint_ground_count =
        constraint_ground_count * aligned_scene_count;
    const size_t aligned_constraint_rigid_count =
        constraint_rigid_count * aligned_scene_count;
    const size_t aligned_collision_rigid_count =
        collision_rigid_count * aligned_scene_count;

    const size_t total_buffer_size =
        _SOAStoreBodyRigid::size(aligned_body_rigid_count) +
        _SOAStoreConstraintGround::size(aligned_constraint_ground_count) +
        _SOAStoreConstraintRigid::size(aligned_constraint_rigid_count) +
        _SOAStoreCollision::size(aligned_collision_rigid_count);
    byte *const data_store = alloc_device<byte>(total_buffer_size);

    size_t offset = 0;
    this->BodyRigid =
        _SOAStoreBodyRigid(data_store, offset, aligned_body_rigid_count);
    this->ConstraintGround = _SOAStoreConstraintGround(
        data_store, offset, aligned_constraint_ground_count);
    this->ConstraintRigid = _SOAStoreConstraintRigid(
        data_store, offset, aligned_constraint_rigid_count);
    this->Collision =
        _SOAStoreCollision(data_store, offset, aligned_collision_rigid_count);
    DEBUG_ASSERT(offset == total_buffer_size, "Allocation of incorrect size!");
}

inline void SOAStore::deallocate() {}

inline _SOAStoreConstraintGround::_SOAStoreConstraintGround(byte *data_store,
                                                            size_t &offset,
                                                            size_t count)
    : lambda(data_store, offset, count),
      nw(data_store, offset, count),
      xl(data_store, offset, count),
      xw(data_store, offset, count),
      d(data_store, offset, count),
      body(data_store, offset, count),
      contactFrame(data_store, offset, count),
      w1(data_store, offset, count),
      delLinVel1(data_store, offset, count),
      angDelta1(data_store, offset, count),
      raXnI1(data_store, offset, count),
      raXn(data_store, offset, count),
      collision(data_store, offset, count) {}

inline _SOAStoreConstraintRigid::_SOAStoreConstraintRigid(byte *data_store,
                                                          size_t &offset,
                                                          size_t count)
    : lambda(data_store, offset, count),
      nw(data_store, offset, count),
      x1(data_store, offset, count),
      x2(data_store, offset, count),
      dlambdaSP(data_store, offset, count),
      d(data_store, offset, count),
      body1(data_store, offset, count),
      body2(data_store, offset, count),
      contactFrame(data_store, offset, count),
      w1(data_store, offset, count),
      delLinVel1(data_store, offset, count),
      angDelta1(data_store, offset, count),
      raXnI1(data_store, offset, count),
      w2(data_store, offset, count),
      delLinVel2(data_store, offset, count),
      angDelta2(data_store, offset, count),
      raXnI2(data_store, offset, count),
      collision(data_store, offset, count) {}

inline _SOAStoreBodyRigid::_SOAStoreBodyRigid(byte *data_store, size_t &offset,
                                              size_t count)
    : position(data_store, offset, count),
      rotation(data_store, offset, count),
      x0(data_store, offset, count),
      collide(data_store, offset, count),
      mu(data_store, offset, count),
      shape(data_store, offset, count),
      density(data_store, offset, count),
      Mr(data_store, offset, count),
      Mp(data_store, offset, count),
      layer(data_store, offset, count),
      v(data_store, offset, count),
      w(data_store, offset, count),
      deltaBody2Worldp(data_store, offset, count),
      deltaBody2Worldq(data_store, offset, count),
      deltaAngDt(data_store, offset, count),
      deltaLinDt(data_store, offset, count),
      dxJacobi(data_store, offset, count),
      dphiJacobi(data_store, offset, count),
      LTx(data_store, offset, count) {}


inline _SOAStoreCollision::_SOAStoreCollision(byte *data_store, size_t &offset,
                                              size_t count)
    : contactNum(data_store, offset, count),
      broken(data_store, offset, count),
      body1(data_store, offset, count),
      body2(data_store, offset, count),
      mIndices(data_store, offset, count),
      layer(data_store, offset, count),
      J1I(data_store, offset, count),
      J2I(data_store, offset, count),
      b(data_store, offset, count),
      mu(data_store, offset, count),
      lambda(data_store, offset, count),
      lambdac(data_store, offset, count),
      lambdad(data_store, offset, count),
      t_bar(data_store, offset, count),
      g(data_store, offset, count),
      p(data_store, offset, count),
      Ax(data_store, offset, count),
      freeIndex(data_store, offset, count),
      r_cg(data_store, offset, count),
      b_cg(data_store, offset, count),
      J1I_cg(data_store, offset, count),
      J2I_cg(data_store, offset, count),
      Minv_cg(data_store, offset, count),
      g_cg(data_store, offset, count),
      d_cg(data_store, offset, count) {}

inline void _SOAStoreBodyRigid::set(unsigned int index,
                                    const apbd::BodyRigid &data) {
    position.set(index, data.x.block<3, 1>(4, 0));
    rotation.set(index, Eigen::Quaternionf(data.x.block<4, 1>(0, 0)));
    x0.set(index, data.x0);
    collide.set(index, data.collide);
    mu.set(index, data.mu);
    shape.set(index, data.shape);
    density.set(index, data.density);
    Mr.set(index, data.Mr);
    Mp.set(index, data.Mp);
    layer.set(index, data.layer);
    v.set(index, data.v);
    w.set(index, data.w);
    deltaBody2Worldp.set(index, Eigen::Vector3f::Zero());
    deltaBody2Worldq.set(index, Eigen::Quaternionf(1.0f, 0.0f, 0.0f, 0.0f));
    deltaAngDt.set(index, Eigen::Vector3f::Zero());
    deltaLinDt.set(index, Eigen::Vector3f::Zero());
    dxJacobi.set(index, data.dxJacobi);
    dphiJacobi.set(index, data.dphiJacobi);
    LTx.set(index, data.LTx);
}

#ifdef __CUDA_ARCH__
#define global_store device_global_store
#else
#define global_store host_global_store
#endif
extern __device__ SOAStore device_global_store;
// dummy for host/device functions
extern SOAStore host_global_store;

}  // namespace data
