#pragma once
#include "apbd/Body.h"
#include "apbd/Contact.h"
#include "data/utilities.h"
#include "se3/lib.h"
#include "util.h"

namespace apbd {

// Aliases used to prevent errors when expanding macros using types with commas
using NarrowphaseReturn = cuda::std::pair<cuda::std::array<Contact, 8>, size_t>;

/**
 * Defines a function for the BodyReference that calls the implementations on
 * the underlying type
 */
#define IMPLEMENT_DELEGATED_BODY_FUNCTION(signature, call)            \
    signature {                                                       \
        /* printf("thisbody: %u %u", index, type);*/                  \
        DEBUG_ASSERT(type != BODY_INVALID, "recieved invalid body!"); \
        switch (type) {                                               \
            case BODY_RIGID: {                                        \
                auto data = get_rigid();                              \
                return call;                                          \
            }                                                         \
            default: {                                                \
                unreachable();                                        \
            }                                                         \
        }                                                             \
    }

class alignas(8) BodyRigidReference {
   public:
    unsigned int index;

    __host__ __device__ BodyRigidReference(const unsigned int index)
        : index(data::soa_index(index)) {}
    // access the data elements in BodyRigid
    // TODO: remove unnecessary elements

    __host__ __device__ vec7 x() const;
    __host__ __device__ void x(const vec7 new_val);
    __host__ __device__ Eigen::Vector3f position() const;
    __host__ __device__ void position(const Eigen::Vector3f new_val);
    __host__ __device__ Eigen::Quaternionf rotation() const;
    __host__ __device__ void rotation(const Eigen::Quaternionf new_val);
    __host__ __device__ vec7 x0() const;
    __host__ __device__ void x0(const vec7 new_val);
    __host__ __device__ Eigen::Vector3f Mr() const;
    __host__ __device__ void Mr(const Eigen::Vector3f new_val);
    __host__ __device__ float Mp() const;
    __host__ __device__ void Mp(const float new_val);
    __host__ __device__ unsigned int layer() const;
    __host__ __device__ void layer(unsigned int new_val);
    __host__ __device__ Eigen::Vector3f v() const;
    __host__ __device__ void v(const Eigen::Vector3f new_val);
    __host__ __device__ Eigen::Vector3f w() const;
    __host__ __device__ void w(const Eigen::Vector3f new_val);
    __host__ __device__ Eigen::Matrix<float, 6, 1> vw() const;
    __host__ __device__ void vw(const Eigen::Matrix<float, 6, 1> new_val);
    __host__ __device__ Eigen::Vector3f deltaBody2Worldp() const;
    __host__ __device__ void deltaBody2Worldp(const Eigen::Vector3f new_val);
    __host__ __device__ Eigen::Quaternionf deltaBody2Worldq() const;
    __host__ __device__ void deltaBody2Worldq(const Eigen::Quaternionf new_val);
    __host__ __device__ Eigen::Vector3f deltaAngDt() const;
    __host__ __device__ void deltaAngDt(const Eigen::Vector3f new_val);
    __host__ __device__ Eigen::Vector3f deltaLinDt() const;
    __host__ __device__ void deltaLinDt(const Eigen::Vector3f new_val);
    // readonly elements
    __host__ __device__ bool collide() const;
    __host__ __device__ float mu() const;
    __host__ __device__ Shape shape() const;
    __host__ __device__ float density() const;

    // delegated implementations
    __host__ __device__ void init(const vec7 xInit);

    __host__ __device__ void stepBDF1(const float hs,
                                      const Eigen::Vector3f gravity);

    __host__ __device__ void clearShock();

    __host__ __device__ void setInitTransform(const Eigen::Matrix4f transform);

    __host__ __device__ void setInitVelocity(
        const Eigen::Matrix<float, 6, 1> velocity);

    __host__ __device__ bool broadphaseGround(const Eigen::Matrix4f E) const;
    __host__ __device__ NarrowphaseReturn
    narrowphaseGround(const Eigen::Matrix4f E) const;
    __host__ __device__ bool broadphaseRigid(
        const BodyRigidReference other) const;
    __host__ __device__ NarrowphaseReturn
    narrowphaseRigid(const BodyRigidReference other) const;

    __host__ __device__ Eigen::Matrix4f computeTransform() const;

    __host__ __device__ vec7 computeVelocity(const unsigned int step,
                                             const unsigned int substep,
                                             const float hs);
    __host__ __device__ void computeInertiaConst();

    __host__ __device__ Eigen::Vector3f computePointVel(
        const Eigen::Vector3f xl, const float hs) const;
    __host__ __device__ void write_state();

    __host__ __device__ void updateStates(float hs);
    __host__ __device__ void integrateStates();
    __host__ __device__ Eigen::Vector3f transformPoint(Eigen::Vector3f xl);
};

class alignas(8) BodyAffineReference { /* TODO */
};

/**
 * Represents a "pointer" to a body in the global SOAStore.
 *
 * @see Body for the behavior of the Body object itself, which this implements.
 */
class alignas(8) BodyReference {
   public:
    // using 32 bit because it is large enough
    /// The index of the body within this thread's simulation.
    unsigned int index;
    BODY_TYPE type;

    __host__ __device__ BodyReference() {}
    __host__ __device__ BodyReference(const BodyRigidReference &br)
        : index(0), type(BODY_RIGID) {
#ifdef __CUDA_ARCH__
        const size_t scene_count = blockDim.x * gridDim.x;
        index = br.index / scene_count;
#else
        index = br.index / _global_scene_count;
#endif
    }
    __host__ __device__ BodyReference(const unsigned int index,
                                      const BODY_TYPE type)
        : index(index), type(type) {}
    __host__ __device__ BodyRigidReference get_rigid() const {
        return BodyRigidReference(index);
    }
    __host__ __device__ BodyAffineReference get_affine() const {
        return BodyAffineReference(/*TODO*/);
    }

    __host__ __device__ bool operator==(const BodyReference &other) const {
        return this->index == other.index && this->type == other.type;
    }

    IMPLEMENT_DELEGATED_BODY_FUNCTION(__host__ __device__ bool collide() const,
                                      data.collide());
    IMPLEMENT_DELEGATED_BODY_FUNCTION(__host__ __device__ unsigned int layer()
                                          const,
                                      data.layer());
    IMPLEMENT_DELEGATED_BODY_FUNCTION(
        __host__ __device__ void layer(unsigned int new_val),
        data.layer(new_val));
    IMPLEMENT_DELEGATED_BODY_FUNCTION(
        __host__ __device__ void stepBDF1(const float hs,
                                          const Eigen::Vector3f gravity),
        data.stepBDF1(hs, gravity));
    /**
     * Unsets the body's layer
     */
    IMPLEMENT_DELEGATED_BODY_FUNCTION(__host__ __device__ void clearShock(),
                                      data.clearShock());
    IMPLEMENT_DELEGATED_BODY_FUNCTION(__host__ __device__ void setInitTransform(
                                          const Eigen::Matrix4f transform),
                                      data.setInitTransform(transform));
    IMPLEMENT_DELEGATED_BODY_FUNCTION(__host__ __device__ void setInitVelocity(
                                          Eigen::Matrix<float, 6, 1> velocity),
                                      data.setInitVelocity(velocity));

    /**
     * Returns whether this body might be intersecting the ground.
     */
    IMPLEMENT_DELEGATED_BODY_FUNCTION(
        __host__ __device__ bool broadphaseGround(const Eigen::Matrix4f E),
        data.broadphaseGround(E));
    /**
     * Calculates collisions with the ground
     */
    IMPLEMENT_DELEGATED_BODY_FUNCTION(
        __host__ __device__ NarrowphaseReturn
            narrowphaseGround(const Eigen::Matrix4f E),
        data.narrowphaseGround(E));
    /**
     * Returns whether this body might be intersecting the other body.
     */
    // TODO: handle other body types
    IMPLEMENT_DELEGATED_BODY_FUNCTION(
        __host__ __device__ bool broadphaseRigid(const BodyReference other),
        data.broadphaseRigid(other.get_rigid()));
    /**
     * Calculates collisions with the other body
     */
    IMPLEMENT_DELEGATED_BODY_FUNCTION(
        __host__ __device__ NarrowphaseReturn
            narrowphaseRigid(const BodyReference other),
        data.narrowphaseRigid(other.get_rigid()));

    IMPLEMENT_DELEGATED_BODY_FUNCTION(__host__ __device__
                                          Eigen::Matrix4f computeTransform(),
                                      data.computeTransform());
    /**
     * Writes the state of this object out to stdout for visualization. Uses the
     * format:
     * ```
     * {x} {y} {z} r {q.x} {q.y} {q.z} {q.w}
     * ```
     */
    IMPLEMENT_DELEGATED_BODY_FUNCTION(__host__ __device__ void write_state(),
                                      data.write_state());
    IMPLEMENT_DELEGATED_BODY_FUNCTION(
        __host__ __device__ void updateStates(float hs), data.updateStates(hs));
    IMPLEMENT_DELEGATED_BODY_FUNCTION(
        __host__ __device__ void integrateStates(), data.integrateStates());
};

#define NULL_BODY BodyReference(0, BODY_INVALID)

}  // namespace apbd
