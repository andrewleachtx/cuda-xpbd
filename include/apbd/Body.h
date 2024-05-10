#pragma once
#include "apbd/Collisions.h"
#include "apbd/Shape.h"
#include <cstddef>
#include <cuda/std/array>

namespace apbd {

// aliases for convenience
typedef Eigen::Matrix<float, 7, 1> vec7;
typedef Eigen::Matrix<float, 12, 1> vec12;

/**
 * The different types a body could be. 0 is reserved for an invalid type to
 * allow for initializing the body to 0 as an uninitialized object.
 */
enum BODY_TYPE {
  BODY_INVALID = 0,
  BODY_AFFINE,
  BODY_RIGID,
};

/**
 * A Rigid body that uses a 3-part position and 4-part rotation in quaternion
 * format. Quaternions are stored in x,y,z,w format.
 */
struct BodyRigid {
  const static size_t DOF = 7;
  /// Initial position. Not used after `init()` is called
  vec7 xInit;
  /// Initial velocity. Not used after the first step velocity is calculated
  vec7 xdotInit;
  /// Current position/rotation vector
  vec7 x;
  /// Previous position
  vec7 x0;
  /// Next position. Used to accumulate changes while a constraint's effect is
  /// being calculated.
  vec7 x1;
  /// The "previous next" position. Used to save the object's rotation after
  /// integration for use in constraints while the next position (x1) is being
  /// calculated
  vec7 x1_0;
  /// The change in x due to constraints
  vec7 dxJacobi;
  /// The change in x due to collision constraints that is being delayed for
  /// shock propagation
  vec7 dxJacobiShock;
  /// Whether or not this object has collision enabled
  bool collide;
  /// The friction coefficient of this body
  float mu;
  /// The collision layer this body is on; used for constraint graph creation
  unsigned int layer;
  /// Shape of the body
  Shape shape;
  /// Density of the body
  float density;
  /// Rotational Inertia
  Eigen::Vector3f Mr;
  /// Mass/Inertia
  float Mp;
  /// Translational velocity
  Eigen::Vector3f v;
  /// Rotational velocity
  Eigen::Vector3f w;

  BodyRigid(Shape shape, float density);
  BodyRigid(Shape shape, float density, bool collide, float mu);

  __host__ __device__ vec7 computeVelocity(unsigned int step,
                                           unsigned int substep, float hs);
  __host__ __device__ void computeInertiaConst();

  __host__ __device__ Eigen::Vector3f computePointVel(Eigen::Vector3f xl,
                                                      float hs);
  __host__ __device__ Eigen::Matrix4f computeTransform();
  __host__ __device__ void applyJacobi();
  /**
   * Ensures the rotation is normalized
   */
  __host__ __device__ void regularize();
};

struct BodyAffine {
  const static size_t DOF = 12;
  vec12 xInit;
  vec12 xdotInit;
  vec12 x;
  vec12 x0;
  vec12 x1;
  vec12 x1_0;
  vec12 dxJacobi;
  vec12 dxJacobiShock;
  bool collide;
  float mu;
  unsigned int layer;
  Shape shape;
  float density;
  Eigen::Vector3f Wa;
  float Wp;

  __host__ __device__ vec12 computeVelocity(unsigned int step,
                                            unsigned int substep, float hs);
  __host__ __device__ void computeInertiaConst();

  /*
   * Can only be called after calling setInitTransform
   */
  __host__ __device__ Eigen::Matrix4f computeInitTransform();
};

/**
 * An internal union not meant to be used without the Body class.
 * The `_dummy` member is used for basic invalid initialization.
 */
union _BodyInner {
  int _dummy;
  BodyAffine affine;
  BodyRigid rigid;
};

/**
 * A container for all body types. Represents a physics object that can interact
 * with other objects, and has some shape.
 */
class Body {
public:
  BODY_TYPE type;
  _BodyInner data;

  Body();
  Body(BodyRigid rigid);
  Body(BodyAffine affine);
  Body &operator=(const apbd::Body &&);

  void init();

  __host__ __device__ bool collide();

  __host__ __device__ void stepBDF1(unsigned int step, unsigned int substep,
                                    float hs, Eigen::Vector3f gravity);

  /**
   * Unsets the body's layer
   */
  __host__ __device__ void clearShock();

  __host__ __device__ void applyJacobiShock();

  __host__ __device__ void regularize();

  __host__ __device__ void setInitTransform(Eigen::Matrix4f transform);

  __host__ __device__ void setInitVelocity(Eigen::Matrix<float, 6, 1> velocity);

  /**
   * Returns whether this body might be intersecting the ground.
   */
  __host__ __device__ bool broadphaseGround(Eigen::Matrix4f E);
  /**
   * Calculates collisions with the ground
   */
  __host__
      __device__ cuda::std::pair<cuda::std::array<CollisionGround, 8>, size_t>
      narrowphaseGround(Eigen::Matrix4f E);
  /**
   * Returns whether this body might be intersecting the other body.
   */
  __host__ __device__ bool broadphaseRigid(Body *other);
  /**
   * Calculates collisions with the other body
   */
  __host__
      __device__ cuda::std::pair<cuda::std::array<CollisionRigid, 8>, size_t>
      narrowphaseRigid(Body *other);

  __host__ __device__ Eigen::Matrix4f computeTransform();

  /**
   * Writes the state of this object out to stdout for visualization. Uses the
   * format:
   * ```
   * {x} {y} {z} r {q.x} {q.y} {q.z} {q.w}
   * ```
   */
  __host__ __device__ void write_state();
};

} // namespace apbd
