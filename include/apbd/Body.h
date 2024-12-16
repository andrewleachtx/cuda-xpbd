#pragma once
#include "apbd/Contact.h"
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
  /// Current position/rotation vector
  vec7 x;
  /// Previous position
  vec7 x0;
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
};

struct BodyAffine {
  const static size_t DOF = 12;
  vec12 xInit;
  vec12 x;
  vec12 x0;
  bool collide;
  float mu;
  unsigned int layer;
  Shape shape;
  float density;
  Eigen::Vector3f Wa;
  float Wp;
};

/**
 * An internal union not meant to be used without the Body class.
 * The `_dummy` member is used for basic invalid initialization.
 */
union _BodyInner {
  int _dummy;
  BodyAffine affine;
  BodyRigid rigid;

  ~_BodyInner() {}
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

  __host__ __device__ void setInitTransform(Eigen::Matrix4f transform);

  __host__ __device__ void setInitVelocity(Eigen::Matrix<float, 6, 1> velocity);
};

} // namespace apbd
