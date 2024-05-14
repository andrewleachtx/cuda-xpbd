#pragma once

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#include "apbd/BodyReference.h"
#include "apbd/Collisions.h"
#include <Eigen/Dense>

namespace apbd {

enum CONSTRAINT_TYPE {
  CONSTRAINT_INVALID = 0,
  CONSTRAINT_COLLISION_GROUND,
  CONSTRAINT_COLLISION_RIGID,
  CONSTRAINT_JOINT_REVOLVE,
};

/**
 * Represents a collision with the ground.
 */
struct ConstraintGround {
  Eigen::Vector3f lambda;
  Eigen::Vector3f nw;
  Eigen::Vector3f xl;
  Eigen::Vector3f xw;
  Eigen::Vector3f d;
  BodyRigidReference body;

  Eigen::Matrix3f contactFrame;

  Eigen::Vector3f w1;
  Eigen::Matrix3f delLinVel1;
  Eigen::Matrix3f angDelta1;
  Eigen::Matrix3f raXnI1;

  Collision *collision;

  __host__ __device__ ConstraintGround(BodyRigidReference body, Contact c,
                                       Collision *collision);
};

/**
 * Represents a collision between a rigid object and another rigid object.
 */
struct ConstraintRigid {
  Eigen::Vector3f lambda;
  Eigen::Vector3f nw;
  Eigen::Vector3f x1;
  Eigen::Vector3f x2;
  Eigen::Vector3f dlambdaSP;
  Eigen::Vector3f d;
  BodyRigidReference body1;
  BodyRigidReference body2;

  Eigen::Matrix3f contactFrame;

  Eigen::Vector3f w1;
  Eigen::Matrix3f delLinVel1;
  Eigen::Matrix3f angDelta1;
  Eigen::Matrix3f raXnI1;

  Eigen::Vector3f w2;
  Eigen::Matrix3f delLinVel2;
  Eigen::Matrix3f angDelta2;
  Eigen::Matrix3f raXnI2;

  Collision *collision;
  __host__ __device__ ConstraintRigid(BodyRigidReference body1,
                                      BodyRigidReference body2, Contact c,
                                      Collision *collision);
};

struct ConstraintJointRevolve {
  Eigen::Vector3f C;
  Eigen::Vector3f lambda;
  BodyRigidReference body1;
  BodyRigidReference body2;
  Eigen::Vector4f ql1;
  Eigen::Vector4f pl1;
  Eigen::Vector4f ql2;
  Eigen::Vector4f pl2;

  __host__ __device__ void solve();
};

union ConstraintInner {
  ConstraintGround ground;
  ConstraintRigid rigid;
  ConstraintJointRevolve joint_revolve;
};

class Constraint {
public:
  CONSTRAINT_TYPE type;
  ConstraintInner data;

  __host__ __device__ Constraint(ConstraintRigid rigid);
  __host__ __device__ Constraint(ConstraintGround ground);
  __host__ __device__ Constraint(ConstraintJointRevolve revolve);
  __host__ __device__ Constraint &operator=(const Constraint &);

  __host__ __device__ void init();

  /// Sets C and lambda to 0
  __host__ __device__ void clear();
  __host__ __device__ void applyLambdaSP();

  __host__ __device__ void solve();
};

} // namespace apbd
