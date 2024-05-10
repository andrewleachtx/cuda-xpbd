#pragma once

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#include "apbd/BodyReference.h"
#include <Eigen/Dense>

namespace apbd {

enum CONSTRAINT_TYPE {
  CONSTRAINT_COLLISION_GROUND,
  CONSTRAINT_COLLISION_RIGID,
  CONSTRAINT_JOINT_REVOLVE,
};

/**
 * Represents a collision with the ground.
 */
struct ConstraintGround {
  // TODO: remove unnecessary data
  Eigen::Vector3f C;
  Eigen::Vector3f lambda;
  Eigen::Vector3f nw;
  Eigen::Vector3f lambdaSF;
  float d;
  BodyRigidReference body;
  Eigen::Matrix4f Eg;
  Eigen::Vector3f xl;
  Eigen::Vector3f xw;
  Eigen::Vector3f vw;

  __host__ __device__ ConstraintGround(BodyRigidReference body,
                                       Eigen::Matrix4f Eg, float d,
                                       Eigen::Vector3f xl, Eigen::Vector3f xw,
                                       Eigen::Vector3f nw, Eigen::Vector3f vw);

  __host__ __device__ vec7
  computeDx(float dlambda, Eigen::Vector3f frictionalContactNormal) const;
  __host__ __device__ float solvePosDir1(float c, Eigen::Vector3f nw) const;
  __host__ __device__ void solveNorPos(float hs);
  __host__ __device__ void solve2(const float hs, const float biasCoef,
                                  const unsigned int iters, const bool doTanVel,
                                  const bool doInit);
  __host__ __device__ void applyJacobi();
  __host__ __device__ void applyLambdaSP();
};

/**
 * Represents a collision between a rigid object and another rigid object.
 */
struct ConstraintRigid {
  // TODO: remove unnecessary data
  Eigen::Vector3f C;
  Eigen::Vector3f lambda;
  Eigen::Vector3f nw;
  Eigen::Vector3f lambdaSF;
  float d;
  BodyRigidReference body1;
  BodyRigidReference body2;
  Eigen::Vector3f x1;
  Eigen::Vector3f x2;

  Eigen::Vector3f shockDv;
  Eigen::Vector3f shockDw;

  __host__ __device__ ConstraintRigid(BodyRigidReference body1,
                                      BodyRigidReference body2, float d,
                                      Eigen::Vector3f nw, Eigen::Vector3f x1,
                                      Eigen::Vector3f x2);

  __host__ __device__ void solveNorPos(float hs, bool doShockProp);
  __host__ __device__ void solve2(const float hs, const float biasCoef,
                                  const unsigned int iters, const bool doTanVel,
                                  const bool shockProp, const bool doInit);
  __host__ __device__ float solvePosDir2(float c, Eigen::Vector3f nw);
  __host__ __device__ void computeDx(float dlambda, Eigen::Vector3f nw,
                                     Eigen::Vector4f *dq1, Eigen::Vector3f *dp1,
                                     Eigen::Vector4f *dq2,
                                     Eigen::Vector3f *dp2);
  __host__ __device__ void applyJacobi();
  __host__ __device__ void applyLambdaSP();
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

  __host__ __device__ void solve(float hs, bool doShockProp);
  __host__ __device__ void solve2(const float hs, const float biasCoef,
                                  const unsigned int iters, const bool doTanVel,
                                  const bool shockProp, const bool doInit);
  __host__ __device__ bool handle_layer(unsigned int layer,
                                        BodyReference *body_layers,
                                        size_t *body_layer_sizes,
                                        size_t &body_count);
};

inline bool Constraint::handle_layer(unsigned int layer,
                                     BodyReference *body_layers,
                                     size_t *body_layer_sizes,
                                     size_t &body_count) {
  // if any bodies affected by this constraint are on `layer-1`,
  //   and the other body is on a higher layer,
  //   then set the other body to this layer, and add it to the layer list
  switch (this->type) {
  case CONSTRAINT_COLLISION_RIGID: {
    auto &data = this->data.rigid;
    unsigned int b1l = data.body1.layer();
    unsigned int b2l = data.body2.layer();
    if (b1l == layer - 1) {
      if (b2l > b1l) {
        data.body2.layer(layer);
        DEBUG_ASSERT(body_count < MAX_LAYER_OBJECTS,
                     "Layer object storage overflow!");
        body_layers[body_count++] = data.body2;
        body_layer_sizes[layer]++;
        return true;
      }
    }
    if (b2l == layer - 1) {
      if (b1l > b1l) {
        data.body1.layer(layer);
        DEBUG_ASSERT(body_count < MAX_LAYER_OBJECTS,
                     "Layer object storage overflow!");
        body_layers[body_count++] = data.body1;
        body_layer_sizes[layer]++;
        return true;
      }
    }
    return false;
  }
  case CONSTRAINT_JOINT_REVOLVE: {
    // TODO
    return false;
  }
  case CONSTRAINT_COLLISION_GROUND:
  default:
    return false;
  }
}

} // namespace apbd
