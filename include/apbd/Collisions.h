#pragma once
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#include "apbd/ConstraintReference.h"
#include "apbd/Contact.h"
#include <Eigen/Dense>
#include <cuda/std/array>
#include <cuda/std/utility>

namespace apbd {

struct Collision {
  /// Number of contacts that exist
  unsigned char contactNum;
  /// Indicates whether this collision has been broken;
  /// i.e. the two bodies are no longer colliding
  bool broken;
  /// The first body in the collision
  BodyReference body1;
  /// The second body in the collision. If this is null, this is a ground
  /// collision.
  BodyReference body2;
  /// A stored sequence of contacts between the bodies
  cuda::std::array<Contact, 8> contacts;
  /// References to the constraints created from each contact
  cuda::std::array<ConstraintReference, 8> constraints;

  __host__ __device__ void setContacts(
      const cuda::std::pair<cuda::std::array<Contact, 8>, size_t> cdata);

  __host__ __device__ bool is_ground();

  __host__ __device__ void getConstraints(size_t &ground_count,
                                          size_t &rigid_count);

  __host__ __device__ void solveCollisionNor(float hs, float biasCoeff,
                                             bool withSP);

  __host__ __device__ void solveCollisionTan(float hs, float biasCoeff,
                                             bool withSP);

  __host__ __device__ void applyLambdaSP();

  __host__ __device__ void initConstraints();
};

} // namespace apbd
