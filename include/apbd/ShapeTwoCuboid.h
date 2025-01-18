#pragma once

#include "JGT-float/smits_mul.h"
#ifndef EIGEN_DEFAULT_DENSE_INDEX_TYPE
    #define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#endif
#include <Eigen/Dense>
#include <cuda/std/array>
#include <cuda/std/utility>

#include "Contact.h"
#include "ShapeCuboid.h"

namespace apbd {

class ShapeTwoCuboid {
   public:
    ShapeCuboid cuboid1;
    ShapeCuboid cuboid2;
    Eigen::Matrix4f E1;
    Eigen::Matrix4f E2;

    ShapeTwoCuboid(const Eigen::Vector3f &sides1, const Eigen::Vector3f &sides2,
                   float halfDis, float halfAngle);

    virtual ~ShapeTwoCuboid() {}

    __host__ __device__ Eigen::Matrix<float, 6, 1> computeInertia(
        float density) const;

    __host__ __device__ Eigen::Vector3f toCenterLocal(
        const Eigen::Matrix4f &E, const Eigen::Vector3f &xl) const;

    __host__ __device__ bool broadphaseGround(const Eigen::Matrix4f &E,
                                              const Eigen::Matrix4f &Eg) const;

    __host__ __device__ cdata_t narrowphaseGround(
        const Eigen::Matrix4f &E, const Eigen::Matrix4f &Eg) const;

    __host__ __device__ bool broadphaseShape(const Eigen::Matrix4f &E1,
                                             const ShapeTwoCuboid &other,
                                             const Eigen::Matrix4f &E2) const;

    __host__ __device__ cdata_t
    narrowphaseShape(const Eigen::Matrix4f &E1, const ShapeTwoCuboid &other,
                     const Eigen::Matrix4f &E2) const;
};

}  // namespace apbd
