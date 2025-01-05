#pragma once

#include "JGT-float/smits_mul.h"
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#include <Eigen/Dense>
#include <cuda/std/array>
#include <cuda/std/utility>

#include "Contact.h"
#include "util.h"

namespace apbd {

class ShapeCuboid {
   public:
    Eigen::Vector3f sides;

    ShapeCuboid() : sides(Eigen::Vector3f::Ones()) {}
    ShapeCuboid(Eigen::Vector3f sides) : sides(sides) {}

    __host__ __device__ bool broadphaseGround(const Eigen::Matrix4f &E,
                                              const Eigen::Matrix4f &Eg) const;

    __host__ __device__ cdata_t narrowphaseGround(
        const Eigen::Matrix4f &E, const Eigen::Matrix4f &Eg) const;

    __host__ __device__ bool broadphaseShapeCuboid(
        const Eigen::Matrix4f E1, const ShapeCuboid &other,
        const Eigen::Matrix4f E2) const;

    __host__ __device__ cdata_t
    narrowphaseShapeCuboid(const Eigen::Matrix4f E1, const ShapeCuboid &other,
                           const Eigen::Matrix4f E2) const;
    __host__ __device__ float raycast(Eigen::Vector3f x,
                                      Eigen::Vector3f n) const;
};

}  // namespace apbd