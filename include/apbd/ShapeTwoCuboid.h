#pragma once

#include "JGT-float/smits_mul.h"
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#include <Eigen/Dense>
#include <cuda/std/array>
#include <cuda/std/utility>

#include "Contact.h"

namespace apbd {

class ShapeTwoCuboid {
public:
    ShapeCuboid cuboid1;
    ShapeCuboid cuboid2;
    Eigen::Matrix4f E1;
    Eigen::Matrix4f E2;

    ShapeTwoCuboid(const Eigen::Vector3f &sides1,
                   const Eigen::Vector3f &sides2,
                   float halfDis,
                   float halfAngle);

    virtual ~ShapeTwoCuboid() {}

    Eigen::Matrix<float, 6, 1> computeInertia(float density) const;

    bool broadphaseGround(const Eigen::Matrix4f &E,
                          const Eigen::Matrix4f &Eg) const;

    cuda::std::pair<cuda::std::array<Contact, 8>, size_t>
    narrowphaseGround(const Eigen::Matrix4f &E,
                      const Eigen::Matrix4f &Eg) const;

    bool broadphaseShape(const Eigen::Matrix4f &E1,
                         const ShapeTwoCuboid &other,
                         const Eigen::Matrix4f &E2) const;

    cuda::std::pair<cuda::std::array<Contact, 8>, size_t>
    narrowphaseShape(const Eigen::Matrix4f &E1,
                     const ShapeTwoCuboid &other,
                     const Eigen::Matrix4f &E2) const;

    Eigen::Vector3f toCenterLocal(const Eigen::Matrix4f &E,
                                  const Eigen::Vector3f &xl) const;
};

} // namespace apbd
