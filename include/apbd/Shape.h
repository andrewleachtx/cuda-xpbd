#pragma once
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#include "Contact.h"
#include <Eigen/Dense>
#include <cuda/std/array>
#include <cuda/std/utility>

namespace apbd {

struct ShapeCuboid {
  Eigen::Vector3f sides;

  __host__ __device__ bool
  broadphaseShapeCuboid(const Eigen::Matrix4f E1, const ShapeCuboid &other,
                        const Eigen::Matrix4f E2) const;
  __host__ __device__ cuda::std::pair<cuda::std::array<Contact, 8>, size_t>
  narrowphaseShapeCuboid(const Eigen::Matrix4f E1, const ShapeCuboid &other,
                         const Eigen::Matrix4f E2) const;
  __host__ __device__ float raycast(Eigen::Vector3f x, Eigen::Vector3f n) const;
};

enum SHAPE_TYPE {
  SHAPE_CUBOID,
};

union _ShapeInner {
  ShapeCuboid cuboid;
};

class alignas(16) Shape {
public:
  _ShapeInner data;
  SHAPE_TYPE type;

  // Default to cuboid so we can call Shape() from ShapeMeshObj
  __host__ __device__ Shape() : 
        type(SHAPE_CUBOID), data{.cuboid = ShapeCuboid{Eigen::Vector3f(1, 1, 1)}} {}

  // TODO: handle other shapes
  __host__ __device__ Shape(const Shape &other)
      : type(other.type), data{.cuboid = other.data.cuboid} {}
  __host__ __device__ Shape(ShapeCuboid cuboid);
  __host__ __device__ Shape &operator=(const Shape &);

  __host__ __device__ bool broadphaseGround(const Eigen::Matrix4f E,
                                            const Eigen::Matrix4f Eg) const;
  __host__ __device__ cuda::std::pair<cuda::std::array<Contact, 8>, size_t>
  narrowphaseGround(const Eigen::Matrix4f E, const Eigen::Matrix4f Eg) const;
  __host__ __device__ bool broadphaseShape(const Eigen::Matrix4f E1,
                                           const Shape &other,
                                           const Eigen::Matrix4f E2) const;
  __host__ __device__ cuda::std::pair<cuda::std::array<Contact, 8>, size_t>
  narrowphaseShape(const Eigen::Matrix4f E1, const Shape &other,
                   const Eigen::Matrix4f E2) const;
  __host__ __device__ Eigen::Matrix<float, 6, 1>
  computeInertia(const float density) const;
};

} // namespace apbd
