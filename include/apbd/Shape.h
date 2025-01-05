#pragma once
#include <Eigen/Dense>
#include <cuda/std/array>
#include <cuda/std/utility>

#include "Contact.h"
#include "ShapeCuboid.h"
#include "ShapeMeshObj.h"
#include "ShapeTwoCuboid.h"
#include "util.h"

namespace apbd {

enum SHAPE_TYPE { SHAPE_CUBOID, SHAPE_MESHOBJ, SHAPE_TWOCUBOID };

union _ShapeInner {
    ShapeCuboid cuboid;
    ShapeMeshObj *meshObj;
    ShapeTwoCuboid twoCuboid;

    _ShapeInner() : cuboid(Eigen::Vector3f(1, 1, 1)) {}
    _ShapeInner(ShapeCuboid cuboid) : cuboid(cuboid) {}
    _ShapeInner(ShapeMeshObj *meshObj) : meshObj(meshObj) {}
    ~_ShapeInner() {}
};

class alignas(16) Shape {
   public:
    _ShapeInner data;
    SHAPE_TYPE type;

    __host__ __device__ Shape() : type(SHAPE_CUBOID) {
        data.cuboid = ShapeCuboid(Eigen::Vector3f(1.0f, 1.0f, 1.0f));
    }
    __host__ __device__ Shape(ShapeCuboid cuboid);
    __host__ __device__ Shape(ShapeMeshObj meshObj);
    __host__ __device__ Shape(ShapeTwoCuboid twoCuboid);
    __host__ __device__ Shape(const Shape &other);
    __host__ __device__ Shape &operator=(const Shape &);
    __host__ __device__ ~Shape() {
        if (type == SHAPE_MESHOBJ && data.meshObj) {
            delete data.meshObj;
        }
    }

    __host__ __device__ bool broadphaseGround(const Eigen::Matrix4f E,
                                              const Eigen::Matrix4f Eg) const;
    __host__ __device__ cdata_t
    narrowphaseGround(const Eigen::Matrix4f E, const Eigen::Matrix4f Eg) const;
    __host__ __device__ bool broadphaseShape(const Eigen::Matrix4f E1,
                                             const Shape &other,
                                             const Eigen::Matrix4f E2) const;
    __host__ __device__ cdata_t
    narrowphaseShape(const Eigen::Matrix4f E1, const Shape &other,
                     const Eigen::Matrix4f E2) const;
    __host__ __device__ Eigen::Matrix<float, 6, 1> computeInertia(
        const float density) const;
};

}  // namespace apbd
