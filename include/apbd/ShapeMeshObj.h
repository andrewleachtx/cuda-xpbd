#pragma once

#include "JGT-float/smits_mul.h"
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#include "Contact.h"
#include <Eigen/Dense>
#include <cuda/std/array>
#include <cuda/std/utility>

#include "Shape.h"

/*
    Inherits from Shape superclass, most notably is an unknown size at compile time. My initial processing
    is to overwrite existing methods that may use the tagged union for effective polymorphism for now,
    so in cases such as computeInertia, instead of relying on the Shape.cu implementation which
    checks enum, I will first just overwrite here.
*/
namespace apbd {
    class ShapeMeshObj : public Shape {
        public:
            // New members are the face, vertices, E_oi, E_io, and radius
            Eigen::Matrix<int, 3, Eigen::Dynamic> F;
            Eigen::Matrix<float, 3, Eigen::Dynamic> V;
            Eigen::Matrix4f E_oi;
            Eigen::Matrix4f E_io;
            float radius;
            std::string filename;

            __host__ __device__ ShapeMeshObj();
            __host__ __device__ ShapeMeshObj(const std::string &filename);
            __host__ __device__ ~ShapeMeshObj();

            __host__ __device__ Eigen::Matrix<float, 6, 1> computeInertia(const float density);
            __host__ __device__ float getAxisSize() const;
            __host__ __device__ Eigen::Vector3f toCenterLocal(const Eigen::Matrix4f E, Eigen::Vector4f xl) const;
            __host__ __device__ bool broadphaseGround(const Eigen::Matrix4f E, const Eigen::Matrix4f Eg) const;
            __host__ __device__ cuda::std::array<Contact, 8> narrowphaseGround(const Eigen::Matrix4f E, const Eigen::Matrix4f Eg) const;
            __host__ __device__ bool broadphaseShape(const Eigen::Matrix4f E1, const Shape &other, const Eigen::Matrix4f E2) const;
            __host__ __device__ cuda::std::array<Contact, 8> narrowphaseShape(const Eigen::Matrix4f E1, const Shape &other, const Eigen::Matrix4f E2) const;

            // Static methods
            static void readOBJ(const std::string &filename, Eigen::Matrix<float, 3, Eigen::Dynamic> &V, Eigen::Matrix<int, 3, Eigen::Dynamic> &F);
            static void VolumeIntegration(const Eigen::Matrix<float, 3, Eigen::Dynamic> &V, const Eigen::Matrix<int, 3, Eigen::Dynamic> &F, float &T0, Eigen::Vector3f &T1, Eigen::Vector3f &T2, Eigen::Vector3f &TP);
    };


} // namespace apbd