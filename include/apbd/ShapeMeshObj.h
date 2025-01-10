#pragma once

#include "JGT-float/smits_mul.h"
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#include <Eigen/Dense>
#include <cuda/std/array>
#include <cuda/std/utility>

#include "Contact.h"
#include "util.h"
#include "coal/BVH/BVH_model.h"

/*
    Inherits from Shape superclass, most notably is an unknown size at compile
    time. My initial processing is to overwrite existing methods that may use
   the tagged union for effective polymorphism for now, so in cases such as
    computeInertia, instead of relying on the Shape.cu implementation which
    checks enum, I will first just overwrite here.
*/
namespace apbd {
class ShapeMeshObj {
   public:
    // New members are the face, vertices, E_oi, E_io, and radius
    Eigen::Matrix<int, 3, Eigen::Dynamic> F;
    Eigen::Matrix<float, 3, Eigen::Dynamic> V;
    Eigen::Matrix4f E_oi;
    Eigen::Matrix4f E_io;
    float radius;
    std::string filename;

    // Convex hull decomposition - also cache optimization
    std::vector<std::shared_ptr<coal::ConvexBase> > cv_hulls;

    __host__ ShapeMeshObj();
    __host__ ShapeMeshObj(const std::string &filename);
    __host__ ShapeMeshObj(const ShapeMeshObj &mesh);
    __host__ ~ShapeMeshObj();

    __host__ ShapeMeshObj& operator=(const ShapeMeshObj& other);

    __host__ Eigen::Matrix<float, 6, 1> computeInertia(const float density);
    __host__ float getAxisSize() const;
    __host__ Eigen::Vector3f toCenterLocal(const Eigen::Matrix4f E,
                                           Eigen::Vector4f xl) const;
    __host__ bool broadphaseGround(const Eigen::Matrix4f E,
                                   const Eigen::Matrix4f Eg) const;
    __host__ cdata_t narrowphaseGround(const Eigen::Matrix4f E,
                                       const Eigen::Matrix4f Eg) const;
    __host__ bool broadphaseShapeMesh(const Eigen::Matrix4f E1,
                                      const ShapeMeshObj &other,
                                      const Eigen::Matrix4f E2) const;
    __host__ cdata_t narrowphaseShapeMesh(const Eigen::Matrix4f E1,
                                          const ShapeMeshObj &other,
                                          const Eigen::Matrix4f E2) const;

    // Static methods
    static void readOBJ(const std::string &filename,
                        Eigen::Matrix<float, 3, Eigen::Dynamic> &V,
                        Eigen::Matrix<int, 3, Eigen::Dynamic> &F);
    static void VolumeIntegration(
        const Eigen::Matrix<float, 3, Eigen::Dynamic> &V,
        const Eigen::Matrix<int, 3, Eigen::Dynamic> &F, float &T0,
        Eigen::Vector3f &T1, Eigen::Vector3f &T2, Eigen::Vector3f &TP);
};

}  // namespace apbd