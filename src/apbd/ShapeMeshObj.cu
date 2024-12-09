#include "apbd/ShapeMeshObj.h"
#include "apbd/Shape.h"
#include <fstream>
#include <sstream>
#include <stdexcept>


namespace apbd {
    
    __host__ __device__ ShapeMeshObj::ShapeMeshObj() :
        Shape(), F(), V(), E_oi(), E_io(), radius(1.0f) {}

    __host__ __device__ ShapeMeshObj::ShapeMeshObj(const std::string &filename) :
        Shape(), F(), V(), E_oi(), E_io(), radius(1.0f) {
            readOBJ(filename, V, F);
        }

    __host__ __device__ ShapeMeshObj::~ShapeMeshObj() {}

    __host__ __device__ Eigen::Matrix<float, 6, 1> ShapeMeshObj::computeInertia(const float density) {
        /*
            The density argument is optional. We can easily multiply the resulting
            inertia (element-by-element) by any density to get the same result.
        */

        // Call VolInt 
        float T0;
        Eigen::Vector3f T1, T2, TP;
        VolumeIntegration(V, F, T0, T1, T2, TP);

        float mass = density * T0;
        
        // compute center of mass
        Eigen::Vector3f r = T1 / T0;

        // compute inertia tensor
        Eigen::Matrix3f J;
        J(0, 0) = density * (T2(1) + T2(2));
        J(1, 1) = density * (T2(2) + T2(0));
        J(2, 2) = density * (T2(0) + T2(1));
        J(0, 1) = -density * TP(0);
        J(1, 2) = -density * TP(1);
        J(2, 0) = -density * TP(2);
        J(1, 0) = J(0, 1);
        J(2, 1) = J(1, 2);
        J(0, 2) = J(2, 0);

        // translate inertia tensor to center of mass
        J(0, 0) -= mass * (r(1) * r(1) + r(2) * r(2));
        J(1, 1) -= mass * (r(2) * r(2) + r(0) * r(0));
        J(2, 2) -= mass * (r(0) * r(0) + r(1) * r(1));
        J(1, 0) += mass * r(0) * r(1);
        J(2, 1) += mass * r(1) * r(2);
        J(0, 2) += mass * r(2) * r(0);

        // Eigenvalue decomposition to get aligned frame
        Eigen::Matrix4f E = Eigen::Matrix4f::Identity();
        Eigen::Matrix<float, 6, 1> I = Eigen::Matrix<float, 6, 1>::Zero();

        // eig(J) -> [Mat3 JV, Mat3 JD] does eigenvalue decomposition, in Eigen we can use a solver
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(J);
        if (es.info() != Eigen::Success) {
            throw std::runtime_error("eig(J) decomposition failed @ line " + std::to_string(__LINE__));
        }
        
        /*
            JD is the diagonal of eigenvalues, and JV eigenvectors. JV, JD are Mat3 in MATLAB, but in Eigen
            .eigenvalues() returns a Vec3f. We can use .asDiagonal but we actually don't use JD anywhere else

            Also, the line "I(4:6) = mass;" really just sets each value to mass
        */
        Eigen::Matrix3f JV = es.eigenvectors();
        Eigen::Vector3f JD = es.eigenvalues();

        I.head<3>() = JD;
        I.tail<3>() = Eigen::Vector3f::Constant(mass);

        // This sets top left 3x3
        E.block<3, 3>(0, 0) = JV;
        E.block<3, 1>(0, 3) = r;

        // TODO: CONTINUE ADDING FROM LINE 80 IN SHAPEMESHOBJ.M

        // Check for right-handedness
        Eigen::Vector3f x = E.block<3, 1>(0, 0);
        Eigen::Vector3f y = E.block<3, 1>(0, 1);
        Eigen::Vector3f z = E.block<3, 1>(0, 2);
        if (x.cross(y).dot(z) < 0.0f) {
            E.block<3, 1>(0, 2) = -z;
        }

        E_oi = E;
        E_io = E.inverse();

        /*
            V_ is as if you transformed the vertices to the inverse of the E_oi matrix, and then
            took the max of the 3D norm, which is the radius

            We don't know the size so at compile time - use Eigen::Dynamic and for slicing just
            update separately
        */
        int nverts = V.cols();
        Eigen::Matrix<float, 4, Eigen::Dynamic> V_(4, nverts);
        V_.topRows<3>() = V;
        V_.row(3).setOnes();
        V_ = E_io * V_;

        Eigen::Matrix<float, 3, Eigen::Dynamic> V_sliced = V_.topRows<3>();
        radius = 0.0f;
        for (int i = 0; i < V_sliced.cols(); i++) {
            float vecnorm = V_sliced.col(i).norm();
            if (vecnorm > radius) {
                radius = vecnorm;
            }            
        }

        return I;
    }

    __host__ __device__ float ShapeMeshObj::getAxisSize() const {
        return 1.0f;
    }
    
    __host__ __device__ Eigen::Vector3f ShapeMeshObj::toCenterLocal(Eigen::Matrix4f E, Eigen::Vector4f xl) const {
        Eigen::Vector4f xlc = E * Eigen::Vector4f(xl(0), xl(1), xl(2), 1.0f);
        return xlc.head<3>();
    }

    __host__ __device__ bool ShapeMeshObj::broadphaseGround(const Eigen::Matrix4f E, const Eigen::Matrix4f Eg) const {
        Eigen::Vector4f xl(0.0f, 0.0f, 0.0f, 1.0f);
        Eigen::Vector4f xw = E * xl;

        // TODO: Be careful about Eg \ xw 
        Eigen::Vector4f xg = Eg.inverse() * xw;
        float r = radius;
        return xg(2) < 1.2f * r;
    }

    __host__ __device__ cuda::std::array<Contact, 8> ShapeMeshObj::narrowphaseGround(const Eigen::Matrix4f E, const Eigen::Matrix4f Eg) const {
        cuda::std::array<Contact, 8> cdata;
        // TODO: CONTINUE FROM LINE 127 OF SHAPEMESHOBJ.m


    }

    __host__ __device__ bool ShapeMeshObj::broadphaseShape(const Eigen::Matrix4f E1, const Shape &other, const Eigen::Matrix4f E2) const {

    }

    __host__ __device__ cuda::std::array<Contact, 8> ShapeMeshObj::narrowphaseShape(const Eigen::Matrix4f E1, const Shape &other, const Eigen::Matrix4f E2) const {

    }


} // namespace apbd