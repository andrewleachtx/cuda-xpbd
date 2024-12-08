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

    __host__ __device__ Eigen::Matrix<float, 6, 1> ShapeMeshObj::computeInertia(const float density) const {
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
    

} // namespace apbd