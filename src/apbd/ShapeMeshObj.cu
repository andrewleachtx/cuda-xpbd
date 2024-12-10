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

        Eigen::Vector4f xg = Eg.inverse() * xw;
        float r = radius;
        return xg(2) < 1.2f * r;
    }

    // TODO: This one I am a bit unsure on, reference ShapeMeshObj : 125
    __host__ __device__ cuda::std::array<Contact, 8> ShapeMeshObj::narrowphaseGround(const Eigen::Matrix4f E, const Eigen::Matrix4f Eg) const {
        cuda::std::array<Contact, 8> cdata;
        int nverts = V.cols();
        Eigen::Matrix<float, 4, Eigen::Dynamic> xl(4, nverts);
        xl.topRows<3>() = E_io * V;
        xl.row(3).setOnes();
        Eigen::Matrix<float, 4, Eigen::Dynamic> xw = E * xl;
        Eigen::Matrix<float, 4, Eigen::Dynamic> xg = Eg.inverse() * xw;
        Eigen::Matrix<float, 1, Eigen::Dynamic> depth = xg.row(2);
        float maxdepth = depth.minCoeff();

        // find(depth < maxdepth + 5e-2) is an abstracted method to get indices where all values in depth are < 5e-2
        if (maxdepth < 0.2f) {
            std::vector<int> cindices;
            for (size_t i = 0; i < depth.size(); i++) {
                if (depth(i) < maxdepth + 5e-2f) {
                    cindices.push_back(i);
                }
            }

            // if there are more than 8 such indices, create subindices
            if (cindices.size() > 8) {
                std::vector<int> subindices(4, 0);
                // [~, subindices(1)] = min(xg(1, cindices)) means find idx of minimum value in xg(0, i)
                size_t min_idx = 0;
                float min_val = std::numeric_limits<float>::max();
                for (size_t i = 0; i < cindices.size(); i++) {
                    if (xg(0, cindices[i]) < min_val) {
                        min_val = xg(0, cindices[i]);
                        min_idx = i;
                    }
                }
                subindices[0] = min_idx;

                size_t max_idx = 0;
                float max_val = std::numeric_limits<float>::min();
                for (size_t i = 0; i < cindices.size(); i++) {
                    if (xg(1, cindices[i]) > max_val) {
                        max_val = xg(1, cindices[i]);
                        max_idx = i;
                    }
                }
                subindices[1] = max_idx;

                min_idx = 0;
                min_val = std::numeric_limits<float>::max();
                for (size_t i = 0; i < cindices.size(); i++) {
                    if (xg(2, cindices[i]) < min_val) {
                        min_val = xg(2, cindices[i]);
                        min_idx = i;
                    }
                }
                subindices[2] = min_idx;

                max_idx = 0;
                max_val = std::numeric_limits<float>::min();
                for (size_t i = 0; i < cindices.size(); i++) {
                    if (xg(2, cindices[i]) > max_val) {
                        max_val = xg(2, cindices[i]);
                        max_idx = i;
                    }
                }
                subindices[3] = max_idx;

                cindices = subindices;                
            }

            int cdata_count = 0;
            for (int idx : cindices) {
                float d = xg(2, idx);
                Eigen::Vector3f xgproj = xg.col(idx);
                xgproj(2) = 0.0f;
                
                // all vec3f: nw, x1, x2 - it seems like d and vw are ignored even for cuboid Contact case, may need to refactor Contact class
                cdata[cdata_count++] = Contact{
                    Eg.block<3, 1>(0, 2),
                    xl.col(idx).head<3>(),
                    Eg.block<3, 1>(0, 0) * xgproj
                }
            }
        }
    }

    __host__ __device__ bool ShapeMeshObj::broadphaseShape(const Eigen::Matrix4f E1, const Shape &other, const Eigen::Matrix4f E2) const {
        // Must be a ShapeMeshObj as well
        const ShapeMeshObj* other_mesh = dynamic_cast<const ShapeMeshObj*>(&other);
        if (other_mesh == nullptr) {
            throw std::runtime_error("Unsupported shape");
        }

        Eigen::Vector3f p1 = E1.block<3, 1>(0, 3);
        Eigen::Vector3f p2 = E2.block<3, 1>(0, 3);

        float d = (p1 - p2).norm();

        float r1 = radius;
        float r2 = other_mesh->radius;

        return d <= 1.2f * (r1 + r2);
    }

    // TODO: Needs Coal integration
    __host__ __device__ cuda::std::array<Contact, 8> ShapeMeshObj::narrowphaseShape(const Eigen::Matrix4f E1, const Shape &other, const Eigen::Matrix4f E2) const {

    }


} // namespace apbd