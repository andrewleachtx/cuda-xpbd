#include "apbd/ShapeMeshObj.h"
#include "collideBoxBox/coalMeshMesh.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace apbd {
    
    __host__ __device__ ShapeMeshObj::ShapeMeshObj() :
        F(), V(), E_oi(), E_io(), radius(1.0f) {}

    __host__ __device__ ShapeMeshObj::ShapeMeshObj(const std::string &filename) :
        F(), V(), E_oi(), E_io(), radius(1.0f), filename(filename) {
            readOBJ(filename, this->V, this->F);
        }

    __host__ __device__ ShapeMeshObj::~ShapeMeshObj() {}

    __host__ __device__ Eigen::Matrix<float, 6, 1> ShapeMeshObj::computeInertia(const float density) {
        // Instead of calling readOBJ here I will do it in constructor to initialize V, F
        // readOBJ(filename, this->V, this->F);

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
            // throw std::runtime_error("eig(J) decomposition failed @ line " + std::to_string(__LINE__));
            printf("eig(J) decomposition failed @ line %d\n", __LINE__);
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
    __host__ __device__ cuda::std::pair<cuda::std::array<Contact, 8>, size_t> ShapeMeshObj::narrowphaseGround(const Eigen::Matrix4f E, const Eigen::Matrix4f Eg) const {
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
                };
            }

            return cuda::std::pair<cuda::std::array<Contact, 8>, size_t>(cdata, static_cast<size_t>(cdata_count));
        }

        return cuda::std::pair<cuda::std::array<Contact, 8>, size_t>(cdata, 0);
    }

    __host__ __device__ bool ShapeMeshObj::broadphaseShapeMesh(const Eigen::Matrix4f E1, const ShapeMeshObj &other, const Eigen::Matrix4f E2) const {
        Eigen::Vector3f p1 = E1.block<3, 1>(0, 3);
        Eigen::Vector3f p2 = E2.block<3, 1>(0, 3);

        float d = (p1 - p2).norm();

        float r1 = radius;
        float r2 = other.radius;

        return d <= 1.2f * (r1 + r2);
    }

    __host__ __device__ cuda::std::pair<cuda::std::array<Contact, 8>, size_t> ShapeMeshObj::narrowphaseShapeMesh(const Eigen::Matrix4f E1, const ShapeMeshObj &other, const Eigen::Matrix4f E2) const {
        cuda::std::array<Contact, 8> cdata;
        
        Eigen::Matrix3f R1 = E1.block<3, 3>(0, 0);
        Eigen::Matrix3f R2 = E2.block<3, 3>(0, 0);
        Eigen::Vector3f p1 = E1.block<3, 1>(0, 3);
        Eigen::Vector3f p2 = E2.block<3, 1>(0, 3);

        const auto collisions = coalMeshMesh(E1 * E_io, this->filename, E2 * other.E_io, other.filename);
        const Eigen::Vector3f& nw = collisions.normal;
        for (int i = 0; i < collisions.count; i++) {
            Eigen::Vector3f xw = collisions.positions[i];
            float d = collisions.depths[i];

            // Compute local point on body 1
            Eigen::Vector3f xw1 = xw - 0.5f * nw * d;
            Eigen::Vector3f x1 = R1.transpose() * (xw1 - p1);

            // Compute local point on body 2
            Eigen::Vector3f xw2 = xw + 0.5f * nw * d;
            Eigen::Vector3f x2 = R2.transpose() * (xw2 - p2);

            cdata[i] = Contact{
                nw,
                x1,
                x2
            };
        }
        
        return cuda::std::pair<cuda::std::array<Contact, 8>, size_t>(cdata, static_cast<size_t>(collisions.count));
    }

    static void readOBJ(const std::string &filename, Eigen::Matrix<float, 3, Eigen::Dynamic> &V, Eigen::Matrix<int, 3, Eigen::Dynamic> &F) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Couldn't open OBJ file in readOBJ: " + filename);
        }

        std::vector<Eigen::Vector3f> vertices;
        std::vector<Eigen::Vector3i> faces;

        std::string line;
        while (std::getline(file, line)) {
            // vertex
            {
                float x, y, z;
                int matches = sscanf(line.c_str(), "v %f %f %f", &x, &y, &z);
                if (matches == 3) {
                    vertices.emplace_back(x, y, z);
                    continue;
                }
            }

            // f_v
            {
                int i1, i2, i3;
                int matches = sscanf(line.c_str(), "f %d %d %d", &i1, &i2, &i3);
                if (matches == 3) {
                    faces.emplace_back(i1, i2, i3);
                    continue;
                }
            }

            // f_vt (f i/j i/j i/j)
            {
                int i1, i2, i3;
                int j1, j2, j3;
                int matches = sscanf(line.c_str(), "f %d/%d %d/%d %d/%d", &i1, &j1, &i2, &j2, &i3, &j3);
                if (matches == 6) {
                    faces.emplace_back(i1, i2, i3);
                    continue;
                }
            }

            // f_vn (i//n i//n i//n)
            {
                int i1, i2, i3;
                int n1, n2, n3;
                int matches = sscanf(line.c_str(), "f %d//%d %d//%d %d//%d", &i1, &n1, &i2, &n2, &i3, &n3);
                if (matches == 6) {
                    faces.emplace_back(i1, i2, i3);
                    continue;
                }
            }

            // f_vtn (f i/j/n i/j/n i/j/n)
            {
                int i1, i2, i3;
                int j1, j2, j3;
                int n1, n2, n3;
                int matches = sscanf(line.c_str(), "f %d/%d/%d %d/%d/%d %d/%d/%d", &i1, &j1, &n1, &i2, &j2, &n2, &i3, &j3, &n3);
                if (matches == 9) {
                    faces.emplace_back(i1, i2, i3);
                    continue;
                }
            }
        }

        file.close();

        V.resize(3, vertices.size());
        for (size_t i = 0; i < vertices.size(); i++) {
            V.col(i) = vertices[i];
        }

        // FIXME: Not sure if it really matters whether we are in 1-based or 0-based, can mess with this later
        F.resize(3, faces.size());
        for (size_t i = 0; i < faces.size(); i++) {
            Eigen::Vector3i f = faces[i] - Eigen::Vector3i(1, 1, 1);
            F.col(i) = f;
        }
    }

    // Based on volInt.c https://people.eecs.berkeley.edu/~jfc/mirtich/massProps.html
    static void VolumeIntegration(const Eigen::Matrix<float, 3, Eigen::Dynamic> &V, const Eigen::Matrix<int, 3, Eigen::Dynamic> &F, float &T0, Eigen::Vector3f &T1, Eigen::Vector3f &T2, Eigen::Vector3f &TP) {
        Eigen::Matrix<float, 3, Eigen::Dynamic> Xn = V.transpose();
        Eigen::Matrix<int, 3, Eigen::Dynamic> Triangles = F.transpose();

        // The Tx, Ty, Tz, ..., Tzx stuff is unused, but you could add the floats if needed later
        T0 = 0.0f;
        T1 = Eigen::Vector3f::Zero();
        T2 = Eigen::Vector3f::Zero();
        TP = Eigen::Vector3f::Zero();

        for (size_t i = 0; i < Triangles.rows(); i++) {
            // Compute face normal - the indices are 1-based in MATLAB but we converted to 0 in readOBJ
            Eigen::Vector3i tri = Triangles.row(i);

            // TODO: Is it Xn.row or Xn.col
            Eigen::Vector3f v0 = Xn.row(tri(0));
            Eigen::Vector3f v1 = Xn.row(tri(1));
            Eigen::Vector3f v2 = Xn.row(tri(2));
            Eigen::Vector3f d10 = v1 - v0;
            Eigen::Vector3f d20 = v2 - v0;
            Eigen::Vector3f normal = d10.cross(d20);
            Eigen::Vector3f Normal = normal / normal.norm();
            if (normal.norm() < 1e-9f) {
                continue;
            }

            float nx = std::abs(Normal(0));
            float ny = std::abs(Normal(1));
            float nz = std::abs(Normal(2));
            int C;

            if ((nx > ny) && (nx > nz)) {
                C = 0;
            }
            else if (ny > nz) {
                C = 1;
            }
            else {
                C = 2;
            }

            int A = (C + 1) % 3;
            int B = (A + 1) % 3;
            A += 1;
            B += 1;
            C += 1;

            float N_A(Normal(A - 1)), N_B(Normal(B - 1)), N_C(Normal(C - 1));
            float w = -Normal(0) * Xn(tri(0), 0) - Normal(1) * Xn(tri(0), 1) - Normal(2) * Xn(tri(0), 2);

            float P1(0), Pa(0), Paa(0), Paaa(0), Pb(0), Pbb(0), Pbbb(0), Pab(0), Paab(0), Pabb(0);

            for (int j = 0; j < 3; j++) {
                int curr_idx = j;
                int next_idx = (j+1)%3;

                float a0 = Xn(tri(curr_idx), A-1);
                float b0 = Xn(tri(curr_idx), B-1);
                float a1 = Xn(tri(next_idx), A-1);
                float b1 = Xn(tri(next_idx), B-1);
                float da = a1 - a0;
                float db = b1 - b0;

                float a0_2 = a0*a0;
                float a0_3 = a0_2*a0;
                float a0_4 = a0_3*a0;
                float b0_2 = b0*b0;
                float b0_3 = b0_2*b0;
                float b0_4 = b0_3*b0;
                float a1_2 = a1*a1;
                float a1_3 = a1_2*a1;
                float b1_2 = b1*b1;
                float b1_3 = b1_2*b1;

                float C1 = a1+a0;
                float Ca = a1*C1 + a0_2;
                float Caa = a1*Ca + a0_3;
                float Caaa = a1*Caa + a0_4;
                float Cb = b1*(b1+b0)+b0_2;
                float Cbb = b1*Cb + b0_3;
                float Cbbb = b1*Cbb + b0_4;
                float Cab = 3*a1_2+2*a1*a0+a0_2;
                float Kab = a1_2+2*a1*a0+3*a0_2;
                float Caab = a0*Cab+4*a1_3;
                float Kaab = a1*Kab+4*a0_3;
                float Cabb = 4*b1_3+3*b1_2*b0+2*b1*b0_2+b0_3;
                float Kabb = b1_3+2*b1_2*b0+3*b1*b0_2+4*b0_3;

                P1 += (db*C1);
                Pa += (db*Ca);
                Paa += db*Caa;
                Paaa += db*Caaa;
                Pb += (da*Cb);
                Pbb += da*Cbb;
                Pbbb += da*Cbbb;
                Pab += db*(b1*Cab+b0*Kab);
                Paab += db*(b1*Caab+b0*Kaab);
                Pabb += da*(a1*Cabb+a0*Kabb);
            }

            P1 /= 2.0f;
            Pa /= 6.0f;
            Paa /= 12.0f;
            Paaa /= 20.0f;
            Pb /= -6.0f;
            Pbb /= -12.0f;
            Pbbb /= -20.0f;
            Pab /= 24.0f;
            Paab /= 60.0f;
            Pabb /= -60.0f;

            // float N_A = Normal(A - 1);
            // float N_B = Normal(B - 1);
            // float N_C = Normal(C - 1);

            float k1 = 1.0f / N_C;
            float k2 = k1 * k1;
            float k3 = k2 * k1;
            float k4 = k3 * k1;

            float Fa = k1 * Pa;
            float Fb = k1 * Pb;
            float Fc = -k2 * (N_A * Pa + N_B * Pb + w * P1);

            float Faa = k1 * Paa;
            float Fbb = k1 * Pbb;
            float Fcc = k3 * (N_A * N_A * Paa + 2 * N_A * N_B * Pab + N_B * N_B * Pbb + w * (2 * (N_A * Pa + N_B * Pb) + w * P1));

            float Faaa = k1 * Paaa;
            float Fbbb = k1 * Pbbb;
            float Fccc = -k4 * (N_A * N_A * N_A * Paaa + 3 * N_A * N_A * N_B * Paab + 3 * N_A * N_B * N_B * Pabb + N_B * N_B * N_B * Pbbb + 3 * w * (N_A * N_A * Paa + 2 * N_A * N_B * Pab + N_B * N_B * Pbb) + w * w * (3 * (N_A * Pa + N_B * Pb) + w * P1));

            float Faab = k1 * Paab;
            float Fbbc = -k2 * (N_A * Pabb + N_B * Pbbb + w * Pbb);
            float Fcca = k3 * (N_A * N_A * Paaa + 2 * N_A * N_B * Paab + N_B * N_B * Pabb + w * (2 * (N_A * Paa + N_B * Pab) + w * Pa));
            
            float Part;
            if (A == 1) {
                Part = Fa;
            }
            else if (B == 1) {
                Part = Fb;
            }
            else {
                Part = Fc;
            }

            T0 += Normal(0) * Part;
            T1(A - 1) += N_A * Faa;
            T1(B - 1) += N_B * Fbb;
            T1(C - 1) += N_C * Fcc;
            T2(A - 1) += N_A * Faaa;
            T2(B - 1) += N_B * Fbbb;
            T2(C - 1) += N_C * Fccc;
            TP(A - 1) += N_A * Faab;
            TP(B - 1) += N_B * Fbbc;
            TP(C - 1) += N_C * Fcca;
        }
    }
} // namespace apbd