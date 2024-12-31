#pragma once
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#include <Eigen/Dense>

namespace se3 {
const float THRESH = 1e-9;

__host__ __device__ Eigen::Vector3f qdotToW(Eigen::Vector4f q,
                                            Eigen::Vector4f qdot);

__host__ __device__ Eigen::Vector4f wToQdot(Eigen::Vector4f q,
                                            Eigen::Vector3f w);

__host__ __device__ Eigen::Matrix3f aaToMat(Eigen::Vector3f axis, float angle);

__host__ __device__ Eigen::Matrix4f brac(Eigen::Matrix<float, 6, 1> x);

/**
 * Gets the diagonal inertia of a cuboid with (width, height, depth)
 */
__host__ __device__ Eigen::Matrix<float, 6, 1> inertiaCuboid(
    Eigen::Vector3f whd, float density);

__host__ __device__ Eigen::Matrix4f inv(Eigen::Matrix4f E);

/*
    Returns inverted transform matrix of E
*/
inline Eigen::Matrix4f inv(Eigen::Matrix4f E) {
    Eigen::Matrix3f R = E.block<3, 3>(0, 0);
    Eigen::Vector3f p = E.block<3, 1>(0, 3);

    // Ei = [R', -R' * p; 0 0 0 1];
    Eigen::Matrix4f Ei = Eigen::Matrix4f::Identity();
    Ei.block<3, 3>(0, 0) = R.transpose();
    Ei.block<3, 1>(0, 3) = -R.transpose() * p;
    return Ei;
}

inline Eigen::Vector3f qdotToW(Eigen::Vector4f q, Eigen::Vector4f qdot) {
    // https://ahrs.readthedocs.io/en/latest/filters/angular.html#quaternion-derivative
    // Q = [
    // 	 q(3)  q(2) -q(1) -q(0)
    // 	-q(2)  q(3)  q(0) -q(1)
    // 	 q(1) -q(0)  q(3) -q(2)
    // 	];
    // w = 2*Q*qdot;
    return 2 * Eigen::Vector3f(q(3) * qdot(0) + q(2) * qdot(1) -
                                   q(1) * qdot(2) - q(0) * qdot(3),
                               -q(2) * qdot(0) + q(3) * qdot(1) +
                                   q(0) * qdot(2) - q(1) * qdot(3),
                               q(1) * qdot(0) - q(0) * qdot(1) +
                                   q(3) * qdot(2) - q(2) * qdot(3));
}

inline Eigen::Vector4f wToQdot(Eigen::Vector4f q, Eigen::Vector3f w) {
    // https://ahrs.readthedocs.io/en/latest/filters/angular.html#quaternion-derivative
    // W = [
    // 	    0  w(2) -w(1)  w(0)
    // 	-w(2)     0  w(0)  w(1)
    // 	 w(1) -w(0)     0  w(2)
    // 	-w(0) -w(1) -w(2)     0
    // 	];
    // qdot = 0.5*W*q;
    return 0.5 * Eigen::Vector4f(w(2) * q(1) - w(1) * q(2) + w(0) * q(3),
                                 -w(2) * q(0) + w(0) * q(2) + w(1) * q(3),
                                 w(1) * q(0) - w(0) * q(1) + w(2) * q(3),
                                 -w(0) * q(0) - w(1) * q(1) - w(2) * q(2));
}

inline Eigen::Matrix3f aaToMat(Eigen::Vector3f axis, float angle) {
    // Create a rotation matrix from an (axis,angle) pair
    // From vecmath
    Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
    auto ax = axis(0);
    auto ay = axis(1);
    auto az = axis(2);
    auto mag = sqrt(ax * ax + ay * ay + az * az);
    if (mag > se3::THRESH) {
        mag = 1.0 / mag;
        ax = ax * mag;
        ay = ay * mag;
        az = az * mag;
        if (abs(ax) < se3::THRESH && abs(ay) < se3::THRESH) {
            // Rotation about Z
            if (az < 0) {
                angle = -angle;
            }
            auto sinTheta = sin(angle);
            auto cosTheta = cos(angle);
            R(0, 0) = cosTheta;
            R(0, 1) = -sinTheta;
            R(1, 0) = sinTheta;
            R(1, 1) = cosTheta;
        } else if (abs(ay) < se3::THRESH && abs(az) < se3::THRESH) {
            // Rotation about X
            if (ax < 0) {
                angle = -angle;
            }
            auto sinTheta = sin(angle);
            auto cosTheta = cos(angle);
            R(1, 1) = cosTheta;
            R(1, 2) = -sinTheta;
            R(2, 1) = sinTheta;
            R(2, 2) = cosTheta;
        } else if (abs(az) < se3::THRESH && abs(ax) < se3::THRESH) {
            // Rotation about Y
            if (ay < 0) {
                angle = -angle;
            }
            auto sinTheta = sin(angle);
            auto cosTheta = cos(angle);
            R(0, 0) = cosTheta;
            R(0, 2) = sinTheta;
            R(2, 0) = -sinTheta;
            R(2, 2) = cosTheta;
        } else {
            // General rotation
            auto sinTheta = sin(angle);
            auto cosTheta = cos(angle);
            auto t = 1.0 - cosTheta;
            auto xz = ax * az;
            auto xy = ax * ay;
            auto yz = ay * az;
            R(0, 0) = t * ax * ax + cosTheta;
            R(0, 1) = t * xy - sinTheta * az;
            R(0, 2) = t * xz + sinTheta * ay;
            R(1, 0) = t * xy + sinTheta * az;
            R(1, 1) = t * ay * ay + cosTheta;
            R(1, 2) = t * yz - sinTheta * ax;
            R(2, 0) = t * xz - sinTheta * ay;
            R(2, 1) = t * yz + sinTheta * ax;
            R(2, 2) = t * az * az + cosTheta;
        }
    }
    return R;
}

inline Eigen::Matrix4f brac(Eigen::Matrix<float, 6, 1> x) {
    Eigen::Matrix4f S = Eigen::Matrix4f::Zero();
    Eigen::Matrix3f tmp;
    tmp << 0, -x(2), x(1), x(2), 0, -x(0), -x(1), x(0), 0;
    S.block<3, 3>(0, 0) = tmp;
    S.block<3, 1>(0, 3) = Eigen::Vector3f(x(3), x(4), x(5));
    return S;
}

inline Eigen::Matrix<float, 6, 1> inertiaCuboid(Eigen::Vector3f whd,
                                                float density) {
    Eigen::Matrix<float, 6, 1> m = Eigen::Matrix<float, 6, 1>::Zero();
    float mass = density * whd.prod();
    m(0) = (1. / 12.) * mass * Eigen::Vector2f(whd(1), whd(2)).transpose() *
           Eigen::Vector2f(whd(1), whd(2));
    m(1) = (1. / 12.) * mass * Eigen::Vector2f(whd(2), whd(0)).transpose() *
           Eigen::Vector2f(whd(2), whd(0));
    m(2) = (1. / 12.) * mass * Eigen::Vector2f(whd(0), whd(1)).transpose() *
           Eigen::Vector2f(whd(0), whd(1));
    m(3) = mass;
    m(4) = mass;
    m(5) = mass;
    return m;
}

}  // namespace se3
