#pragma once
#include "BodyReference.h"
#include "data/soa.h"

namespace apbd {

IMPLEMENT_ACCESS_FUNCTIONS(Eigen::Vector3f, BodyRigidReference, BodyRigid,
                           position)
IMPLEMENT_ACCESS_FUNCTIONS(Eigen::Quaternionf, BodyRigidReference, BodyRigid,
                           rotation)
// Some attributes have an alternate way to get and set them for convenience
inline vec7 BodyRigidReference::x() const {
    auto p = data::global_store.BodyRigid.position.get(index);
    auto q = data::global_store.BodyRigid.rotation.get(index).coeffs();
    return vec7(q(0), q(1), q(2), q(3), p(0), p(1), p(2));
}
inline void BodyRigidReference::x(const vec7 new_val) {
    data::global_store.BodyRigid.rotation.set(
        index, Eigen::Quaternionf(new_val.block<4, 1>(0, 0)));
    data::global_store.BodyRigid.position.set(
        index, Eigen::Vector3f(new_val.block<3, 1>(4, 0)));
}
IMPLEMENT_ACCESS_FUNCTIONS(vec7, BodyRigidReference, BodyRigid, x0)
IMPLEMENT_READONLY_ACCESS_FUNCTIONS(bool, BodyRigidReference, BodyRigid,
                                    collide)
IMPLEMENT_READONLY_ACCESS_FUNCTIONS(float, BodyRigidReference, BodyRigid, mu)
IMPLEMENT_READONLY_ACCESS_FUNCTIONS(Shape, BodyRigidReference, BodyRigid, shape)
IMPLEMENT_READONLY_ACCESS_FUNCTIONS(float, BodyRigidReference, BodyRigid,
                                    density)
IMPLEMENT_ACCESS_FUNCTIONS(Eigen::Vector3f, BodyRigidReference, BodyRigid, Mr)
IMPLEMENT_ACCESS_FUNCTIONS(float, BodyRigidReference, BodyRigid, Mp)
IMPLEMENT_ACCESS_FUNCTIONS(unsigned int, BodyRigidReference, BodyRigid, layer)
// TODO: test making vw the canonical storage method and splitting data into v
// and w
IMPLEMENT_ACCESS_FUNCTIONS(Eigen::Vector3f, BodyRigidReference, BodyRigid, v)
IMPLEMENT_ACCESS_FUNCTIONS(Eigen::Vector3f, BodyRigidReference, BodyRigid, w)
inline Eigen::Matrix<float, 6, 1> BodyRigidReference::vw() const {
    auto v = data::global_store.BodyRigid.v.get(index);
    auto w = data::global_store.BodyRigid.w.get(index);
    return Eigen::Matrix<float, 6, 1>(v(0), v(1), v(2), w(0), w(1), w(2));
}
inline void BodyRigidReference::vw(const Eigen::Matrix<float, 6, 1> new_val) {
    data::global_store.BodyRigid.v.set(
        index, Eigen::Vector3f(new_val.block<3, 1>(0, 0)));
    data::global_store.BodyRigid.w.set(
        index, Eigen::Vector3f(new_val.block<3, 1>(3, 0)));
}
IMPLEMENT_ACCESS_FUNCTIONS(Eigen::Vector3f, BodyRigidReference, BodyRigid,
                           deltaBody2Worldp)
IMPLEMENT_ACCESS_FUNCTIONS(Eigen::Quaternionf, BodyRigidReference, BodyRigid,
                           deltaBody2Worldq)
IMPLEMENT_ACCESS_FUNCTIONS(Eigen::Vector3f, BodyRigidReference, BodyRigid,
                           deltaAngDt)
IMPLEMENT_ACCESS_FUNCTIONS(Eigen::Vector3f, BodyRigidReference, BodyRigid,
                           deltaLinDt)

inline void BodyRigidReference::init(vec7 xInit) {
    this->computeInertiaConst();
    this->position(xInit.block<3, 1>(4, 0));
    this->rotation(Eigen::Quaternionf(xInit.block<4, 1>(0, 0)));

    // printf("Initializing rot with xInit.block<4, 1>(0, 0) aka %f %f %f %f\n",
    //        xInit(0), xInit(1), xInit(2), xInit(3));

    this->x0(xInit);
}

inline void BodyRigidReference::stepBDF1(const float h,
                                         const Eigen::Vector3f gravity) {
    this->x0(this->x());
    auto v = this->v();
    auto q = this->rotation();
    auto p = this->position();
    auto R = q.matrix();

    auto w = this->w();
    Eigen::Vector3f f = Eigen::Vector3f::Zero();
    Eigen::Vector3f t = Eigen::Vector3f::Zero();
    auto m = this->Mp();

    Eigen::Matrix3f diag(Eigen::Matrix3f::Zero());
    Eigen::Vector3f Mr = this->Mr();
    diag(0, 0) = Mr(0);
    diag(1, 1) = Mr(1);
    diag(2, 2) = Mr(2);

    Eigen::Matrix3f I = R * diag * R.transpose();
    Eigen::Vector3f Iw = I * w;
    f = f + m * gravity;
    t = t + Iw.cross(w);
    // Note: this needs to be saved in a Matrix3f before multiplication,
    // otherwise a non-cuda compatible function is called
    Eigen::Matrix3f Iinv = I.inverse();
    w = w + h * (Iinv * t);
    v = v + h * (f / m);
    Eigen::Vector3f sqrtMr = Mr.array().sqrt();
    diag(0, 0) = sqrtMr(0);
    diag(1, 1) = sqrtMr(1);
    diag(2, 2) = sqrtMr(2);

    auto sqrtIntertia = R * diag * R.transpose();

    // printf("m = %f\n", m);
    // printf("f = %f %f %f\n", f(0), f(1), f(2));
    // printf("v = %f %f %f\n", v(0), v(1), v(2));
    // printf("this->w = %f %f %f\n", this->w()(0), this->w()(1), this->w()(2));
    // printf("sqrtInertia = %f %f %f\n", sqrtIntertia(0, 0), sqrtIntertia(1,
    // 1),
    //        sqrtIntertia(2, 2));
    // printf("R = %f %f %f\n", R(0, 0), R(1, 1), R(2, 2));
    // printf("diag = %f %f %f\n", diag(0, 0), diag(1, 1), diag(2, 2));

    this->w(sqrtIntertia * w);
    this->v(v);
}

constexpr unsigned int UNSIGNED_MAX = std::numeric_limits<unsigned int>::max();
inline void BodyRigidReference::clearShock() { this->layer(UNSIGNED_MAX); }

inline bool BodyRigidReference::broadphaseGround(
    const Eigen::Matrix4f Eg) const {
    const Eigen::Matrix4f E = this->computeTransform();
    return this->shape().broadphaseGround(E, Eg);
}
inline NarrowphaseReturn BodyRigidReference::narrowphaseGround(
    const Eigen::Matrix4f Eg) const {
    const Eigen::Matrix4f E = this->computeTransform();
    return this->shape().narrowphaseGround(E, Eg);
}
inline bool BodyRigidReference::broadphaseRigid(
    const BodyRigidReference other) const {
    const Eigen::Matrix4f E1 = this->computeTransform();
    const Eigen::Matrix4f E2 = other.computeTransform();
    return this->shape().broadphaseShape(E1, other.shape(), E2);
}
inline NarrowphaseReturn BodyRigidReference::narrowphaseRigid(
    const BodyRigidReference other) const {
    const Eigen::Matrix4f E1 = this->computeTransform();
    const Eigen::Matrix4f E2 = other.computeTransform();
    return this->shape().narrowphaseShape(E1, other.shape(), E2);
}

inline Eigen::Matrix4f BodyRigidReference::computeTransform() const {
    Eigen::Matrix4f E = Eigen::Matrix4f::Identity();
    E.block<3, 3>(0, 0) = this->rotation().toRotationMatrix();
    E.block<3, 1>(0, 3) = this->position();
    return E;
}

inline vec7 BodyRigidReference::computeVelocity(const unsigned int step,
                                                const unsigned int substep,
                                                const float hs) {
    const vec7 _x = this->x();
    const vec7 _x0 = this->x0();
    vec7 xdot = (_x - _x0);
    this->v(Eigen::Vector3f(xdot.block<3, 1>(4, 0) / hs));
    xdot.block<4, 1>(0, 0) =
        (Eigen::Quaternionf(_x.block<4, 1>(0, 0)) *
         Eigen::Quaternionf(_x0.block<4, 1>(0, 0)).inverse())
            .coeffs();
    Eigen::Vector3f _w(2 * xdot.block<3, 1>(0, 0) / hs);
    if (xdot(3) < 0) {
        _w = -_w;
    }
    this->w(_w);
    return xdot;
}
inline void BodyRigidReference::computeInertiaConst() {
    const auto d = this->density();
    const auto s = this->shape();
    const auto I = s.computeInertia(d);
    this->Mr(I.block<3, 1>(0, 0));

    if (I.x() < 0 || I.y() < 0 || I.z() < 0 || I.hasNaN()) {
        printf(
            "WARNING - NEGATIVE MOMENT OF INERTIA: [%f %f %f] - using "
            "identity, "
            "check your computeInertia!\n",
            I.x(), I.y(), I.z());
        this->Mr(Eigen::Vector3f(1.0, 1.0, 1.0));
    }

    this->Mp(I(4));
}

inline Eigen::Vector3f BodyRigidReference::computePointVel(
    const Eigen::Vector3f xl, const float hs) const {
    Eigen::Vector3f rw = this->rotation() * xl;
    return this->w().cross(rw) + this->v();
}

inline void BodyRigidReference::write_state() {
    auto r = rotation().coeffs();
    printf("%f %f %f r %f %f %f %f", position()(0), position()(1),
           position()(2), r(0), r(1), r(2), r(3));
}

inline void BodyRigidReference::setInitTransform(const Eigen::Matrix4f E) {
    this->rotation(Eigen::Quaternionf(E.block<3, 3>(0, 0)));
    if (this->rotation().coeffs()(3) < 0) {
        this->rotation(Eigen::Quaternionf(-this->rotation().coeffs()));
    }
    this->position(E.block<3, 1>(0, 3));
}

inline void BodyRigidReference::setInitVelocity(
    const Eigen::Matrix<float, 6, 1> velocity) {
    this->v(velocity.block<3, 1>(3, 0));
    this->w(velocity.block<3, 1>(0, 0));
}

inline void BodyRigidReference::updateStates(float hs) {
    const Eigen::Vector4f q = this->x0().block<4, 1>(0, 0);
    auto R = Eigen::Quaternionf(q).matrix();
    Eigen::Matrix3f diag(Eigen::Matrix3f::Zero());
    Eigen::Vector3f Mr = this->Mr();
    Eigen::Vector3f isqrtMr = (1.0 / Mr.array()).sqrt();
    diag(0, 0) = isqrtMr(0);
    diag(1, 1) = isqrtMr(1);
    diag(2, 2) = isqrtMr(2);
    Eigen::Matrix3f invsqrtI = R * diag * R.transpose();
    Eigen::Vector3f angularMotionVel = invsqrtI * this->w();
    float wNorm = angularMotionVel.norm();
    if (wNorm > 1e-9) {
        float halfWDt = 0.5 * wNorm * hs;
        Eigen::Vector3f dqvec = angularMotionVel * sin(halfWDt) / wNorm;
        Eigen::Quaternionf dq(0.0, dqvec(0), dqvec(1), dqvec(2));
        Eigen::Quaternionf _deltaBody2Worldq = this->deltaBody2Worldq();
        Eigen::Vector4f result = (dq * _deltaBody2Worldq).coeffs();
        result += _deltaBody2Worldq.coeffs() * cos(halfWDt);
        this->deltaBody2Worldq(Eigen::Quaternionf(result).normalized());
    }
    this->deltaBody2Worldp(this->deltaBody2Worldp() + this->v() * hs);

    this->deltaAngDt(this->deltaAngDt() + this->w() * hs);
    this->deltaLinDt(this->deltaLinDt() + this->v() * hs);

    this->rotation(this->deltaBody2Worldq() *
                   Eigen::Quaternionf(this->x0().block<4, 1>(0, 0)));
    this->position(this->x0().block<3, 1>(4, 0) + this->deltaBody2Worldp());
}

inline void BodyRigidReference::integrateStates() {
    const Eigen::Quaternionf q =
        Eigen::Quaternionf(this->x0().block<4, 1>(0, 0));
    auto R = q.matrix();
    Eigen::Matrix3f diag(Eigen::Matrix3f::Zero());
    Eigen::Vector3f Mr = this->Mr();
    Eigen::Vector3f isqrtMr = (1.0 / Mr.array()).sqrt();
    diag(0, 0) = isqrtMr(0);
    diag(1, 1) = isqrtMr(1);
    diag(2, 2) = isqrtMr(2);
    Eigen::Matrix3f invsqrtI = R * diag * R.transpose();
    this->w(invsqrtI * this->w());
    this->rotation(this->deltaBody2Worldq() * q);
    this->position(this->x0().block<3, 1>(4, 0) + this->deltaBody2Worldp());
    this->deltaBody2Worldp(Eigen::Vector3f::Zero());
    this->deltaBody2Worldq(Eigen::Quaternionf(1.0, 0.0, 0.0, 0.0));

    this->deltaLinDt(Eigen::Vector3f::Zero());
    this->deltaAngDt(Eigen::Vector3f::Zero());

    this->layer(UNSIGNED_MAX);
    // TODO: clear this neighbors?
}

inline Eigen::Vector3f BodyRigidReference::transformPoint(Eigen::Vector3f xl) {
    return this->rotation() * xl + this->position();
}

}  // namespace apbd
