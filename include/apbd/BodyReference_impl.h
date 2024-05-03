#pragma once
#include "BodyReference.h"
#include "data/soa.h"

namespace apbd {

/**
 * Implements only the getter for an SOAStore stored attribute.
 */
#define IMPLEMENT_READONLY_ACCESS_FUNCTIONS(AttributeType, RefType, Type,      \
                                            attribute)                         \
  inline AttributeType RefType::attribute() const {                            \
    return data::global_store.Type.attribute.get(index);                       \
  }
/**
 * Implements the getter and setter for an SOAStore stored attribute.
 */
#define IMPLEMENT_ACCESS_FUNCTIONS(AttributeType, RefType, Type, attribute)    \
  inline AttributeType RefType::attribute() const {                            \
    return data::global_store.Type.attribute.get(index);                       \
  }                                                                            \
  inline void RefType::attribute(const AttributeType new_val) {                \
    data::global_store.Type.attribute.set(index, new_val);                     \
  }

IMPLEMENT_ACCESS_FUNCTIONS(vec7, BodyRigidReference, BodyRigid, xdotInit)
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
IMPLEMENT_ACCESS_FUNCTIONS(vec7, BodyRigidReference, BodyRigid, x0)
IMPLEMENT_ACCESS_FUNCTIONS(vec7, BodyRigidReference, BodyRigid, x1)
inline void BodyRigidReference::x1(Eigen::Vector4f new_q,
                                   Eigen::Vector3f new_p) {
  data::global_store.BodyRigid.x1.set(index, vec7(new_q(0), new_q(1), new_q(2),
                                                  new_q(3), new_p(0), new_p(1),
                                                  new_p(2)));
}
IMPLEMENT_ACCESS_FUNCTIONS(Eigen::Quaternionf, BodyRigidReference, BodyRigid,
                           x1_0_rot)
IMPLEMENT_ACCESS_FUNCTIONS(vec7, BodyRigidReference, BodyRigid, dxJacobi)
inline void BodyRigidReference::dxJacobi(Eigen::Vector4f new_q,
                                         Eigen::Vector3f new_p) {
  data::global_store.BodyRigid.dxJacobi.set(
      index, vec7(new_q(0), new_q(1), new_q(2), new_q(3), new_p(0), new_p(1),
                  new_p(2)));
}
IMPLEMENT_ACCESS_FUNCTIONS(vec7, BodyRigidReference, BodyRigid, dxJacobiShock)
inline void BodyRigidReference::dxJacobiShock(Eigen::Vector4f new_q,
                                              Eigen::Vector3f new_p) {
  data::global_store.BodyRigid.dxJacobiShock.set(
      index, vec7(new_q(0), new_q(1), new_q(2), new_q(3), new_p(0), new_p(1),
                  new_p(2)));
}
IMPLEMENT_READONLY_ACCESS_FUNCTIONS(bool, BodyRigidReference, BodyRigid,
                                    collide)
IMPLEMENT_READONLY_ACCESS_FUNCTIONS(float, BodyRigidReference, BodyRigid, mu)
IMPLEMENT_READONLY_ACCESS_FUNCTIONS(Shape, BodyRigidReference, BodyRigid, shape)
IMPLEMENT_READONLY_ACCESS_FUNCTIONS(float, BodyRigidReference, BodyRigid,
                                    density)
IMPLEMENT_ACCESS_FUNCTIONS(Eigen::Vector3f, BodyRigidReference, BodyRigid, Mr)
IMPLEMENT_ACCESS_FUNCTIONS(float, BodyRigidReference, BodyRigid, Mp)
IMPLEMENT_ACCESS_FUNCTIONS(unsigned int, BodyRigidReference, BodyRigid, layer)

inline void BodyRigidReference::init(vec7 xInit) {
  this->computeInertiaConst();
  this->position(xInit.block<3, 1>(4, 0));
  this->rotation(Eigen::Quaternionf(xInit.block<4, 1>(0, 0)));
  this->x0(xInit);
}

inline void BodyRigidReference::stepBDF1(const unsigned int step,
                                         const unsigned int substep,
                                         const float hs,
                                         const Eigen::Vector3f gravity) {
  const auto xdot = this->computeVelocity(step, substep, hs);
  Eigen::Vector4f qdot = xdot.block<4, 1>(0, 0);
  Eigen::Vector3f v = xdot.block<3, 1>(4, 0); // pdot
  this->x0(this->x());
  Eigen::Vector4f q = this->rotation().coeffs();
  Eigen::Vector3f p = this->position();
  auto w = se3::qdotToW(q, qdot); // angular velocity in body coords
  Eigen::Vector3f f =
      Eigen::Vector3f::Zero(); // translational force in world space
  Eigen::Vector3f t = Eigen::Vector3f::Zero(); // angular torque in body space
  const auto m = this->Mp();                   // scalar mass
  const auto I = this->Mr();                   // inertia in body space
  const Eigen::Vector3f Iw =
      I.array() * w.array(); // angular momentum in body space
  f = f + m * gravity;       // Gravity
  t = t + Iw.cross(w);       // Coriolis
  // Integrate velocities
  w = w + hs * Eigen::Vector3f(t.array() / I.array());
  v = v + hs * (f / m);
  qdot = se3::wToQdot(q, w);
  // Integrate positions
  q = q + hs * qdot;
  p = p + hs * v;
  q = q / q.norm();
  this->rotation(Eigen::Quaternionf(q));
  this->position(p);
  this->x1_0_rot(this->rotation());
  this->x1(this->rotation().coeffs(), this->position());
}

constexpr unsigned int UNSIGNED_MAX = std::numeric_limits<unsigned int>::max();
inline void BodyRigidReference::clearShock() {
  this->layer(UNSIGNED_MAX);
  this->dxJacobiShock(vec7::Zero());
}

inline void BodyRigidReference::applyJacobiShock() {
  this->x1(this->x1() + this->dxJacobiShock());
  this->dxJacobiShock(vec7::Zero());
}

inline void BodyRigidReference::regularize() {
  const auto x1_ = this->x1();
  this->position(x1_.block<3, 1>(4, 0));
  Eigen::Vector4f q = x1_.block<4, 1>(0, 0);
  q /= q.norm();
  this->rotation(Eigen::Quaternionf(q));
}

inline bool
BodyRigidReference::broadphaseGround(const Eigen::Matrix4f Eg) const {
  const Eigen::Matrix4f E = this->computeTransform();
  return this->shape().broadphaseGround(E, Eg);
}
inline GroundNarrowphaseReturn
BodyRigidReference::narrowphaseGround(const Eigen::Matrix4f Eg) const {
  const Eigen::Matrix4f E = this->computeTransform();
  return this->shape().narrowphaseGround(E, Eg);
}
inline bool
BodyRigidReference::broadphaseRigid(const BodyRigidReference other) const {
  const Eigen::Matrix4f E1 = this->computeTransform();
  const Eigen::Matrix4f E2 = other.computeTransform();
  return this->shape().broadphaseShape(E1, other.shape(), E2);
}
inline NarrowphaseReturn
BodyRigidReference::narrowphaseRigid(const BodyRigidReference other) const {
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
                                                const float hs) const {
  if (step == 0 && substep == 0)
    return this->xdotInit();
  else
    return (this->x() - this->x0()) / hs;
}
inline void BodyRigidReference::computeInertiaConst() {
  const auto d = this->density();
  const auto s = this->shape();
  const auto I = s.computeInertia(d);
  this->Mr(I.block<3, 1>(0, 0));
  this->Mp(I(4));
}

inline Eigen::Vector3f
BodyRigidReference::computePointVel(const Eigen::Vector3f xl,
                                    const float hs) const {
  const vec7 xdot = (this->x() - this->x0()) / hs;
  const Eigen::Vector4f qdot = xdot.block<4, 1>(0, 0);
  const Eigen::Vector3f pdot = xdot.block<3, 1>(4, 0); // in world coords
  const Eigen::Quaternionf q = this->rotation();
  const Eigen::Vector3f w =
      se3::qdotToW(q.coeffs(), qdot); // angular velocity in body coords
  return (q * w.cross(xl)) + pdot;
}

inline void BodyRigidReference::applyJacobi() {
  this->x1(this->x1() + this->dxJacobi());
  this->dxJacobi(vec7::Zero());
  this->regularize();
}

inline void BodyRigidReference::write_state() {
  auto r = rotation().coeffs();
  printf("%f %f %f r %f %f %f %f", position()(0), position()(1), position()(2),
         r(0), r(1), r(2), r(3));
}

inline void BodyRigidReference::setInitTransform(const Eigen::Matrix4f E) {
  this->rotation(Eigen::Quaternionf(E.block<3, 3>(0, 0)));
  if (this->rotation().coeffs()(3) < 0) {
    this->rotation(Eigen::Quaternionf(-this->rotation().coeffs()));
  }
  this->position(E.block<3, 1>(0, 3));
}

} // namespace apbd
