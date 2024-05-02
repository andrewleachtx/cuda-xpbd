#include "util.h"
#include <apbd/Collider.h>
#include <apbd/Constraint.h>
#include <se3/lib.h>

using Eigen::Vector2f, Eigen::Vector3f, Eigen::Vector4f, Eigen::Quaternionf;

namespace apbd {

Constraint::Constraint(ConstraintRigid rigid)
    : type(CONSTRAINT_COLLISION_RIGID), data{.rigid = {rigid}} {}
Constraint::Constraint(ConstraintGround ground)
    : type(CONSTRAINT_COLLISION_GROUND), data{.ground = {ground}} {}
Constraint::Constraint(ConstraintJointRevolve revolve)
    : type(CONSTRAINT_JOINT_REVOLVE), data{.joint_revolve = {revolve}} {}

ConstraintGround::ConstraintGround(BodyRigidReference body, Eigen::Matrix4f Eg,
                                   float d, Eigen::Vector3f xl,
                                   Eigen::Vector3f xw, Eigen::Vector3f nw,
                                   Eigen::Vector3f vw)
    : C(Vector3f::Zero()), lambda(Vector3f::Zero()), nw(nw),
      lambdaSF(Vector3f::Zero()), d(d), body(body), Eg(Eg), xl(xl), xw(xw),
      vw(vw) {}

ConstraintRigid::ConstraintRigid(BodyRigidReference body1,
                                 BodyRigidReference body2, float d,
                                 Eigen::Vector3f nw, Eigen::Vector3f x1,
                                 Eigen::Vector3f x2)
    : C(Vector3f::Zero()), lambda(Vector3f::Zero()), nw(nw),
      lambdaSF(Vector3f::Zero()), d(d), body1(body1), body2(body2), x1(x1),
      x2(x2) {}

Constraint &Constraint::operator=(const Constraint &other) {
  this->type = other.type;
  switch (type) {
  case CONSTRAINT_COLLISION_GROUND: {
    this->data.ground = other.data.ground;
    break;
  }
  case CONSTRAINT_COLLISION_RIGID: {
    this->data.rigid = other.data.rigid;
    break;
  }
  case CONSTRAINT_JOINT_REVOLVE: {
    this->data.joint_revolve = other.data.joint_revolve;
    break;
  }

  default:
    break;
  }
  return *this;
}

void Constraint::init() {
  // TODO: handle joint_revolve
}

void Constraint::clear() {
  switch (type) {
  case CONSTRAINT_COLLISION_GROUND: {
    ConstraintGround *c = &data.ground;
    c->C = Eigen::Vector3f::Zero();
    c->lambda = Eigen::Vector3f::Zero();
    break;
  }
  case CONSTRAINT_COLLISION_RIGID: {
    ConstraintRigid *c = &data.rigid;
    c->C = Eigen::Vector3f::Zero();
    c->lambda = Eigen::Vector3f::Zero();
    break;
  }
  case CONSTRAINT_JOINT_REVOLVE: {
    ConstraintJointRevolve *c = &data.joint_revolve;
    c->C = Eigen::Vector3f::Zero();
    c->lambda = Eigen::Vector3f::Zero();
    break;
  }

  default:
    break;
  }
}

void Constraint::solve(const float hs, const bool doShockProp) {
  switch (type) {
  case CONSTRAINT_COLLISION_GROUND: {
    ConstraintGround *c = &data.ground;
    c->solveNorPos(hs);
    c->applyJacobi();
    break;
  }
  case CONSTRAINT_COLLISION_RIGID: {
    ConstraintRigid *c = &data.rigid;
    c->solveNorPos(hs, doShockProp);
    c->applyJacobi();
    break;
  }
  case CONSTRAINT_JOINT_REVOLVE: {
    ConstraintJointRevolve *c = &data.joint_revolve;
    c->solve();
    break;
  }

  default:
    break;
  }
}

void ConstraintGround::solveNorPos(const float hs) {
  const float penetration_resolution_speed = 0.1;
  const Vector3f v = hs * body.computePointVel(xl, hs) +
                     penetration_resolution_speed * this->d * this->nw;
  const float vNorm = v.norm();
  const Vector3f vNormalized = v / vNorm;
  const Vector3f tx = Eg.block<3, 1>(0, 0);
  const Vector3f ty = Eg.block<3, 1>(0, 1);
  Eigen::Matrix3f frame_tmp;
  frame_tmp << nw, tx, ty;
  const Vector3f vNormalizedContactFrame = frame_tmp.transpose() * vNormalized;

  float dlambda = solvePosDir1(vNorm, vNormalized);
  C = vNorm * vNormalizedContactFrame;

  float dlambdaNor = dlambda * vNormalizedContactFrame(0);
  Vector3f lambda = this->lambda;
  const float lambdaNor = lambda(0) + dlambdaNor;
  if (lambdaNor < 0) {
    dlambdaNor = -lambda(0);
  }
  lambda(0) += dlambdaNor;
  const float mu = this->body.mu();
  Vector2f dlambdaTan = Vector2f::Zero();
  if (mu > 0) {
    const float dlambdaTx = dlambda * vNormalizedContactFrame(1);
    const float dlambdaTy = dlambda * vNormalizedContactFrame(2);
    const float lambdaNorLenMu = mu * lambda(0);
    const Vector2f lambdaTan =
        Vector2f(lambda(1) + dlambdaTx, lambda(2) + dlambdaTy);
    const float lambdaTanLen = lambdaTan.norm();
    dlambdaTan = Vector2f(dlambdaTx, dlambdaTy);
    if (lambdaTanLen > lambdaNorLenMu) {
      dlambdaTan = (lambdaTan / lambdaTanLen * lambdaNorLenMu -
                    Vector2f(lambda(1), lambda(2)));
    }
    lambda.block<2, 1>(1, 0) += dlambdaTan;
    this->lambda = lambda;
  }

  Vector3f frictionalContactLambda =
      Vector3f(dlambdaNor, dlambdaTan(0), dlambdaTan(1));
  dlambda = frictionalContactLambda.norm();
  if (dlambda > 0) {
    // frictionalContactNormal = [this->nw, tx, ty] * frictionalContactLambda ./
    // dlambda;
    Eigen::Matrix3f tmp;
    tmp << nw, tx, ty;
    const Vector3f frictionalContactNormal =
        tmp * frictionalContactLambda / dlambda;
    const vec7 dq = computeDx(dlambda, frictionalContactNormal);
    this->body.dxJacobi(this->body.dxJacobi() + dq);
  }
}

float ConstraintGround::solvePosDir1(const float c,
                                     const Eigen::Vector3f nw) const {
  // Use the provided normal rather than normalizing
  const auto m1 = this->body.Mp();
  const auto I1 = this->body.Mr();
  const Quaternionf q1 = this->body.rotation();
  const Vector3f nl1 = q1.inverse() * nw;
  const Vector3f rl1 = this->xl;
  const Vector3f rnl1 = rl1.cross(nl1);
  const float w1 =
      (1 / m1) + rnl1.transpose() * Vector3f(rnl1.array() / I1.array());
  const float numerator = -c;
  const float denominator = w1;
  return numerator / denominator;
}

vec7 ConstraintGround::computeDx(const float dlambda,
                                 const Eigen::Vector3f nw) const {
  const float m1 = this->body.Mp();
  const Vector3f I1 = this->body.Mr();
  // Position update
  const Vector3f dpw = dlambda * nw;
  const Vector3f dp = dpw / m1;
  // Quaternion update
  const Quaternionf q1 = this->body.x1_0_rot();
  const auto dpl1 = q1.inverse() * dpw;
  const Vector3f q2vec = q1 * (xl.cross(dpl1).array() / I1.array());
  const Quaternionf q2(0, q2vec(0), q2vec(1), q2vec(2));
  const Vector4f dq = 0.5 * (q2 * q1).coeffs();
  return vec7(dq(0), dq(1), dq(2), dq(3), dp(0), dp(1), dp(2));
}

void ConstraintGround::applyJacobi() { this->body.applyJacobi(); }
void ConstraintRigid::applyJacobi() {
  this->body1.applyJacobi();
  this->body2.applyJacobi();
}

void ConstraintRigid::solveNorPos(const float hs, bool shockProp) {
  const float penetration_resolution_speed = 0.1;
  const float allowable_penetration = 1e-3;
  const Vector3f v1w = this->body1.computePointVel(this->x1, hs);
  const Vector3f v2w = this->body2.computePointVel(this->x2, hs);
  const Vector3f v = hs * (v1w - v2w) - penetration_resolution_speed *
                                            (this->d + allowable_penetration) *
                                            this->nw;
  const float vNorm = v.norm();
  const Vector3f vNormalized = v / vNorm;
  Vector3f tx, ty;
  Collider::generateTangents(this->nw, &tx, &ty);
  Eigen::Matrix3f tmp;
  tmp << -this->nw, tx, ty;
  const Vector3f vNormalizedContactFrame = tmp.transpose() * vNormalized;

  float dlambda = this->solvePosDir2(vNorm, vNormalized);
  this->C = vNorm * vNormalizedContactFrame;

  float dlambdaNor = dlambda * vNormalizedContactFrame(0);
  Vector3f lambda = this->lambda;
  const float lambdaNor = lambda(0) + dlambdaNor;
  if (lambdaNor < 0) {
    dlambdaNor = -lambda(0);
  }
  lambda(0) += dlambdaNor;
  const float mu1 = this->body1.mu();
  const float mu2 = this->body2.mu();
  const float mu = 0.5 * (mu1 + mu2);
  Vector2f dlambdaTan{0, 0};
  if (mu > 0) {
    const float dlambdaTx = dlambda * vNormalizedContactFrame(1);
    const float dlambdaTy = dlambda * vNormalizedContactFrame(2);
    const float lambdaNorLenMu = mu * lambda(0);
    const Vector2f lambdaTan{lambda(1) + dlambdaTx, lambda(2) + dlambdaTy};
    const float lambdaTanLen = lambdaTan.norm();
    dlambdaTan = Vector2f(dlambdaTx, dlambdaTy);
    if (lambdaTanLen > lambdaNorLenMu) {
      dlambdaTan = lambdaTan / lambdaTanLen * lambdaNorLenMu -
                   Vector2f(lambda(1), lambda(2));
    }
    lambda.block<2, 1>(1, 0) += dlambdaTan;
    this->lambda = lambda;
  }

  Vector3f frictionalContactLambda;
  frictionalContactLambda << dlambdaNor, dlambdaTan;
  dlambda = frictionalContactLambda.norm();
  if (dlambda > 0) {
    Eigen::Matrix3f tmp;
    tmp << -this->nw, tx, ty;
    const Vector3f frictionalContactNormal =
        tmp * frictionalContactLambda / dlambda;
    Vector4f dq1, dq2;
    Vector3f dp1, dp2;
    this->computeDx(dlambda, frictionalContactNormal, &dq1, &dp1, &dq2, &dp2);
    if (shockProp) {
      this->body1.dxJacobiShock(
          this->body1.dxJacobiShock().block<4, 1>(0, 0) + dq1,
          this->body1.dxJacobiShock().block<3, 1>(4, 0) + dp1);
    } else {
      this->body1.dxJacobi(this->body1.dxJacobi().block<4, 1>(0, 0) + dq1,
                           this->body1.dxJacobi().block<3, 1>(4, 0) + dp1);
    }
    this->body2.dxJacobi(this->body2.dxJacobi().block<4, 1>(0, 0) + dq2,
                         this->body2.dxJacobi().block<3, 1>(4, 0) + dp2);
  }
}
float ConstraintRigid::solvePosDir2(const float c, const Eigen::Vector3f nw) {
  // Use the provided normal rather than normalizing
  const auto m1 = this->body1.Mp();
  const auto m2 = this->body2.Mp();
  const auto I1 = this->body1.Mr();
  const auto I2 = this->body2.Mr();
  const Quaternionf q1 = this->body1.rotation();
  const Quaternionf q2 = this->body2.rotation();
  const Vector3f nl1 = q1.inverse() * nw;
  const Vector3f nl2 = q2.inverse() * nw;
  const Vector3f rl1 = this->x1;
  const Vector3f rl2 = this->x2;
  const Vector3f rnl1 = rl1.cross(nl1);
  const Vector3f rnl2 = rl2.cross(nl2);
  const float w1 =
      (1 / m1) + rnl1.transpose() * Vector3f(rnl1.array() / I1.array());
  const float w2 =
      (1 / m2) + rnl2.transpose() * Vector3f(rnl2.array() / I2.array());
  const float numerator = -c;
  const float denominator = w1 + w2;
  return numerator / denominator;
}

void ConstraintRigid::computeDx(const float dlambda, const Eigen::Vector3f nw,
                                Vector4f *dq1, Vector3f *dp1, Vector4f *dq2,
                                Vector3f *dp2) {
  const auto m1 = this->body1.Mp();
  const auto m2 = this->body2.Mp();
  const auto I1 = this->body1.Mr();
  const auto I2 = this->body2.Mr();
  // Position update
  const Vector3f dpw = dlambda * nw;
  *dp1 = dpw / m1;
  *dp2 = -dpw / m2;
  // Quaternion update
  const Quaternionf q1 = this->body1.x1_0_rot();
  const Quaternionf q2 = this->body2.x1_0_rot();
  const Vector3f dpl1 = q1.inverse() * dpw;
  const Vector3f dpl2 = q2.inverse() * dpw;

  const Vector3f tmp1 = q1 * (this->x1.cross(dpl1).array() / I1.array());
  const Quaternionf qtmp1(0, tmp1.x(), tmp1.y(), tmp1.z());

  const Vector3f tmp2 = q2 * (this->x2.cross(dpl2).array() / I2.array());
  const Quaternionf qtmp2(0, tmp2.x(), tmp2.y(), tmp2.z());

  *dq1 = 0.5 * (qtmp1 * q1).coeffs();
  *dq2 = -0.5 * (qtmp2 * q2).coeffs();
}

void ConstraintJointRevolve::solve() {
  // TODO
}
} // namespace apbd
