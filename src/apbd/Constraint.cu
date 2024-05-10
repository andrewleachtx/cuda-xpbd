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
      x2(x2), shockDv(Vector3f::Zero()), shockDw(Vector3f::Zero()) {}

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

void Constraint::applyLambdaSP() {
  switch (type) {
  case CONSTRAINT_COLLISION_GROUND: {
    ConstraintGround *c = &data.ground;
    c->applyLambdaSP();
    break;
  }
  case CONSTRAINT_COLLISION_RIGID: {
    ConstraintRigid *c = &data.rigid;
    c->applyLambdaSP();
    break;
  }
  case CONSTRAINT_JOINT_REVOLVE: {
    // TODO
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

void Constraint::solve2(const float hs, const float biasCoef,
                        const unsigned int iters, const bool doTanVel,
                        const bool shockProp, const bool doInit) {
  switch (type) {
  case CONSTRAINT_COLLISION_GROUND: {
    ConstraintGround *c = &data.ground;
    c->solve2(hs, biasCoef, iters, doTanVel, doInit);
    break;
  }
  case CONSTRAINT_COLLISION_RIGID: {
    ConstraintRigid *c = &data.rigid;
    c->solve2(hs, biasCoef, iters, doTanVel, shockProp, doInit);
    break;
  }
  case CONSTRAINT_JOINT_REVOLVE: {
    // TODO
    break;
  }

  default:
    break;
  }
}

void ConstraintGround::solve2(const float hs, const float biasCoef,
                              const unsigned int iters, const bool doTanVel,
                              const bool doInit) {
  // initialization

  // cached values
  Vector3f nw = this->nw;
  auto body = this->body;

  Vector3f rl1 = this->xl;
  Vector3f d = body.transformPoint(rl1) - this->xw;
  float scale = min(0.8, biasCoef);
  float biasCoefficient;
  if (nw.transpose() * d) {
    biasCoefficient = -scale / hs;
  } else {
    biasCoefficient = -1 / hs;
  }

  if (doInit) {
    this->lambda = Vector3f::Zero();
  }
  Vector3f lambda = this->lambda;

  Vector3f tx, ty;
  Collider::generateTangents(nw, &tx, &ty);
  Eigen::Matrix3f contactFrame;
  contactFrame << nw, tx, ty;
  float mu = body.mu();

  float m1 = body.Mp();
  Vector3f I1 = body.Mr();
  Quaternionf q1(body.x0().block<4, 1>(0, 0));

  Eigen::Matrix3f raXnI1;
  Vector3f w1;
  Eigen::Matrix3f delLinVel1;
  Eigen::Matrix3f angDelta1;
  for (unsigned int i = 0; i < 3; i++) {
    Vector3f nl1 = q1.inverse() * contactFrame.block<3, 1>(0, i);
    Vector3f rnl1 = rl1.cross(nl1);
    raXnI1.block<3, 1>(0, i) = q1 * (rnl1.array() / I1.array().sqrt());
    w1(i) = (1 / m1) +
            raXnI1.block<3, 1>(0, i).transpose() * raXnI1.block<3, 1>(0, i);

    delLinVel1.block<3, 1>(0, i) = contactFrame.block<3, 1>(0, i) / m1;
    angDelta1.block<3, 1>(0, i) = q1 * (rnl1.array() / I1.array());
  }

  // solving NorPos
  for (unsigned int i = 0; i < iters; i++) {
    float sep =
        float(nw.transpose() * body.deltaLinDt()) +
        float(raXnI1.block<3, 1>(0, 0).transpose() * body.deltaAngDt()) +
        float(nw.transpose() * d);
    float bias = sep * biasCoefficient;

    Vector3f normalVel = nw.array() * body.v().array() +
                         body.w().array() * raXnI1.block<3, 1>(0, 0).array();
    float dlambdaNor = bias / w1(0) - normalVel.sum() / w1(0);
    float nplambda = lambda(0) + dlambdaNor;
    if (nplambda < 0) {
      dlambdaNor = -lambda(0);
      // TODO: this collision has been broken
    }
    lambda(0) = lambda(0) + dlambdaNor;
    body.v(body.v() + dlambdaNor * delLinVel1.block<3, 1>(0, 0));
    body.w(body.w() + dlambdaNor * raXnI1.block<3, 1>(0, 0));
  }

  // solving TanVel
  if (doTanVel) {
    for (unsigned int i = 0; i < iters; i++) {
      Vector2f dlambdaTan = Vector2f::Zero();
      for (unsigned int i = 1; i < 3; i++) {
        float sep =
            float(contactFrame.block<3, 1>(0, i).transpose() *
                  body.deltaLinDt()) +
            float(raXnI1.block<3, 1>(0, i).transpose() * body.deltaAngDt()) +
            float(contactFrame.block<3, 1>(0, i).transpose() * d);
        float bias = sep * biasCoefficient;
        Vector3f normalVel =
            contactFrame.block<3, 1>(0, i).array() * body.v().array() +
            body.w().array() * raXnI1.block<3, 1>(0, i).array();
        dlambdaTan(i - 1) = (bias / w1(i) - normalVel.sum() / w1(i)) * 0.8;
      }
      Vector3f dlambdas(0, dlambdaTan(0), dlambdaTan(1));
      Vector3f lambdas = lambda + dlambdas;
      float frictionRadius = mu * lambdas(0);
      if (lambdas.block<2, 1>(1, 0).norm() > frictionRadius) {
        lambdas.block<2, 1>(1, 0) = frictionRadius * lambdas.block<2, 1>(1, 0) /
                                    lambdas.block<2, 1>(1, 0).norm();
        dlambdas = lambdas - lambda;
        // TODO: this collision has been broken
      }
      lambda = lambda + dlambdas;
      body.v(body.v() + delLinVel1 * dlambdas);
      body.w(body.w() + raXnI1 * dlambdas);
    }
  }
  this->lambda = lambda;
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
  this->C = vNorm * vNormalizedContactFrame;

  float dlambdaNor = dlambda * vNormalizedContactFrame(0);
  Vector3f lambda_local = this->lambda;
  const float lambdaNor = lambda_local(0) + dlambdaNor;
  if (lambdaNor < 0) {
    dlambdaNor = -lambda_local(0);
  }
  lambda_local(0) += dlambdaNor;
  const float mu = this->body.mu();
  Vector2f dlambdaTan = Vector2f::Zero();
  if (mu > 0) {
    const float dlambdaTx = dlambda * vNormalizedContactFrame(1);
    const float dlambdaTy = dlambda * vNormalizedContactFrame(2);
    const float lambdaNorLenMu = mu * lambda_local(0);
    const Vector2f lambdaTan =
        Vector2f(lambda_local(1) + dlambdaTx, lambda_local(2) + dlambdaTy);
    const float lambdaTanLen = lambdaTan.norm();
    dlambdaTan = Vector2f(dlambdaTx, dlambdaTy);
    if (lambdaTanLen > lambdaNorLenMu) {
      dlambdaTan = (lambdaTan / lambdaTanLen * lambdaNorLenMu -
                    Vector2f(lambda_local(1), lambda_local(2)));
    }
    lambda_local.block<2, 1>(1, 0) += dlambdaTan;
    this->lambda = lambda_local;
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

void ConstraintGround::applyLambdaSP() {}
void ConstraintRigid::applyLambdaSP() {
  this->body2.v(this->body2.v() - this->shockDv);
  this->body2.w(this->body2.w() - this->shockDw);
}

void ConstraintRigid::solve2(const float hs, const float biasCoef,
                             const unsigned int iters, const bool doTanVel,
                             const bool shockProp, const bool doInit) {
  // initialize

  // Contact normal always points from body2 to body1
  // TODO: might be redundant??
  if (this->body1.layer() < this->body2.layer()) {
    auto temp = this->body1;
    this->body1 = this->body2;
    this->body2 = temp;

    auto temp2 = this->x1;
    this->x1 = this->x2;
    this->x2 = temp2;
    this->nw = -this->nw;
  }

  // read and cache data to prevent unnecessary reads
  Vector3f nw = this->nw;
  auto body1 = this->body1;
  auto body2 = this->body2;
  // now we can assume body1 is always above body2
  Vector3f d = body1.transformPoint(this->x1) - body2.transformPoint(this->x2);
  float scale = min(0.8, biasCoef);
  float biasCoefficient;
  if (float(nw.transpose() * d) <= 0) {
    biasCoefficient = -scale / hs;
  } else {
    biasCoefficient = -1 / hs;
  }

  if (doInit) {
    this->lambda = Vector3f::Zero();
    this->lambdaSF = Vector3f::Zero();
  }
  Vector3f lambda = this->lambda;
  Vector3f dlambdaSP = this->lambdaSF;

  Vector3f tx, ty;
  Collider::generateTangents(nw, &tx, &ty);
  Eigen::Matrix3f contactFrame;
  contactFrame << nw, tx, ty;
  float mu = 0.5 * (body1.mu() + body2.mu());

  float m1 = body1.Mp();
  Vector3f I1 = body1.Mr();
  Quaternionf q1(body1.x0().block<4, 1>(0, 0));
  Vector3f rl1 = this->x1;

  float m2 = body2.Mp();
  Vector3f I2 = body2.Mr();
  Quaternionf q2(body2.x0().block<4, 1>(0, 0));
  Vector3f rl2 = this->x2;

  Eigen::Matrix3f raXnI1;
  Vector3f w1;
  Eigen::Matrix3f delLinVel1;
  Eigen::Matrix3f angDelta1;
  Eigen::Matrix3f raXnI2;
  Vector3f w2;
  Eigen::Matrix3f delLinVel2;
  Eigen::Matrix3f angDelta2;
  for (unsigned int i = 0; i < 3; i++) {
    Vector3f nl1 = q1.inverse() * contactFrame.block<3, 1>(0, i);
    Vector3f rnl1 = rl1.cross(nl1);
    raXnI1.block<3, 1>(0, i) = q1 * (rnl1.array() / I1.array().sqrt());
    w1(i) = (1 / m1) +
            raXnI1.block<3, 1>(0, i).transpose() * raXnI1.block<3, 1>(0, i);
    delLinVel1.block<3, 1>(0, i) = contactFrame.block<3, 1>(0, i) / m1;
    angDelta1.block<3, 1>(0, i) = q1 * (rnl1.array() / I1.array());

    Vector3f nl2 = q2.inverse() * contactFrame.block<3, 1>(0, i);
    Vector3f rnl2 = rl2.cross(nl2);
    raXnI2.block<3, 1>(0, i) = q2 * (rnl2.array() / I2.array().sqrt());
    w2(i) = (1 / m2) +
            raXnI2.block<3, 1>(0, i).transpose() * raXnI2.block<3, 1>(0, i);
    delLinVel2.block<3, 1>(0, i) = contactFrame.block<3, 1>(0, i) / m2;
    angDelta2.block<3, 1>(0, i) = q2 * (rnl2.array() / I2.array());
  }

  // solve NorPos
  if (!shockProp) {
    for (unsigned int i = 0; i < iters; i++) {
      float sep =
          float(nw.transpose() * body1.deltaLinDt()) +
          float(raXnI1.block<3, 1>(0, 0).transpose() * body1.deltaAngDt()) +
          float(nw.transpose() * d);
      sep -= float(nw.transpose() * body2.deltaLinDt()) +
             float(raXnI2.block<3, 1>(0, 0).transpose() * body2.deltaAngDt());
      float bias = sep * biasCoefficient;

      Vector3f normalVel = nw.array() * body1.v().array() +
                           body1.w().array() * raXnI1.block<3, 1>(0, 0).array();
      normalVel -=
          Eigen::Vector3f(nw.array() * body2.v().array() +
                          body2.w().array() * raXnI2.block<3, 1>(0, 0).array());
      float w = w1(0) + w2(0);
      float dlambdaNor = bias / w - normalVel.sum() / w;
      float nplambda = lambda(0) + dlambdaNor;
      if (nplambda < 0) {
        dlambdaNor = -lambda(0);
        // TODO: this collision has been broken
      }
      lambda(0) = lambda(0) + dlambdaNor;
      body1.v(body1.v() + dlambdaNor * delLinVel1.block<3, 1>(0, 0));
      body1.w(body1.w() + dlambdaNor * raXnI1.block<3, 1>(0, 0));
      body2.v(body2.v() + dlambdaNor * delLinVel2.block<3, 1>(0, 0));
      body2.w(body2.w() + dlambdaNor * raXnI2.block<3, 1>(0, 0));
    }
  } else {
    for (unsigned int i = 0; i < iters; i++) {
      Vector3f normalVel = nw.array() * body1.v().array() +
                           body1.w().array() * raXnI1.block<3, 1>(0, 0).array();
      float dlambdaNor = -normalVel.sum() / w1(0);
      float nplambda = lambda(0) + dlambdaNor;
      if (nplambda < 0) {
        dlambdaNor = -lambda(0);
        // TODO: this collision has been broken
      }
      lambda(0) = lambda(0) + dlambdaNor;
      body1.v(body1.v() + dlambdaNor * delLinVel1.block<3, 1>(0, 0));
      body1.w(body1.w() + dlambdaNor * raXnI1.block<3, 1>(0, 0));
      dlambdaSP(0) += dlambdaNor;
    }
  }

  // solve TanVel
  if (doTanVel) {
    if (!shockProp) {
      for (unsigned int i = 0; i < iters; i++) {
        Vector2f dlambdaTan = Vector2f::Zero();
        for (unsigned int i = 1; i < 3; i++) {
          float sep =
              float(contactFrame.block<3, 1>(0, i).transpose() *
                    body1.deltaLinDt()) +
              float(raXnI1.block<3, 1>(0, i).transpose() * body1.deltaAngDt()) +
              float(contactFrame.block<3, 1>(0, i).transpose() * d);
          sep -=
              float(contactFrame.block<3, 1>(0, i).transpose() *
                    body2.deltaLinDt()) +
              float(raXnI2.block<3, 1>(0, i).transpose() * body2.deltaAngDt());
          float bias = sep * biasCoefficient;
          Vector3f normalVel =
              contactFrame.block<3, 1>(0, i).array() * body1.v().array() +
              body1.w().array() * raXnI1.block<3, 1>(0, i).array();
          normalVel -= Eigen::Vector3f(
              contactFrame.block<3, 1>(0, i).array() * body2.v().array() +
              body2.w().array() * raXnI2.block<3, 1>(0, i).array());
          float w = w1(i) + w2(i);
          dlambdaTan(i - 1) = (bias / w - normalVel.sum() / w) * 0.8;
        }
        Vector3f dlambdas(0, dlambdaTan(0), dlambdaTan(1));
        Vector3f lambdas = lambda + dlambdas;
        float frictionRadius = mu * lambdas(0);
        if (lambdas.block<2, 1>(1, 0).norm() > frictionRadius) {
          lambdas.block<2, 1>(1, 0) = frictionRadius *
                                      lambdas.block<2, 1>(1, 0) /
                                      lambdas.block<2, 1>(1, 0).norm();
          dlambdas = lambdas - lambda;
          // TODO: this collision has been broken
        }
        lambda = lambda + dlambdas;
        body1.v(body1.v() + delLinVel1 * dlambdas);
        body1.w(body1.w() + raXnI1 * dlambdas);
        body2.v(body2.v() + delLinVel2 * dlambdas);
        body2.w(body2.w() + raXnI2 * dlambdas);
      }
    } else {
      for (unsigned int i = 0; i < iters; i++) {
        Vector2f dlambdaTan = Vector2f::Zero();
        for (unsigned int i = 1; i < 3; i++) {
          Vector3f normalVel =
              contactFrame.block<3, 1>(0, i).array() * body1.v().array() +
              body1.w().array() * raXnI1.block<3, 1>(0, i).array();
          dlambdaTan(i - 1) = -normalVel.sum() / w1(0) * 0.8;
        }
        Vector3f dlambdas(0, dlambdaTan(0), dlambdaTan(1));
        Vector3f lambdas = lambda + dlambdas;
        float frictionRadius = mu * lambdas(0);
        if (lambdas.block<2, 1>(1, 0).norm() > frictionRadius) {
          lambdas.block<2, 1>(1, 0) = frictionRadius *
                                      lambdas.block<2, 1>(1, 0) /
                                      lambdas.block<2, 1>(1, 0).norm();
          dlambdas = lambdas - lambda;
          // TODO: this collision has been broken
        }
        lambda = lambda + dlambdas;
        body1.v(body1.v() + delLinVel1 * dlambdas);
        body1.w(body1.w() + raXnI1 * dlambdas);
        dlambdaSP += dlambdas;
      }
    }
  }

  this->lambda = lambda;
  this->lambdaSF = dlambdaSP;
  this->shockDv = delLinVel2 * dlambdaSP;
  this->shockDw = raXnI2 * dlambdaSP;
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
  Vector3f lambda_local = this->lambda;
  const float lambdaNor = lambda_local(0) + dlambdaNor;
  if (lambdaNor < 0) {
    dlambdaNor = -lambda_local(0);
  }
  lambda_local(0) += dlambdaNor;
  const float mu1 = this->body1.mu();
  const float mu2 = this->body2.mu();
  const float mu = 0.5 * (mu1 + mu2);
  Vector2f dlambdaTan{0, 0};
  if (mu > 0) {
    const float dlambdaTx = dlambda * vNormalizedContactFrame(1);
    const float dlambdaTy = dlambda * vNormalizedContactFrame(2);
    const float lambdaNorLenMu = mu * lambda_local(0);
    const Vector2f lambdaTan{lambda_local(1) + dlambdaTx,
                             lambda_local(2) + dlambdaTy};
    const float lambdaTanLen = lambdaTan.norm();
    dlambdaTan = Vector2f(dlambdaTx, dlambdaTy);
    if (lambdaTanLen > lambdaNorLenMu) {
      dlambdaTan = lambdaTan / lambdaTanLen * lambdaNorLenMu -
                   Vector2f(lambda_local(1), lambda_local(2));
    }
    lambda_local.block<2, 1>(1, 0) += dlambdaTan;
    this->lambda = lambda_local;
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
