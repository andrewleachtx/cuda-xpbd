#pragma once
#include "Collider.h"
#include "ConstraintReference.h"
#include "data/soa.h"

namespace apbd {

IMPLEMENT_ACCESS_FUNCTIONS(Eigen::Vector3f, ConstraintGroundReference,
                           ConstraintGround, lambda)
IMPLEMENT_ACCESS_FUNCTIONS(Eigen::Vector3f, ConstraintGroundReference,
                           ConstraintGround, nw)
IMPLEMENT_ACCESS_FUNCTIONS(Eigen::Vector3f, ConstraintGroundReference,
                           ConstraintGround, xl)
IMPLEMENT_ACCESS_FUNCTIONS(Eigen::Vector3f, ConstraintGroundReference,
                           ConstraintGround, xw)
IMPLEMENT_ACCESS_FUNCTIONS(Eigen::Vector3f, ConstraintGroundReference,
                           ConstraintGround, d)
IMPLEMENT_ACCESS_FUNCTIONS(BodyRigidReference, ConstraintGroundReference,
                           ConstraintGround, body)

IMPLEMENT_ACCESS_FUNCTIONS(Eigen::Matrix3f, ConstraintGroundReference,
                           ConstraintGround, contactFrame)

IMPLEMENT_ACCESS_FUNCTIONS(Eigen::Vector3f, ConstraintGroundReference,
                           ConstraintGround, w1)
IMPLEMENT_ACCESS_FUNCTIONS(Eigen::Matrix3f, ConstraintGroundReference,
                           ConstraintGround, delLinVel1)
IMPLEMENT_ACCESS_FUNCTIONS(Eigen::Matrix3f, ConstraintGroundReference,
                           ConstraintGround, angDelta1)
IMPLEMENT_ACCESS_FUNCTIONS(Eigen::Matrix3f, ConstraintGroundReference,
                           ConstraintGround, raXnI1)

IMPLEMENT_ACCESS_FUNCTIONS(CollisionReference, ConstraintGroundReference,
                           ConstraintGround, collision)

IMPLEMENT_ACCESS_FUNCTIONS(Eigen::Vector3f, ConstraintRigidReference,
                           ConstraintRigid, lambda)
IMPLEMENT_ACCESS_FUNCTIONS(Eigen::Vector3f, ConstraintRigidReference,
                           ConstraintRigid, nw)
IMPLEMENT_ACCESS_FUNCTIONS(Eigen::Vector3f, ConstraintRigidReference,
                           ConstraintRigid, x1)
IMPLEMENT_ACCESS_FUNCTIONS(Eigen::Vector3f, ConstraintRigidReference,
                           ConstraintRigid, x2)
IMPLEMENT_ACCESS_FUNCTIONS(Eigen::Vector3f, ConstraintRigidReference,
                           ConstraintRigid, dlambdaSP)
IMPLEMENT_ACCESS_FUNCTIONS(Eigen::Vector3f, ConstraintRigidReference,
                           ConstraintRigid, d)
IMPLEMENT_ACCESS_FUNCTIONS(BodyRigidReference, ConstraintRigidReference,
                           ConstraintRigid, body1)
IMPLEMENT_ACCESS_FUNCTIONS(BodyRigidReference, ConstraintRigidReference,
                           ConstraintRigid, body2)

IMPLEMENT_ACCESS_FUNCTIONS(Eigen::Matrix3f, ConstraintRigidReference,
                           ConstraintRigid, contactFrame)

IMPLEMENT_ACCESS_FUNCTIONS(Eigen::Vector3f, ConstraintRigidReference,
                           ConstraintRigid, w1)
IMPLEMENT_ACCESS_FUNCTIONS(Eigen::Matrix3f, ConstraintRigidReference,
                           ConstraintRigid, delLinVel1)
IMPLEMENT_ACCESS_FUNCTIONS(Eigen::Matrix3f, ConstraintRigidReference,
                           ConstraintRigid, angDelta1)
IMPLEMENT_ACCESS_FUNCTIONS(Eigen::Matrix3f, ConstraintRigidReference,
                           ConstraintRigid, raXnI1)

IMPLEMENT_ACCESS_FUNCTIONS(Eigen::Vector3f, ConstraintRigidReference,
                           ConstraintRigid, w2)
IMPLEMENT_ACCESS_FUNCTIONS(Eigen::Matrix3f, ConstraintRigidReference,
                           ConstraintRigid, delLinVel2)
IMPLEMENT_ACCESS_FUNCTIONS(Eigen::Matrix3f, ConstraintRigidReference,
                           ConstraintRigid, angDelta2)
IMPLEMENT_ACCESS_FUNCTIONS(Eigen::Matrix3f, ConstraintRigidReference,
                           ConstraintRigid, raXnI2)

IMPLEMENT_ACCESS_FUNCTIONS(CollisionReference, ConstraintRigidReference,
                           ConstraintRigid, collision)

inline void ConstraintGroundReference::init() {
  // cached values
  Eigen::Vector3f nw = this->nw();
  auto body = this->body();

  Eigen::Vector3f rl1 = this->xl();
  this->d(body.transformPoint(rl1) - this->xw());

  this->lambda(Eigen::Vector3f::Zero());

  Eigen::Vector3f tx, ty;
  Collider::generateTangents(nw, &tx, &ty);
  Eigen::Matrix3f tmp;
  tmp << nw, tx, ty;
  this->contactFrame(tmp);

  float m1 = body.Mp();
  Eigen::Vector3f I1 = body.Mr();
  Eigen::Quaternionf q1(body.x0().block<4, 1>(0, 0));

  auto raXnI1 = this->raXnI1();
  auto w1 = this->w1();
  auto delLinVel1 = this->delLinVel1();
  auto angDelta1 = this->angDelta1();

  for (unsigned int i = 0; i < 3; i++) {
    Eigen::Vector3f contactFrame_row = this->contactFrame().block<3, 1>(0, i);
    Eigen::Vector3f nl1 = q1.inverse() * contactFrame_row;
    Eigen::Vector3f rnl1 = rl1.cross(nl1);
    raXnI1.block<3, 1>(0, i) =
        q1 * Eigen::Vector3f(rnl1.array() / I1.array().sqrt());

    w1(i) = (1 / m1) +
            raXnI1.block<3, 1>(0, i).transpose() * raXnI1.block<3, 1>(0, i);
    delLinVel1.block<3, 1>(0, i) = contactFrame_row / m1;
    angDelta1.block<3, 1>(0, i) = q1 * (rnl1.array() / I1.array());
  }

  this->raXnI1(raXnI1);
  this->w1(w1);
  this->delLinVel1(delLinVel1);
  this->angDelta1(angDelta1);
}

inline void ConstraintGroundReference::solveNorPos(float hs, float biasCoef,
                                                   float minpenetration) {
  // cached values
  Eigen::Vector3f nw = this->nw();
  auto body = this->body();
  Eigen::Vector3f lambda = this->lambda();

  float scale = min(0.8, biasCoef);
  float biasCoefficient;
  if (nw.transpose() * this->d()) {
    biasCoefficient = -scale / hs;
  } else {
    biasCoefficient = -1 / hs;
  }

  Eigen::Vector3f raXnI1_0 = this->raXnI1().block<3, 1>(0, 0);
  float sep = float(nw.transpose() * body.deltaLinDt()) +
              float(raXnI1_0.transpose() * body.deltaAngDt()) +
              float(nw.transpose() * this->d());
  sep = max(minpenetration, sep);
  float bias = sep * biasCoefficient;

  Eigen::Vector3f normalVel =
      nw.array() * body.v().array() + body.w().array() * raXnI1_0.array();
  float dlambdaNor = bias / this->w1()(0) - normalVel.sum() / this->w1()(0);
  float nplambda = lambda(0) + dlambdaNor;
  if (nplambda < 0) {
    dlambdaNor = -lambda(0);
    this->collision().broken(true);
  }
  lambda(0) = lambda(0) + dlambdaNor;
  body.v(body.v() + dlambdaNor * this->delLinVel1().block<3, 1>(0, 0));
  body.w(body.w() + dlambdaNor * this->raXnI1().block<3, 1>(0, 0));
  this->lambda(lambda);
}

inline void ConstraintGroundReference::solveTanVel(float hs, float biasCoef) {
  // cached values
  Eigen::Vector3f nw = this->nw();
  auto body = this->body();
  Eigen::Vector3f lambda = this->lambda();

  float scale = min(0.8, biasCoef);
  float biasCoefficient;
  if (nw.transpose() * this->d()) {
    biasCoefficient = -scale / hs;
  } else {
    biasCoefficient = -1 / hs;
  }
  float mu = body.mu();
  Eigen::Vector2f dlambdaTan = Eigen::Vector2f::Zero();
  for (unsigned int i = 1; i < 3; i++) {
    Eigen::Vector3f contactFrame_row = this->contactFrame().block<3, 1>(0, i);
    Eigen::Vector3f raXnI1_row = this->raXnI1().block<3, 1>(0, i);
    float sep = float(contactFrame_row.transpose() * body.deltaLinDt()) +
                float(raXnI1_row.transpose() * body.deltaAngDt()) +
                float(contactFrame_row.transpose() * this->d());
    float bias = sep * biasCoefficient;
    Eigen::Vector3f normalVel = contactFrame_row.array() * body.v().array() +
                                body.w().array() * raXnI1_row.array();
    dlambdaTan(i - 1) =
        (bias / this->w1()(i) - normalVel.sum() / this->w1()(i)) * 0.8;
  }
  Eigen::Vector3f dlambdas(0, dlambdaTan(0), dlambdaTan(1));
  Eigen::Vector3f lambdas = lambda + dlambdas;
  float frictionRadius = mu * lambdas(0);
  if (lambdas.block<2, 1>(1, 0).norm() > frictionRadius) {
    lambdas.block<2, 1>(1, 0) = frictionRadius * lambdas.block<2, 1>(1, 0) /
                                lambdas.block<2, 1>(1, 0).norm();
    dlambdas = lambdas - lambda;
    this->collision().broken(true);
  }
  lambda = lambda + dlambdas;
  body.v(body.v() + this->delLinVel1() * dlambdas);
  body.w(body.w() + this->raXnI1() * dlambdas);
  this->lambda(lambda);
}

inline void ConstraintGroundReference::applyLambdaSP() {}

inline void ConstraintRigidReference::applyLambdaSP() {
  this->body2().v(this->body2().v() - this->delLinVel2() * this->dlambdaSP());
  this->body2().w(this->body2().w() - this->raXnI2() * this->dlambdaSP());
}

inline void ConstraintRigidReference::init() {
  // Contact normal always points from body2 to body1
  // TODO: might be redundant??
  if (this->body1().layer() < this->body2().layer()) {
    auto temp = this->body1();
    this->body1(this->body2());
    this->body2(temp);

    auto temp2 = this->x1();
    this->x1(this->x2());
    this->x2(temp2);
  } else {
    this->nw(-this->nw());
  }

  // read and cache data to prevent unnecessary reads
  Eigen::Vector3f nw = this->nw();
  auto body1 = this->body1();
  auto body2 = this->body2();
  // now we can assume body1 is always above body2
  this->d(body1.transformPoint(this->x1()) - body2.transformPoint(this->x2()));

  this->lambda(Eigen::Vector3f::Zero());
  this->dlambdaSP(Eigen::Vector3f::Zero());

  Eigen::Vector3f tx, ty;
  Collider::generateTangents(nw, &tx, &ty);
  Eigen::Matrix3f contactFrame;
  contactFrame << nw, tx, ty;

  float m1 = body1.Mp();
  Eigen::Vector3f I1 = body1.Mr();
  Eigen::Quaternionf q1(body1.x0().block<4, 1>(0, 0));
  Eigen::Vector3f rl1 = this->x1();

  float m2 = body2.Mp();
  Eigen::Vector3f I2 = body2.Mr();
  Eigen::Quaternionf q2(body2.x0().block<4, 1>(0, 0));
  Eigen::Vector3f rl2 = this->x2();

  auto raXnI1 = this->raXnI1();
  auto w1 = this->w1();
  auto delLinVel1 = this->delLinVel1();
  auto angDelta1 = this->angDelta1();

  auto raXnI2 = this->raXnI2();
  auto w2 = this->w2();
  auto delLinVel2 = this->delLinVel2();
  auto angDelta2 = this->angDelta2();

  for (unsigned int i = 0; i < 3; i++) {
    Eigen::Vector3f contactFrame_row = contactFrame.block<3, 1>(0, i);
    Eigen::Vector3f nl1 = q1.inverse() * contactFrame_row;
    Eigen::Vector3f rnl1 = rl1.cross(nl1);
    raXnI1.block<3, 1>(0, i) =
        q1 * Eigen::Vector3f(rnl1.array() / I1.array().sqrt());

    w1(i) = (1 / m1) +
            raXnI1.block<3, 1>(0, i).transpose() * raXnI1.block<3, 1>(0, i);
    delLinVel1.block<3, 1>(0, i) = contactFrame_row / m1;
    angDelta1.block<3, 1>(0, i) = q1 * (rnl1.array() / I1.array());

    Eigen::Vector3f nl2 = q2.inverse() * contactFrame_row;
    Eigen::Vector3f rnl2 = rl2.cross(nl2);
    raXnI2.block<3, 1>(0, i) = q2 * (rnl2.array() / I2.array().sqrt());
    w2(i) = (1 / m2) +
            raXnI2.block<3, 1>(0, i).transpose() * raXnI2.block<3, 1>(0, i);
    delLinVel2.block<3, 1>(0, i) = contactFrame_row / m2;
    angDelta2.block<3, 1>(0, i) = q2 * (rnl2.array() / I2.array());
  }
  this->raXnI1(raXnI1);
  this->w1(w1);
  this->delLinVel1(delLinVel1);
  this->angDelta1(angDelta1);

  this->raXnI2(raXnI2);
  this->w2(w2);
  this->delLinVel2(delLinVel2);
  this->angDelta2(angDelta2);

  this->contactFrame(contactFrame);
}

inline void ConstraintRigidReference::solveNorPos(float hs, float biasCoef,
                                                  float minpenetration,
                                                  bool doShockProp) {
  // read and cache data to prevent unnecessary reads
  Eigen::Vector3f nw = this->nw();
  auto body1 = this->body1();
  auto body2 = this->body2();
  Eigen::Vector3f lambda = this->lambda();
  float scale = min(0.8, biasCoef);
  float biasCoefficient;
  if (float(nw.transpose() * this->d()) <= 0) {
    biasCoefficient = -scale / hs;
  } else {
    biasCoefficient = -1 / hs;
  }
  if (!doShockProp) {
    float sep = float(nw.transpose() * body1.deltaLinDt()) +
                float(this->raXnI1().block<3, 1>(0, 0).transpose() *
                      body1.deltaAngDt()) +
                float(nw.transpose() * this->d());
    sep -= float(nw.transpose() * body2.deltaLinDt()) +
           float(this->raXnI2().block<3, 1>(0, 0).transpose() *
                 body2.deltaAngDt());
    sep = max(minpenetration, sep);
    float bias = sep * biasCoefficient;

    Eigen::Vector3f normalVel =
        nw.array() * body1.v().array() +
        body1.w().array() * this->raXnI1().block<3, 1>(0, 0).array();
    normalVel -= Eigen::Vector3f(nw.array() * body2.v().array() +
                                 body2.w().array() *
                                     this->raXnI2().block<3, 1>(0, 0).array());
    float w = this->w1()(0) + this->w2()(0);
    float dlambdaNor = bias / w - normalVel.sum() / w;
    float nplambda = lambda(0) + dlambdaNor;
    if (nplambda < 0) {
      dlambdaNor = -lambda(0);
      this->collision().broken(true);
    }
    lambda(0) = lambda(0) + dlambdaNor;
    body1.v(body1.v() + dlambdaNor * this->delLinVel1().block<3, 1>(0, 0));
    body1.w(body1.w() + dlambdaNor * this->raXnI1().block<3, 1>(0, 0));
    body2.v(body2.v() - dlambdaNor * this->delLinVel2().block<3, 1>(0, 0));
    body2.w(body2.w() - dlambdaNor * this->raXnI2().block<3, 1>(0, 0));
  } else {
    float sep = float(nw.transpose() * body1.deltaLinDt()) +
                float(this->raXnI1().block<3, 1>(0, 0).transpose() *
                      body1.deltaAngDt()) +
                float(nw.transpose() * this->d());
    sep = max(minpenetration, sep);
    float bias = sep * biasCoefficient;
    Eigen::Vector3f normalVel =
        nw.array() * body1.v().array() +
        body1.w().array() * this->raXnI1().block<3, 1>(0, 0).array();
    float dlambdaNor = bias / this->w1()(0) - normalVel.sum() / this->w1()(0);
    float nplambda = lambda(0) + dlambdaNor;
    if (nplambda < 0) {
      dlambdaNor = -lambda(0);
      this->collision().broken(true);
    }
    lambda(0) = lambda(0) + dlambdaNor;
    body1.v(body1.v() + dlambdaNor * this->delLinVel1().block<3, 1>(0, 0));
    body1.w(body1.w() + dlambdaNor * this->raXnI1().block<3, 1>(0, 0));
    auto dlambdaSP = this->dlambdaSP();
    dlambdaSP(0) += dlambdaNor;
    this->dlambdaSP(dlambdaSP);
  }

  this->lambda(lambda);
}

inline void ConstraintRigidReference::solveTanVel(float hs, float biasCoef,
                                                  bool doShockProp) {
  // read and cache data to prevent unnecessary reads
  Eigen::Vector3f nw = this->nw();
  auto body1 = this->body1();
  auto body2 = this->body2();
  Eigen::Vector3f lambda = this->lambda();
  float scale = min(0.8, biasCoef);
  float biasCoefficient;
  if (float(nw.transpose() * this->d()) <= 0) {
    biasCoefficient = -scale / hs;
  } else {
    biasCoefficient = -1 / hs;
  }
  float mu = 0.5 * (body1.mu() + body2.mu());
  if (!doShockProp) {
    Eigen::Vector2f dlambdaTan = Eigen::Vector2f::Zero();
    for (unsigned int i = 1; i < 3; i++) {
      float sep =
          float(this->contactFrame().block<3, 1>(0, i).transpose() *
                body1.deltaLinDt()) +
          float(this->raXnI1().block<3, 1>(0, i).transpose() *
                body1.deltaAngDt()) +
          float(this->contactFrame().block<3, 1>(0, i).transpose() * this->d());
      sep -= float(this->contactFrame().block<3, 1>(0, i).transpose() *
                   body2.deltaLinDt()) +
             float(this->raXnI2().block<3, 1>(0, i).transpose() *
                   body2.deltaAngDt());
      float bias = sep * biasCoefficient;
      Eigen::Vector3f normalVel =
          this->contactFrame().block<3, 1>(0, i).array() * body1.v().array() +
          body1.w().array() * this->raXnI1().block<3, 1>(0, i).array();
      normalVel -= Eigen::Vector3f(
          this->contactFrame().block<3, 1>(0, i).array() * body2.v().array() +
          body2.w().array() * this->raXnI2().block<3, 1>(0, i).array());
      float w = this->w1()(i) + this->w2()(i);
      dlambdaTan(i - 1) = (bias / w - normalVel.sum() / w) * 0.8;
    }
    Eigen::Vector3f dlambdas(0, dlambdaTan(0), dlambdaTan(1));
    Eigen::Vector3f lambdas = lambda + dlambdas;
    float frictionRadius = mu * lambdas(0);
    if (lambdas.block<2, 1>(1, 0).norm() > frictionRadius) {
      lambdas.block<2, 1>(1, 0) = frictionRadius * lambdas.block<2, 1>(1, 0) /
                                  lambdas.block<2, 1>(1, 0).norm();
      dlambdas = lambdas - lambda;
      this->collision().broken(true);
    }
    lambda = lambda + dlambdas;
    body1.v(body1.v() + this->delLinVel1() * dlambdas);
    body1.w(body1.w() + this->raXnI1() * dlambdas);
    body2.v(body2.v() - this->delLinVel2() * dlambdas);
    body2.w(body2.w() - this->raXnI2() * dlambdas);
  } else {
    Eigen::Vector2f dlambdaTan = Eigen::Vector2f::Zero();
    for (unsigned int i = 1; i < 3; i++) {
      float sep =
          float(this->contactFrame().block<3, 1>(0, i).transpose() *
                body1.deltaLinDt()) +
          float(this->raXnI1().block<3, 1>(0, i).transpose() *
                body1.deltaAngDt()) +
          float(this->contactFrame().block<3, 1>(0, i).transpose() * this->d());
      float bias = sep * biasCoefficient;
      Eigen::Vector3f normalVel =
          this->contactFrame().block<3, 1>(0, i).array() * body1.v().array() +
          body1.w().array() * this->raXnI1().block<3, 1>(0, i).array();
      dlambdaTan(i - 1) =
          (bias / this->w1()(i)) - normalVel.sum() / this->w1()(i) * 0.8;
    }
    Eigen::Vector3f dlambdas(0, dlambdaTan(0), dlambdaTan(1));
    Eigen::Vector3f lambdas = lambda + dlambdas;
    float frictionRadius = mu * lambdas(0);
    if (lambdas.block<2, 1>(1, 0).norm() > frictionRadius) {
      lambdas.block<2, 1>(1, 0) = frictionRadius * lambdas.block<2, 1>(1, 0) /
                                  lambdas.block<2, 1>(1, 0).norm();
      dlambdas = lambdas - lambda;
      this->collision().broken(true);
    }
    lambda = lambda + dlambdas;
    body1.v(body1.v() + this->delLinVel1() * dlambdas);
    body1.w(body1.w() + this->raXnI1() * dlambdas);
    this->dlambdaSP(this->dlambdaSP() + dlambdas);
  }

  this->lambda(lambda);
}

} // namespace apbd
