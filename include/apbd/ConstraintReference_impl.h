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

    Eigen::Matrix3f raXnI1;
    Eigen::Vector3f w1;
    Eigen::Matrix3f delLinVel1;
    Eigen::Matrix3f angDelta1;

    for (unsigned int i = 0; i < 3; i++) {
        Eigen::Vector3f contactFrame_row =
            this->contactFrame().block<3, 1>(0, i);
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
    // cached values, read now to inform the compiler that they do not change in
    // the following code
    Eigen::Vector3f nw = this->nw();
    auto body = this->body();
    Eigen::Vector3f lambda = this->lambda();
    Eigen::Vector3f bodyv = body.v();
    Eigen::Vector3f bodyw = body.w();
    Eigen::Vector3f d = this->d();
    float w1_0 = this->w1()(0);
    Eigen::Vector3f raXnI1_0 = this->raXnI1().block<3, 1>(0, 0);
    Eigen::Vector3f delLinVel1_0 = this->delLinVel1().block<3, 1>(0, 0);
    Eigen::Vector3f bodydeltaLinDt = body.deltaLinDt();
    Eigen::Vector3f bodydeltaAngDt = body.deltaAngDt();

    float scale = min(0.8, biasCoef);
    float biasCoefficient;
    if (nw.transpose() * d) {
        biasCoefficient = -scale / hs;
    } else {
        biasCoefficient = -1 / hs;
    }

    float sep = float(nw.transpose() * bodydeltaLinDt) +
                float(raXnI1_0.transpose() * bodydeltaAngDt) +
                float(nw.transpose() * d);
    sep = max(minpenetration, sep);
    float bias = sep * biasCoefficient;

    Eigen::Vector3f normalVel =
        nw.array() * bodyv.array() + bodyw.array() * raXnI1_0.array();
    float dlambdaNor = bias / w1_0 - normalVel.sum() / w1_0;
    float nplambda = lambda(0) + dlambdaNor;
    if (nplambda < 0) {
        dlambdaNor = -lambda(0);
        this->collision().broken(true);
    }
    lambda(0) = lambda(0) + dlambdaNor;
    body.v(bodyv + dlambdaNor * delLinVel1_0);
    body.w(bodyw + dlambdaNor * raXnI1_0);
    this->lambda(lambda);
}

inline void ConstraintGroundReference::solveTanVel(float hs, float biasCoef) {
    // cached values
    Eigen::Vector3f nw = this->nw();
    auto body = this->body();
    Eigen::Vector3f lambda = this->lambda();
    Eigen::Vector3f bodyv = body.v();
    Eigen::Vector3f bodyw = body.w();
    Eigen::Vector3f d = this->d();
    Eigen::Vector3f w1 = this->w1();
    Eigen::Matrix3f raXnI1 = this->raXnI1();
    Eigen::Matrix3f delLinVel1 = this->delLinVel1();
    Eigen::Vector3f bodydeltaLinDt = body.deltaLinDt();
    Eigen::Vector3f bodydeltaAngDt = body.deltaAngDt();

    float scale = min(0.8, biasCoef);
    float biasCoefficient;
    if (nw.transpose() * d) {
        biasCoefficient = -scale / hs;
    } else {
        biasCoefficient = -1 / hs;
    }
    float mu = body.mu();
    Eigen::Vector2f dlambdaTan = Eigen::Vector2f::Zero();
    for (unsigned int i = 1; i < 3; i++) {
        Eigen::Vector3f contactFrame_row =
            this->contactFrame().block<3, 1>(0, i);
        Eigen::Vector3f raXnI1_row = raXnI1.block<3, 1>(0, i);
        float sep = float(contactFrame_row.transpose() * bodydeltaLinDt) +
                    float(raXnI1_row.transpose() * bodydeltaAngDt) +
                    float(contactFrame_row.transpose() * d);
        float bias = sep * biasCoefficient;
        Eigen::Vector3f normalVel = contactFrame_row.array() * bodyv.array() +
                                    bodyw.array() * raXnI1_row.array();
        dlambdaTan(i - 1) = (bias / w1(i) - normalVel.sum() / w1(i)) * 0.8;
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
    body.v(bodyv + delLinVel1 * dlambdas);
    body.w(bodyw + raXnI1 * dlambdas);
    this->lambda(lambda);
}

inline void ConstraintGroundReference::applyLambdaSP() {}

inline void ConstraintRigidReference::applyLambdaSP() {
    this->body2().v(this->body2().v() - this->delLinVel2() * this->dlambdaSP());
    this->body2().w(this->body2().w() - this->raXnI2() * this->dlambdaSP());
}

inline void ConstraintRigidReference::init() {
    // Contact normal always points from body2 to body1
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
    // now we can assume body1 is always above body2

    // read and cache data to prevent unnecessary reads
    Eigen::Vector3f nw = this->nw();
    auto body1 = this->body1();
    auto body2 = this->body2();
    Eigen::Vector3f rl1 = this->x1();
    Eigen::Vector3f rl2 = this->x2();
    this->d(body1.transformPoint(rl1) - body2.transformPoint(rl2));

    this->lambda(Eigen::Vector3f::Zero());
    this->dlambdaSP(Eigen::Vector3f::Zero());

    Eigen::Vector3f tx, ty;
    Collider::generateTangents(nw, &tx, &ty);
    Eigen::Matrix3f contactFrame;
    contactFrame << nw, tx, ty;

    float m1 = body1.Mp();
    Eigen::Vector3f I1 = body1.Mr();
    Eigen::Quaternionf q1(body1.x0().block<4, 1>(0, 0));

    float m2 = body2.Mp();
    Eigen::Vector3f I2 = body2.Mr();
    Eigen::Quaternionf q2(body2.x0().block<4, 1>(0, 0));

    // These are always overwritten, so they don't need to be read. We just
    // start uninitialized
    Eigen::Matrix3f raXnI1;
    Eigen::Vector3f w1;
    Eigen::Matrix3f delLinVel1;
    Eigen::Matrix3f angDelta1;

    Eigen::Matrix3f raXnI2;
    Eigen::Vector3f w2;
    Eigen::Matrix3f delLinVel2;
    Eigen::Matrix3f angDelta2;

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
    Eigen::Vector3f body1v = body1.v();
    Eigen::Vector3f body1w = body1.w();
    Eigen::Vector3f body2v = body2.v();
    Eigen::Vector3f body2w = body2.w();
    Eigen::Vector3f d = this->d();
    Eigen::Vector3f lambda = this->lambda();
    Eigen::Vector3f raXnI1_0 = this->raXnI1().block<3, 1>(0, 0);
    Eigen::Vector3f raXnI2_0 = this->raXnI2().block<3, 1>(0, 0);
    Eigen::Vector3f delLinVel1_0 = this->delLinVel1().block<3, 1>(0, 0);
    Eigen::Vector3f delLinVel2_0 = this->delLinVel2().block<3, 1>(0, 0);
    Eigen::Vector3f body1deltaLinDt = body1.deltaLinDt();
    Eigen::Vector3f body1deltaAngDt = body1.deltaAngDt();
    Eigen::Vector3f body2deltaLinDt = body2.deltaLinDt();
    Eigen::Vector3f body2deltaAngDt = body2.deltaAngDt();
    float scale = min(0.8, biasCoef);
    float biasCoefficient;
    if (float(nw.transpose() * d) <= 0) {
        biasCoefficient = -scale / hs;
    } else {
        biasCoefficient = -1 / hs;
    }
    if (!doShockProp) {
        float w = this->w1()(0) + this->w2()(0);
        float sep = float(nw.transpose() * body1deltaLinDt) +
                    float(raXnI1_0.transpose() * body1deltaAngDt) +
                    float(nw.transpose() * d);
        sep -= float(nw.transpose() * body2deltaLinDt) +
               float(raXnI2_0.transpose() * body2deltaAngDt);
        sep = max(minpenetration, sep);
        float bias = sep * biasCoefficient;

        Eigen::Vector3f normalVel =
            nw.array() * body1v.array() + body1w.array() * raXnI1_0.array();
        normalVel -= Eigen::Vector3f(nw.array() * body2v.array() +
                                     body2w.array() * raXnI2_0.array());
        float dlambdaNor = bias / w - normalVel.sum() / w;
        float nplambda = lambda(0) + dlambdaNor;
        if (nplambda < 0) {
            dlambdaNor = -lambda(0);
            this->collision().broken(true);
        }
        lambda(0) = lambda(0) + dlambdaNor;
        body1.v(body1v + dlambdaNor * delLinVel1_0);
        body1.w(body1w + dlambdaNor * raXnI1_0);
        body2.v(body2v - dlambdaNor * delLinVel2_0);
        body2.w(body2w - dlambdaNor * raXnI2_0);
    } else {
        float w1_0 = this->w1()(0);
        auto dlambdaSP = this->dlambdaSP();
        float sep = float(nw.transpose() * body1deltaLinDt) +
                    float(raXnI1_0.transpose() * body1deltaAngDt) +
                    float(nw.transpose() * d);
        sep = max(minpenetration, sep);
        float bias = sep * biasCoefficient;
        Eigen::Vector3f normalVel =
            nw.array() * body1v.array() + body1w.array() * raXnI1_0.array();
        float dlambdaNor = bias / w1_0 - normalVel.sum() / w1_0;
        // printf("w1_0 = %f\n", w1_0);
        // printf("normalVel.sum() = %f\n", normalVel.sum());
        float nplambda = lambda(0) + dlambdaNor;
        if (nplambda < 0) {
            dlambdaNor = -lambda(0);
            this->collision().broken(true);
        }
        lambda(0) = lambda(0) + dlambdaNor;
        // printf("Previous velocity for body 0 = [%f, %f, %f]\n",
        // body1.v().x(), body1.v().y(), body1.v().z()); printf("dlambdaNor =
        // %f\n", dlambdaNor); printf("delLinVel1_0 = [%f, %f, %f]\n",
        // delLinVel1_0.x(), delLinVel1_0.y(), delLinVel1_0.z());
        body1.v(body1v + dlambdaNor * delLinVel1_0);
        // printf("After velocity for body 0 = [%f, %f, %f]\n", body1.v().x(),
        // body1.v().y(), body1.v().z());
        body1.w(body1w + dlambdaNor * raXnI1_0);
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
    Eigen::Vector3f body1v = body1.v();
    Eigen::Vector3f body1w = body1.w();
    Eigen::Vector3f body2v = body2.v();
    Eigen::Vector3f body2w = body2.w();
    Eigen::Vector3f d = this->d();
    Eigen::Vector3f lambda = this->lambda();
    Eigen::Matrix3f raXnI1 = this->raXnI1();
    Eigen::Matrix3f raXnI2 = this->raXnI2();
    Eigen::Matrix3f delLinVel1 = this->delLinVel1();
    Eigen::Matrix3f delLinVel2 = this->delLinVel2();
    Eigen::Vector3f body1deltaLinDt = body1.deltaLinDt();
    Eigen::Vector3f body1deltaAngDt = body1.deltaAngDt();
    Eigen::Vector3f body2deltaLinDt = body2.deltaLinDt();
    Eigen::Vector3f body2deltaAngDt = body2.deltaAngDt();
    float scale = min(0.8, biasCoef);
    float biasCoefficient;
    if (float(nw.transpose() * d) <= 0) {
        biasCoefficient = -scale / hs;
    } else {
        biasCoefficient = -1 / hs;
    }
    float mu = 0.5 * (body1.mu() + body2.mu());
    if (!doShockProp) {
        Eigen::Vector2f dlambdaTan = Eigen::Vector2f::Zero();
        for (unsigned int i = 1; i < 3; i++) {
            float w = this->w1()(i) + this->w2()(i);
            Eigen::Vector3f contactFrame_row =
                this->contactFrame().block<3, 1>(0, i);
            Eigen::Vector3f raXnI1_row = raXnI1.block<3, 1>(0, i);
            Eigen::Vector3f raXnI2_row = raXnI2.block<3, 1>(0, i);
            float sep = float(contactFrame_row.transpose() * body1deltaLinDt) +
                        float(raXnI1_row.transpose() * body1deltaAngDt) +
                        float(contactFrame_row.transpose() * d);
            sep -= float(contactFrame_row.transpose() * body2deltaLinDt) +
                   float(raXnI2_row.transpose() * body2deltaAngDt);
            float bias = sep * biasCoefficient;
            Eigen::Vector3f normalVel =
                contactFrame_row.array() * body1v.array() +
                body1w.array() * raXnI1_row.array();
            normalVel -=
                Eigen::Vector3f(contactFrame_row.array() * body2v.array() +
                                body2w.array() * raXnI2_row.array());
            dlambdaTan(i - 1) = (bias / w - normalVel.sum() / w) * 0.8;
        }
        Eigen::Vector3f dlambdas(0, dlambdaTan(0), dlambdaTan(1));
        Eigen::Vector3f lambdas = lambda + dlambdas;
        float frictionRadius = mu * lambdas(0);
        if (lambdas.block<2, 1>(1, 0).norm() > frictionRadius) {
            lambdas.block<2, 1>(1, 0) = frictionRadius *
                                        lambdas.block<2, 1>(1, 0) /
                                        lambdas.block<2, 1>(1, 0).norm();
            dlambdas = lambdas - lambda;
            this->collision().broken(true);
        }
        lambda = lambda + dlambdas;
        body1.v(body1v + delLinVel1 * dlambdas);
        body1.w(body1w + raXnI1 * dlambdas);
        body2.v(body2v - delLinVel2 * dlambdas);
        body2.w(body2w - raXnI2 * dlambdas);
    } else {
        Eigen::Vector2f dlambdaTan = Eigen::Vector2f::Zero();
        for (unsigned int i = 1; i < 3; i++) {
            float w1_i = this->w1()(i);
            Eigen::Vector3f contactFrame_row =
                this->contactFrame().block<3, 1>(0, i);
            Eigen::Vector3f raXnI1_row = raXnI1.block<3, 1>(0, i);
            float sep = float(contactFrame_row.transpose() * body1deltaLinDt) +
                        float(raXnI1_row.transpose() * body1deltaAngDt) +
                        float(contactFrame_row.transpose() * d);
            float bias = sep * biasCoefficient;
            Eigen::Vector3f normalVel =
                contactFrame_row.array() * body1.v().array() +
                body1w.array() * raXnI1_row.array();
            dlambdaTan(i - 1) = (bias / w1_i) - normalVel.sum() / w1_i * 0.8;
        }
        Eigen::Vector3f dlambdas(0, dlambdaTan(0), dlambdaTan(1));
        Eigen::Vector3f lambdas = lambda + dlambdas;
        float frictionRadius = mu * lambdas(0);
        if (lambdas.block<2, 1>(1, 0).norm() > frictionRadius) {
            lambdas.block<2, 1>(1, 0) = frictionRadius *
                                        lambdas.block<2, 1>(1, 0) /
                                        lambdas.block<2, 1>(1, 0).norm();
            dlambdas = lambdas - lambda;
            this->collision().broken(true);
        }
        lambda = lambda + dlambdas;
        body1.v(body1v + delLinVel1 * dlambdas);
        body1.w(body1w + raXnI1 * dlambdas);
        this->dlambdaSP(this->dlambdaSP() + dlambdas);
    }

    this->lambda(lambda);
}

}  // namespace apbd
