#include "util.h"
#include <apbd/Collider.h>
#include <apbd/Constraint.h>
#include <se3/lib.h>

using Eigen::Vector2f, Eigen::Vector3f, Eigen::Vector4f, Eigen::Matrix3f,
    Eigen::Quaternionf;

namespace apbd {

Constraint::Constraint(ConstraintRigid rigid)
    : type(CONSTRAINT_COLLISION_RIGID), data{.rigid = {rigid}} {}
Constraint::Constraint(ConstraintGround ground)
    : type(CONSTRAINT_COLLISION_GROUND), data{.ground = {ground}} {}
Constraint::Constraint(ConstraintJointRevolve revolve)
    : type(CONSTRAINT_JOINT_REVOLVE), data{.joint_revolve = {revolve}} {}
ConstraintGround::ConstraintGround(BodyRigidReference body, Contact c,
                                   Collision *collision)
    : lambda(Vector3f::Zero()), nw(c.nw), d(Vector3f::Zero()), body(body),
      xl(c.x1), xw(c.x2), contactFrame(Matrix3f::Zero()), w1(Vector3f::Zero()),
      raXnI1(Matrix3f::Zero()), delLinVel1(Matrix3f::Zero()),
      angDelta1(Matrix3f::Zero()), collision(collision) {}

ConstraintRigid::ConstraintRigid(BodyRigidReference body1,
                                 BodyRigidReference body2, Contact c,
                                 Collision *collision)
    : lambda(Vector3f::Zero()), nw(c.nw), dlambdaSP(Vector3f::Zero()),
      d(Vector3f::Zero()), body1(body1), body2(body2), x1(c.x1), x2(c.x2),
      contactFrame(Matrix3f::Zero()), w1(Vector3f::Zero()),
      raXnI1(Matrix3f::Zero()), delLinVel1(Matrix3f::Zero()),
      angDelta1(Matrix3f::Zero()), w2(Vector3f::Zero()),
      raXnI2(Matrix3f::Zero()), delLinVel2(Matrix3f::Zero()),
      angDelta2(Matrix3f::Zero()), collision(collision) {}

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
  switch (type) {
  case CONSTRAINT_JOINT_REVOLVE: {
    // TODO
    break;
  }

  default:
    break;
  }
}

void Constraint::solve() {
  switch (type) {
  case CONSTRAINT_JOINT_REVOLVE: {
    // TODO
    break;
  }

  default:
    break;
  }
}

void Constraint::clear() {
  switch (type) {
  case CONSTRAINT_JOINT_REVOLVE: {
    ConstraintJointRevolve *c = &data.joint_revolve;
    c->lambda = Eigen::Vector3f::Zero();
    break;
  }

  default:
    break;
  }
}

} // namespace apbd
