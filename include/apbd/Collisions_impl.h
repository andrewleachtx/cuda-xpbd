#pragma once
#include "apbd/BodyReference_impl.h"
#include "apbd/Collisions.h"
#include "apbd/ConstraintReference_impl.h"

namespace apbd {

inline void Collision::setContacts(
    const cuda::std::pair<cuda::std::array<Contact, 8>, size_t> cdata) {
  this->contacts = cdata.first;
  this->data.contactNum(cdata.second);
}

inline bool Collision::is_ground() { return this->data.body2() == NULL_BODY; }

inline void Collision::getConstraints(size_t &ground_count,
                                      size_t &rigid_count) {
  if (this->is_ground()) {
    for (unsigned int i = 0; i < this->data.contactNum(); i++) {
      ConstraintReference cr(ground_count++);
      // TODO: handle other body types
      cr.get_ground().create(this->data.body1().get_rigid(), contacts[i], this);
      this->constraints[i] = cr;
    }
  } else {
    for (unsigned int i = 0; i < this->data.contactNum(); i++) {
      ConstraintReference cr(rigid_count++);
      cr.get_rigid().create(this->data.body1().get_rigid(),
                            this->data.body2().get_rigid(), contacts[i], this);
      this->constraints[i] = cr;
    }
  }
}

inline void Collision::solveCollisionNor(float hs, float biasCoeff,
                                         float minpenetration, bool withSP) {
  for (unsigned int i = 0; i < this->data.contactNum(); i++) {
    if (this->is_ground()) {
      this->constraints[i].get_ground().solveNorPos(hs, biasCoeff,
                                                    minpenetration);
    } else {
      this->constraints[i].get_rigid().solveNorPos(hs, biasCoeff,
                                                   minpenetration, withSP);
    }
  }
}

inline void Collision::solveCollisionTan(float hs, float biasCoeff,
                                         bool withSP) {
  for (unsigned int i = 0; i < this->data.contactNum(); i++) {
    if (this->is_ground()) {
      this->constraints[i].get_ground().solveTanVel(hs, biasCoeff);
    } else {
      this->constraints[i].get_rigid().solveTanVel(hs, biasCoeff, withSP);
    }
  }
}

inline void Collision::applyLambdaSP() {
  for (unsigned int i = 0; i < this->data.contactNum(); i++) {
    if (this->is_ground()) {
      this->constraints[i].get_ground().applyLambdaSP();
    } else {
      this->constraints[i].get_rigid().applyLambdaSP();
    }
  }
}

inline void Collision::initConstraints() {
  for (unsigned int i = 0; i < this->data.contactNum(); i++) {
    if (this->is_ground()) {
      this->constraints[i].get_ground().init();
    } else {
      this->constraints[i].get_rigid().init();
    }
  }
}
} // namespace apbd
