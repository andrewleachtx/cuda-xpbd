#pragma once
#include "apbd/BodyReference.h"
#include "apbd/CollisionReference.h"
#include "data/utilities.h"
#include "util.h"

namespace apbd {

class Collision;

#define DECLARE_CONSTRAINT_ACCESS_FUNCTIONS(data_type, element) \
    __host__ __device__ data_type element() const;              \
    __host__ __device__ void element(data_type const new_val);

class ConstraintGroundReference {
   public:
    unsigned int index;

    __host__ __device__ ConstraintGroundReference(const unsigned int index)
        : index(data::soa_index(index)) {}

    __host__ __device__ void create(BodyRigidReference body, Contact c,
                                    CollisionReference collision) {
        this->body(body);
        this->nw(c.nw);
        this->xl(c.x1);
        this->xw(c.x2);
        this->collision(collision);
    }

    // access the data elements in ConstraintGround
    DECLARE_CONSTRAINT_ACCESS_FUNCTIONS(Eigen::Vector3f, lambda)
    DECLARE_CONSTRAINT_ACCESS_FUNCTIONS(Eigen::Vector3f, nw)
    DECLARE_CONSTRAINT_ACCESS_FUNCTIONS(Eigen::Vector3f, xl)
    DECLARE_CONSTRAINT_ACCESS_FUNCTIONS(Eigen::Vector3f, xw)
    DECLARE_CONSTRAINT_ACCESS_FUNCTIONS(Eigen::Vector3f, d)
    DECLARE_CONSTRAINT_ACCESS_FUNCTIONS(BodyRigidReference, body)

    DECLARE_CONSTRAINT_ACCESS_FUNCTIONS(Eigen::Matrix3f, contactFrame)

    DECLARE_CONSTRAINT_ACCESS_FUNCTIONS(Eigen::Vector3f, w1)
    DECLARE_CONSTRAINT_ACCESS_FUNCTIONS(Eigen::Matrix3f, delLinVel1)
    DECLARE_CONSTRAINT_ACCESS_FUNCTIONS(Eigen::Matrix3f, angDelta1)
    DECLARE_CONSTRAINT_ACCESS_FUNCTIONS(Eigen::Matrix3f, raXnI1)

    DECLARE_CONSTRAINT_ACCESS_FUNCTIONS(CollisionReference, collision)

    // functions on the constraint type
    __host__ __device__ void applyLambdaSP();
    __host__ __device__ void init();
    __host__ __device__ void solveNorPos(float hs, float biasCoef,
                                         float minpenetration);
    __host__ __device__ void solveTanVel(float hs, float biasCoef);
    __host__ __device__ Eigen::Vector3f evalCs(float h);
};

class ConstraintRigidReference {
   public:
    unsigned int index;

    __host__ __device__ ConstraintRigidReference(const unsigned int index)
        : index(data::soa_index(index)) {}

    __host__ __device__ void create(BodyRigidReference body1,
                                    BodyRigidReference body2, Contact c,
                                    CollisionReference collision) {
        this->body1(body1);
        this->body2(body2);
        this->nw(c.nw);
        this->x1(c.x1);
        this->x2(c.x2);
        this->collision(collision);
    }

    // access the data elements in ConstraintRigid

    DECLARE_CONSTRAINT_ACCESS_FUNCTIONS(Eigen::Vector3f, lambda)
    DECLARE_CONSTRAINT_ACCESS_FUNCTIONS(Eigen::Vector3f, nw)
    DECLARE_CONSTRAINT_ACCESS_FUNCTIONS(Eigen::Vector3f, x1)
    DECLARE_CONSTRAINT_ACCESS_FUNCTIONS(Eigen::Vector3f, x2)
    DECLARE_CONSTRAINT_ACCESS_FUNCTIONS(Eigen::Vector3f, dlambdaSP)
    DECLARE_CONSTRAINT_ACCESS_FUNCTIONS(Eigen::Vector3f, d)
    DECLARE_CONSTRAINT_ACCESS_FUNCTIONS(BodyRigidReference, body1)
    DECLARE_CONSTRAINT_ACCESS_FUNCTIONS(BodyRigidReference, body2)

    DECLARE_CONSTRAINT_ACCESS_FUNCTIONS(Eigen::Matrix3f, contactFrame)

    DECLARE_CONSTRAINT_ACCESS_FUNCTIONS(Eigen::Vector3f, w1)
    DECLARE_CONSTRAINT_ACCESS_FUNCTIONS(Eigen::Matrix3f, delLinVel1)
    DECLARE_CONSTRAINT_ACCESS_FUNCTIONS(Eigen::Matrix3f, angDelta1)
    DECLARE_CONSTRAINT_ACCESS_FUNCTIONS(Eigen::Matrix3f, raXnI1)

    DECLARE_CONSTRAINT_ACCESS_FUNCTIONS(Eigen::Vector3f, w2)
    DECLARE_CONSTRAINT_ACCESS_FUNCTIONS(Eigen::Matrix3f, delLinVel2)
    DECLARE_CONSTRAINT_ACCESS_FUNCTIONS(Eigen::Matrix3f, angDelta2)
    DECLARE_CONSTRAINT_ACCESS_FUNCTIONS(Eigen::Matrix3f, raXnI2)

    DECLARE_CONSTRAINT_ACCESS_FUNCTIONS(CollisionReference, collision)

    // functions on the constraint type
    __host__ __device__ void applyLambdaSP();
    __host__ __device__ void init();
    __host__ __device__ void solveNorPos(float hs, float biasCoef,
                                         float minpenetration,
                                         bool doShockProp);
    __host__ __device__ void solveTanVel(float hs, float biasCoef,
                                         bool doShockProp);
    __host__ __device__ Eigen::Vector3f evalCs(float h);
};

class ConstraintReference {
   public:
    // we don't store the type for the constraint because the owner of this
    // reference should keep track of it.
    unsigned int index;

    __host__ __device__ ConstraintReference() {}
    __host__ __device__ ConstraintReference(const unsigned int index)
        : index(index) {
        DEBUG_ASSERT(index < MAX_COLLISION_CONSTRAINTS,
                     "ConstraintReference overflow!");
    }
    __host__ __device__ ConstraintGroundReference get_ground() const {
        return ConstraintGroundReference(index);
    }
    __host__ __device__ ConstraintRigidReference get_rigid() const {
        return ConstraintRigidReference(index);
    }

    __host__ __device__ bool operator==(const BodyReference &other) const {
        return this->index == other.index;
    }
};

}  // namespace apbd
