#pragma once
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#include <Eigen/Dense>
#include <cuda/std/array>
#include <cuda/std/utility>

#include "apbd/CollisionReference.h"
#include "apbd/ConstraintReference.h"
#include "apbd/Contact.h"

namespace apbd {

struct alignas(16) Collision {
    /// A stored sequence of contacts between the bodies
    cuda::std::array<Contact, 8> contacts;
    /// References to the constraints created from each contact
    cuda::std::array<ConstraintReference, 8> constraints;

    __host__ __device__ void setContacts(
        CollisionReference clr,
        const cuda::std::pair<cuda::std::array<Contact, 8>, size_t> cdata);

    __host__ __device__ bool is_ground(CollisionReference clr);

    __host__ __device__ void getConstraints(CollisionReference clr,
                                            size_t &ground_count,
                                            size_t &rigid_count);

    __host__ __device__ void solveCollisionNor(CollisionReference clr, float hs,
                                               float biasCoeff,
                                               float minpenetration,
                                               bool withSP);

    __host__ __device__ void solveCollisionTan(CollisionReference clr, float hs,
                                               float biasCoeff, bool withSP);

    __host__ __device__ void applyLambdaSP(CollisionReference clr);

    __host__ __device__ void initConstraints(CollisionReference clr);

    // GPQP
    __host__ __device__ void computeJ_b(CollisionReference clr, float h);
    __host__ __device__ void compute_LTlambda(CollisionReference clr);
    __host__ __device__ void compute_degenerate_J1I_J2I_b(CollisionReference clr);
    __host__ __device__ void compute_degenerate_LTlambda(CollisionReference clr);
    __host__ __device__ void compute_LTd_cg(CollisionReference clr);
    __host__ __device__ void compute_LTp(CollisionReference clr);
    __host__ __device__ void compute_LLTx(CollisionReference clr);
    __host__ __device__ void compute_degenerate_LLTx(CollisionReference clr);

    // FIXME: compute_tbar returns a "tbar_i"
    __host__ __device__ void compute_tbar(CollisionReference clr);
    __host__ __device__ void compute_lambdac(CollisionReference clr, float t);
    __host__ __device__ void compute_lambdad(CollisionReference clr, float t);
    __host__ __device__ void compute_p(CollisionReference clr, float t);

    __host__ __device__ void project(CollisionReference clr);
    __host__ __device__ bool update_cg(CollisionReference clr, float alpha);
};

}  // namespace apbd
