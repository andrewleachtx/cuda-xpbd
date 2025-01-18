#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>

#include "apbd/BodyReference_impl.h"
#include "apbd/CollisionReference_impl.h"
#include "apbd/Collisions_impl.h"
#include "apbd/Model.h"
#include "util.h"

// thrust for duplicate removal and sorting ()
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

namespace apbd {

Model::Model()
    : h(1. / 30.),
      tEnd(1),
      substeps(10),
      bodies(nullptr),
      body_count(0),
      constraints(nullptr),
      constraint_count(0),
      solver_type(Solver_Type::S_2PSP),
      gravity(0.0, 0.0, -980.0),
      forward_iters(0),
      reverse_iters(0),
      ground_E(Eigen::Matrix4f::Zero()),
      ground_size(10),
      steps(0) {}

Model::Model(const Model &other)
    : h(other.h),
      tEnd(other.tEnd),
      substeps(other.substeps),
      bodies(other.bodies),
      body_count(other.body_count),
      constraints(nullptr),
      constraint_count(other.constraint_count),
      solver_type(other.solver_type),
      gravity(other.gravity),
      forward_iters(other.forward_iters),
      reverse_iters(other.reverse_iters),
      ground_E(other.ground_E),
      ground_size(other.ground_size),
      steps(other.steps) {}

Model::Model(const Model &&other)
    : h(other.h),
      tEnd(other.tEnd),
      substeps(other.substeps),
      bodies(other.bodies),
      body_count(other.body_count),
      constraints(other.constraints),
      constraint_count(other.constraint_count),
      solver_type(other.solver_type),
      gravity(other.gravity),
      forward_iters(other.forward_iters),
      reverse_iters(other.reverse_iters),
      ground_E(other.ground_E),
      ground_size(other.ground_size),
      steps(other.steps) {}

void Model::create_store(size_t scene_count) {
    // TODO: handle this constraints
    data::SOAStore data_store(
        this->body_count, MAX_COLLISION_CONSTRAINTS, MAX_COLLISION_CONSTRAINTS,
        this->body_count * (this->body_count + 1) / 2, scene_count);

#ifdef USE_CUDA
    cudaMemcpyToSymbol(data::device_global_store, &data_store,
                       sizeof(data::SOAStore), size_t(0),
                       cudaMemcpyHostToDevice);
#else
    data::global_store = std::move(data_store);
#endif
}

void Model::copy_data_to_store(Body *body_array) {
    for (size_t i = 0; i < this->body_count; i++) {
        auto &body = body_array[i];
        switch (body.type) {
            case BODY_RIGID: {
                data::global_store.BodyRigid.set(data::soa_index(i),
                                                 body.data.rigid);
                auto br = BodyReference(i, body.type);
                br.get_rigid().init(body.data.rigid.xInit);
                this->bodies[i] = br;
                break;
            }
            default:
                break;
        }
    }
    this->write_state(0);
}

ModelBuffers Model::allocate_buffers(size_t count, const Model &model) {
    ModelBuffers buffers;
    if (model.constraint_count > 0) {
        buffers.constraints =
            alloc_device<Constraint>(count * model.constraint_count);
    } else {
        buffers.constraints = nullptr;
    }
    return buffers;
}

Model Model::clone_with_buffers(const ModelBuffers &buffers, size_t offset) {
    Model new_model = Model(*this);
    DEBUG_ASSERT(&new_model != this, "Model not cloned!");
    if (this->constraint_count > 0) {
        new_model.constraints =
            &buffers.constraints[offset * this->constraint_count];
        memcpy_device(new_model.constraints, this->constraints,
                      this->constraint_count);
    }
    return new_model;
}

void Model::init() {
    this->t = 0.0f;

    // bodies are initialized when data is copied to store
    for (size_t i = 0; i < this->constraint_count; i++) {
        this->constraints[i].init();
    }
    // calculate parameters
    this->steps = ceil(this->tEnd / this->h);
    this->print_config();
}

void Model::move_to_device() {
    bodies = move_array_to_device(bodies, body_count);
    constraints = move_array_to_device(constraints, constraint_count);
}

/*
float comouteResiduals(){
    Eigen::VectorXf rs(num_constraints*3);
    for(int i = 0; i< num_constraints;i++){
        rs(seq(i*3,i*3+2)) = constraints[i].evalCs(h);
    }
    float residual = rs.array().max(0).matrix().norm();
    return residual;
}
*/
float Model::computeResiduals(Collider *collider, float h) {
    // Should be num_constraints * 3
    Eigen::VectorXf rs = Eigen::VectorXf::Zero(collider->active_collision_count * 3);

    // 'Constraint' doesn't work, it only points to ConstraintReference and we need all collisions first to access them
    for (size_t i = 0; i < collider->active_collision_count; i++) {
        CollisionReference clr(collider->activeCollisions[i]);

        /*
        if (this->is_ground(clr)) {
                for (unsigned int i = 0; i < clr.contactNum(); i++) {
                    ConstraintReference cr(ground_count++);
                    cr.get_ground().create(clr.body1().get_rigid(), contacts[i], clr);
                    this->constraints[i] = cr;
                }
            } else {
                for (unsigned int i = 0; i < clr.contactNum(); i++) {
                    ConstraintReference cr(rigid_count++);
                    cr.get_rigid().create(clr.body1().get_rigid(),
                                        clr.body2().get_rigid(), contacts[i], clr);
                    this->constraints[i] = cr;
                }
            }

    collider->collisions[collider->activeCollisions[i]]

    collider->collisions[collider->activeCollisions[i]]
        */
        for (unsigned int j = 0; j < clr.contactNum(); j++) {
            Eigen::Vector3f Cs;
            if (collider->collisions[collider->activeCollisions[i]].is_ground(clr)) {
                Cs = collider->collisions[collider->activeCollisions[i]].constraints[j].get_ground().evalCs(this->h);
            }
            else {
                Cs = collider->collisions[collider->activeCollisions[i]].constraints[j].get_rigid().evalCs(this->h);
            }

            // add to rs
            rs.segment(i * 3, 3) = Cs;
        }
    }

    float residual = rs.array().max(0).matrix().norm();
    return residual;
}

void Model::simulate(Collider *collider) {
    float hs = this->h / static_cast<float>(this->substeps);
    // printf("Simulating a total of %u steps.\n", this->steps);
    for (unsigned int step = 0; step < this->steps; step++) {
        // printf("==== Step %u starting ====\n", step);
        if (this->solver_type == Solver_Type::S_2PSP) {
            this->solveConTGS(collider, hs);
        } else if (this->solver_type == Solver_Type::S_GPQP) {
            this->solveConGPQP(collider, hs);
        }

        this->write_state(step + 1);
        float resid = this->computeResiduals(collider, hs);
        printf("Residuals: %f\n", resid);
    }
}

/** Private Functions **/

void Model::stepBDF1(float hs) {
    for (size_t body_i = 0; body_i < this->body_count; body_i++) {
        this->bodies[body_i].stepBDF1(hs, this->gravity);
    }
}

// TODO: In addition to the GPQP changes, it appears this method in Model.m has
// changed as well.
void Model::solveConTGS(Collider *collider, float hs) {
    this->stepBDF1(this->h);

    // Run collider; generate contact points
    collider->run(this);
    float biasCoefficient = 2 * sqrt(hs / this->h);
    // 1e20 is used instead of oo for maximum hardware
    // compatibilty/predictability
    const float Inf = 1e20f;

    // We solve constraints in the layer order. The exact layer sizes don't
    // matter at this step, so we don't bother walking through each layer
    // individually.

    // 2 Pass Shock Propagation
    for (size_t i = 0; i < collider->active_collision_count; i++) {
        CollisionReference clr(collider->activeCollisions[i]);
        collider->collisions[collider->activeCollisions[i]].initConstraints(
            clr);
    }
    for (size_t i = 0; i < collider->active_collision_count; i++) {
        CollisionReference clr(collider->activeCollisions[i]);
        for (unsigned int j = 0; j < this->forward_iters; j++) {
            collider->collisions[collider->activeCollisions[i]]
                .solveCollisionNor(clr, hs, biasCoefficient, -Inf, true);
        }
    }

    // work backward now
    // order within layers might matter, but this is much simpler
    for (long int i = collider->active_collision_count - 1; i >= 0; i--) {
        CollisionReference clr(collider->activeCollisions[i]);
        for (unsigned int j = 0; j < this->reverse_iters; j++) {
            collider->collisions[collider->activeCollisions[i]]
                .solveCollisionNor(clr, hs, biasCoefficient, -Inf, true);
        }
        collider->collisions[collider->activeCollisions[i]].applyLambdaSP(clr);
    }

    for (size_t i = 0; i < this->body_count; i++) {
        this->bodies[i].updateStates(hs);
    }

    unsigned int ks = 0;
    while (ks < this->substeps) {
        for (size_t constraint_i = 0; constraint_i < this->constraint_count;
             constraint_i++) {
            this->constraints[constraint_i].clear();
        }

        // Gauss-Seidel solve for non-collision constraints
        for (size_t constraint_i = 0; constraint_i < this->constraint_count;
             constraint_i++) {
            this->constraints[constraint_i].solve();
        }

        // Gauss-Seidel for collisions
        for (size_t i = 0; i < collider->active_collision_count; i++) {
            CollisionReference clr(collider->activeCollisions[i]);
            collider->collisions[collider->activeCollisions[i]]
                .solveCollisionNor(clr, hs, biasCoefficient, -Inf, false);
            collider->collisions[collider->activeCollisions[i]]
                .solveCollisionTan(clr, hs, biasCoefficient, false);
        }

        for (size_t i = 0; i < this->body_count; i++) {
            this->bodies[i].updateStates(hs);
        }

        ks++;
    }

    for (size_t i = 0; i < collider->active_collision_count; i++) {
        CollisionReference clr(collider->activeCollisions[i]);
        collider->collisions[collider->activeCollisions[i]].solveCollisionNor(
            clr, hs, biasCoefficient, 0, false);
        collider->collisions[collider->activeCollisions[i]].solveCollisionTan(
            clr, hs, biasCoefficient, false);
    }

    for (size_t i = 0; i < this->body_count; i++) {
        this->bodies[i].integrateStates();
    }
}

/* FIXME: Still in progress! */
void Model::solveConGPQP(Collider *collider, float hs) {
    const float POSINF(1e20), biasCoefficient(2 * sqrt(hs / this->h));
    collider->run(this);

    printf("line: %d\n", __LINE__);

    for (size_t i = 0; i < collider->active_collision_count; i++) {
        CollisionReference clr(collider->activeCollisions[i]);
        collider->collisions[collider->activeCollisions[i]].initConstraints(
            clr);
    }

    this->stepBDF1(this->h);

    for (size_t i = 0; i < this->constraint_count; i++) {
        this->constraints[i].clear();
    }

    for (size_t i = 0; i < this->body_count; i++) {
        this->bodies[i].clearJacobi();
    }

    /*
        TODO: The MATLAB solver calls an empty .solve method:

            for (size_t i = 0; i < this->constraint_count; i++) {
                this->constraints[i].solve();
            }

        That said, it tells us to use a Gauss-Seidel solve instead. But in
       MATLAB it doesn't even do that, so we will leave out for now

        for (size_t i = 0; i < collider->active_collision_count; i++) {
            CollisionReference clr(collider->activeCollisions[i]);
            collider->collisions[collider->activeCollisions[i]]
                .solveCollisionNor(clr, hs, biasCoefficient, -POSINF, false);
            collider->collisions[collider->activeCollisions[i]]
                .solveCollisionTan(clr, hs, biasCoefficient, false);
        }
    */

    unsigned int n = 0;
    unsigned int ci = 0;
    for (size_t i = 0; i < collider->active_collision_count; i++) {
        CollisionReference clr(collider->activeCollisions[i]);
        clr.index = ci;
        clr.mIndices(n);

        // computeJ_b
        collider->collisions[collider->activeCollisions[i]].computeJ_b(clr,
                                                                       this->h);

        {
            // e.g. if clr.b() is an Eigen::VectorXf
            if (hasNaN(clr.b())) {
                printf("NaN detected in collision b() at collision i=%zu.\n",
                       i);
                exit(1);
            }
        }

        ci++;
        n += 3 * clr.contactNum();
    }

    GPQPOutput output = this->GPQP(collider, n);
    auto lambdas = output.lambdas;

    if (hasNaN(lambdas)) {
        printf("NaN detected in GPQP output lambdas.\n");
        exit(1);
    }

    // TODO: The use of mIndices is a bit odd. It doesn't really need to exist
    // in C++, we can just use offsetting. Should investigate that
    for (unsigned int i = 0; i < collider->active_collision_count; i++) {
        CollisionReference clr(collider->activeCollisions[i]);

        unsigned int dim = 3 * clr.contactNum();
        // unsigned int start = clr.mIndices();
        unsigned int start = i * dim;

        // Copy out the sub-vector from lambdas
        Eigen::VectorXf lambdai = lambdas.segment(start, dim);

        // Check lambdai for NaNs
        if (hasNaN(lambdai)) {
            printf("NaN found in lambdai for collision i=%u.\n", i);
            exit(1);
        }

        for (unsigned int k = 0; k < clr.contactNum(); k++) {
            Eigen::VectorXf contactLambda = lambdai.segment(3 * k, 3);

            if (hasNaN(contactLambda)) {
                printf(
                    "NaN found in contactLambda at collision i=%u, contact "
                    "k=%u\n",
                    i, k);
                exit(1);
            }

            collider->collisions[collider->activeCollisions[i]]
                .constraints[k]
                .get_rigid()
                .applyLambda(contactLambda);
        }
    }

    for (size_t i = 0; i < this->body_count; i++) {
        this->bodies[i].updateStatesDirect(this->h);
        Eigen::Vector3f pos = this->bodies[i].get_rigid().position();
        if (pos.hasNaN()) {
            printf("NaN in body %zu position after updateStatesDirect.\n", i);
            exit(1);
        }
    }

    this->t += this->h;
    for (size_t i = 0; i < this->body_count; i++) {
        this->bodies[i].integrateStates();
        Eigen::Vector3f pos = this->bodies[i].get_rigid().position();
        if (pos.hasNaN()) {
            printf("NaN in body %zu position after integrateStates.\n", i);
            exit(1);
        }
    }
}

/*
    This seeks to replicate the MATLAB call to `unique(tList, 'sorted')`.

    Main issue using std is device compilation -- potential implementation
   (seems to work) with thrust for CUDA usage.

    Should 1) remove dupes 2) sort. Note this is done in backwards because of
   how thrust::unique works.
*/
__host__ __device__ void uniqueDeviceVector(thrust::device_vector<float> &vec,
                                            size_t &used_ct) {
    if (used_ct > vec.size()) {
        used_ct = vec.size();
    }
    auto begin = vec.begin();
    auto end = begin + used_ct;

    thrust::sort(begin, end);

    auto new_end = thrust::unique(begin, end);
    size_t unique_count = new_end - begin;

    if (unique_count < used_ct) {
        thrust::fill(new_end, end, 1e20f);
    }

    used_ct = unique_count;
}

GPQPOutput Model::GPQP(Collider *collider, int n) {
    DEBUG_ASSERT(n <= MAX_COLLISION_CONSTRAINTS,
                 "GPQP n exceeds constraint limit; overflow");

    GPQPOutput output;
    output.iterations = 0;
    output.cgiterations_ct = 0;

    unsigned int collision_indices[MAX_COLLISION_CONSTRAINTS];
    size_t coll_ct = 0;
    for (size_t i = 0; i < collider->active_collision_count; i++) {
        collision_indices[coll_ct++] = collider->activeCollisions[i];
    }

    vecGPQPf gPrev = vecGPQPf::Zero();
    vecGPQPf g = vecGPQPf::Zero();
    vecGPQPf lambda = vecGPQPf::Zero();
    if (gPrev.hasNaN()) {
        printf("NaN in gPrev (initial) at line %d\n", __LINE__);
        exit(1);
    }

    float fPrev = 0.0f;

    for (size_t i = 0; i < this->body_count; i++) {
        this->bodies[i].get_rigid().LTx(vec6f::Zero());
    }
    // TODO: A bit weird that collison_indices[i] and activeCollisions[i] are
    // interchanged, my fault
    for (size_t i = 0; i < coll_ct; i++) {
        CollisionReference clr(collider->activeCollisions[i]);
        collider->collisions[collision_indices[i]].compute_LTlambda(clr);
    }
    for (size_t i = 0; i < coll_ct; i++) {
        CollisionReference clr(collider->activeCollisions[i]);
        collider->collisions[collision_indices[i]].compute_LLTx(clr);

        const auto &Ax_cpy = clr.Ax();
        const auto &b_cpy = clr.b();
        if (Ax_cpy.hasNaN()) {
            printf("NaN detected in Ax_local at coll i=%zu, line=%d\n", i,
                   __LINE__);
            exit(1);
        }
        if (b_cpy.hasNaN()) {
            printf("NaN detected in b_local at coll i=%zu, line=%d\n", i,
                   __LINE__);
            exit(1);
        }

        clr.g(Ax_cpy - b_cpy);

        vec24f g_local = Ax_cpy - b_cpy;
        if (g_local.hasNaN()) {
            printf("NaN in g_local=Ax-b at coll i=%zu, line=%d\n", i, __LINE__);
            exit(1);
        }

        unsigned int dim = 3 * clr.contactNum();
        // unsigned int start = clr.mIndices();
        unsigned int start = i * dim;

        gPrev.segment(start, dim) = g_local.head(dim);
        if (gPrev.hasNaN()) {
            printf("NaN introduced in gPrev segment at coll i=%zu, line=%d\n",
                   i, __LINE__);
            exit(1);
        }

        float fPart = clr.lambda().head(dim).dot(0.5f * Ax_cpy.head(dim) -
                                                 b_cpy.head(dim));
        if (std::isnan(fPart)) {
            printf("NaN in fPart at coll i=%zu, line=%d\n", i, __LINE__);
            exit(1);
        }
        fPrev += fPart;
    }
    const float Inf = 1e20f;
    float tol(1e-8f), eps(1e-8f);
    unsigned int iterMax = this->substeps;
    unsigned int CGiterMax = 200;

    size_t iter = 0;
    for (iter = 0; iter < iterMax; iter++) {
        vecGPQPf tList = vecGPQPf::Constant(Inf);
        if (tList.hasNaN()) {
            printf("NaN in tList (initialized) at iteration %zu, line=%d\n",
                   iter, __LINE__);
            exit(1);
        }

        // used_ct = sum(3 * contactNum)
        size_t used_ct = 0;
        for (size_t c = 0; c < coll_ct; c++) {
            CollisionReference clr(collider->activeCollisions[c]);
            used_ct += 3 * clr.contactNum();
        }
        for (size_t i = 0; i < this->body_count; i++) {
            this->bodies[i].get_rigid().LTx(vec6f::Zero());
        }
        for (size_t i = 0; i < coll_ct; i++) {
            CollisionReference clr(collider->activeCollisions[i]);
            collider->collisions[collision_indices[i]].compute_LTlambda(clr);
        }
        for (size_t i = 0; i < coll_ct; i++) {
            CollisionReference clr(collider->activeCollisions[i]);
            collider->collisions[collision_indices[i]].compute_LLTx(clr);

            vec24f Ax_local = clr.Ax();
            vec24f b_local = clr.b();
            clr.g(Ax_local - b_local);

            if (Ax_local.hasNaN()) {
                printf("NaN in Ax_local (LLTx) at coll c=%zu, line=%d\n", i,
                       __LINE__);
                exit(1);
            }
            if (b_local.hasNaN()) {
                printf("NaN in b_local (LLTx) at coll c=%zu, line=%d\n", i,
                       __LINE__);
                exit(1);
            }

            /*
                FIXME:

                At this point, for some reason clr.b() is getting set to only
               zeroes, unlike the first round of this when calculating
               clr.g(...).

                Because of that, Ax_local - b_local == vec24(0.0f), and the norm
               of that is of course zero.

                Then a divide by zero occurs in
               compute_tbar->rayConeIntersection->g_norm = g / g.norm() which
               produces a NaN.
            */
            const auto &tbar_local =
                collider->collisions[collision_indices[i]].compute_tbar(clr);
            if (tbar_local.hasNaN()) {
                printf("NaN in tbar_local at coll c=%zu, line=%d\n", i,
                       __LINE__);
                exit(1);
            }

            unsigned int dim = 3 * clr.contactNum();
            // unsigned int start = clr.mIndices();
            unsigned int start = i * dim;
            if (start + dim <= tList.size()) {
                tList.segment(start, dim) = tbar_local.head(dim);
                if (tList.hasNaN()) {
                    printf(
                        "NaN introduced in tList after segment at coll c=%zu, "
                        "line=%d\n",
                        i, __LINE__);
                    exit(1);
                }
            }
        }

        /*
            FIXME: The line tUniqueList = unique(tList, 'sorted') returns a list
           of all unique values found in tList, in sorted order. This means the
           size is refactored as well, and it guarantees that Inf is the last
           value.

            This needs to be changed, without library abstraction (thrust) it is
           complex to add efficient sorts, and dynamic resize.
        */
        thrust::device_vector<float> tUniqueListVec(tList.data(),
                                                    tList.data() + used_ct);
        uniqueDeviceVector(tUniqueListVec, used_ct);
        vecGPQPf tUniqueList;
        for (size_t i = 0; i < used_ct; i++) {
            tUniqueList(i) = tUniqueListVec[i];
        }
        if (used_ct > 0 && tUniqueList(used_ct - 1) != Inf &&
            used_ct < MAX_COLLISION_CONSTRAINTS) {
            tUniqueList(used_ct) = Inf;
        }
        float tc = 0.0f;
        for (size_t tIndex = 0; tIndex < used_ct; tIndex++) {
            float t_cur = tUniqueList(tIndex);
            if (t_cur >= Inf) {
                break;
            }

            for (size_t i = 0; i < coll_ct; i++) {
                CollisionReference clr(collider->activeCollisions[i]);
                collider->collisions[collision_indices[i]].compute_p(clr, tc);
                collider->collisions[collision_indices[i]].compute_lambdac(clr,
                                                                           tc);
            }

            float fPrime = 0.0f;
            float fPrimePrime = 0.0f;

            for (size_t b = 0; b < this->body_count; b++) {
                this->bodies[b].get_rigid().LTx(vec6f::Zero());
            }
            for (size_t i = 0; i < coll_ct; i++) {
                CollisionReference clr(collider->activeCollisions[i]);
                collider->collisions[collision_indices[i]].compute_LTp(clr);
            }
            for (size_t i = 0; i < coll_ct; i++) {
                CollisionReference clr(collider->activeCollisions[i]);
                collider->collisions[collision_indices[i]].compute_LLTx(clr);

                auto b_local = clr.b();
                auto p_local = clr.p();
                auto Ax_local = clr.Ax();
                auto lambdac_local = clr.lambdac();

                unsigned int dim = 3 * clr.contactNum();
                float tmp1 = 0.0f, tmp2 = 0.0f;
                for (unsigned int dd = 0; dd < dim; dd++) {
                    tmp1 -= b_local(dd) * p_local(dd);
                    tmp1 += lambdac_local(dd) * Ax_local(dd);
                    tmp2 += p_local(dd) * Ax_local(dd);
                }
                fPrime += tmp1;
                fPrimePrime += tmp2;
            }

            float deltaTStar = -fPrime / fPrimePrime;
            if (std::isnan(deltaTStar)) {
                printf("NaN in deltaTStar at line %d\n", __LINE__);
                exit(1);
            }
            if (fPrime > 0.0f) {
                break;
            } else if (deltaTStar >= 0.0f && deltaTStar < (t_cur - tc)) {
                tc += deltaTStar;
                break;
            }

            tc = t_cur;
        }
        for (size_t i = 0; i < coll_ct; i++) {
            CollisionReference clr(collider->activeCollisions[i]);
            collider->collisions[collision_indices[i]].compute_lambdac(clr, tc);
            collider->collisions[collision_indices[i]].compute_lambdad(clr, tc);
            clr.lambda(clr.lambdac());
            // check lambda for NaN
            auto lam_local = clr.lambda();
            if (lam_local.hasNaN()) {
                printf("NaN in clr.lambda() after finalize, i=%zu, line=%d\n",
                       i, __LINE__);
                exit(1);
            }
        }
        // PCG
        for (size_t i = 0; i < coll_ct; i++) {
            CollisionReference clr(collider->activeCollisions[i]);
            collider->collisions[collision_indices[i]]
                .compute_degenerate_J1I_J2I_b(clr);
        }
        for (size_t i = 0; i < this->body_count; i++) {
            this->bodies[i].get_rigid().LTx(vec6f::Zero());
        }
        for (size_t i = 0; i < coll_ct; i++) {
            CollisionReference clr(collider->activeCollisions[i]);
            collider->collisions[collision_indices[i]]
                .compute_degenerate_LTlambda(clr);
        }
        for (size_t c = 0; c < coll_ct; c++) {
            CollisionReference clr(collider->activeCollisions[c]);
            collider->collisions[collision_indices[c]].compute_degenerate_LLTx(
                clr);

            clr.r_cg(clr.Ax() - clr.b_cg());
            clr.g_cg(clr.Minv_cg().cwiseProduct(clr.r_cg()));
            clr.d_cg(-clr.g_cg());

            if (clr.r_cg().hasNaN()) {
                printf("NaN in r_cg at coll c=%zu, line=%d\n", c, __LINE__);
                exit(1);
            }
            if (clr.g_cg().hasNaN()) {
                printf("NaN in g_cg at coll c=%zu, line=%d\n", c, __LINE__);
                exit(1);
            }
            if (clr.d_cg().hasNaN()) {
                printf("NaN in d_cg at coll c=%zu, line=%d\n", c, __LINE__);
                exit(1);
            }
        }
        // CG iterations
        unsigned int CGiter = 0;
        for (; CGiter < CGiterMax; CGiter++) {
            float r_val = 0.0f;

            for (size_t i = 0; i < coll_ct; i++) {
                CollisionReference clr(collider->activeCollisions[i]);
                vec24f M = clr.Minv_cg().cwiseInverse();
                auto r_cg_local = clr.r_cg();
                auto freeIndex = clr.freeIndex();

                for (int j = 0; j < 24; j++) {
                    if (freeIndex(j)) {
                        r_val += r_cg_local(j) * M(j) * r_cg_local(j);
                    }
                }
            }
            if (std::isnan(r_val)) {
                printf("NaN in r_val during PCG at line=%d, CGiter=%u\n",
                       __LINE__, CGiter);
                exit(1);
            }
            if (r_val < tol) {
                break;
            }

            for (size_t b = 0; b < this->body_count; b++) {
                this->bodies[b].get_rigid().LTx(vec6f::Zero());
            }
            for (size_t i = 0; i < coll_ct; i++) {
                CollisionReference clr(collider->activeCollisions[i]);
                collider->collisions[collision_indices[i]].compute_LTd_cg(clr);
            }

            float numerator = 0.0f;
            float denominator = 0.0f;
            for (size_t i = 0; i < coll_ct; i++) {
                CollisionReference clr(collider->activeCollisions[i]);
                collider->collisions[collision_indices[i]]
                    .compute_degenerate_LLTx(clr);

                const auto &r_cg_local = clr.r_cg();
                const auto &g_cg_local = clr.g_cg();
                const auto &d_cg_local = clr.d_cg();
                const auto &Ax_local = clr.Ax();
                const auto &freeIndex = clr.freeIndex();

                for (int j = 0; j < 24; j++) {
                    if (freeIndex(j)) {
                        numerator += r_cg_local(j) * g_cg_local(j);
                        denominator += d_cg_local(j) * Ax_local(j);
                    }
                }
            }
            if (std::isnan(numerator) || std::isnan(denominator)) {
                printf(
                    "NaN in numerator/denominator in PCG at line=%d, "
                    "CGiter=%u\n",
                    __LINE__, CGiter);
                exit(1);
            }

            float alpha = numerator / denominator;
            if (std::isnan(alpha)) {
                printf("NaN in alpha in PCG at line=%d, CGiter=%u\n", __LINE__,
                       CGiter);
                exit(1);
            }

            bool feasible = true;
            for (size_t i = 0; i < coll_ct; i++) {
                CollisionReference clr(collider->activeCollisions[i]);
                bool ok = collider->collisions[collision_indices[i]].update_cg(
                    clr, alpha);
                if (!ok) {
                    feasible = false;
                }
            }
            if (!feasible) {
                break;
            }

            numerator = 0.0f;
            denominator = 0.0f;

            for (size_t i = 0; i < coll_ct; i++) {
                CollisionReference clr(collider->activeCollisions[i]);
                auto freeIndex = clr.freeIndex();

                if (std::isnan(alpha)) {
                    printf("NaN in alpha at line=%d, CGiter=%u, i=%zu\n",
                           __LINE__, CGiter, i);
                    exit(1);
                }

                vec24f Ax_local = clr.Ax();
                if (Ax_local.hasNaN()) {
                    printf("NaN in Ax_local at line=%d, CGiter=%u, i=%zu\n",
                           __LINE__, CGiter, i);
                    exit(1);
                }

                vec24f new_r = clr.r_cg() + alpha * Ax_local;
                if (new_r.hasNaN()) {
                    printf(
                        "NaN in new_r (r_cg + alpha * Ax) at line=%d, "
                        "CGiter=%u, i=%zu\n",
                        __LINE__, CGiter, i);
                    printf("alpha = %.6f\n", alpha);
                    exit(1);
                }
                clr.r_cg(new_r);

                vec24f Minv_cg_local = clr.Minv_cg();
                if (Minv_cg_local.hasNaN()) {
                    printf("NaN in Minv_cg at line=%d, CGiter=%u, i=%zu\n",
                           __LINE__, CGiter, i);
                    exit(1);
                }

                vec24f g_cg_local = Minv_cg_local.cwiseProduct(new_r);
                if (g_cg_local.hasNaN()) {
                    printf(
                        "NaN in g_cg (Minv_cg * new_r) at line=%d, CGiter=%u, "
                        "i=%zu\n",
                        __LINE__, CGiter, i);
                    exit(1);
                }
                clr.g_cg(g_cg_local);

                for (int j = 0; j < 24; j++) {
                    if (freeIndex(j)) {
                        denominator += new_r(j) * g_cg_local(j);
                    }
                }
            }
            if (std::isnan(denominator)) {
                printf(
                    "NaN in denominator after update at line=%d, CGiter=%u\n",
                    __LINE__, CGiter);
                exit(1);
            }

            numerator = denominator;

            float beta = numerator / denominator;
            if (std::isnan(beta)) {
                printf("NaN in beta in PCG at line=%d, CGiter=%u\n", __LINE__,
                       CGiter);
                exit(1);
            }
            for (size_t i = 0; i < coll_ct; i++) {
                CollisionReference clr(collider->activeCollisions[i]);
                clr.d_cg(-clr.g_cg() + beta * clr.d_cg());
            }
        }
        if (output.cgiterations_ct < MAX_COLLISION_CONSTRAINTS) {
            output.cgiterations[output.cgiterations_ct++] = CGiter;
        }

        for (size_t i = 0; i < coll_ct; i++) {
            CollisionReference clr(collider->activeCollisions[i]);
            collider->collisions[collision_indices[i]].project(clr);
        }
        float f = 0.0f;
        for (size_t i = 0; i < this->body_count; i++) {
            this->bodies[i].get_rigid().LTx(vec6f::Zero());
        }
        for (size_t i = 0; i < coll_ct; i++) {
            CollisionReference clr(collider->activeCollisions[i]);
            collider->collisions[collision_indices[i]].compute_LTlambda(clr);
        }
        for (size_t i = 0; i < coll_ct; i++) {
            CollisionReference clr(collider->activeCollisions[i]);
            collider->collisions[collision_indices[i]].compute_LLTx(clr);

            vec24f Ax_local = clr.Ax();
            vec24f b_local = clr.b();
            vec24f lam_local = clr.lambda();

            vec24f g_local = Ax_local - b_local;
            clr.g(g_local);

            unsigned int dim = 3 * clr.contactNum();
            float fPart = 0.0f;
            for (unsigned int dd = 0; dd < dim; dd++) {
                fPart += lam_local(dd) * (0.5f * Ax_local(dd) - b_local(dd));
            }
            if (std::isnan(fPart)) {
                printf("NaN in fPart for collision i=%zu, line=%d\n", i,
                       __LINE__);
                exit(1);
            }
            f += fPart;

            // unsigned int start = clr.mIndices();
            unsigned int start = i * dim;
            if (start + dim <= n) {
                g.segment(start, dim) = g_local.head(dim);
                lambda.segment(start, dim) = lam_local.head(dim);

                if (g.hasNaN()) {
                    printf("NaN introduced in g during KKT check, line=%d\n",
                           __LINE__);
                    exit(1);
                }
                if (lambda.hasNaN()) {
                    printf(
                        "NaN introduced in lambda during KKT check, line=%d\n",
                        __LINE__);
                    exit(1);
                }
            }
        }

        if (g.norm() < eps) {
            printf("line: %d - g.norm() < eps, break.\n", __LINE__);
            break;
        }
        if ((g - gPrev).norm() < eps) {
            printf("line: %d - (g - gPrev).norm() < eps, break.\n", __LINE__);
            break;
        }
        float df = f - fPrev;
        if (df > -eps && df < eps) {
            printf("line: %d - df in [-eps, eps], break.\n", __LINE__);
            break;
        }

        fPrev = f;
        gPrev = g;
        output.rs[iter] = g.norm();
    }

    output.iterations = static_cast<unsigned int>(iter);
    output.lambdas = lambda;

    return output;
}

void Model::write_state(unsigned int step) {
#ifdef WRITE
#ifdef __CUDA_ARCH__
    if (threadIdx.x == 0) printf("Step %d\n", step);
    // print up to 8 simulations in parallel
    for (size_t i = 0; i < body_count * 8; i++) {
        if (i / body_count != threadIdx.x) continue;
        printf("%lu ", i);
        bodies[i % body_count].write_state();
        printf("\n");
    }
#else
    using namespace std::chrono_literals;
    if (_thread_scene_id == 0) printf("Step %d\n", step);
    for (size_t i = 0; i < body_count * 8; i++) {
        if (i / body_count != _thread_scene_id) {
            // std::this_thread::sleep_for(10ms);
            continue;
        }
        printf("%lu ", i);
        bodies[i % body_count].write_state();
        printf("\n");
    }
#endif
#endif
}

void Model::print_config() {
    printf(
        "# Body count: %lu\n"
        "# Constraint count: %lu\n"
        "# Gravity: [%f %f %f]\n"
        "# Ground size: %f\n"
        "# Time Step: %f\n"
        "# End Time: %f\n"
        "# Steps: %u\n"
        "# Substeps: %u\n"
        "# Iterations: ->%u <-%u\n",
        body_count, constraint_count, gravity(0), gravity(1), gravity(2),
        ground_size, h, tEnd, steps, substeps, forward_iters, reverse_iters);
}

size_t Model::get_shared_memory_size() {
    return sizeof(BodyReference) * this->body_count;
}

__device__ void Model::populate_shared_mem(void *shared_memory) {
    BodyReference *shared_bodies =
        reinterpret_cast<BodyReference *>(shared_memory);

    if (threadIdx.x == 0)
        for (size_t i = 0; i < this->body_count; i++) {
            shared_bodies[i] = this->bodies[i];
        }
    __syncthreads();

    this->bodies = shared_bodies;
}

unsigned int Model::get_collision_index(BodyReference body1,
                                        BodyReference body2) {
    DEBUG_ASSERT(body1.index != body2.index,
                 "Cannot find collision index for a body and itself.");
    unsigned int x, y;
    unsigned int N = this->body_count;
    if (body1.index < body2.index) {
        x = body2.index;
        y = body1.index;
    } else {
        x = body1.index;
        y = body2.index;
    }
    // Gives the index of an element in a triangle grid where y < x,
    // plus a full row at the bottom (reserved for the ground)
    return N + (x + y * N) - ((y + 2) * (y + 1)) / 2;
}

}  // namespace apbd
