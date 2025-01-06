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

void Model::simulate(Collider *collider) {
    float hs = this->h / static_cast<float>(this->substeps);
    // printf("Simulating a total of %u steps.\n", this->steps);
    for (unsigned int step = 0; step < this->steps; step++) {
        // printf("==== Step %u starting ====\n", step);
        // this->solveConTGS(collider, hs);
        this->solveConGPQP(collider);

        // for (size_t i = 0; i < this->body_count; i++) {
        //     auto pos = this->bodies[i].get_rigid().position();
        //     auto vel = this->bodies[i].get_rigid().v();
        //     auto w = this->bodies[i].get_rigid().w();
        //     auto r = this->bodies[i].get_rigid().rotation().coeffs();
        //     printf("Body %zu: Position = [%f, %f, %f], Velocity = [%f, %f,
        //     %f]\n",
        //            i, pos(0), pos(1), pos(2), vel(0), vel(1), vel(2));
        //     printf("Body Angular Velocity: [%f, %f, %f]\n", w(0), w(1),
        //     w(2)); printf("Body Rotation: [%f, %f, %f, %f]\n", r(0), r(1),
        //     r(2), r(3));
        // }

        this->write_state(step + 1);
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
    collider->run(this);
    float biasCoefficient = 2 * sqrt(hs / this->h);
    // 1e20 is used instead of oo for maximum hardware
    // compatibilty/predictability
    const float Inf = 1e20f;

    // We solve contstraints in the layer order. The exact layer sizes don't
    // matter at this step, so we don't bother walking through each layer
    // individually.

    // Shock propagation
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

/*
function solveConGPQP(this)
    this.collider.run();

    for i = 1 : length(this.collider.activeCollisions)
        for j = this.collider.activeCollisions{i}
            this.collider.collisions{j}.initConstraints(this.h, this.hs);
        end
    end
    this.draw();

    this.stepBDF1();

    %this.draw();
    %fprintf('substep %d\n',this.ks);
    for i = 1 : length(this.constraints)
        this.constraints{i}.clear();
    end

    %fprintf('  iter %d\n',iter);
    % Clear the Jacobi updates
    for i = 1 : length(this.bodies)
        this.bodies{i}.clearJacobi();
    end
    % Gauss-Seidel solve for non-collision constraints
    for j = 1 : length(this.constraints)
        this.constraints{j}.solve();
    end
    % Solve all collision normals at the position level
    %fprintf('    ');
    %this.collider.run();

    n = 1;
    ci = 1;
    for i = 1 : length(this.collider.activeCollisions)
        for j = this.collider.activeCollisions{i}
            this.collider.collisions{j}.index = ci;
            this.collider.collisions{j}.mIndces = n : n - 1 +
this.collider.collisions{j}.contactNum * 3; n = n +
this.collider.collisions{j}.contactNum * 3;
            this.collider.collisions{j}.computeJ_b();
            ci = ci + 1;
        end
    end

    n = n-1;

    output = this.GPQP(n);
    lambdas = output.lambdas;
    %lambdas = solver.Gauss_Sidiel(A, b, mu);

    for i = 1 : length(this.collider.activeCollisions)
        for j = this.collider.activeCollisions{i}
            l = this.collider.collisions{j}.contactNum*3 - 1;
            start = this.collider.collisions{j}.mIndces(1);
            lambdai = lambdas(start:start+l);
            for k = 1: this.collider.collisions{j}.contactNum
                this.collider.collisions{j}.constraints{k}.applyLambda(lambdai(3*(k-1)
+ 1: 3*k)); end end end

    for i = 1 : length(this.bodies)
        this.bodies{i}.updateStatesDirect(this.h);
    end

    this.t = this.t + this.h;
    for i = 1 : length(this.bodies)
        this.bodies{i}.integrateStates();
    end
end
*/
void Model::solveConGPQP(Collider *collider) {
    collider->run(this);

    for (size_t i = 0; i < collider->active_collision_count; i++) {
        CollisionReference clr(collider->activeCollisions[i]);
        collider->collisions[collider->activeCollisions[i]].initConstraints(
            clr);
    }

    this->stepBDF1(this->h);

    for (size_t i = 0; i < this->constraint_count; i++) {
        this->constraints[i].clear();
    }

    // Clear Jacobi updates
    for (size_t i = 0; i < this->body_count; i++) {
        this->bodies[i].clearJacobi();
    }

    // Gauss-Seidel solve for non-collision constraints
    for (size_t i = 0; i < this->constraint_count; i++) {
        this->constraints[i].solve();
    }

    unsigned int n = 1;
    unsigned int ci = 1;
    for (size_t i = 0; i < collider->active_collision_count; i++) {
        CollisionReference clr(collider->activeCollisions[i]);
        clr.index = ci;
        clr.mIndices(n);
        collider->collisions[collider->activeCollisions[i]].computeJ_b(clr,
                                                                       this->h);

        ci++;
        n += 3 * clr.contactNum();
    }

    n--;

    GPQPOutput output = this->GPQP(collider, n);
    auto lambdas = output.lambdas;

    for (unsigned int i = 0; i < collider->active_collision_count; i++) {
        CollisionReference clr(collider->activeCollisions[i]);

        unsigned int l = 3 * clr.contactNum() - 1;
        unsigned int start = clr.mIndices();
        // FIXME: Doubt this device compiles
        Eigen::VectorXf lambdai = lambdas.segment(start, l);

        for (unsigned int k = 0; k < clr.contactNum(); k++) {
            collider->collisions[collider->activeCollisions[i]].constraints[k].get_rigid().applyLambda(lambdai.segment(3 * k, 3));
        }
    }

    for (size_t i = 0; i < this->body_count; i++) {
        this->bodies[i].updateStatesDirect(this->h);
    }

    this->t += this->h;
    for (size_t i = 0; i < this->body_count; i++) {
        this->bodies[i].integrateStates();
    }
}

/*
function output = GPQP(this, n)
    tol = 1e-8;
    eps = 1e-8;
    iterMax = this.substeps;
    CGiterMax = 200;
    rs = zeros(iterMax,1);
    collisions = [];
    for i = 1 : length(this.collider.activeCollisions)
        for j = this.collider.activeCollisions{i}
            collisions(end+1) = j;
        end
    end


    fPrev = 0;
    gPrev = zeros(n,1);
    g = zeros(n,1);
    lambda = zeros(n,1);

    for i = 1:length(this.bodies)
        this.bodies{i}.LTx = zeros(6,1);
    end
    for i = collisions
        this.collider.collisions{i}.compute_LTlambda();
    end
    for i = collisions
        this.collider.collisions{i}.compute_LLTx();
        this.collider.collisions{i}.g = this.collider.collisions{i}.Ax -
this.collider.collisions{i}.b; fPrev = fPrev +
this.collider.collisions{i}.lambda' * ( 0.5 * this.collider.collisions{i}.Ax -
this.collider.collisions{i}.b); gPrev(this.collider.collisions{i}.mIndces) =
this.collider.collisions{i}.g; end

    CGiterVec = [];

    for iter = 1:iterMax
        if(iter == 3)
            disp(iter);
        end

        % Compute Cauchy Point
        tList = Inf(n, 1);
        for i = 1:length(this.bodies)
            this.bodies{i}.LTx = zeros(6,1);
        end
        for i = collisions
            this.collider.collisions{i}.compute_LTlambda();
        end
        for i = collisions
            this.collider.collisions{i}.compute_LLTx();
            this.collider.collisions{i}.g = this.collider.collisions{i}.Ax -
this.collider.collisions{i}.b; tList(this.collider.collisions{i}.mIndces) =
this.collider.collisions{i}.compute_tbar(); end

        tUniqueList = unique(tList,'sorted');
        if(tUniqueList(end) ~= Inf)
            tUniqueList(end + 1) = Inf;
        end
        tc = 0;
        for tIndex = 1: size(tUniqueList,1)
            for i = collisions
                this.collider.collisions{i}.compute_p(tc);
                this.collider.collisions{i}.compute_lambdac(tc);
            end

            fPrime = 0;
            fPrimePrime = 0;

            for i = 1:length(this.bodies)
                this.bodies{i}.LTx = zeros(6,1);
            end
            for i = collisions
                this.collider.collisions{i}.compute_LTp();
            end
            for i = collisions
                this.collider.collisions{i}.compute_LLTx();
                fPrime = fPrime - this.collider.collisions{i}.b' *
this.collider.collisions{i}.p + ... this.collider.collisions{i}.lambdac' *
this.collider.collisions{i}.Ax; fPrimePrime = fPrimePrime +
this.collider.collisions{i}.p' * this.collider.collisions{i}.Ax; end

            deltaTStar = - fPrime / fPrimePrime;
            if(fPrime >0)
                break;
            elseif(deltaTStar >=0 && deltaTStar < tUniqueList(tIndex) - tc)
                tc = tc + deltaTStar;
                break;
            end
            tc = tUniqueList(tIndex);
        end

        for i = collisions
            this.collider.collisions{i}.compute_lambdac(tc);
            this.collider.collisions{i}.compute_lambdad(tc);
            this.collider.collisions{i}.lambda =
this.collider.collisions{i}.lambdac; end

        % PCG
        % Compute b_cg
        for i = collisions
                this.collider.collisions{i}.compute_degenerate_J1I_J2I_b();
        end

        % Init CG
        for i = 1:length(this.bodies)
            this.bodies{i}.LTx = zeros(6,1);
        end
        for i = collisions
            this.collider.collisions{i}.compute_degenerate_LTlambda();
        end
        for i = collisions
            this.collider.collisions{i}.compute_degenerate_LLTx();
            this.collider.collisions{i}.r_cg = this.collider.collisions{i}.Ax -
this.collider.collisions{i}.b_cg; this.collider.collisions{i}.g_cg =
this.collider.collisions{i}.Minv_cg .* this.collider.collisions{i}.r_cg;
            this.collider.collisions{i}.d_cg = -
this.collider.collisions{i}.g_cg; end

% CG iterations
for CGiter = 1: CGiterMax
    r = 0;
    for i = collisions
        M = 1 ./ this.collider.collisions{i}.Minv_cg;
        r = r +
this.collider.collisions{i}.r_cg(this.collider.collisions{i}.freeIndex)' *
diag(M(this.collider.collisions{i}.freeIndex)) ...
            * this.collider.collisions{i}.r_cg(this.collider.collisions{i}.freeIndex);
    end
    if(r < tol)
        break;
    end

    numerator = 0;
    denominator = 0;

    for i = 1:length(this.bodies)
        this.bodies{i}.LTx = zeros(6,1);
    end
    for i = collisions
        this.collider.collisions{i}.compute_LTd_cg();
    end
    for i = collisions
        this.collider.collisions{i}.compute_degenerate_LLTx();
        numerator = numerator +
this.collider.collisions{i}.r_cg(this.collider.collisions{i}.freeIndex)' ...
                    * this.collider.collisions{i}.g_cg(this.collider.collisions{i}.freeIndex);
        denominator = denominator +
this.collider.collisions{i}.d_cg(this.collider.collisions{i}.freeIndex)' ...
                        * this.collider.collisions{i}.Ax(this.collider.collisions{i}.freeIndex);
    end
    alpha = numerator / denominator;

    feasible = true;
    for i = collisions
        feasible = feasible & this.collider.collisions{i}.update_cg(alpha);
    end
    if(~feasible)
        break;
    end

    numerator = 0;
    denominator = 0;
    for i = collisions
        denominator = denominator +
this.collider.collisions{i}.r_cg(this.collider.collisions{i}.freeIndex)' ...
                        * this.collider.collisions{i}.g_cg(this.collider.collisions{i}.freeIndex);
        this.collider.collisions{i}.r_cg = this.collider.collisions{i}.r_cg +
alpha * this.collider.collisions{i}.Ax; this.collider.collisions{i}.g_cg =
this.collider.collisions{i}.Minv_cg .* this.collider.collisions{i}.r_cg;
        numerator = numerator +
this.collider.collisions{i}.r_cg(this.collider.collisions{i}.freeIndex)' ...
                    * this.collider.collisions{i}.g_cg(this.collider.collisions{i}.freeIndex);
    end
    beta = numerator / denominator;
    for i = collisions
        this.collider.collisions{i}.d_cg = -this.collider.collisions{i}.g_cg +
beta * this.collider.collisions{i}.d_cg; end end

CGiterVec = [CGiterVec, CGiter];

% Project to feasible region
for i = collisions
    this.collider.collisions{i}.project();
end

% Test if results satisfy the KKT conditions
f = 0;
for i = 1:length(this.bodies)
    this.bodies{i}.LTx = zeros(6,1);
end
for i = collisions
    this.collider.collisions{i}.compute_LTlambda();
end
for i = collisions
    this.collider.collisions{i}.compute_LLTx();
    this.collider.collisions{i}.g = this.collider.collisions{i}.Ax -
this.collider.collisions{i}.b; f = f + this.collider.collisions{i}.lambda' * (
0.5 * this.collider.collisions{i}.Ax -  this.collider.collisions{i}.b);
    g(this.collider.collisions{i}.mIndces) = this.collider.collisions{i}.g;
    lambda(this.collider.collisions{i}.mIndces) =
this.collider.collisions{i}.lambda; end

if norm(g) < eps
    break;
end
if norm(g - gPrev) < eps
    break;
end
if norm(f-fPrev) < eps
    break;
end
fPrev = f;
gPrev = g;
rs(iter) = norm(g);
end


output.iterations = iter;
output.lambdas = lambda;
output.cgiterations = CGiterVec;
output.rs = rs;
end
*/
__host__ __device__ inline vecGPQPf uniqueTList(vecGPQPf &tList,
                                                size_t used_count,
                                                bool do_sort) {
    thrust::device_ptr<float> tPtr(tList.data());

    if (do_sort) {
        thrust::sort(tPtr, tPtr + used_count);
    }

    auto new_end = thrust::unique(tPtr, tPtr + used_count);

    size_t unique_count = static_cast<size_t>(new_end - tPtr);

    if (unique_count < used_count) {
        thrust::fill(new_end, tPtr + used_count, 1e20f);
    }

    return tList;
}

GPQPOutput Model::GPQP(Collider *collider, int n) {
    GPQPOutput output;

    if (n > MAX_COLLISION_CONSTRAINTS) {
        DEBUG_ASSERT(n < MAX_COLLISION_CONSTRAINTS, "GPQP overflow!");
    }

    float tol(1e-8f), eps(1e-8f);
    unsigned int iterMax = this->substeps;
    unsigned int CGiterMax = 200;

    unsigned int collision_indices[MAX_COLLISION_CONSTRAINTS];
    size_t coll_ct = 0;

    for (size_t i = 0; i < collider->active_collision_count; i++) {
        collision_indices[coll_ct++] = collider->activeCollisions[i];
    }

    float fPrev = 0.0f;
    vecGPQPf gPrev = vecGPQPf::Zero();
    vecGPQPf g = vecGPQPf::Zero();
    vecGPQPf lambda = vecGPQPf::Zero();

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
        clr.g(clr.Ax() - clr.b());
        fPrev += clr.lambda().dot(0.5f * clr.Ax() - clr.b());
        gPrev.segment(clr.mIndices(), 3 * clr.contactNum()) = clr.g();
    }

    /*
        Eigen::Matrix<float, MAX_COLLISION_CONSTRAINTS, 1> iterations;
        Eigen::Matrix<float, MAX_COLLISION_CONSTRAINTS, 1> lambdas;
        unsigned int cgiterations[MAX_COLLISION_CONSTRAINTS];
        unsigned int rs[MAX_COLLISION_CONSTRAINTS];
    */

    output.cgiterations_ct = 0;
    const float Inf = 1e20f;

    size_t iter;
    for (iter = 0; iter < iterMax; iter++) {
        vecGPQPf tList = vecGPQPf::Constant(Inf);
        size_t used_ct = 0;
        for (size_t i = 0; i < coll_ct; i++) {
            CollisionReference clr(collider->activeCollisions[i]);
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
            clr.g(clr.Ax() - clr.b());
            tList.segment(clr.mIndices(), 3 * clr.contactNum()) =
                collider->collisions[collision_indices[i]].compute_tbar(clr);
        }

        // using thrust to abstract sort and duplicate detection
        vecGPQPf tUniqueList = uniqueTList(tList, used_ct, true);
        float end = tUniqueList(used_ct - 1);
        if (end != Inf && used_ct < MAX_COLLISION_CONSTRAINTS) {
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

            for (size_t i = 0; i < this->body_count; i++) {
                this->bodies[i].get_rigid().LTx(vec6f::Zero());
            }
            for (size_t i = 0; i < coll_ct; i++) {
                CollisionReference clr(collider->activeCollisions[i]);
                collider->collisions[collision_indices[i]].compute_LTp(clr);
            }
            for (size_t i = 0; i < coll_ct; i++) {
                CollisionReference clr(collider->activeCollisions[i]);
                collider->collisions[collision_indices[i]].compute_LLTx(clr);

                const auto &b_cpy = clr.b();
                const auto &p_cpy = clr.p();
                const auto &Ax_cpy = clr.Ax();
                const auto &lambac_cpy = clr.lambdac();

                fPrime -= b_cpy.dot(p_cpy) + lambac_cpy.dot(Ax_cpy);
                fPrimePrime += p_cpy.dot(Ax_cpy);
            }

            float deltaTStar = -fPrime / fPrimePrime;
            if (fPrime > 0) {
                break;
            } else if (deltaTStar >= 0 && deltaTStar < t_cur - tc) {
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
        }

        // PCG, compute b_cg
        for (size_t i = 0; i < coll_ct; i++) {
            CollisionReference clr(collider->activeCollisions[i]);
            collider->collisions[collision_indices[i]]
                .compute_degenerate_J1I_J2I_b(clr);
        }

        // Init CG
        for (size_t i = 0; i < this->body_count; i++) {
            this->bodies[i].get_rigid().LTx(vec6f::Zero());
        }
        for (size_t i = 0; i < coll_ct; i++) {
            CollisionReference clr(collider->activeCollisions[i]);
            collider->collisions[collision_indices[i]]
                .compute_degenerate_LTlambda(clr);
        }
        for (size_t i = 0; i < coll_ct; i++) {
            CollisionReference clr(collider->activeCollisions[i]);
            collider->collisions[collision_indices[i]].compute_degenerate_LLTx(
                clr);
            clr.r_cg(clr.Ax() - clr.b_cg());
            // this.collider.collisions{i}.g_cg =
            // this.collider.collisions{i}.Minv_cg .*
            // this.collider.collisions{i}.r_cg;
            // FIXME: Minv_cg .* r_cg means element-wise multiplication, not
            // sure if this works
            clr.g_cg(clr.Minv_cg().cwiseProduct(clr.r_cg()));
            clr.d_cg(-clr.g_cg());
        }

        // CG iterations
        unsigned int CGiter;
        for (CGiter = 0; CGiter < CGiterMax; CGiter++) {
            float r = 0.0f;
            for (size_t i = 0; i < coll_ct; i++) {
                // The A ./ B divides all elements in A by B, so this is the
                // inverse
                CollisionReference clr(collider->activeCollisions[i]);
                auto M = clr.Minv_cg().cwiseInverse();
                auto r_cg = clr.r_cg();

                // Indexing into a vec24f (r_cg) with a vec24b (freeIndex) in
                // MATLAB will only select the true values, so we need to do
                // this manually
                // FIXME: r_cg(j)' * M(j) * r_cg might be more like r_cg.dot(M)
                // * r_cg, although these should be equivalent
                const vec24b &freeIndex = clr.freeIndex();
                for (size_t j = 0; j < 24; j++) {
                    if (freeIndex(j)) {
                        r += r_cg(j) * M(j) * r_cg(j);
                    }
                }
            }

            if (r < tol) {
                break;
            }

            for (size_t i = 0; i < this->body_count; i++) {
                this->bodies[i].get_rigid().LTx(vec6f::Zero());
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

                // FIXME: same as last
                const auto &freeIndex = clr.freeIndex();
                for (size_t j = 0; j < 24; j++) {
                    if (freeIndex(j)) {
                        numerator += clr.r_cg()(j) * clr.g_cg()(j);
                        denominator += clr.d_cg()(j) * clr.Ax()(j);
                    }
                }
            }

            float alpha = numerator / denominator;

            bool feasible = true;
            for (size_t i = 0; i < coll_ct; i++) {
                CollisionReference clr(collider->activeCollisions[i]);
                feasible &=
                    collider->collisions[collision_indices[i]].update_cg(clr,
                                                                         alpha);
            }
            if (!feasible) {
                break;
            }

            numerator = 0.0f;
            denominator = 0.0f;
            for (size_t i = 0; i < coll_ct; i++) {
                CollisionReference clr(collider->activeCollisions[i]);
                const auto &freeIndex = clr.freeIndex();

                for (size_t j = 0; j < 24; j++) {
                    if (freeIndex(j)) {
                        denominator += clr.r_cg()(j) * clr.g_cg()(j);
                        clr.r_cg(clr.r_cg() + alpha * clr.Ax());
                        clr.g_cg(clr.Minv_cg().cwiseProduct(clr.r_cg()));
                        numerator += clr.r_cg()(j) * clr.g_cg()(j);
                    }
                }
            }

            float beta = numerator / denominator;
            for (size_t i = 0; i < coll_ct; i++) {
                CollisionReference clr(collider->activeCollisions[i]);
                clr.d_cg(-clr.g_cg() + beta * clr.d_cg());
            }
        }

        // FIXME: eh
        output.cgiterations[output.cgiterations_ct++] = CGiter;

        // Project to feasible region
        for (size_t i = 0; i < coll_ct; i++) {
            CollisionReference clr(collider->activeCollisions[i]);
            collider->collisions[collision_indices[i]].project(clr);
        }

        // Test if results satisfy the KKT conditions
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
            clr.g(clr.Ax() - clr.b());
            f += clr.lambda().dot(0.5f * clr.Ax() - clr.b());
            g.segment(clr.mIndices(), 3 * clr.contactNum()) = clr.g();
            lambda.segment(clr.mIndices(), 3 * clr.contactNum()) = clr.lambda();
        }

        if (g.norm() < eps) {
            break;
        }
        if ((g - gPrev).norm() < eps) {
            break;
        }
        if ((f - fPrev) > -eps && (f - fPrev) < eps) {
            break;
        }

        fPrev = f;
        gPrev = g;
        output.rs[iter] = g.norm();
    }

    // FIXME: All that we really need back is output.lambdas for now, but all
    // will be updated
    output.iterations = iter;
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
