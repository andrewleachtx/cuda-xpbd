#include "apbd/BodyReference_impl.h"
#include "apbd/CollisionReference_impl.h"
#include "apbd/Collisions_impl.h"
#include "apbd/Model.h"
#include "util.h"
#include <chrono>
#include <thread>

#include <iostream>
#include <stdexcept>

namespace apbd
{

    Model::Model()
        : h(1. / 30.), tEnd(1), substeps(10), bodies(nullptr), body_count(0),
          constraints(nullptr), constraint_count(0), gravity(0.0, 0.0, -980.0),
          forward_iters(0), reverse_iters(0), ground_E(Eigen::Matrix4f::Zero()),
          ground_size(10), steps(0) {}

    Model::Model(const Model &other)
        : h(other.h), tEnd(other.tEnd), substeps(other.substeps),
          bodies(other.bodies), body_count(other.body_count), constraints(nullptr),
          constraint_count(other.constraint_count), gravity(other.gravity),
          forward_iters(other.forward_iters), reverse_iters(other.reverse_iters),
          ground_E(other.ground_E), ground_size(other.ground_size),
          steps(other.steps) {}

    Model::Model(const Model &&other)
        : h(other.h), tEnd(other.tEnd), substeps(other.substeps),
          bodies(other.bodies), body_count(other.body_count),
          constraints(other.constraints), constraint_count(other.constraint_count),
          gravity(other.gravity), forward_iters(other.forward_iters),
          reverse_iters(other.reverse_iters), ground_E(other.ground_E),
          ground_size(other.ground_size), steps(other.steps) {}

    void Model::create_store(size_t scene_count)
    {
        // TODO: handle this constraints
        data::SOAStore data_store(
            this->body_count, MAX_COLLISION_CONSTRAINTS, MAX_COLLISION_CONSTRAINTS,
            this->body_count * (this->body_count + 1) / 2, scene_count);

#ifdef USE_CUDA
        cudaMemcpyToSymbol(data::device_global_store, &data_store,
                           sizeof(data::SOAStore), size_t(0), cudaMemcpyHostToDevice);
#else
        data::global_store = std::move(data_store);
#endif
    }

    void Model::copy_data_to_store(Body *body_array)
    {
        for (size_t i = 0; i < this->body_count; i++)
        {
            auto &body = body_array[i];
            switch (body.type)
            {
            case BODY_RIGID:
            {
                data::global_store.BodyRigid.set(data::soa_index(i), body.data.rigid);
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

    ModelBuffers Model::allocate_buffers(size_t count, const Model &model)
    {
        ModelBuffers buffers;
        if (model.constraint_count > 0)
            buffers.constraints =
                alloc_device<Constraint>(count * model.constraint_count);
        else
            buffers.constraints = nullptr;
        return buffers;
    }

    Model Model::clone_with_buffers(const ModelBuffers &buffers, size_t offset)
    {
        Model new_model = Model(*this);
        DEBUG_ASSERT(&new_model != this, "Model not cloned!");
        if (this->constraint_count > 0)
        {
            new_model.constraints =
                &buffers.constraints[offset * this->constraint_count];
            memcpy_device(new_model.constraints, this->constraints,
                          this->constraint_count);
        }
        return new_model;
    }

    void Model::init()
    {
        // bodies are initialized when data is copied to store
        for (size_t i = 0; i < this->constraint_count; i++)
        {
            this->constraints[i].init();
        }
        // calculate parameters
        this->steps = ceil(this->tEnd / this->h);
        this->print_config();
    }

    void Model::move_to_device()
    {
        bodies = move_array_to_device(bodies, body_count);
        constraints = move_array_to_device(constraints, constraint_count);
    }

    void Model::simulate(Collider *collider)
    {
        float hs = this->h / static_cast<float>(this->substeps);
        // printf("Simulating a total of %u steps.\n", this->steps);
        for (unsigned int step = 0; step < this->steps; step++)
        {
            // printf("==== Step %u starting ====\n", step);
            this->solveConTGS(collider, hs);

            for (size_t i = 0; i < this->body_count; i++)
            {
                auto pos = this->bodies[i].get_rigid().position();
                auto vel = this->bodies[i].get_rigid().v();
                auto w = this->bodies[i].get_rigid().w();
                auto r = this->bodies[i].get_rigid().rotation().coeffs();
                // printf("Body %zu: Position = [%f, %f, %f], Velocity = [%f, %f, %f]\n",
                //        i, pos(0), pos(1), pos(2), vel(0), vel(1), vel(2));
                // printf("Body Angular Velocity: [%f, %f, %f]\n", w(0), w(1), w(2));
                // printf("Body Rotation: [%f, %f, %f, %f]\n", r(0), r(1), r(2), r(3));
            }

            this->write_state(step + 1);
        }
    }

    /** Private Functions **/

    void Model::stepBDF1(float hs)
    {
        for (size_t body_i = 0; body_i < this->body_count; body_i++)
        {
            this->bodies[body_i].stepBDF1(hs, this->gravity);
        }
    }

    void Model::solveConTGS(Collider *collider, float hs)
    {
        // ---- START DEBUG PRINT ----
        // printf("==== solveConTGS start, hs = %f ====\n", hs);
        // Print basic info about the bodies before we do anything
        // for (size_t i = 0; i < this->body_count; i++)
        // {
        //     auto pos = this->bodies[i].get_rigid().position();
        //     auto vel = this->bodies[i].get_rigid().v();
        //     auto w = this->bodies[i].get_rigid().w();
        //     printf("Initial Body %zu: pos=[%f, %f, %f], vel=[%f, %f, %f]\n",
        //            i, pos.x(), pos.y(), pos.z(), vel.x(), vel.y(), vel.z());
        //     printf("Body Angular Velocity: [%f, %f, %f]\n", w(0), w(1), w(2));
        // }
        // ---- END DEBUG PRINT ----

        this->stepBDF1(this->h);
        collider->run(this);
        float biasCoefficient = 2 * sqrt(hs / this->h);
        // 1e20 is used instead of oo for maximum hardware compatibilty/predictability
        const float Inf = 1e20;

        // We solve contstraints in the layer order. The exact layer sizes don't
        // matter at this step, so we don't bother walking through each layer
        // individually.

        // ---- START DEBUG PRINT ----
        // printf("[After stepBDF1(%f)]\n", this->h);
        // for (size_t i = 0; i < this->body_count; i++)
        // {
        //     auto pos = this->bodies[i].get_rigid().position();
        //     auto vel = this->bodies[i].get_rigid().v();
        //     // Check for NaN
        //     if (!pos.allFinite() || !vel.allFinite())
        //     {
        //         printf("** NaN DETECTED after stepBDF1 for body %zu **\n", i);
        //     }
        //     printf("Body %zu: pos=[%f, %f, %f], vel=[%f, %f, %f]\n",
        //            i, pos.x(), pos.y(), pos.z(), vel.x(), vel.y(), vel.z());
        // }
        // ---- END DEBUG PRINT ----

        // Shock propagation
        for (size_t i = 0; i < collider->active_collision_count; i++)
        {
            CollisionReference clr(collider->activeCollisions[i]);
            collider->collisions[collider->activeCollisions[i]].initConstraints(clr);
        }
        for (size_t i = 0; i < collider->active_collision_count; i++)
        {
            CollisionReference clr(collider->activeCollisions[i]);
            for (unsigned int j = 0; j < this->forward_iters; j++)
            {
                collider->collisions[collider->activeCollisions[i]].solveCollisionNor(
                    clr, hs, biasCoefficient, -Inf, true);
            }
        }

        // ---- START DEBUG PRINT ----
        // printf("[After forward_iters, shock propagation]\n");
        // for (size_t i = 0; i < this->body_count; i++)
        // {
        //     auto pos = this->bodies[i].get_rigid().position();
        //     auto vel = this->bodies[i].get_rigid().v();
        //     if (!pos.allFinite() || !vel.allFinite())
        //     {
        //         printf("** NaN DETECTED after forward_iters for body %zu **\n", i);
        //     }
        //     printf("Body %zu: pos=[%f, %f, %f], vel=[%f, %f, %f]\n",
        //            i, pos.x(), pos.y(), pos.z(), vel.x(), vel.y(), vel.z());
        // }
        // ---- END DEBUG PRINT ----

        // work backward now
        // order within layers might matter, but this is much simpler
        for (long int i = collider->active_collision_count - 1; i >= 0; i--)
        {
            CollisionReference clr(collider->activeCollisions[i]);
            for (unsigned int j = 0; j < this->reverse_iters; j++)
            {
                collider->collisions[collider->activeCollisions[i]].solveCollisionNor(
                    clr, hs, biasCoefficient, -Inf, true);
            }
            collider->collisions[collider->activeCollisions[i]].applyLambdaSP(clr);
        }

        // ---- START DEBUG PRINT ----
        // printf("[After reverse_iters]\n");
        // for (size_t i = 0; i < this->body_count; i++)
        // {
        //     auto pos = this->bodies[i].get_rigid().position();
        //     auto vel = this->bodies[i].get_rigid().v();
        //     if (!pos.allFinite() || !vel.allFinite())
        //     {
        //         printf("** NaN DETECTED after reverse_iters for body %zu **\n", i);
        //     }
        //     printf("Body %zu: pos=[%f, %f, %f], vel=[%f, %f, %f]\n",
        //            i, pos.x(), pos.y(), pos.z(), vel.x(), vel.y(), vel.z());
        // }
        // ---- END DEBUG PRINT ----

        for (size_t i = 0; i < this->body_count; i++)
        {
            this->bodies[i].updateStates(hs);
        }

        // printf("[After first updateStates]\n");
        // for (size_t i = 0; i < this->body_count; i++)
        // {
        //     auto pos = this->bodies[i].get_rigid().position();
        //     auto vel = this->bodies[i].get_rigid().v();
        //     if (!pos.allFinite() || !vel.allFinite())
        //     {
        //         printf("** NaN DETECTED after first updateStates for body %zu **\n", i);
        //     }
        //     printf("Body %zu: pos=[%f, %f, %f], vel=[%f, %f, %f]\n",
        //            i, pos.x(), pos.y(), pos.z(), vel.x(), vel.y(), vel.z());
        // }

        unsigned int ks = 0;
        while (ks < this->substeps)
        {
            for (size_t constraint_i = 0; constraint_i < this->constraint_count;
                 constraint_i++)
            {
                this->constraints[constraint_i].clear();
            }

            // Gauss-Seidel solve for non-collision constraints
            for (size_t constraint_i = 0; constraint_i < this->constraint_count;
                 constraint_i++)
            {
                this->constraints[constraint_i].solve();
            }

            // Gauss-Seidel for collisions
            for (size_t i = 0; i < collider->active_collision_count; i++)
            {
                CollisionReference clr(collider->activeCollisions[i]);
                collider->collisions[collider->activeCollisions[i]].solveCollisionNor(
                    clr, hs, biasCoefficient, -Inf, false);
                collider->collisions[collider->activeCollisions[i]].solveCollisionTan(
                    clr, hs, biasCoefficient, false);
            }

            for (size_t i = 0; i < this->body_count; i++)
            {
                this->bodies[i].updateStates(hs);
            }

            // printf("[Substep %u / %u]\n", ks + 1, this->substeps);
            // for (size_t i = 0; i < this->body_count; i++)
            // {
            //     auto pos = this->bodies[i].get_rigid().position();
            //     auto vel = this->bodies[i].get_rigid().v();
            //     if (!pos.allFinite() || !vel.allFinite())
            //     {
            //         printf("** NaN DETECTED in substep for body %zu **\n", i);
            //     }
            //     printf("Body %zu: pos=[%f, %f, %f], vel=[%f, %f, %f]\n",
            //            i, pos.x(), pos.y(), pos.z(), vel.x(), vel.y(), vel.z());
            // }

            ks++;
        }

        for (size_t i = 0; i < collider->active_collision_count; i++)
        {
            CollisionReference clr(collider->activeCollisions[i]);
            collider->collisions[collider->activeCollisions[i]].solveCollisionNor(
                clr, hs, biasCoefficient, 0, false);
            collider->collisions[collider->activeCollisions[i]].solveCollisionTan(
                clr, hs, biasCoefficient, false);
        }

        for (size_t i = 0; i < this->body_count; i++)
        {
            this->bodies[i].integrateStates();
        }

        // // ---- START DEBUG PRINT ----
        // printf("[After final integrateStates]\n");
        // for (size_t i = 0; i < this->body_count; i++)
        // {
        //     auto pos = this->bodies[i].get_rigid().position();
        //     auto vel = this->bodies[i].get_rigid().v();
        //     if (!pos.allFinite() || !vel.allFinite())
        //     {
        //         printf("** NaN DETECTED after final integrateStates for body %zu **\n", i);
        //     }
        //     printf("Body %zu final: pos=[%f, %f, %f], vel=[%f, %f, %f]\n",
        //            i, pos.x(), pos.y(), pos.z(), vel.x(), vel.y(), vel.z());
        // }
        // // ---- END DEBUG PRINT ----

        // printf("==== solveConTGS end ====\n");
    }

    void Model::write_state(unsigned int step)
    {
#ifdef WRITE
#ifdef __CUDA_ARCH__
        if (threadIdx.x == 0)
            printf("Step %d\n", step);
        // print up to 8 simulations in parallel
        for (size_t i = 0; i < body_count * 8; i++)
        {
            if (i / body_count != threadIdx.x)
                continue;
            printf("%lu ", i);
            bodies[i % body_count].write_state();
            printf("\n");
        }
#else
        using namespace std::chrono_literals;
        if (_thread_scene_id == 0)
            printf("Step %d\n", step);
        for (size_t i = 0; i < body_count * 8; i++)
        {
            if (i / body_count != _thread_scene_id)
            {
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

    void Model::print_config()
    {
        printf("# Body count: %lu\n"
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

    size_t Model::get_shared_memory_size()
    {
        return sizeof(BodyReference) * this->body_count;
    }

    __device__ void Model::populate_shared_mem(void *shared_memory)
    {
        BodyReference *shared_bodies =
            reinterpret_cast<BodyReference *>(shared_memory);

        if (threadIdx.x == 0)
            for (size_t i = 0; i < this->body_count; i++)
            {
                shared_bodies[i] = this->bodies[i];
            }
        __syncthreads();

        this->bodies = shared_bodies;
    }

    unsigned int Model::get_collision_index(BodyReference body1,
                                            BodyReference body2)
    {
        DEBUG_ASSERT(body1.index != body2.index,
                     "Cannot find collision index for a body and itself.");
        unsigned int x, y;
        unsigned int N = this->body_count;
        if (body1.index < body2.index)
        {
            x = body2.index;
            y = body1.index;
        }
        else
        {
            x = body1.index;
            y = body2.index;
        }
        // Gives the index of an element in a triangle grid where y < x,
        // plus a full row at the bottom (reserved for the ground)
        return N + (x + y * N) - ((y + 2) * (y + 1)) / 2;
    }

} // namespace apbd
