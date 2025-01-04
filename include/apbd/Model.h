#pragma once
#include <cstddef>

#include "BodyReference.h"
#include "Collider.h"
#include "Constraint.h"
#include "data/soa.h"

namespace apbd {
class Collider;

struct ModelBuffers {
    Constraint *constraints;
};

/*
    output.iterations
    output.lambdas
    output.cgiterations
    output.rs

    Because we can't use dynamic sized arrays, we will instead store an upper
   bound (in config.h) and a size for each member that represents one
*/
struct GPQPOutput {
    Eigen::Matrix<float, MAX_COLLISION_CONSTRAINTS, 1> iterations;
    Eigen::Matrix<float, MAX_COLLISION_CONSTRAINTS, 1> lambdas;
    unsigned int cgiterations[MAX_COLLISION_CONSTRAINTS];
    unsigned int rs[MAX_COLLISION_CONSTRAINTS];

    int iterations_ct;
    int lambdas_ct;
    int cgiterations_ct;
    int rs_ct;
};

/**
 * A simulation model, contains all information necessary to run a single
 * simulation. Designed to be copied to each thread and modified with any
 * per-thread differences.
 */
class Model {
   public:
    /// Duration of one simulation step
    float h;
    /// Duration of the entire simulation
    float tEnd;
    unsigned int substeps;
    /// A list of references to the bodies for this simulation, will be
    /// initialized when data is copied to the global store.
    BodyReference *bodies;
    size_t body_count;
    Constraint *constraints;
    size_t constraint_count;

    Eigen::Vector3f gravity;
    unsigned int forward_iters;
    unsigned int reverse_iters;

    Eigen::Matrix4f ground_E;
    float ground_size;

    /// The number of simulation steps. Set when the model is initialized from h
    /// and tEnd
    unsigned int steps;

    __host__ __device__ void stepBDF1(float hs);
    __host__ __device__ void constructConstraintGraph(Collider *collider);
    __host__ __device__ void solveConSP(float hs);
    __host__ __device__ void solveConGS(Collider *collider, float hs);
    __host__ __device__ void solveConTGS(Collider *collider, float hs);
    __host__ __device__ void solveConGPQP(Collider *collider);
    __host__ __device__ GPQPOutput GPQP(Collider *collider, int n);

    /**
     * Constructs default data structures
     */
    Model();
    /**
     * Constructs a copy of the model, only duplicating data that cannot be
     * shared
     */
    __host__ __device__ Model(const Model &other);
    /**
     * Constructs a copy of the model, moving all data from other to this.
     * Note: does not handle layer counts
     */
    __host__ __device__ Model(const Model &&other);
    /**
     * Moves the arrays allocated in this model to device storage.
     */
    void move_to_device();
    /**
     * Initializes the model objects based on configuration
     */
    __host__ __device__ void init();
    /**
     * Runs all simulations to completion
     */
    __host__ __device__ void simulate(Collider *collider);
    /**
     * Writes current state out for debugging
     */
    __host__ __device__ void write_state(unsigned int step);
    /**
     * Prints the configuration of this model to stdout
     */
    __host__ __device__ void print_config();
    /**
     * Creates the global data store object based on this model and the number
     * of scenes.
     */
    void create_store(size_t scene_count);
    /**
     * Copies the data in this model that uses SOA to the global store
     */
    __host__ __device__ void copy_data_to_store(Body *body_array);
    /**
     * Allocates a set of buffers necessary to hold `count` number of `model`'s
     * data.
     */
    static ModelBuffers allocate_buffers(size_t count, const Model &model);

    /**
     * Clones a model using the given buffers as a backing data store.
     */
    __host__ __device__ Model clone_with_buffers(const ModelBuffers &buffers,
                                                 size_t offset);

    /**
     * Gets the total shared memory size needed to store data for this model.
     * Shared memory size includes all threads in a block, as all data is
     * identical between threads and treated as read-only.
     */
    __host__ __device__ size_t get_shared_memory_size();

    /**
     * Stores data from this model into shared memory. Assumes that this
     * function is being run by all threads, and that shared_memory is the
     * size given by get_shared_memory_size.
     */
    __device__ void populate_shared_mem(void *shared_memory);

    /**
     * Gets the index of a collision within the Collider that corresponds to
     * these bodies
     */
    __host__ __device__ unsigned int get_collision_index(BodyReference body1,
                                                         BodyReference body2);
};
}  // namespace apbd
