#include <getopt.h>

#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "apbd/BodyReference_impl.h"
#include "model_samples.h"

/* THESE SHOULD BE THE SAME AS IN cmaes.cpp */
#define NUM_ENVIRONMENTS 4096
#define DIM 6
#define GOAL_X 8.0f
#define GOAL_Y 0.0f
#define GOAL_Z 0.5f
// Although it is much faster(?), ptxas error   : File uses too much global constant data (0x18000 bytes, 0x10000 max)
// __constant__ float d_initialVels[DIM * NUM_ENVIRONMENTS];

using std::cout, std::endl, std::string, std::runtime_error, std::vector;
typedef std::chrono::high_resolution_clock Clock;

__global__ void __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS_PER_SM)
    simKernel(apbd::Model model, apbd::ModelBuffers buffers,
              apbd::Body *body_buffer, apbd::BodyReference *body_ptr_buffer,
              apbd::Collision *collision_buffer,
              unsigned int *active_collision_buffer, int sims,
              bool do_variations, float* d_initVels) {
    extern __shared__ unsigned char shared_memory[];
    // get this scene ID
    size_t scene_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (scene_id >= sims) {
        return;
    }

    model.copy_data_to_store(body_buffer);
    model.populate_shared_mem(shared_memory);
    Eigen::Matrix4f E = Eigen::Matrix4f::Identity();

    // This thread has its own copy of the model including bodies, this is what
    // we should modify
    apbd::Model thread_model = model.clone_with_buffers(buffers, scene_id);

    /*
        Each scene / env / world makes its own copy of the models based on the
       root model.

        Because we want different conditions for each scene, we can preload
       those conditions for each thread by passing in the buffer of floats to
       constant memory and indexing in.

        We should only do this for the 0th (lowest) box, but it is important to
       be aware that body_buffer was originally global; only after the
       clone_with_buffers have we branched into our new code where local[0] is
       really scene_id * model.body_count.
    */
    Eigen::Vector3f vel = Eigen::Vector3f(d_initVels[scene_id * DIM + 0],
                                          d_initVels[scene_id * DIM + 1],
                                          d_initVels[scene_id * DIM + 2]);
    thread_model.bodies[0].get_rigid().w(vel);

    vel = Eigen::Vector3f(d_initVels[scene_id * DIM + 3],
                          d_initVels[scene_id * DIM + 4],
                          d_initVels[scene_id * DIM + 5]);
    thread_model.bodies[0].get_rigid().v(vel);

    // create a thread-local collider
    auto collider = apbd::Collider(&thread_model, scene_id, body_ptr_buffer,
                                   collision_buffer, active_collision_buffer);
    // simulate
    thread_model.simulate(&collider);
}

/*
This kernel calculates for a scene or environment the L2 norm of the distances

FIXME: I am not taking the sqrt for now
https://docs.nvidia.com/cuda/cuda-c-programming-guide/#launch-bounds

TODO: Allocate space for dp, should be == NUM_ENVIRONMENTS (or sims)

One thing I reason is, do I return a single float for the objective function as
the sum? Wouldn't that make every initial velocity in x the same? Or should I
say minimum is zero?
*/
__global__ void __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS_PER_SM)
    computeL2Kernel(apbd::Model model, apbd::ModelBuffers buffers,
                    apbd::Body *body_buffer,
                    apbd::BodyReference *body_ptr_buffer,
                    apbd::Collision *collision_buffer,
                    unsigned int *active_collision_buffer, int sims,
                    float *dp) {
    // By call time device memory is populated, we just need to iterate over the
    // bodies to get dp
    int scene_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (scene_idx >= sims) {
        return;
    }

    // TODO: Make sure you aren't erroneously counting bodies towards the objective function!
    float dpTdp = 0.0f;
    for (size_t i = 0; i < model.body_count; i++) {
        if (i == 1) {
            continue;
        }

        Eigen::Vector3f pos = model.bodies[i].get_rigid().position();

        dpTdp += (pos(0) - GOAL_X) * (pos(0) - GOAL_X) +
                 (pos(1) - GOAL_Y) * (pos(1) - GOAL_Y) +
                 (pos(2) - GOAL_Z) * (pos(2) - GOAL_Z);
    }

    dp[scene_idx] = dpTdp;
}

void launchCMAESKernels(apbd::Model model, apbd::Body *bodies, int sims,
                        bool do_variations) {
    cout << "# thread blocks: " << (sims + BLOCK_SIZE - 1) / BLOCK_SIZE << endl;

    const size_t shared_size = model.get_shared_memory_size();

    apbd::BodyReference *body_ptr_buffer = nullptr;
    apbd::Collision *collision_buffer = nullptr;
    unsigned int *active_collision_buffer = nullptr;

    /* CMAES VELOCITIES */
    // Read in the float x* and update each model's initial velocities
    fs::path p = fs::current_path();
    fs::path VELOCITIES_PATH = fs::current_path() / "cmaes/data/velocities.txt";
    cout << "# Trying to read velocities from " << VELOCITIES_PATH << endl;
    std::ifstream fin(VELOCITIES_PATH);
    if (!fin.is_open()) {
        cout << VELOCITIES_PATH << endl;
        exit(1);
    }

    float *h_initVels = new float[DIM * NUM_ENVIRONMENTS];
    float *d_initVels = nullptr;
    CUDA_CHECK(cudaMalloc(&d_initVels, sizeof(float) * DIM * NUM_ENVIRONMENTS));
    for (int i = 0; i < DIM * NUM_ENVIRONMENTS; i++) {
        float tmp;
        if (!(fin >> tmp)) {
            cout << "Failed to read value at index " << i << endl;
            exit(1);
        }

        h_initVels[i] = tmp;
        // printf("h_init[%d] = %f\n", i, h_initVels[i]);
    }

    cudaMemcpy(d_initVels, h_initVels, sizeof(float) * DIM * NUM_ENVIRONMENTS, cudaMemcpyHostToDevice);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // We don't need it anymore
    delete[] h_initVels;

    // Inside allocate_buffer alloc_device ... cudaMalloc(&d_dp, sims *
    // sizeof(float)) is called
    float *h_dp = new float[sims];
    float *d_dp = nullptr;
    apbd::Collider::allocate_buffers(model, sims, body_ptr_buffer,
                                     collision_buffer, active_collision_buffer,
                                     d_dp);
    auto buffers = apbd::Model::allocate_buffers(sims, model);
    if (d_dp == nullptr) {
        printf("d_dp not allocated properly, exiting\n");
        exit(1);
    }
    cudaMemcpy(d_dp, h_dp, sizeof(float) * sims, cudaMemcpyHostToDevice);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    model.move_to_device();
    bodies = move_array_to_device(bodies, model.body_count);

    auto t1 = Clock::now();

    simKernel<<<(sims + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE,
                shared_size>>>(model, buffers, bodies, body_ptr_buffer,
                               collision_buffer, active_collision_buffer, sims,
                               do_variations, d_initVels);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t2 = Clock::now();
    std::cout << "# Kernel took: " << (t2 - t1).count() << endl;

    // Run L2 kernel to find how far we are from the boxes
    t1 = Clock::now();
    // (a + b - 1) / b is the same as ceil(a / b)
    computeL2Kernel<<<(sims + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE,
                      shared_size>>>(model, buffers, bodies, body_ptr_buffer,
                                     collision_buffer, active_collision_buffer,
                                     sims, d_dp);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    t2 = Clock::now();
    // std::cout << "L2 Kernel took: " << (t2 - t1).count() << endl;

    cudaMemcpy(h_dp, d_dp, sizeof(float) * sims, cudaMemcpyDeviceToHost);
    // for (int i = 0; i < sims; i++) {
    //     cout << "# dp[" << i << "]: " << h_dp[i] << endl;
    // }
    cout << "# dp[0] = " << h_dp[0] << endl;

    // Write to cmaes/data/objective.txt
    fs::path OBJECTIVE_PATH = fs::current_path() / "cmaes/data/objective.txt";
    std::ofstream fout(OBJECTIVE_PATH);
    if (!fout.is_open()) {
        throw std::runtime_error("Couldn't open " + OBJECTIVE_PATH.string() +
                                 " for writing");
    }
    for (size_t i = 0; i < NUM_ENVIRONMENTS - 1; i++) {
        fout << h_dp[i] << ' ';
    }
    fout << h_dp[NUM_ENVIRONMENTS - 1];
    fout.close();

    delete[] h_dp;
}

// TODO: Move all this to its own util folder
struct MainState {
    int model_id;
    unsigned long scene_count;
    unsigned long substeps;
    bool variations;
};

const char *HELP =
    "\
arguments:          \n\
  -i, --model_id ID    The model scene to use\n\
  -s, --scene-count N  The number of simulations to run\n\
  -t, --substeps N     The number of substeps to use\n\
  -v, --variations     Adds different variations to each scene\n\
  -h, --help           Show this help text\n\
";

MainState parse_arguments(int argc, char *argv[]) {
    struct MainState state = {
        .model_id = 0,
        .scene_count = 1,
        .substeps = 20,
        .variations = false,
    };

    option longopts[] = {{"model_id", required_argument, NULL, 'm'},
                         {"scene-count", required_argument, NULL, 's'},
                         {"substeps", required_argument, NULL, 't'},
                         {"variations", no_argument, NULL, 'v'},
                         {"help", no_argument, NULL, 'h'},
                         {0}};

    while (1) {
        const int opt = getopt_long(argc, argv, "hm:s:t:v", longopts, 0);

        if (opt == -1) {
            break;
        }
        string o;
        switch (opt) {
            case 'h':
                cout << HELP << endl;
                exit(0);
            case 'm':
                if (optarg == NULL) {
                    break;
                }
                cout << "# model_id: " << optarg << endl;
                state.model_id = std::stoi(optarg);
                break;
            case 's':
                if (optarg == NULL) {
                    break;
                }
                cout << "# scene-count: " << optarg << endl;
                state.scene_count = std::stoul(optarg);
                break;
            case 't':
                if (optarg == NULL) {
                    break;
                }
                cout << "# substeps: " << optarg << endl;
                state.substeps = std::stoul(optarg);
                break;
            case 'v':
                cout << "# variations: true" << endl;
                state.variations = true;
                break;
            case '?':
            default:
                cout << "unknown option." << endl;
        }
    }
    return state;
}

int main(int argc, char *argv[]) {
    auto state = parse_arguments(argc, argv);
    apbd::Body *bodies;

    auto model = createModelSample(state.model_id, 1e-2, state.substeps, bodies,
                                   state.scene_count);

    auto t1 = Clock::now();
#ifdef USE_CUDA
    cout << "# Running with CUDA #" << endl;
    launchCMAESKernels(model, bodies, state.scene_count, state.variations);
#else
    throw std::runtime_error(
        "# Rebuild with -DUSE_CUDA=ON, CPU is not supported!");
#endif
    auto t2 = Clock::now();
    cout << "# Simulation took: " << (t2 - t1).count() << endl;
}
