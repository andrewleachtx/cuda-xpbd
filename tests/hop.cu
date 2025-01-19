#include <getopt.h>

#include <exception>
#include <iostream>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "apbd/BodyReference_impl.h"
#include "model_samples.h"

/* THESE SHOULD BE THE SAME AS cmaes.cpp */
const int NUM_ENVIRONMENTS = 1;
const int DIM              = 6;

using std::cout, std::endl, std::string, std::runtime_error, std::vector;
typedef std::chrono::high_resolution_clock Clock;

__global__ void __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS_PER_SM)
    kernel(apbd::Model model, apbd::ModelBuffers buffers,
           apbd::Body *body_buffer, apbd::BodyReference *body_ptr_buffer,
           apbd::Collision *collision_buffer,
           unsigned int *active_collision_buffer, int sims,
           bool do_variations) {

    extern __shared__ unsigned char shared_memory[];
    // get this scene ID
    size_t scene_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (scene_id >= sims) {
        return;
    }

    // make a copy of the model
    model.copy_data_to_store(body_buffer);
    model.populate_shared_mem(shared_memory);
    Eigen::Matrix4f E = Eigen::Matrix4f::Identity();

    apbd::Model thread_model = model.clone_with_buffers(buffers, scene_id);

    // create a thread-local collider
    auto collider = apbd::Collider(&thread_model, scene_id, body_ptr_buffer,
                                   collision_buffer, active_collision_buffer);
    // simulate
    thread_model.simulate(&collider);
}

float g_x[DIM * NUM_ENVIRONMENTS] = {0.0f};

void run_kernelCMAES(apbd::Model model, apbd::Body *bodies, int sims,
                     bool do_variations) {
    cout << "# thread blocks: " << (sims + BLOCK_SIZE - 1) / BLOCK_SIZE << endl;

    const size_t shared_size = model.get_shared_memory_size();

    apbd::BodyReference *body_ptr_buffer = nullptr;
    apbd::Collision *collision_buffer = nullptr;
    unsigned int *active_collision_buffer = nullptr;
    apbd::Collider::allocate_buffers(model, sims, body_ptr_buffer,
                                     collision_buffer, active_collision_buffer);
    auto buffers = apbd::Model::allocate_buffers(sims, model);

    // Read in the float x* and update each model's initial velocities
    std::ifstream fin("./cmaes/data/velocities.txt");
    if (!fin.is_open()) {
        printf("Failed to open velocities.txt\n");
        exit(1);
    }

    for (int i = 0; i < DIM * NUM_ENVIRONMENTS; i++) {
        fin >> g_x[i];
    }

    // Now update each bodies velocity
    for (size_t i = 0; i < model.body_count; i++) {
        Eigen::Matrix<float, 6, 1> init_velocity;
        for (int j = 0; j < DIM; j++) {
            // 0,1,2nd body * 6 + 0 = index 12, or 0 6 12
            init_velocity(j) = g_x[i * DIM + j];
        }
        model.bodies[i].setInitVelocity(init_velocity);
    }

    model.move_to_device();
    bodies = move_array_to_device(bodies, model.body_count);

    auto t1 = Clock::now();

    kernel<<<(sims + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, shared_size>>>(
        model, buffers, bodies, body_ptr_buffer, collision_buffer,
        active_collision_buffer, sims, do_variations);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t2 = Clock::now();
    std::cout << "# Kernel took: " << (t2 - t1).count() << '\t';
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

    // Guaranteed to use model # 99
    assert(state.model_id == 99 && "Model ID must be 99");
    
    auto model = createModelSample(state.model_id, 1e-2, state.substeps, bodies,
                                   state.scene_count);

    auto t1 = Clock::now();
#ifdef USE_CUDA
    cout << "# Running with CUDA #" << endl;
    run_kernelCMAES(model, bodies, state.scene_count, state.variations);
#else
    std::runtime_error("# Rebuild with -DUSE_CUDA=ON, CPU is not supported!");
#endif
    auto t2 = Clock::now();
    cout << " Simulation took: " << (t2 - t1).count() << '\n';
}
