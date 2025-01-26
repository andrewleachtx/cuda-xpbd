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

/* FIXME: QUADRUPLE CHECK THESE */
#define GOAL_X 0.0f
#define GOAL_Y 0.0f
#define GOAL_Z 10.5f

#define GOAL_IDX 10
#define LAUNCH_IDX 10
#define W1 1e-3f
#define W2 1e0f

using std::cout, std::endl, std::string, std::runtime_error, std::vector;
typedef std::chrono::high_resolution_clock Clock;

__global__ void __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS_PER_SM)
    simKernel(apbd::Model model, apbd::ModelBuffers buffers,
              apbd::Body *body_buffer, apbd::BodyReference *body_ptr_buffer,
              apbd::Collision *collision_buffer,
              unsigned int *active_collision_buffer, int sims,
              bool do_variations, float *d_initVels, float *d_finalVels,
              float *d_objectives) {
    extern __shared__ unsigned char shared_memory[];
    // get this scene ID
    size_t scene_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (scene_idx >= sims) {
        return;
    }

    model.copy_data_to_store(body_buffer);
    model.populate_shared_mem(shared_memory);
    Eigen::Matrix4f E = Eigen::Matrix4f::Identity();

    // This thread has its own copy of the model including bodies, this is what
    // we should modify
    apbd::Model thread_model = model.clone_with_buffers(buffers, scene_idx);

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
    Eigen::Vector3f vel = Eigen::Vector3f(d_initVels[scene_idx * DIM + 0],
                                          d_initVels[scene_idx * DIM + 1],
                                          d_initVels[scene_idx * DIM + 2]);
    thread_model.bodies[LAUNCH_IDX].get_rigid().w(vel);

    vel = Eigen::Vector3f(d_initVels[scene_idx * DIM + 3],
                          d_initVels[scene_idx * DIM + 4],
                          d_initVels[scene_idx * DIM + 5]);
    thread_model.bodies[LAUNCH_IDX].get_rigid().v(vel);

    // create a thread-local collider
    auto collider = apbd::Collider(&thread_model, scene_idx, body_ptr_buffer,
                                   collision_buffer, active_collision_buffer);
    // simulate
    thread_model.simulate(&collider);

    /////////////// CALCULATE DP ///////////////
    /* CALCULATE DP */
    // TODO: Make sure you aren't erroneously counting bodies towards the
    // objective function!
    float dpTdp = 0.0f;
    Eigen::Vector3f pos = thread_model.bodies[GOAL_IDX].get_rigid().position();

    dpTdp += (pos(0) - GOAL_X) * (pos(0) - GOAL_X) +
             (pos(1) - GOAL_Y) * (pos(1) - GOAL_Y) +
             (pos(2) - GOAL_Z) * (pos(2) - GOAL_Z);

    // pos = model.bodies[2].get_rigid().position();

    // dpTdp += (pos(0) - GOAL_2X) * (pos(0) - GOAL_2X) +
    //          (pos(1) - GOAL_2Y) * (pos(1) - GOAL_2Y) +
    //          (pos(2) - GOAL_2Z) * (pos(2) - GOAL_2Z);

    d_objectives[scene_idx] = dpTdp;

    /////////////// STORE FINAL VELOCITIES ///////////////
    // store final velocities
    // FIXME: Should unroll(?), strides are weird
    for (size_t j = 0; j < 3; j++) {
        d_finalVels[scene_idx * DIM + j] =
            thread_model.bodies[GOAL_IDX].get_rigid().w()(j);
        d_finalVels[scene_idx * DIM + j + 3] =
            thread_model.bodies[GOAL_IDX].get_rigid().v()(j);
    }
}

void launchCMAESKernels(apbd::Model model, apbd::Body *bodies, int sims,
                        bool do_variations, int64_t &kernel_time) {
    cout << "# thread blocks: " << (sims + BLOCK_SIZE - 1) / BLOCK_SIZE << endl;

    const size_t shared_size = model.get_shared_memory_size();

    apbd::BodyReference *body_ptr_buffer = nullptr;
    apbd::Collision *collision_buffer = nullptr;
    unsigned int *active_collision_buffer = nullptr;

    ////////////////////////////// CMAES VELOCITY READ //////////////////////////////
    // TODO: Make inline func do this
    // Read in the float x* and update each model's initial velocities
    float *h_initVels  = new float[DIM * NUM_ENVIRONMENTS];
    float *h_finalVels = new float[DIM * NUM_ENVIRONMENTS];
    // cudaMallocHost(&h_initVels, sizeof(float) * DIM * NUM_ENVIRONMENTS);

    fs::path p = fs::current_path();
    fs::path VELOCITIES_PATH = fs::current_path() / "cmaes/data/velocities.txt";
    cout << "# Trying to read velocities from " << VELOCITIES_PATH << endl;
    std::ifstream fin(VELOCITIES_PATH);
    if (!fin.is_open()) {
        cout << VELOCITIES_PATH << endl;
        exit(1);
    }
    for (int i = 0; i < DIM * NUM_ENVIRONMENTS; i++) {
        float tmp;
        if (!(fin >> tmp)) {
            cout << "Failed to read value at index " << i << endl;
            exit(1);
        }

        h_initVels[i] = tmp;
    }
    fin.close();

    ////////////////////////////// DEV PTR ALLOC & COPY //////////////////////////////

    float *d_initVels = nullptr;
    float *d_finalVels = nullptr;
    CUDA_CHECK(cudaMalloc(&d_initVels, sizeof(float) * DIM * NUM_ENVIRONMENTS));
    CUDA_CHECK(cudaMalloc(&d_finalVels, sizeof(float) * DIM * NUM_ENVIRONMENTS));

    cudaMemcpy(d_initVels, h_initVels, sizeof(float) * DIM * NUM_ENVIRONMENTS, cudaMemcpyHostToDevice);
    CUDA_CHECK(cudaGetLastError());

    cudaMemcpy(d_finalVels, h_initVels, sizeof(float) * DIM * NUM_ENVIRONMENTS, cudaMemcpyHostToDevice);
    CUDA_CHECK(cudaGetLastError());

    float *h_objectives = new float[NUM_ENVIRONMENTS];
    // cudaMallocHost(&h_objectives, sizeof(float) * DIM * NUM_ENVIRONMENTS);
    float *d_objectives = nullptr;
    apbd::Collider::allocate_buffers(model, sims, body_ptr_buffer,
                                     collision_buffer, active_collision_buffer,
                                     d_objectives);
    auto buffers = apbd::Model::allocate_buffers(sims, model);
    if (d_objectives == nullptr) {
        printf("d_objs not allocated properly, exiting\n");
        exit(1);
    }
    cudaMemcpy(d_objectives, h_objectives, sizeof(float) * sims, cudaMemcpyHostToDevice);
    CUDA_CHECK(cudaGetLastError());

    model.move_to_device();
    bodies = move_array_to_device(bodies, model.body_count);

    ////////////////////////////// KERNEL LAUNCH //////////////////////////////
    auto t1 = Clock::now();
    simKernel<<<(sims + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE,
                shared_size>>>(model, buffers, bodies, body_ptr_buffer,
                               collision_buffer, active_collision_buffer, sims,
                               do_variations, d_initVels, d_finalVels, d_objectives);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t2 = Clock::now();
    std::cout << "# Kernel Took: " << (t2 - t1).count() << endl;
    kernel_time += (t2 - t1).count();

    // cudaMallocHost(&h_finalVels, sizeof(float) * DIM * NUM_ENVIRONMENTS);
    cudaMemcpy(h_finalVels, d_finalVels, sizeof(float) * DIM * NUM_ENVIRONMENTS, cudaMemcpyDeviceToHost);
    CUDA_CHECK(cudaGetLastError());

    cudaMemcpy(h_objectives, d_objectives, sizeof(float) * sims, cudaMemcpyDeviceToHost);
    CUDA_CHECK(cudaGetLastError());

    ////////////////////////////// WRITE BACK //////////////////////////////
    // Write to cmaes/data/objective.txt
    fs::path OBJECTIVE_PATH = fs::current_path() / "cmaes/data/objective.txt";
    std::ofstream fout(OBJECTIVE_PATH);
    if (!fout.is_open()) {
        throw std::runtime_error("Couldn't open " + OBJECTIVE_PATH.string() +
                                 " for writing");
    }

    /*
        fine tune with w: f(x) = dp'*dp + w*0.5*x0'*M*x0 + w*0.5*xf'*M*xf

        (xf = xfinal = final velocity)
    */
    // Eigen::Matrix<float, DIM, 1> I = Eigen::Matrix<float, DIM, 1>::Zero();
    const Eigen::Matrix<float, DIM, 1> I = {1.66666667f, 1.66666667f, 1.66666667f, 1.0f, 1.0f, 1.0f};
    const auto &M = I.asDiagonal();

    fout << std::fixed << std::setprecision(8);

    for (size_t i = 0; i < NUM_ENVIRONMENTS; i++) {
        Eigen::Matrix<float, DIM, 1> v0 = Eigen::Matrix<float, DIM, 1>::Zero();
        Eigen::Matrix<float, DIM, 1> vf = Eigen::Matrix<float, DIM, 1>::Zero();
        for (int j = 0; j < DIM; j++) {
            v0[j] = h_initVels[i * DIM + j];
            vf[j] = h_finalVels[i * DIM + j];
        }

        // DEBUG_VEC(v0, 6);
        // DEBUG_VEC(vf, 6);
        // printf("term1 (v0) = %f\n", (W_OBJ * 0.5f * v0.dot(I.asDiagonal() *
        // v0))); printf("term1 (vf) = %f\n", (W_OBJ * 0.5f *
        // vf.dot(I.asDiagonal() * vf)));

        h_objectives[i] += 0.5f * (W1 * v0.dot(M * v0) + W2 * vf.dot(M * vf));
        fout << h_objectives[i] << ' ';
    }
    fout.close();

    cout << "# h_dp[0] = " << h_objectives[0] << endl;

    delete[] h_objectives;
    delete[] h_initVels;
    delete[] h_finalVels;

    // cuda dealloc
    cudaFree(d_initVels);
    cudaFree(d_finalVels);
    cudaFree(d_objectives);
}

void run_cpu_thread(apbd::Model *model, apbd::Body *bodies, int sims,
                    int processor_count, int id, bool do_variations) {
    for (int i = id; i < sims; i += processor_count) {
        _thread_scene_id = i;
        model->copy_data_to_store(bodies);
        Eigen::Matrix4f E = Eigen::Matrix4f::Identity();

        if (model->body_count > 1 && do_variations)
            model->bodies[1].setInitVelocity(Eigen::Matrix<float, 6, 1>(
                0, 0, 0, float(_thread_scene_id % 1000), 0, 0));
        auto collider = apbd::Collider(model);
        model->simulate(&collider);
    }
}

void cpu_run_group(apbd::Model model, apbd::Body *bodies, int sims,
                   bool do_variations) {
    _global_scene_count = (size_t)sims;
    const auto processor_count = std::thread::hardware_concurrency();
    if (processor_count == 0) {
        throw runtime_error("Failed to detect concurrency.");
    }

    fs::path p = fs::current_path();
    fs::path VELOCITIES_PATH = fs::current_path() / "cmaes/data/velocities.txt";
    cout << "# Trying to read velocities from " << VELOCITIES_PATH << endl;
    float *h_initVels = new float[DIM * NUM_ENVIRONMENTS];

    std::ifstream fin(VELOCITIES_PATH);
    if (!fin.is_open()) {
        cout << VELOCITIES_PATH << endl;
        exit(1);
    }
    for (int i = 0; i < DIM * NUM_ENVIRONMENTS; i++) {
        float tmp;
        if (!(fin >> tmp)) {
            cout << "Failed to read value at index " << i << endl;
            exit(1);
        }

        h_initVels[i] = tmp;
        // printf("h_init[%d] = %f\n", i, h_initVels[i]);
    }

    auto handles = std::vector<std::thread>();
    auto t1 = Clock::now();
    auto buffers = apbd::Model::allocate_buffers(sims, model);
    for (int i = 0; i < processor_count; i++) {
        if (i < sims) {
            apbd::Model *thread_model = new apbd::Model(
                std::move(model.clone_with_buffers(buffers, i)));

            // Update the model for this sim
            Eigen::Vector3f vel = Eigen::Vector3f(h_initVels[i * DIM + 0],
                                                  h_initVels[i * DIM + 1],
                                                  h_initVels[i * DIM + 2]);
            bodies[GOAL_IDX].data.rigid.w = vel;

            vel = Eigen::Vector3f(h_initVels[i * DIM + 3],
                                  h_initVels[i * DIM + 4],
                                  h_initVels[i * DIM + 5]);
            bodies[GOAL_IDX].data.rigid.v = vel;

            handles.push_back(std::thread(run_cpu_thread, thread_model, bodies,
                                          sims, processor_count, i,
                                          do_variations));
        }
    }
    for (auto &h : handles) {
        h.join();
    }
    auto t2 = Clock::now();
    cout << "# Kernel took: " << (t2 - t1).count() << '\t';
}

// TODO: Move all this to its own util folder
struct MainState {
    int model_id;
    unsigned long scene_count;
    unsigned long substeps;
    bool variations;
    bool do_write;
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
        const int opt = getopt_long(argc, argv, "hm:s:t:vw", longopts, 0);

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
            case 'w': {
                cout << "# write: true" << endl;
                state.do_write = true;
                break;
            }
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

    auto model = createModelSample(state.model_id, 1e-2f, state.substeps,
                                   bodies, state.scene_count);

    auto t1 = Clock::now();
#ifdef USE_CUDA
    cout << "# Running with CUDA #" << endl;
    int64_t kernel_time = 0;
    launchCMAESKernels(model, bodies, state.scene_count, state.variations,
                       kernel_time);

    // Write aggregate time to times.txt
    fs::path p = fs::current_path();
    fs::path TIMES_PATH = fs::current_path() / "cmaes/data/times.txt";
    cout << "# Trying to write times to " << TIMES_PATH << endl;
    std::ofstream fout(TIMES_PATH, std::ios_base::app);
    if (!fout.is_open()) {
        throw std::runtime_error("Couldn't open " + TIMES_PATH.string() +
                                 " for writing");
    }
    fout << kernel_time << endl;
#else
    cout << "# Running on CPU #" << endl;
    // cpu_run_group(model, bodies, state.scene_count, state.variations);
#endif
    auto t2 = Clock::now();
    cout << "# Simulation took: " << (t2 - t1).count() << endl;

    delete bodies;
}