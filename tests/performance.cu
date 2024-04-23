#include "model_samples.h"
#include <exception>
#include <getopt.h>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

using std::cout, std::endl, std::string, std::runtime_error;
typedef std::chrono::high_resolution_clock Clock;

__global__ void kernel(apbd::Model model, apbd::ModelBuffers buffers,
                       apbd::Body *body_buffer,
                       apbd::BodyReference *body_ptr_buffer,
                       apbd::Constraint *constraint_buffer, int sims) {
  // get this scene ID
  size_t scene_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (scene_id >= sims)
    return;
  // make a copy of the model
  model.copy_data_to_store(body_buffer);
  Eigen::Matrix4f E = Eigen::Matrix4f::Identity();

  // Eigen::Matrix3f R = se3::aaToMat(
  //     Eigen::Vector3f(1, 1, 1), static_cast<float>(scene_id) * 0.5 * M_PI /
  //     4);
  // E.block<3, 3>(0, 0) = R;

  // for (size_t index = 0; index < model.body_count; index++) {
  //   auto &body = model.bodies[index];
  //   E.block<3, 1>(0, 3) = body.get_rigid().position() +
  //                         Eigen::Vector3f(0,
  //                                         (static_cast<float>(scene_id) - 4)
  //                                         *
  //                                             static_cast<float>(index) *
  //                                             0.1,
  //                                         0);
  //   body.setInitTransform(E);
  // }
  apbd::Model thread_model = model.clone_with_buffers(buffers, scene_id);

  // create a thread-local collider
  auto collider = apbd::Collider(&thread_model, scene_id, body_ptr_buffer,
                                 constraint_buffer);
  // simulate
  thread_model.simulate(&collider);
}

void run_kernel(apbd::Model model, apbd::Body *bodies, int sims) {
  cout << "# thread blocks: " << (sims + BLOCK_SIZE - 1) / BLOCK_SIZE << endl;

  const size_t shared_size = 0;

  apbd::BodyReference *body_ptr_buffer = nullptr;
  apbd::Constraint *constraint_buffer = nullptr;
  apbd::Collider::allocate_buffers(model, sims, body_ptr_buffer,
                                   constraint_buffer);
  auto buffers = apbd::Model::allocate_buffers(sims, model);

  model.move_to_device();
  bodies = move_array_to_device(bodies, model.body_count);

  auto t1 = Clock::now();

  kernel<<<(sims + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, shared_size>>>(
      model, buffers, bodies, body_ptr_buffer, constraint_buffer, sims);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  auto t2 = Clock::now();
  std::cout << "# Kernel took: " << (t2 - t1).count() << '\t';
}

void run_cpu_thread(apbd::Model *model, apbd::Body *bodies, int sims,
                    int processor_count, int id) {
  for (int i = id; i < sims; i += processor_count) {
    _thread_scene_id = i;
    model->copy_data_to_store(bodies);
    Eigen::Matrix4f E = Eigen::Matrix4f::Identity();

    // Eigen::Matrix3f R = se3::aaToMat(
    //     Eigen::Vector3f(1, 1, 1), static_cast<float>(i) * 0.5 * M_PI / 4);
    // E.block<3, 3>(0, 0) = R;

    for (size_t index = 0; index < model->body_count; index++) {
      auto &body = model->bodies[index];
      E.block<3, 1>(0, 3) = body.get_rigid().position() +
                            Eigen::Vector3f(0,
                                            (static_cast<float>(i) - 4) *
                                                static_cast<float>(index) * 0.1,
                                            0);
      body.setInitTransform(E);
    }
    auto collider = apbd::Collider(model);
    model->simulate(&collider);
  }
}

void cpu_run_group(apbd::Model model, apbd::Body *bodies, int sims) {
  _global_scene_count = (size_t)sims;
  const auto processor_count = std::thread::hardware_concurrency();
  if (processor_count == 0) {
    throw runtime_error("Failed to detect concurrency.");
  }
  auto handles = std::vector<std::thread>();
  auto t1 = Clock::now();
  auto buffers = apbd::Model::allocate_buffers(sims, model);
  for (int i = 0; i < processor_count; i++) {
    if (i < sims) {
      apbd::Model *thread_model =
          new apbd::Model(std::move(model.clone_with_buffers(buffers, i)));
      handles.push_back(std::thread(run_cpu_thread, thread_model, bodies, sims,
                                    processor_count, i));
    }
  }
  for (auto &h : handles) {
    h.join();
  }
  auto t2 = Clock::now();
  cout << "# Kernel took: " << (t2 - t1).count() << '\t';
}

struct MainState {
  bool visualize;
  string output_file;
  int model_id;
  unsigned long scene_count;
};

const char *HELP = "\
arguments:          \n\
  -i, --model_id ID    The model scene to use\n\
  -s, --scene-count N  The number of simulations to run\n\
  -h, --help           Show this help text\n\
";

MainState parse_arguments(int argc, char *argv[]) {
  struct MainState state = {
      .model_id = 0,
      .scene_count = 1,
  };

  option longopts[] = {{"model_id", required_argument, NULL, 'm'},
                       {"scene-count", required_argument, NULL, 's'},
                       {"help", no_argument, NULL, 'h'},
                       {0}};

  while (1) {
    const int opt = getopt_long(argc, argv, "hvo:m:s:a:", longopts, 0);

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
  auto model = createModelSample(state.model_id, bodies, state.scene_count);

  auto t1 = Clock::now();
#ifdef USE_CUDA
  cout << "# Running with CUDA #" << endl;
  run_kernel(model, bodies, state.scene_count);
#else
  cout << "# Running on CPU #" << endl;
  cpu_run_group(model, bodies, state.scene_count);
#endif
  auto t2 = Clock::now();
  cout << " Simulation took: " << (t2 - t1).count() << '\n';
}
