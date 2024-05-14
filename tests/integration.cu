#include "model_samples.h"

int main(int argc, char **argv) {
  if (argc < 2)
    exit(1);
#ifdef USE_CUDA
  printf("CUDA not supported in integration test. See performance.cu\n");
  exit(1);
#endif
  apbd::Body *bodies;
  apbd::Model model = createModelSample(atoi(argv[1]), 1e-2, 20, bodies, 1);
  _global_scene_count = 1;
  _thread_scene_id = 0;
  model.copy_data_to_store(bodies);

  auto collider = apbd::Collider(&model);
  model.simulate(&collider);
}
