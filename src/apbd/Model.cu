#include "apbd/Model.h"
#include "util.h"

#include <iostream>
#include <stdexcept>

namespace apbd {

Model::Model()
    : h(1. / 30.), tEnd(1), substeps(10), bodies(nullptr), body_count(0),
      constraints(nullptr), constraint_count(0), constraint_layers(nullptr),
      layer_count(0), constraint_layer_sizes(), body_layers(nullptr),
      body_layer_sizes(), gravity(0.0, 0.0, -980.0), iters(1),
      ground_E(Eigen::Matrix4f::Zero()), ground_size(10), steps(0) {}

Model::Model(const Model &other)
    : h(other.h), tEnd(other.tEnd), substeps(other.substeps),
      bodies(other.bodies), body_count(other.body_count), constraints(nullptr),
      constraint_count(other.constraint_count), constraint_layers(nullptr),
      layer_count(other.layer_count), constraint_layer_sizes(),
      body_layers(nullptr), body_layer_sizes(), gravity(other.gravity),
      iters(other.iters), ground_E(other.ground_E),
      ground_size(other.ground_size), steps(other.steps) {}

Model::Model(const Model &&other)
    : h(other.h), tEnd(other.tEnd), substeps(other.substeps),
      bodies(other.bodies), body_count(other.body_count),
      constraints(other.constraints), constraint_count(other.constraint_count),
      constraint_layers(other.constraint_layers),
      layer_count(other.layer_count), constraint_layer_sizes(),
      body_layers(other.body_layers), body_layer_sizes(),
      gravity(other.gravity), iters(other.iters), ground_E(other.ground_E),
      ground_size(other.ground_size), steps(other.steps) {}

void Model::create_store(size_t scene_count) {
  // TODO: handle this constraints
  data::SOAStore data_store(this->body_count, MAX_COLLISIONS, MAX_COLLISIONS,
                            scene_count);

#ifdef USE_CUDA
  cudaMemcpyToSymbol(data::device_global_store, &data_store,
                     sizeof(data::SOAStore), size_t(0), cudaMemcpyHostToDevice);
#else
  data::global_store = std::move(data_store);
#endif
}

void Model::copy_data_to_store(Body *body_array) {
  for (size_t i = 0; i < this->body_count; i++) {
    auto &body = body_array[i];
    switch (body.type) {
    case BODY_RIGID: {
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

ModelBuffers Model::allocate_buffers(size_t count, const Model &model) {
  ModelBuffers buffers;
  if (model.constraint_count > 0)
    buffers.constraints =
        alloc_device<Constraint>(count * model.constraint_count);
  else
    buffers.constraints = nullptr;
  buffers.body_layers = alloc_device<BodyReference>(count * MAX_LAYER_OBJECTS);
  buffers.constraint_layers =
      alloc_device<Constraint *>(count * MAX_LAYER_OBJECTS);
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
  new_model.body_layers = &buffers.body_layers[offset * MAX_LAYER_OBJECTS];
  new_model.constraint_layers =
      &buffers.constraint_layers[offset * MAX_LAYER_OBJECTS];
  return new_model;
}

void Model::init(/*Body *body_array*/) {
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
  float time = 0;
  float hs = this->h / static_cast<float>(this->substeps);
  for (unsigned int step = 0; step < this->steps; step++) {
    this->clearBodyShockPropInfo();
    collider->run(this);
    this->constructConstraintGraph(collider);
    for (unsigned int substep = 0; substep < this->substeps; substep++) {
      this->stepBDF1(step, substep, hs);
      this->solveConSP(hs);
      this->solveConGS(collider, hs);
      time += hs;
    }
    this->write_state(step + 1);
  }
}

/** Private Functions **/

void Model::stepBDF1(unsigned int step, unsigned int substep, float hs) {
  for (size_t body_i = 0; body_i < this->body_count; body_i++) {
    this->bodies[body_i].stepBDF1(step, substep, hs, this->gravity);
  }
}
void Model::clearBodyShockPropInfo() {
  // clears the shock propagation info from each body; this may not be necessary
  for (size_t body_i = 0; body_i < this->body_count; body_i++) {
    this->bodies[body_i].clearShock();
  }
}

void Model::constructConstraintGraph(Collider *collider) {
  // Constructs a graph of constraints, working from the ground layer up
  // needs a list of constraints and bodies
  //
  //  collect static constraints and collision constraints
  //  for each constraint:
  //    if it is ground, initialize the body affected to layer 1 and add this
  //    constraint to the list affecting that body if the constraint has 2
  //    bodies (not 1), then tell both bodies that this constraint affects them
  //
  //  working up one layer at a time:
  //    for every body affected in the previous layer:
  //      for each constraint affecting the body:
  //        make sure the body on a higher layer is second
  //        assign the second body to this layer, and add this constraint to the
  //        parent constraints add the second body and this constraint to this
  //        layer

  // in theory, we can do this by keeping a constraint and body layer list, and
  // a last layer constraint list. the shock parent list is more difficult, but
  // does not seem to be used for anything other than setting shockProp to true;
  // so we can do this here
  // first, reset all counts
  this->layer_constraint_count = 0;
  this->layer_body_count = 0;
  for (size_t i = 0; i < this->layer_count; i++) {
    this->body_layer_sizes[i] = 0;
    this->constraint_layer_sizes[i] = 0;
  }
  this->layer_count = 0;

  unsigned int current_layer_body_count = 0;
  for (size_t i = 0; i < collider->ground_collision_count; i++) {
    auto &constraint = collider->collisions[i];
    DEBUG_ASSERT(constraint.type == CONSTRAINT_COLLISION_GROUND,
                 "Wrong constraint detected!");
    constraint.data.ground.body.layer(0);
    DEBUG_ASSERT(this->layer_constraint_count < MAX_LAYER_OBJECTS,
                 "Layer object storage overflow!");
    this->constraint_layers[this->layer_constraint_count++] = &constraint;
    this->constraint_layer_sizes[0]++;
    DEBUG_ASSERT(this->layer_body_count < MAX_LAYER_OBJECTS,
                 "Layer object storage overflow!");
    this->body_layers[this->layer_body_count++] = constraint.data.ground.body;

    this->body_layer_sizes[0] += 1;
    current_layer_body_count++;
  }

  unsigned int layer = 1;
  while (current_layer_body_count > 0 && layer < MAX_LAYERS) {
    current_layer_body_count = 0;
    // loop through all other constraints
    for (size_t i = collider->ground_collision_count;
         i < collider->collision_count; i++) {
      auto &constraint = collider->collisions[i];
      if (constraint.handle_layer(layer, this->body_layers,
                                  this->body_layer_sizes,
                                  this->layer_body_count)) {
        DEBUG_ASSERT(this->layer_constraint_count < MAX_LAYER_OBJECTS,
                     "Layer object storage overflow!");
        this->constraint_layers[this->layer_constraint_count++] = &constraint;
        this->constraint_layer_sizes[layer]++;
        current_layer_body_count++;
      }
    }
    for (size_t i = 0; i < this->constraint_count; i++) {
      auto &constraint = collider->collisions[i];
      if (constraint.handle_layer(layer, this->body_layers,
                                  this->body_layer_sizes,
                                  this->layer_body_count)) {
        DEBUG_ASSERT(this->layer_constraint_count < MAX_LAYER_OBJECTS,
                     "Layer object storage overflow!");
        this->constraint_layers[this->layer_constraint_count++] = &constraint;
        this->constraint_layer_sizes[layer++];
        current_layer_body_count++;
      }
    }
    layer++;
  }
  this->layer_count = layer - 1;
}

void Model::solveConSP(float hs) {
  for (size_t constraint_i = 0; constraint_i < this->constraint_count;
       constraint_i++) {
    this->constraints[constraint_i].clear();
  }

  // Solve all constraints in the graph with shock propagation in order of the
  // constraint layers. The exact layer sizes don't matter at this step, so we
  // don't bother walking through each layer individually.
  for (size_t i = 0; i < this->layer_constraint_count; i++) {
    for (int iter = 0; iter < this->iters; iter++) {
      this->constraint_layers[i]->solve(hs, true);
    }
  }

  size_t current_layer_body_offset = this->layer_body_count;
  size_t current_layer_constraint_offset = this->layer_constraint_count;
  for (long i = this->layer_count - 1; i >= 0; i--) {
    DEBUG_ASSERT(this->body_layer_sizes[i] <= current_layer_body_offset, "");
    DEBUG_ASSERT(
        this->constraint_layer_sizes[i] <= current_layer_constraint_offset, "");
    current_layer_body_offset -= this->body_layer_sizes[i];
    current_layer_constraint_offset -= this->constraint_layer_sizes[i];
    for (size_t j = 0; j < this->body_layer_sizes[i]; j++) {
      DEBUG_ASSERT(j < this->body_layer_sizes[i], "body_layer_sizes changed");
      this->body_layers[current_layer_body_offset + j].applyJacobiShock();
    }
    for (int iter = 0; iter < this->iters; iter++) {
      for (size_t j = 0; j < this->constraint_layer_sizes[i]; j++) {
        this->constraint_layers[current_layer_constraint_offset + j]->solve(
            hs, true);
      }
    }
  }
}

void Model::solveConGS(Collider *collider, float hs) {
  for (int iter = 0; iter < this->iters; iter++) {
    for (size_t constraint_i = 0; constraint_i < this->constraint_count;
         constraint_i++) {
      this->constraints[constraint_i].solve(hs, false);
    }
    for (size_t i = 0; i < collider->collision_count; i++) {
      collider->collisions[i].solve(hs, false);
    }
  }
}

void Model::write_state(unsigned int step) {
#ifdef WRITE
#ifdef __CUDA_ARCH__
  if (threadIdx.x == 0)
    printf("Step %d\n", step);
  // print up to 8 simulations in parallel
  for (size_t i = 0; i < body_count * 8; i++) {
    if (i / body_count != threadIdx.x)
      continue;
    printf("%lu ", i);
    bodies[i % body_count].write_state();
    printf("\n");
  }
#else
  printf("Step %d\n", step);
  for (size_t i = 0; i < body_count; i++) {
    printf("%lu ", i);
    bodies[i].write_state();
    printf("\n");
  }
#endif
#endif
}

void Model::print_config() {
  printf("# Body count: %lu\n"
         "# Constraint count: %lu\n"
         "# Gravity: [%f %f %f]\n"
         "# Ground size: %f\n"
         "# Time Step: %f\n"
         "# End Time: %f\n"
         "# Steps: %u\n"
         "# Substeps: %u\n"
         "# Iterations: %u\n",
         body_count, constraint_count, gravity(0), gravity(1), gravity(2),
         ground_size, h, tEnd, steps, substeps, iters);
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

} // namespace apbd
