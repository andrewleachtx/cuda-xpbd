#include "apbd/Collider.h"
#include "util.h"

namespace apbd {

Collider::Collider(Model *model)
    : bp_cap_1(model->body_count),
      bp_cap_2(model->body_count * model->body_count), bp_count_1(0),
      bp_count_2(0), bpList1(nullptr), bpList2(nullptr), collision_count(0),
      collisions(nullptr) {
  bpList1 = alloc_device<BodyReference>(bp_cap_1);
  bpList2 = alloc_device<BodyReference>(bp_cap_2);
  collisions = alloc_device<Constraint>(MAX_COLLISIONS);
}

Collider::Collider(Model *model, size_t scene_id,
                   BodyReference *body_ptr_buffer,
                   Constraint *constraint_buffer)
    : bp_cap_1(model->body_count),
      bp_cap_2(model->body_count * model->body_count), bp_count_1(0),
      bp_count_2(0), bpList1(&body_ptr_buffer[model->body_count * scene_id *
                                              (model->body_count + 1)]),
      bpList2(&body_ptr_buffer[model->body_count * scene_id *
                                   (model->body_count + 1) +
                               model->body_count]),
      collision_count(0),
      collisions(&constraint_buffer[MAX_COLLISIONS * scene_id]) {}
void Collider::allocate_buffers(Model &model, int sim_count,
                                BodyReference *&body_ptr_buffer,
                                Constraint *&constraint_buffer) {
  body_ptr_buffer = alloc_device<BodyReference>(
      model.body_count * (model.body_count + 1) * sim_count);
  constraint_buffer = alloc_device<Constraint>(MAX_COLLISIONS * sim_count);
}

void Collider::run(Model *model) {
  bp_count_1 = 0;
  bp_count_2 = 0;
  collision_count = 0;
  this->broadphase(model);
  this->narrowphase(model);
}

void Collider::broadphase(Model *model) {
  BodyReference *bodies = model->bodies;

  for (size_t i = 0; i < model->body_count && this->bp_count_1 < this->bp_cap_1;
       i++) {
    BodyReference body = bodies[i];
    if (body.collide()) {
      if (body.broadphaseGround(model->ground_E)) {
        this->bpList1[this->bp_count_1++] = body;
      }
    }
  }
  for (size_t i = 0; i < model->body_count && this->bp_count_2 < this->bp_cap_2;
       i++) {
    BodyReference body = bodies[i];
    if (body.collide()) {
      for (size_t j = i + 1;
           j < model->body_count && this->bp_count_2 < this->bp_cap_2; j++) {
        if (bodies[j].collide()) {
          if (body.broadphaseRigid(bodies[j])) {
            this->bpList2[this->bp_count_2++] = body;
            this->bpList2[this->bp_count_2++] = bodies[j];
          }
        }
      }
    }
  }
}

void Collider::narrowphase(Model *model) {
  auto &Eg = model->ground_E;
  DEBUG_ASSERT(this->collision_count == 0, "");

  for (size_t i = 0; i < this->bp_count_1; i++) {
    auto body = this->bpList1[i];
    auto cpair = body.narrowphaseGround(Eg);
    auto cdata = cpair.first;
    auto c_count = cpair.second;
    for (size_t k = 0; k < c_count && this->collision_count < MAX_COLLISIONS;
         k++) {
      auto &c = cdata[k];
      switch (body.type) {
      case BODY_RIGID: {
        this->collisions[this->collision_count++] = Constraint(ConstraintGround(
            body.get_rigid(), Eg, c.d, c.xl, c.xw, c.nw, c.vw));
        break;
      }
      default:
        break;
      }
    }
  }
  this->ground_collision_count = this->collision_count;

  for (size_t i = 0; i < this->bp_count_2; i += 2) {
    auto body1 = this->bpList2[i];
    auto body2 = this->bpList2[i + 1];
    auto cpair = body1.narrowphaseRigid(body2);
    auto cdata = cpair.first;
    auto c_count = cpair.second;
    for (size_t k = 0; k < c_count && this->collision_count < MAX_COLLISIONS;
         k++) {
      auto &c = cdata[k];
      switch (body1.type) {
      case BODY_RIGID: {
        // we require the other body to be rigid
        if (body2.type == BODY_RIGID) {
          this->collisions[this->collision_count++] =
              Constraint(ConstraintRigid(body1.get_rigid(), body2.get_rigid(),
                                         c.d, c.nw, c.x1, c.x2));
        }
        break;
      }
      default:
        break;
      }
    }
  }
}

} // namespace apbd
