#include "apbd/BodyReference_impl.h"
#include "apbd/Collider.h"
#include "apbd/CollisionReference_impl.h"
#include "apbd/Collisions_impl.h"
#include "util.h"

namespace apbd {

Collider::Collider(Model *model)
    : bp_cap_1(model->body_count),
      bp_cap_2(model->body_count * (model->body_count - 1)),
      bp_count_1(0),
      bp_count_2(0),
      bpList1(nullptr),
      bpList2(nullptr),
      collision_cap(model->body_count * (model->body_count + 1) / 2),
      ground_constraint_count(0),
      rigid_constraint_count(0),
      collisions(nullptr),
      active_collision_count(0),
      activeCollisions(nullptr) {
    bpList1 = alloc_device<BodyReference>(bp_cap_1);
    bpList2 = alloc_device<BodyReference>(bp_cap_2);
    collisions = alloc_device<Collision>(collision_cap);
    activeCollisions = alloc_device<unsigned int>(collision_cap);
    // initialize collisions
    unsigned int N = model->body_count;
    for (unsigned int x = 0; x < N; x++) {
        CollisionReference clr = CollisionReference(x);
        clr.body1(model->bodies[x]);
        clr.body2(NULL_BODY);
        clr.broken(true);
        clr.contactNum(0);
        for (unsigned int y = x + 1; y < N; y++) {
            auto index =
                model->get_collision_index(model->bodies[x], model->bodies[y]);
            CollisionReference clr = CollisionReference(index);
            clr.body1(model->bodies[x]);
            clr.body2(model->bodies[y]);
            clr.broken(true);
            clr.contactNum(0);
        }
    }
}

Collider::Collider(Model *model, size_t scene_id,
                   BodyReference *body_ptr_buffer, Collision *collision_buffer,
                   unsigned int *active_collision_buffer)
    : bp_cap_1(model->body_count),
      bp_cap_2(model->body_count * (model->body_count - 1)),
      bp_count_1(0),
      bp_count_2(0),
      bpList1(nullptr),
      bpList2(nullptr),
      collision_cap(model->body_count * (model->body_count + 1) / 2),
      ground_constraint_count(0),
      rigid_constraint_count(0),
      collisions(&collision_buffer[collision_cap * scene_id]),
      active_collision_count(0),
      activeCollisions(&active_collision_buffer[collision_cap * scene_id]) {
    unsigned int total_bp_els = this->bp_cap_1 + this->bp_cap_2;
    this->bpList1 = &body_ptr_buffer[total_bp_els * scene_id];
    this->bpList2 = &body_ptr_buffer[total_bp_els * scene_id + this->bp_cap_1];
    // initialize collisions
    unsigned int N = model->body_count;
    for (unsigned int x = 0; x < N; x++) {
        CollisionReference clr = CollisionReference(x);
        clr.body1(model->bodies[x]);
        clr.body2(NULL_BODY);
        clr.broken(true);
        clr.contactNum(0);
        for (unsigned int y = x + 1; y < N; y++) {
            auto index =
                model->get_collision_index(model->bodies[x], model->bodies[y]);
            CollisionReference clr = CollisionReference(index);
            clr.body1(model->bodies[x]);
            clr.body2(model->bodies[y]);
            clr.broken(true);
            clr.contactNum(0);
        }
    }
}
void Collider::allocate_buffers(Model &model, int sim_count,
                                BodyReference *&body_ptr_buffer,
                                Collision *&collision_buffer,
                                unsigned int *&active_collision_buffer) {
    // bpList1 - 1 x body_count
    // bpList2 - body_count * (body_count - 1) / 2 * 2
    body_ptr_buffer = alloc_device<BodyReference>(
        (model.body_count * (model.body_count - 1) + model.body_count) *
        sim_count);
    collision_buffer = alloc_device<Collision>(
        model.body_count * (model.body_count + 1) / 2 * sim_count);
    active_collision_buffer = alloc_device<unsigned int>(
        model.body_count * (model.body_count + 1) / 2 * sim_count);
}

void Collider::run(Model *model) {
    bp_count_1 = 0;
    bp_count_2 = 0;
    active_collision_count = 0;
    ground_constraint_count = 0;
    rigid_constraint_count = 0;
    this->broadphase(model);
    this->narrowphase(model);
    this->constructCollisionOrder(model);
}

void Collider::constructCollisionOrder(Model *model) {
    // we need to sort the collisions by distance from the ground in terms of
    // bodies. these will be stored in activeCollisions.

    // start with the ground collisions
    for (size_t i = 0; i < this->bp_count_1; i++) {
        BodyReference body = this->bpList1[i];
        body.layer(0);
        // collisions with the ground are the same index as the body
        auto collision_index = body.index;
        DEBUG_ASSERT(this->active_collision_count < this->collision_cap,
                     "Active collisions overflowing (likely duplicates)!");
        this->activeCollisions[this->active_collision_count++] =
            collision_index;
    }

    // for each layer, if a body has a collision with it, propagate up the layer
    // TODO: see if this can be optimized by not re-checking bodies over and
    // over might be able to keep a cache for the previous layer that we can
    // iterate through
    unsigned int layer_size = 1;
    for (size_t layer = 0; layer < MAX_LAYERS && layer_size != 0; layer++) {
        layer_size = 0;
        // body-body collisions are stored as pairs
        for (size_t i = 0; i < this->bp_count_2; i += 2) {
            BodyReference body1 = this->bpList2[i];
            BodyReference body2 = this->bpList2[i + 1];
            // allow rigid collisions that are on the same layer through
            if (body1.layer() == layer && body2.layer() == layer) {
                // keep the same layer, but add the collision into the active
                // ones
                DEBUG_ASSERT(
                    this->active_collision_count < this->collision_cap,
                    "Active collisions overflowing (likely duplicates)!");
                activeCollisions[active_collision_count++] =
                    model->get_collision_index(body1, body2);
                layer_size++;
            }
            if (body1.layer() == layer && body2.layer() > layer) {
                body2.layer(layer + 1);
                DEBUG_ASSERT(
                    this->active_collision_count < this->collision_cap,
                    "Active collisions overflowing (likely duplicates)!");
                activeCollisions[active_collision_count++] =
                    model->get_collision_index(body1, body2);
                layer_size++;
            }
            // vice-versa
            if (body2.layer() == layer && body1.layer() > layer) {
                body1.layer(layer + 1);
                DEBUG_ASSERT(
                    this->active_collision_count < this->collision_cap,
                    "Active collisions overflowing (likely duplicates)!");
                activeCollisions[active_collision_count++] =
                    model->get_collision_index(body1, body2);
                layer_size++;
            }
        }
    }
}

void Collider::broadphase(Model *model) {
    BodyReference *bodies = model->bodies;

    for (size_t i = 0; i < model->body_count; i++) {
        BodyReference body = bodies[i];
        if (body.collide()) {
            if (body.broadphaseGround(model->ground_E)) {
                DEBUG_ASSERT(this->bp_count_1 < this->bp_cap_1,
                             "bpList1 overflowing!");
                this->bpList1[this->bp_count_1++] = body;
            }
        }
    }
    for (size_t i = 0; i < model->body_count; i++) {
        BodyReference body = bodies[i];
        if (body.collide()) {
            for (size_t j = i + 1; j < model->body_count; j++) {
                if (bodies[j].collide()) {
                    if (body.broadphaseRigid(bodies[j])) {
                        DEBUG_ASSERT(this->bp_count_2 + 1 < this->bp_cap_2,
                                     "bpList2 overflowing!");
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

    unsigned int old_bp_count = this->bp_count_1;
    this->bp_count_1 = 0;

    for (size_t i = 0; i < old_bp_count; i++) {
        auto body = this->bpList1[i];
        auto body_index = body.index;
        CollisionReference clr(body_index);
        if (clr.broken()) {
            auto cdata = body.narrowphaseGround(Eg);
            this->collisions[body_index].setContacts(clr, cdata);
        }
        if (clr.contactNum() != 0) {
            clr.broken(false);
            this->collisions[body_index].getConstraints(
                clr, this->ground_constraint_count,
                this->rigid_constraint_count);
            // we overwrite the old list to help the constriaint construction
            // func this will never overwrite data we care about, since we are
            // at most staying just behind i
            this->bpList1[this->bp_count_1++] = body;
        }
    }

    old_bp_count = this->bp_count_2;
    this->bp_count_2 = 0;

    for (size_t i = 0; i < old_bp_count; i += 2) {
        auto body1 = this->bpList2[i];
        auto body2 = this->bpList2[i + 1];
        auto collision_index = model->get_collision_index(body1, body2);
        Collision &collision = this->collisions[collision_index];
        CollisionReference clr(collision_index);
        if (clr.broken()) {
            // we need to ensure that the bodies are in the same order as the
            // collision expects
            // TODO: one of these branches is already guaranteed by the
            // construction in broadphase
            if (body1.index < body2.index) {
                auto cdata = body1.narrowphaseRigid(body2);
                collision.setContacts(clr, cdata);
            } else {
                auto cdata = body2.narrowphaseRigid(body1);
                collision.setContacts(clr, cdata);
            }
        }

        if (clr.contactNum() != 0) {
            clr.broken(false);
            collision.getConstraints(clr, this->ground_constraint_count,
                                     this->rigid_constraint_count);
            this->bpList2[this->bp_count_2++] = body1;
            this->bpList2[this->bp_count_2++] = body2;
        }
    }
}

}  // namespace apbd
