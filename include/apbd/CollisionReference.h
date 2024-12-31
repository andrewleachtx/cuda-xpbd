#pragma once
#include "util.h"

namespace apbd {

#define DECLARE_COLLISION_ACCESS_FUNCTIONS(data_type, element) \
    __host__ __device__ data_type element() const;             \
    __host__ __device__ void element(data_type const new_val);

class CollisionReference {
   public:
    unsigned int index;

    __host__ __device__ CollisionReference(const unsigned int index)
        : index(data::soa_index(index)) {}

    __host__ __device__ void create(unsigned int contactNum, bool broken,
                                    BodyReference body1, BodyReference body2) {
        this->contactNum(contactNum);
        this->broken(broken);
        this->body1(body1);
        this->body2(body2);
    }

    // access the data elements in Collision

    /// Number of contacts that exist
    DECLARE_COLLISION_ACCESS_FUNCTIONS(unsigned int, contactNum)
    /// Indicates whether this collision has been broken;
    /// i.e. the two bodies are no longer colliding
    DECLARE_COLLISION_ACCESS_FUNCTIONS(bool, broken)
    /// The first body in the collision
    DECLARE_COLLISION_ACCESS_FUNCTIONS(BodyReference, body1)
    /// The second body in the collision. If this is null, this is a ground
    /// collision.
    DECLARE_COLLISION_ACCESS_FUNCTIONS(BodyReference, body2)
};

}  // namespace apbd
