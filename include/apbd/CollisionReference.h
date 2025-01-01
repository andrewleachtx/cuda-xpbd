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

    // TODO: Probably should zero initialize new members or find what calls this thing
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

    // GPQP variables - note because cdata <= 8 we can have at worst 3 *
    // contactNum == 3 * 8 = 24
    DECLARE_COLLISION_ACCESS_FUNCTIONS(unsigned int, mIndices)
    DECLARE_COLLISION_ACCESS_FUNCTIONS(int, layer)

    DECLARE_COLLISION_ACCESS_FUNCTIONS(Mat24x6f, J1I)
    DECLARE_COLLISION_ACCESS_FUNCTIONS(Mat24x6f, J2I)
    DECLARE_COLLISION_ACCESS_FUNCTIONS(Vec24f, b)
    DECLARE_COLLISION_ACCESS_FUNCTIONS(float, mu)

    DECLARE_COLLISION_ACCESS_FUNCTIONS(Vec24f, lambda)
    DECLARE_COLLISION_ACCESS_FUNCTIONS(Vec24f, lambdac)
    DECLARE_COLLISION_ACCESS_FUNCTIONS(Vec24f, lambdad)
    DECLARE_COLLISION_ACCESS_FUNCTIONS(Vec24f, t_bar)
    DECLARE_COLLISION_ACCESS_FUNCTIONS(Vec24f, g)
    DECLARE_COLLISION_ACCESS_FUNCTIONS(Vec24f, p)
    DECLARE_COLLISION_ACCESS_FUNCTIONS(Vec24f, Ax)
    DECLARE_COLLISION_ACCESS_FUNCTIONS(Vec24b, freeIndex)

    DECLARE_COLLISION_ACCESS_FUNCTIONS(Vec24f, r_cg)
    DECLARE_COLLISION_ACCESS_FUNCTIONS(Vec24f, b_cg)
    DECLARE_COLLISION_ACCESS_FUNCTIONS(Mat24x6f, J1I_cg)
    DECLARE_COLLISION_ACCESS_FUNCTIONS(Mat24x6f, J2I_cg)
    DECLARE_COLLISION_ACCESS_FUNCTIONS(Vec24f, Minv_cg)
    DECLARE_COLLISION_ACCESS_FUNCTIONS(Vec24f, g_cg)
    DECLARE_COLLISION_ACCESS_FUNCTIONS(Vec24f, d_cg)
};

}  // namespace apbd
