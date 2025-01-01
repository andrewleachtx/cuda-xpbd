#pragma once
#include "CollisionReference.h"
#include "data/soa.h"

namespace apbd {

IMPLEMENT_ACCESS_FUNCTIONS(unsigned int, CollisionReference, Collision,
                           contactNum)
IMPLEMENT_ACCESS_FUNCTIONS(bool, CollisionReference, Collision, broken)
IMPLEMENT_ACCESS_FUNCTIONS(BodyReference, CollisionReference, Collision, body1)
IMPLEMENT_ACCESS_FUNCTIONS(BodyReference, CollisionReference, Collision, body2)

// GPQP
IMPLEMENT_ACCESS_FUNCTIONS(int, CollisionReference, Collision, layer)
IMPLEMENT_ACCESS_FUNCTIONS(unsigned int, CollisionReference, Collision,
                           mIndices)

IMPLEMENT_ACCESS_FUNCTIONS(Mat24x6f, CollisionReference, Collision, J1I)
IMPLEMENT_ACCESS_FUNCTIONS(Mat24x6f, CollisionReference, Collision, J2I)
IMPLEMENT_ACCESS_FUNCTIONS(Vec24f, CollisionReference, Collision, b)
IMPLEMENT_ACCESS_FUNCTIONS(float, CollisionReference, Collision, mu)

IMPLEMENT_ACCESS_FUNCTIONS(Vec24f, CollisionReference, Collision, lambda)
IMPLEMENT_ACCESS_FUNCTIONS(Vec24f, CollisionReference, Collision, lambdac)
IMPLEMENT_ACCESS_FUNCTIONS(Vec24f, CollisionReference, Collision, lambdad)
IMPLEMENT_ACCESS_FUNCTIONS(Vec24f, CollisionReference, Collision, t_bar)
IMPLEMENT_ACCESS_FUNCTIONS(Vec24f, CollisionReference, Collision, g)
IMPLEMENT_ACCESS_FUNCTIONS(Vec24f, CollisionReference, Collision, p)
IMPLEMENT_ACCESS_FUNCTIONS(Vec24f, CollisionReference, Collision, Ax)
IMPLEMENT_ACCESS_FUNCTIONS(Vec24b, CollisionReference, Collision, freeIndex)

IMPLEMENT_ACCESS_FUNCTIONS(Vec24f, CollisionReference, Collision, r_cg)
IMPLEMENT_ACCESS_FUNCTIONS(Vec24f, CollisionReference, Collision, b_cg)
IMPLEMENT_ACCESS_FUNCTIONS(Mat24x6f, CollisionReference, Collision, J1I_cg)
IMPLEMENT_ACCESS_FUNCTIONS(Mat24x6f, CollisionReference, Collision, J2I_cg)
IMPLEMENT_ACCESS_FUNCTIONS(Vec24f, CollisionReference, Collision, Minv_cg)
IMPLEMENT_ACCESS_FUNCTIONS(Vec24f, CollisionReference, Collision, g_cg)
IMPLEMENT_ACCESS_FUNCTIONS(Vec24f, CollisionReference, Collision, d_cg)

}  // namespace apbd
