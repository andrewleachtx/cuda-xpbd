#pragma once
#include "CollisionReference.h"
#include "data/soa.h"

namespace apbd {

IMPLEMENT_ACCESS_FUNCTIONS(unsigned int, CollisionReference, Collision,
                           contactNum)
IMPLEMENT_ACCESS_FUNCTIONS(bool, CollisionReference, Collision, broken)
IMPLEMENT_ACCESS_FUNCTIONS(BodyReference, CollisionReference, Collision, body1)
IMPLEMENT_ACCESS_FUNCTIONS(BodyReference, CollisionReference, Collision, body2)
}  // namespace apbd
