#pragma once
#include <stddef.h>

/// Maximum number of collision constraints within the entire simulation
const size_t MAX_COLLISION_CONSTRAINTS = 65536;
/// Block size used in CUDA kernel
const size_t BLOCK_SIZE = 256;
/// Number of blocks per SM minimum
// currently limited by register usage
const size_t MIN_BLOCKS_PER_SM = 1;
/// Maximum number of layers in the constraint graph
const size_t MAX_LAYERS = 16;
