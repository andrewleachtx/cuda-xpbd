#pragma once
#include <stddef.h>

/// Maximum number of collisions within the entire simulation
const size_t MAX_COLLISIONS = 64;
/// Block size used in CUDA kernel
const size_t BLOCK_SIZE = 256;
/// Number of blocks per SM minimum
// currently limited by register usage
const size_t MIN_BLOCKS_PER_SM = 1;
/// Maximum total number of pointers to store in constraint layer graph.
/// This is includes all layers.
const size_t MAX_LAYER_OBJECTS = 64;
/// Maximum number of layers in the constraint graph
const size_t MAX_LAYERS = 16;
