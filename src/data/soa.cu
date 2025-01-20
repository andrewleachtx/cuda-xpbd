#include "data/soa.h"

namespace data {

SOAStore host_global_store;
__device__ SOAStore device_global_store;

// implementation moved to header for inlining

}  // namespace data
