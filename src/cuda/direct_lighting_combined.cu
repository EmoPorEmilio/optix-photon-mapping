// Combined direct lighting shaders for OptiX compilation
// This file combines all direct lighting shaders into a single compilation unit

#include "direct_lighting/raygen_direct.cu"
#include "direct_lighting/closesthit_direct.cu"
#include "direct_lighting/miss_direct.cu"
#include "direct_lighting/intersection_direct_sphere.cu"

