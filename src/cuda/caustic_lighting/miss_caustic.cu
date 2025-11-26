#include <optix.h>
#include <sutil/vec_math.h>
#include "caustic_launch_params.h"

extern "C" __constant__ CausticLaunchParams params;

extern "C" __global__ void __miss__caustic()
{
    // Black background
    optixSetPayload_0(__float_as_uint(0.0f));
    optixSetPayload_1(__float_as_uint(0.0f));
    optixSetPayload_2(__float_as_uint(0.0f));
}

