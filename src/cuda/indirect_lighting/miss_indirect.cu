#include <optix.h>
#include <sutil/vec_math.h>
#include "indirect_launch_params.h"

extern "C" __constant__ IndirectLaunchParams params;

extern "C" __global__ void __miss__indirect()
{
    // Background - dark gray
    optixSetPayload_0(__float_as_uint(0.1f));
    optixSetPayload_1(__float_as_uint(0.1f));
    optixSetPayload_2(__float_as_uint(0.1f));
}

