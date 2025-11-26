#include <optix.h>
#include <sutil/vec_math.h>
#include "specular_launch_params.h"

extern "C" __constant__ SpecularLaunchParams params;

extern "C" __global__ void __miss__specular()
{
    // Black background
    optixSetPayload_0(__float_as_uint(0.0f));
    optixSetPayload_1(__float_as_uint(0.0f));
    optixSetPayload_2(__float_as_uint(0.0f));
}

