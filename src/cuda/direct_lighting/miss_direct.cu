#include <optix.h>
#include <sutil/vec_math.h>
#include "direct_launch_params.h"

extern "C" __constant__ DirectLaunchParams params;

// Primary ray miss - return background color
extern "C" __global__ void __miss__direct()
{
    // Dark background
    float3 bg = make_float3(0.02f, 0.02f, 0.03f);

    optixSetPayload_0(__float_as_uint(bg.x));
    optixSetPayload_1(__float_as_uint(bg.y));
    optixSetPayload_2(__float_as_uint(bg.z));
    optixSetPayload_3(__float_as_uint(1e16f));  // Very large distance - no fog for background
}

// Shadow ray miss - no occlusion (light is visible)
extern "C" __global__ void __miss__shadow()
{
    // DEBUG: Force everything to be occluded to test if shadow rays work
    // optixSetPayload_0(1u);  // Force occluded for testing
    optixSetPayload_0(0u);  // Not occluded
}

