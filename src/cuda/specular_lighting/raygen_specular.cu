#include <optix.h>
#include <sutil/vec_math.h>
#include "specular_launch_params.h"

extern "C" __constant__ SpecularLaunchParams params;

extern "C" __global__ void __raygen__specular()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    const float2 d = make_float2(
        (2.0f * (float)idx.x / (float)dim.x) - 1.0f,
        (2.0f * (float)idx.y / (float)dim.y) - 1.0f
    );

    float3 ray_origin = params.eye;
    float3 ray_direction = normalize(d.x * params.U + d.y * params.V + params.W);

    // Payload: RGB color + depth counter
    unsigned int p0 = __float_as_uint(0.0f);
    unsigned int p1 = __float_as_uint(0.0f);
    unsigned int p2 = __float_as_uint(0.0f);
    unsigned int p3 = 0;  // depth

    optixTrace(
        params.handle,
        ray_origin,
        ray_direction,
        0.001f,
        1e16f,
        0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        0, 1, 0,
        p0, p1, p2, p3
    );

    float3 color = make_float3(
        __uint_as_float(p0),
        __uint_as_float(p1),
        __uint_as_float(p2)
    );

    // Gamma correction
    color.x = powf(fmaxf(0.0f, fminf(1.0f, color.x)), 1.0f / 2.2f);
    color.y = powf(fmaxf(0.0f, fminf(1.0f, color.y)), 1.0f / 2.2f);
    color.z = powf(fmaxf(0.0f, fminf(1.0f, color.z)), 1.0f / 2.2f);

    params.frame_buffer[idx.y * params.width + idx.x] = make_float4(color, 1.0f);
}

