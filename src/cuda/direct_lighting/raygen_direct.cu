#include <optix.h>
#include <sutil/vec_math.h>
#include "direct_launch_params.h"

extern "C" __constant__ DirectLaunchParams params;

extern "C" __global__ void __raygen__direct()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    // Compute normalized screen coordinates [-1, 1]
    const float u = (static_cast<float>(idx.x) + 0.5f) / static_cast<float>(dim.x) * 2.0f - 1.0f;
    const float v = (static_cast<float>(idx.y) + 0.5f) / static_cast<float>(dim.y) * 2.0f - 1.0f;

    // Generate camera ray
    const float3 origin = params.eye;
    const float3 direction = normalize(params.W + u * params.U + v * params.V);

    // Payload: color (RGB) + hit distance
    unsigned int p0, p1, p2, p3;
    p0 = __float_as_uint(0.0f);  // R
    p1 = __float_as_uint(0.0f);  // G
    p2 = __float_as_uint(0.0f);  // B
    p3 = __float_as_uint(0.0f);  // hit distance (for debugging)

    optixTrace(
        params.handle,
        origin,
        direction,
        0.001f,           // tmin
        1e16f,            // tmax
        0.0f,             // ray time
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        0,                // SBT offset for primary rays
        1,                // SBT stride (1 record per instance)
        0,                // miss SBT index
        p0, p1, p2, p3
    );

    // Unpack result color
    float3 result;
    result.x = __uint_as_float(p0);
    result.y = __uint_as_float(p1);
    result.z = __uint_as_float(p2);

    // Write to frame buffer
    const unsigned int pixel_idx = idx.y * dim.x + idx.x;
    params.frame_buffer[pixel_idx] = make_float4(result.x, result.y, result.z, 1.0f);
}

