#include <optix.h>
#include <sutil/vec_math.h>
#include "direct_launch_params.h"
#include "../volume_photon_gather.h"

extern "C" __constant__ DirectLaunchParams params;

//=============================================================================
// Fog Helper Functions (Jensen's PDF §1.4, §3.3)
// Uses ray marching with density falloff for smooth fog boundaries
//=============================================================================

// Apply atmospheric fog using ray marching with density-weighted extinction
// This creates smooth, natural fog boundaries instead of hard boxy edges
static __forceinline__ __device__ float3 applyFog(
    const float3 &surface_color,
    const float3 &ray_origin,
    const float3 &ray_dir,
    float hit_distance)
{
    if (!params.enable_fog || hit_distance <= 0.0f || hit_distance > 1e10f)
        return surface_color;

    // Compute intersection with volume bounds (extended for falloff region)
    float t_near, t_far;
    if (!intersectVolumeBounds(ray_origin, ray_dir, params.volume, 0.0f, hit_distance, t_near, t_far))
        return surface_color;  // Ray doesn't pass through fog volume

    if (t_far <= t_near)
        return surface_color;

    // Ray march with density-weighted extinction for smooth boundaries
    const float step_size = 10.0f;  // March step size in world units
    float optical_depth = 0.0f;     // Accumulated optical depth (integral of sigma_t * density)
    float3 in_scatter = make_float3(0.0f);  // Accumulated in-scattering
    float t = fmaxf(t_near, 0.001f);

    // March through the volume
    while (t < t_far)
    {
        float3 sample_pos = ray_origin + t * ray_dir;

        // Get local density (0 at edges, 1 deep inside)
        float density = params.volume.densityAt(sample_pos);

        if (density > 0.0f)
        {
            // Accumulate optical depth: integral of sigma_t * density * dt
            float local_sigma_t = params.volume.sigma_t * density;
            optical_depth += local_sigma_t * step_size;

            // In-scatter contribution with transmittance
            float transmittance_so_far = expf(-optical_depth);
            in_scatter += params.fog_color * density * (1.0f - expf(-local_sigma_t * step_size)) * transmittance_so_far;
        }

        t += step_size;
    }

    // Final transmittance through entire volume
    float transmittance = expf(-optical_depth);

    // Final color: attenuated surface + in-scattered fog
    return surface_color * transmittance + in_scatter;
}

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
    p3 = __float_as_uint(0.0f);  // hit distance

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

    // Unpack result color and hit distance
    float3 result;
    result.x = __uint_as_float(p0);
    result.y = __uint_as_float(p1);
    result.z = __uint_as_float(p2);
    float hit_distance = __uint_as_float(p3);

    // Apply fog (Jensen's participating media) - only if enabled
    // When fog is disabled (for combined mode), the hit_distance is still passed through
    // so the combined renderer can apply fog once at the end
    result = applyFog(result, origin, direction, hit_distance);

    // Write to frame buffer - store hit_distance in w component for combined rendering
    const unsigned int pixel_idx = idx.y * dim.x + idx.x;
    params.frame_buffer[pixel_idx] = make_float4(result.x, result.y, result.z, hit_distance);
}

