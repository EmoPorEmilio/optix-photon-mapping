#include <optix.h>
#include <sutil/vec_math.h>
#include "specular_launch_params.h"
#include "../volume_photon_gather.h"

extern "C" __constant__ SpecularLaunchParams params;

//=============================================================================
// Fog Helper - Ray marching with density falloff for smooth boundaries
//=============================================================================
static __forceinline__ __device__ float3 applyFogSpecular(
    const float3 &surface_color,
    const float3 &ray_origin,
    const float3 &ray_dir,
    float hit_distance)
{
    if (!params.enable_fog || hit_distance <= 0.0f || hit_distance > 1e10f)
        return surface_color;

    float t_near, t_far;
    if (!intersectVolumeBounds(ray_origin, ray_dir, params.volume, 0.0f, hit_distance, t_near, t_far))
        return surface_color;

    if (t_far <= t_near)
        return surface_color;

    // Ray march with density-weighted extinction for smooth boundaries
    const float step_size = 10.0f;
    float optical_depth = 0.0f;
    float3 in_scatter = make_float3(0.0f);
    float t = fmaxf(t_near, 0.001f);

    while (t < t_far)
    {
        float3 sample_pos = ray_origin + t * ray_dir;

        // Get local density (0 at edges, 1 deep inside)
        float density = params.volume.densityAt(sample_pos);

        if (density > 0.0f)
        {
            float local_sigma_t = params.volume.sigma_t * density;
            optical_depth += local_sigma_t * step_size;

            float transmittance_so_far = expf(-optical_depth);
            in_scatter += params.fog_color * density * (1.0f - expf(-local_sigma_t * step_size)) * transmittance_so_far;
        }

        t += step_size;
    }

    float transmittance = expf(-optical_depth);
    return surface_color * transmittance + in_scatter;
}

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

    // Payload: RGB color + depth counter + hit distance
    unsigned int p0 = __float_as_uint(0.0f);
    unsigned int p1 = __float_as_uint(0.0f);
    unsigned int p2 = __float_as_uint(0.0f);
    unsigned int p3 = 0;  // depth
    unsigned int p4 = __float_as_uint(1e16f);  // hit distance for fog

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
        p0, p1, p2, p3, p4
    );

    float3 color = make_float3(
        __uint_as_float(p0),
        __uint_as_float(p1),
        __uint_as_float(p2)
    );
    float hit_distance = __uint_as_float(p4);

    // Apply fog to spheres and everything else (only if enabled)
    color = applyFogSpecular(color, ray_origin, ray_direction, hit_distance);

    // Output linear color - store hit_distance in w for combined rendering
    params.frame_buffer[idx.y * params.width + idx.x] = make_float4(color.x, color.y, color.z, hit_distance);
}

