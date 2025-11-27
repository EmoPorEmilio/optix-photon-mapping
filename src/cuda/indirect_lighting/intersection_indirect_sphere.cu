#include <optix.h>
#include <sutil/vec_math.h>
#include "indirect_launch_params.h"

extern "C" __constant__ IndirectLaunchParams params;

// Hardcoded sphere data - must match configuration.xml
__device__ const float3 sphere_centers[2] = {
    {185.0f, 82.5f, 169.0f}, // Sphere 1 (transmissive/glass)
    {368.0f, 82.5f, 351.0f}  // Sphere 2 (specular/mirror)
};
__device__ const float sphere_radii[2] = {82.5f, 82.5f};

extern "C" __global__ void __intersection__indirect_sphere()
{
    const unsigned int prim_idx = optixGetPrimitiveIndex();

    const float3 ray_origin = optixGetWorldRayOrigin();
    const float3 ray_direction = optixGetWorldRayDirection();

    // Get sphere data based on primitive index
    float3 center = sphere_centers[prim_idx];
    float radius = sphere_radii[prim_idx];

    float3 O = ray_origin - center;
    float b = dot(O, ray_direction);
    float c = dot(O, O) - radius * radius;
    float discriminant = b * b - c;

    if (discriminant >= 0.0f)
    {
        float sqrt_disc = sqrtf(discriminant);
        float t1 = -b - sqrt_disc;
        float t2 = -b + sqrt_disc;

        float t_hit = (t1 > optixGetRayTmin()) ? t1 : t2;

        if (t_hit > optixGetRayTmin() && t_hit < optixGetRayTmax())
        {
            float3 hit_point = ray_origin + t_hit * ray_direction;
            float3 normal = normalize(hit_point - center);

            optixReportIntersection(
                t_hit,
                0,
                __float_as_uint(normal.x),
                __float_as_uint(normal.y),
                __float_as_uint(normal.z));
        }
    }
}
