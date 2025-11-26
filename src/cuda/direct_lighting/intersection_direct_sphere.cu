#include <optix.h>
#include <sutil/vec_math.h>
#include "direct_launch_params.h"

extern "C" __constant__ DirectLaunchParams params;

// Sphere intersection for direct lighting
extern "C" __global__ void __intersection__direct_sphere()
{
    const unsigned int prim_idx = optixGetPrimitiveIndex();

    const float3 ray_origin = optixGetWorldRayOrigin();
    const float3 ray_dir = optixGetWorldRayDirection();

    float3 center;
    float radius;

    if (prim_idx == 0)
    {
        center = params.sphere1_center;
        radius = params.sphere1_radius;
    }
    else
    {
        center = params.sphere2_center;
        radius = params.sphere2_radius;
    }

    const float3 oc = ray_origin - center;
    const float a = dot(ray_dir, ray_dir);
    const float b = 2.0f * dot(oc, ray_dir);
    const float c = dot(oc, oc) - radius * radius;
    const float discriminant = b * b - 4.0f * a * c;

    if (discriminant < 0.0f)
        return;

    const float sqrtD = sqrtf(discriminant);
    float t = (-b - sqrtD) / (2.0f * a);

    if (t < optixGetRayTmin() || t > optixGetRayTmax())
    {
        t = (-b + sqrtD) / (2.0f * a);
        if (t < optixGetRayTmin() || t > optixGetRayTmax())
            return;
    }

    // Compute normal at hit point
    const float3 hit_point = ray_origin + t * ray_dir;
    const float3 normal = (hit_point - center) / radius;

    // Report intersection with normal as attributes
    optixReportIntersection(
        t,
        0,
        __float_as_uint(normal.x),
        __float_as_uint(normal.y),
        __float_as_uint(normal.z));
}

