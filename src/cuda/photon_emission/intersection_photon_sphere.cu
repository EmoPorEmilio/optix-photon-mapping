#include <optix.h>
#include <sutil/vec_math.h>
#include "photon_launch_params.h"

// Custom intersection program for analytic spheres in the photon pass.
// Uses PhotonLaunchParams::sphere1/sphere2 to define geometry.

#ifndef PHOTON_PARAMS_DEFINED
extern "C" __constant__ PhotonLaunchParams params;
#endif

extern "C" __global__ void __intersection__photon_sphere()
{
    const unsigned int primIdx = optixGetPrimitiveIndex(); // 0 or 1

    // Select sphere geometry.
    const PhotonSphereData sphere = (primIdx == 0) ? params.sphere1 : params.sphere2;
    const float3 center = sphere.center;
    const float radius = sphere.radius;

    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir = optixGetWorldRayDirection();
    const float tmin = optixGetRayTmin();
    const float tmax = optixGetRayTmax();

    const float3 oc = ray_orig - center;
    const float a = dot(ray_dir, ray_dir);
    const float b = 2.0f * dot(oc, ray_dir);
    const float c = dot(oc, oc) - radius * radius;
    const float discriminant = b * b - 4.0f * a * c;

    if (discriminant >= 0.0f)
    {
        const float sqrtd = sqrtf(discriminant);
        float t = (-b - sqrtd) / (2.0f * a);

        if (t < tmin)
            t = (-b + sqrtd) / (2.0f * a);

        if (t >= tmin && t <= tmax)
        {
            const float3 hit_point = ray_orig + t * ray_dir;
            const float3 normal = normalize(hit_point - center);

            optixReportIntersection(
                t,
                0,
                __float_as_uint(normal.x),
                __float_as_uint(normal.y),
                __float_as_uint(normal.z));
        }
    }
}


