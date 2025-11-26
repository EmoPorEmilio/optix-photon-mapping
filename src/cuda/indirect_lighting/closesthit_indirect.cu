#include <optix.h>
#include <sutil/vec_math.h>
#include "indirect_launch_params.h"
#include "../photon_gather.h"

extern "C" __constant__ IndirectLaunchParams params;


extern "C" __global__ void __closesthit__indirect_triangle()
{
    const unsigned int prim_idx = optixGetPrimitiveIndex();

    // Check if we hit the light source
    if (prim_idx >= params.quadLightStartIndex)
    {
        // Light source - return white
        optixSetPayload_0(__float_as_uint(1.0f));
        optixSetPayload_1(__float_as_uint(1.0f));
        optixSetPayload_2(__float_as_uint(1.0f));
        return;
    }

    const float3 ray_origin = optixGetWorldRayOrigin();
    const float3 ray_dir = optixGetWorldRayDirection();
    const float t_hit = optixGetRayTmax();
    const float3 hit_point = ray_origin + t_hit * ray_dir;

    // Get surface normal
    float3 normal = computeTriangleNormal(ray_dir);

    // Get material
    Material mat;
    if (params.triangle_materials)
        mat = params.triangle_materials[prim_idx];
    else
    {
        mat.type = MATERIAL_DIFFUSE;
        mat.albedo = make_float3(0.8f, 0.8f, 0.8f);
    }

    // Gather photons for indirect illumination (kd-tree accelerated when available)
    float3 color = gatherPhotonsRadianceWithAlbedo(
        hit_point,
        normal,
        mat.albedo,
        params.photon_map,
        params.photon_count,
        params.kdtree,
        params.gather_radius,
        params.brightness_multiplier);

    // Clamp
    color.x = fminf(1.0f, color.x);
    color.y = fminf(1.0f, color.y);
    color.z = fminf(1.0f, color.z);

    optixSetPayload_0(__float_as_uint(color.x));
    optixSetPayload_1(__float_as_uint(color.y));
    optixSetPayload_2(__float_as_uint(color.z));
    optixSetPayload_3(__float_as_uint(t_hit));
}

extern "C" __global__ void __closesthit__indirect_sphere()
{
    // Specular/transmissive spheres - show black for now
    optixSetPayload_0(__float_as_uint(0.0f));
    optixSetPayload_1(__float_as_uint(0.0f));
    optixSetPayload_2(__float_as_uint(0.0f));
    optixSetPayload_3(__float_as_uint(optixGetRayTmax()));
}

