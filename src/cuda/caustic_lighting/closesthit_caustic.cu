#include <optix.h>
#include <sutil/vec_math.h>
#include "caustic_launch_params.h"
#include "../photon_gather.h"

extern "C" __constant__ CausticLaunchParams params;

// Walls/floor show caustic highlights
extern "C" __global__ void __closesthit__caustic_triangle()
{
    const unsigned int prim_idx = optixGetPrimitiveIndex();

    // Light source still visible
    if (prim_idx >= params.quadLightStartIndex)
    {
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

    // Get material for BRDF - Jensen's Eq. 8 requires f_r(x,ω_p,ω)
    // For diffuse surfaces: f_r = ρ_d/π (albedo divided by π)
    Material mat;
    if (params.triangle_materials)
        mat = params.triangle_materials[prim_idx];
    else
    {
        mat.type = MATERIAL_DIFFUSE;
        mat.albedo = make_float3(0.8f, 0.8f, 0.8f);
    }

    // Gather caustic photons with surface albedo (Jensen's radiance estimate)
    // L_r(x,ω) ≈ (1/πr²) Σ f_r(x,ω_p,ω) * ΔΦ_p
    float3 caustics = gatherPhotonsRadianceWithAlbedo(
        hit_point,
        normal,
        mat.albedo,
        params.caustic_photon_map,
        params.caustic_photon_count,
        params.caustic_kdtree,
        params.gather_radius,
        params.brightness_multiplier);

    // Caustics are bright highlights on dark background
    float3 color = caustics;
    
    color.x = fminf(1.0f, color.x);
    color.y = fminf(1.0f, color.y);
    color.z = fminf(1.0f, color.z);
    
    optixSetPayload_0(__float_as_uint(color.x));
    optixSetPayload_1(__float_as_uint(color.y));
    optixSetPayload_2(__float_as_uint(color.z));
    optixSetPayload_3(__float_as_uint(t_hit));
}

// Spheres are black in caustics mode - they don't show caustics, they CREATE them
extern "C" __global__ void __closesthit__caustic_sphere()
{
    optixSetPayload_0(__float_as_uint(0.0f));
    optixSetPayload_1(__float_as_uint(0.0f));
    optixSetPayload_2(__float_as_uint(0.0f));
    optixSetPayload_3(__float_as_uint(optixGetRayTmax()));
}

