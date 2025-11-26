#include <optix.h>
#include <sutil/vec_math.h>
#include "indirect_launch_params.h"

extern "C" __constant__ IndirectLaunchParams params;

// Compute normal based on which Cornell box wall was hit
__device__ float3 getWallNormal(const float3& hit_point, const float3& ray_dir)
{
    const float eps = 5.0f;
    
    if (hit_point.y < eps) // Floor
        return make_float3(0.0f, 1.0f, 0.0f);
    if (hit_point.y > 548.8f - eps) // Ceiling
        return make_float3(0.0f, -1.0f, 0.0f);
    if (hit_point.x < eps) // Left wall (red)
        return make_float3(1.0f, 0.0f, 0.0f);
    if (hit_point.x > 556.0f - eps) // Right wall (blue)
        return make_float3(-1.0f, 0.0f, 0.0f);
    if (hit_point.z > 559.2f - eps) // Back wall
        return make_float3(0.0f, 0.0f, -1.0f);
    
    return normalize(-ray_dir);
}

// Simple photon gathering - linear search through all photons
__device__ float3 gatherPhotons(const float3& hit_point, const float3& normal, const float3& albedo)
{
    float3 indirect = make_float3(0.0f);
    
    const float radius = params.gather_radius;
    const float radius_sq = radius * radius;
    
    unsigned int gathered = 0;
    
    // Linear search through photon map
    for (unsigned int i = 0; i < params.photon_count; i++)
    {
        Photon photon = params.photon_map[i];
        
        // Distance check
        float3 diff = hit_point - photon.position;
        float dist_sq = dot(diff, diff);
        
        if (dist_sq < radius_sq)
        {
            // Check if photon is on the same side of the surface
            // (incident direction should be roughly opposite to normal)
            float incidentDot = dot(photon.incidentDir, normal);
            if (incidentDot < 0.0f)  // Photon came from the correct side
            {
                // Cone filter weight (gives more weight to closer photons)
                float weight = 1.0f - sqrtf(dist_sq) / radius;
                
                // Add photon contribution
                indirect += photon.power * weight;
                gathered++;
            }
        }
    }
    
    if (gathered > 0)
    {
        // Normalize by the disc area (Jensen's formula)
        // The cone filter has a normalization factor of 3/(pi*r^2)
        float area = 3.14159265f * radius_sq;
        indirect = indirect * albedo / area;
        
        // Scale for visibility - photon power is normalized, need significant boost
        indirect *= 50000.0f;
    }
    
    return indirect;
}

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
    float3 normal = getWallNormal(hit_point, ray_dir);

    // Get material
    Material mat;
    if (params.triangle_materials)
        mat = params.triangle_materials[prim_idx];
    else
    {
        mat.type = MATERIAL_DIFFUSE;
        mat.albedo = make_float3(0.8f, 0.8f, 0.8f);
    }

    // Gather photons for indirect illumination
    float3 color = gatherPhotons(hit_point, normal, mat.albedo);

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

