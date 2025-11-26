#include <optix.h>
#include <sutil/vec_math.h>
#include "indirect_launch_params.h"

extern "C" __constant__ IndirectLaunchParams params;

// Compute geometric triangle normal from vertex positions
__device__ float3 getTriangleNormal(const float3& ray_dir)
{
    float3 vertices[3];
    optixGetTriangleVertexData(
        optixGetGASTraversableHandle(),
        optixGetPrimitiveIndex(),
        optixGetSbtGASIndex(),
        0.0f,
        vertices
    );

    float3 edge1 = vertices[1] - vertices[0];
    float3 edge2 = vertices[2] - vertices[0];
    float3 normal = normalize(cross(edge1, edge2));

    // Ensure normal faces toward the ray origin
    if (dot(normal, ray_dir) > 0.0f)
        normal = -normal;

    return normal;
}

// Photon gathering with correct Jensen radiance estimation formula
// L(x, ω) = Σ(Φ_p * f_r(x, ω_p, ω) * K(d_p)) / (π * r²)
// For diffuse surfaces: f_r = albedo / π
// Cone filter K(d) = (1 - d/r) requires normalization factor of 3
__device__ float3 gatherPhotons(const float3& hit_point, const float3& normal, const float3& albedo)
{
    const float radius = params.gather_radius;
    const float radius_sq = radius * radius;
    const float inv_pi = 0.31830988618f;  // 1/π
    
    float3 flux_sum = make_float3(0.0f);
    
    // Linear search through photon map
    for (unsigned int i = 0; i < params.photon_count; i++)
    {
        Photon photon = params.photon_map[i];
        
        float3 diff = hit_point - photon.position;
        float dist_sq = dot(diff, diff);
        
        if (dist_sq < radius_sq)
        {
            float incidentDot = dot(photon.incidentDir, normal);
            if (incidentDot < 0.0f)
            {
                float weight = 1.0f - sqrtf(dist_sq) / radius;
                flux_sum += photon.power * weight;
            }
        }
    }
    
    // Jensen's radiance estimation formula
    // L = (albedo / π) * (3 / (π * r²)) * Σ(Φ_p * K)
    float cone_normalization = 3.0f;
    float area = 3.14159265f * radius_sq;
    float3 radiance = flux_sum * (cone_normalization / area) * albedo * inv_pi;
    
    // Apply brightness multiplier for display adjustment
    radiance *= params.brightness_multiplier;
    
    return radiance;
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
    float3 normal = getTriangleNormal(ray_dir);

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

