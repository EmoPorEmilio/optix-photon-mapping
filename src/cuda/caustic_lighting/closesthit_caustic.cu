#include <optix.h>
#include <sutil/vec_math.h>
#include "caustic_launch_params.h"

extern "C" __constant__ CausticLaunchParams params;

// Gather caustic photons at a point using Jensen's radiance estimation
// Caustics use the same formula but without albedo modulation since
// the caustic represents specular-transmitted light arriving at a diffuse surface
__device__ float3 gatherCausticPhotons(const float3& hit_point, const float3& normal)
{
    const float radius = params.gather_radius;
    const float radius_sq = radius * radius;
    const float inv_pi = 0.31830988618f;  // 1/π
    
    float3 flux_sum = make_float3(0.0f);
    
    // Linear search through caustic photon map
    for (unsigned int i = 0; i < params.caustic_photon_count; i++)
    {
        Photon photon = params.caustic_photon_map[i];
        
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
    
    // Jensen's radiance estimation with cone filter normalization
    float cone_normalization = 3.0f;
    float area = 3.14159265f * radius_sq;
    float3 radiance = flux_sum * (cone_normalization / area) * inv_pi;
    
    // Apply brightness multiplier for display adjustment
    radiance *= params.brightness_multiplier;
    
    return radiance;
}

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

    if (dot(normal, ray_dir) > 0.0f)
        normal = -normal;

    return normal;
}

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
    float3 normal = getTriangleNormal(ray_dir);
    
    // Gather caustic photons - these are the highlights!
    float3 caustics = gatherCausticPhotons(hit_point, normal);
    
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

