#include <optix.h>
#include <sutil/vec_math.h>
#include "direct_launch_params.h"
#include "../photon_gather.h"  // For shared computeTriangleNormal

extern "C" __constant__ DirectLaunchParams params;

// Simple pseudo-random for soft shadows (optional)
__device__ __forceinline__ float frand(unsigned int &seed)
{
    seed = seed * 1664525u + 1013904223u;
    return (seed & 0x00FFFFFF) / static_cast<float>(0x01000000);
}

// Trace shadow ray using OptiX to test occlusion by ALL geometry (triangles + spheres)
__device__ bool traceOcclusionRay(const float3 &origin, const float3 &direction, float maxDist)
{
    // Use same payload structure as primary rays for consistency
    unsigned int p0 = 0u;
    
    // Trace shadow ray
    // SBT layout: [0]=primary_tri, [1]=primary_sphere, [2]=shadow_tri, [3]=shadow_sphere
    // SBT offset 2 = shadow hit groups start
    // SBT stride 1 = same as primary rays (1 record per instance)
    // Miss index 1 = shadow miss program
    optixTrace(
        params.handle,
        origin,
        direction,
        0.1f,             // tmin - offset to avoid self-intersection
        maxDist * 0.999f, // tmax - stop just before reaching the light (proportional)
        0.0f,             // rayTime
        OptixVisibilityMask(1),  // Match instance visibility mask (both instances use 1)
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        2,                // SBT offset (shadow hit groups start at index 2)
        1,                // SBT stride (1 record per instance, same as primary)
        1,                // miss index (shadow miss)
        p0                // payload 0 - will be set to 1 if occluded
    );
    
    return p0 != 0u;
}

extern "C" __global__ void __closesthit__direct_triangle()
{
    const unsigned int prim_idx = optixGetPrimitiveIndex();

    // Check if we hit the light source itself
    if (prim_idx >= params.quadLightStartIndex)
    {
        // Hit the light - return white light
        float3 emission = make_float3(1.0f, 1.0f, 1.0f);
        optixSetPayload_0(__float_as_uint(emission.x));
        optixSetPayload_1(__float_as_uint(emission.y));
        optixSetPayload_2(__float_as_uint(emission.z));
        optixSetPayload_3(__float_as_uint(optixGetRayTmax()));  // Hit distance for fog
        return;
    }

    const float3 ray_origin = optixGetWorldRayOrigin();
    const float3 ray_dir = optixGetWorldRayDirection();
    const float t_hit = optixGetRayTmax();
    const float3 hit_point = ray_origin + t_hit * ray_dir;

    // Get surface normal based on which wall we hit
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

    // Direct lighting computation
    float3 color = make_float3(0.0f);

    // Sample light center for hard shadows
    float3 lightPos = params.light.position;
    float3 toLight = lightPos - hit_point;
    float lightDist = length(toLight);
    float3 L = toLight / lightDist;

    // Offset the shadow ray origin along the normal to avoid self-intersection
    float3 shadowOrigin = hit_point + normal * 0.1f;

    // Check if occluded by any geometry (triangles including bunny, and spheres)
    bool occluded = traceOcclusionRay(shadowOrigin, L, lightDist);

    if (!occluded)
    {
        // N dot L (use absolute value for two-sided lighting on walls)
        float NdotL = fabsf(dot(normal, L));

        // Simple inverse square attenuation with configurable factor
        float attenuation = 1.0f / (lightDist * lightDist * params.attenuation_factor + 1.0f);

        // Diffuse contribution with configurable intensity
        color = mat.albedo * NdotL * attenuation * params.intensity_multiplier;
    }
    else
    {
        // SHADOW - configurable shadow ambient
        color = mat.albedo * params.shadow_ambient;
    }

    // Add ambient term for visibility (even in shadow)
    color += mat.albedo * params.ambient;

    // Clamp and write result
    color.x = fminf(1.0f, color.x);
    color.y = fminf(1.0f, color.y);
    color.z = fminf(1.0f, color.z);

    optixSetPayload_0(__float_as_uint(color.x));
    optixSetPayload_1(__float_as_uint(color.y));
    optixSetPayload_2(__float_as_uint(color.z));
    optixSetPayload_3(__float_as_uint(t_hit));
}

extern "C" __global__ void __closesthit__direct_sphere()
{
    // Specular/transmissive spheres don't contribute to direct lighting
    // They would need reflection/refraction tracing which is not part of this simple direct lighting pass
    // Show them as black (they block light but don't emit or reflect in this simplified model)

    float3 color = make_float3(0.0f, 0.0f, 0.0f);

    optixSetPayload_0(__float_as_uint(color.x));
    optixSetPayload_1(__float_as_uint(color.y));
    optixSetPayload_2(__float_as_uint(color.z));
    optixSetPayload_3(__float_as_uint(optixGetRayTmax()));
}

// Shadow ray hit on triangle - check if it's actual geometry, not the light
extern "C" __global__ void __closesthit__shadow()
{
    // Don't count light geometry as occlusion
    const unsigned int prim_idx = optixGetPrimitiveIndex();
    if (prim_idx >= params.quadLightStartIndex)
    {
        // Hit the light geometry itself - not an occlusion
        optixSetPayload_0(0u);
        return;
    }
    optixSetPayload_0(1u); // Occluded by actual geometry
}

// Shadow ray hit on sphere - always blocks light
extern "C" __global__ void __closesthit__shadow_sphere()
{
    optixSetPayload_0(1u); // Occluded
}
