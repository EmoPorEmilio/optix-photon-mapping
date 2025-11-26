#include <optix.h>
#include <sutil/vec_math.h>
#include "direct_launch_params.h"

extern "C" __constant__ DirectLaunchParams params;

// Simple pseudo-random for soft shadows (optional)
__device__ __forceinline__ float frand(unsigned int &seed)
{
    seed = seed * 1664525u + 1013904223u;
    return (seed & 0x00FFFFFF) / static_cast<float>(0x01000000);
}

// Check if ray intersects a sphere (for shadow testing)
__device__ bool raySphereIntersect(const float3 &origin, const float3 &dir,
                                   const float3 &center, float radius, float maxDist)
{
    float3 oc = origin - center;
    float a = dot(dir, dir);
    float b = 2.0f * dot(oc, dir);
    float c = dot(oc, oc) - radius * radius;
    float disc = b * b - 4.0f * a * c;

    if (disc < 0.0f)
        return false;

    float sqrtD = sqrtf(disc);
    float t = (-b - sqrtD) / (2.0f * a);
    if (t > 0.001f && t < maxDist - 0.001f)
        return true;

    t = (-b + sqrtD) / (2.0f * a);
    return (t > 0.001f && t < maxDist - 0.001f);
}

// Check if a point is occluded from the light by the spheres
__device__ bool isOccludedBySpheres(const float3 &origin, const float3 &direction, float maxDist)
{
    if (raySphereIntersect(origin, direction, params.sphere1_center, params.sphere1_radius, maxDist))
        return true;
    if (raySphereIntersect(origin, direction, params.sphere2_center, params.sphere2_radius, maxDist))
        return true;
    return false;
}

// Compute triangle normal using barycentric interpolation from built-in attributes
__device__ float3 computeTriangleNormal(const float3 &ray_dir)
{
    // Get barycentric coordinates from built-in function
    const float2 barycentrics = optixGetTriangleBarycentrics();

    // For a flat triangle, the normal is perpendicular to the face
    // We can compute it from the ray direction and the fact that we hit from outside
    // For now, use the geometric normal from OptixGetWorldRayDirection
    // The normal should face opposite to the ray direction (toward the viewer)

    // Simple heuristic: determine normal based on which wall was hit
    // This works for axis-aligned Cornell box walls
    const float3 hit_point = optixGetWorldRayOrigin() + optixGetRayTmax() * ray_dir;

    const float eps = 5.0f; // Tolerance for wall detection

    // Check which wall we hit based on position
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

    // Default: face toward camera
    return normalize(-ray_dir);
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
    float3 shadowOrigin = hit_point + normal * 0.01f;

    // Check if occluded by spheres
    bool occluded = isOccludedBySpheres(shadowOrigin, L, lightDist);

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

// Shadow ray hit - means something is blocking the light
extern "C" __global__ void __closesthit__shadow()
{
    optixSetPayload_0(1u); // Occluded
}
