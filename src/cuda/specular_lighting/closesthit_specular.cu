#include <optix.h>
#include <sutil/vec_math.h>
#include "specular_launch_params.h"

extern "C" __constant__ SpecularLaunchParams params;

// Refract helper - returns false if total internal reflection
__device__ bool refract(const float3& incident, const float3& normal, float eta, float3& refracted)
{
    float cos_i = -dot(incident, normal);
    float sin2_t = eta * eta * (1.0f - cos_i * cos_i);
    
    if (sin2_t > 1.0f)
        return false;  // Total internal reflection
    
    float cos_t = sqrtf(1.0f - sin2_t);
    refracted = eta * incident + (eta * cos_i - cos_t) * normal;
    return true;
}

// Get wall normal based on position
__device__ float3 getWallNormal(const float3& hit_point)
{
    const float eps = 5.0f;
    
    if (hit_point.y < eps) return make_float3(0.0f, 1.0f, 0.0f);      // Floor
    if (hit_point.y > 548.8f - eps) return make_float3(0.0f, -1.0f, 0.0f);  // Ceiling
    if (hit_point.x < eps) return make_float3(1.0f, 0.0f, 0.0f);      // Left wall
    if (hit_point.x > 556.0f - eps) return make_float3(-1.0f, 0.0f, 0.0f); // Right wall
    if (hit_point.z > 559.2f - eps) return make_float3(0.0f, 0.0f, -1.0f); // Back wall
    
    return make_float3(0.0f, 1.0f, 0.0f);
}

// Gather photons for indirect/caustic lighting
__device__ float3 gatherPhotons(const float3& hit_point, const float3& normal, 
                                 const Photon* photon_map, unsigned int count, float radius)
{
    if (count == 0 || photon_map == nullptr)
        return make_float3(0.0f);
    
    float3 result = make_float3(0.0f);
    float radius_sq = radius * radius;
    unsigned int gathered = 0;
    
    for (unsigned int i = 0; i < count; i++)
    {
        float3 diff = hit_point - photon_map[i].position;
        float dist_sq = dot(diff, diff);
        
        if (dist_sq < radius_sq)
        {
            float incidentDot = dot(photon_map[i].incidentDir, normal);
            if (incidentDot < 0.0f)
            {
                float weight = 1.0f - sqrtf(dist_sq) / radius;
                result += photon_map[i].power * weight;
                gathered++;
            }
        }
    }
    
    if (gathered > 0)
    {
        float area = 3.14159265f * radius_sq;
        result = result / area;
    }
    
    return result;
}

// Compute direct lighting at a point
__device__ float3 computeDirectLight(const float3& hit_point, const float3& normal, const float3& albedo)
{
    float3 toLight = params.light_position - hit_point;
    float lightDist = length(toLight);
    float3 L = toLight / lightDist;
    
    float NdotL = fmaxf(0.0f, dot(normal, L));
    if (NdotL <= 0.0f)
        return make_float3(0.0f);
    
    // Simple attenuation
    float attenuation = 1.0f / (lightDist * lightDist * 0.00001f + 1.0f);
    
    return albedo * NdotL * attenuation * 0.3f;
}

// Triangles return FULL lighting when hit by reflected/refracted rays
extern "C" __global__ void __closesthit__specular_triangle()
{
    const unsigned int prim_idx = optixGetPrimitiveIndex();
    unsigned int depth = optixGetPayload_3();
    
    // Primary rays hitting triangles = black (only show spheres)
    if (depth == 0)
    {
        optixSetPayload_0(__float_as_uint(0.0f));
        optixSetPayload_1(__float_as_uint(0.0f));
        optixSetPayload_2(__float_as_uint(0.0f));
        return;
    }
    
    // Light source
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
    
    // Get normal and albedo
    float3 normal = getWallNormal(hit_point);
    float3 albedo = make_float3(0.8f, 0.8f, 0.8f);  // Default gray
    
    if (params.triangle_materials)
        albedo = params.triangle_materials[prim_idx].albedo;
    
    // Compute full lighting: direct + indirect + caustics
    float3 color = make_float3(0.0f);
    
    // Direct lighting (stronger)
    color += computeDirectLight(hit_point, normal, albedo) * 2.0f;
    
    // Indirect (global photon map) - brighter
    float3 indirect = gatherPhotons(hit_point, normal, params.global_photon_map, 
                                     params.global_photon_count, params.gather_radius);
    color += indirect * albedo * 100000.0f;  // Doubled for brightness
    
    // Caustics - brighter
    float3 caustics = gatherPhotons(hit_point, normal, params.caustic_photon_map,
                                     params.caustic_photon_count, params.gather_radius * 0.5f);
    color += caustics * 200000.0f;  // Caustics are brighter
    
    // Stronger ambient for base visibility
    color += albedo * 0.15f;
    
    // Clamp
    color.x = fminf(1.0f, color.x);
    color.y = fminf(1.0f, color.y);
    color.z = fminf(1.0f, color.z);
    
    optixSetPayload_0(__float_as_uint(color.x));
    optixSetPayload_1(__float_as_uint(color.y));
    optixSetPayload_2(__float_as_uint(color.z));
}

// Spheres show reflection/refraction
extern "C" __global__ void __closesthit__specular_sphere()
{
    const unsigned int prim_idx = optixGetPrimitiveIndex();
    unsigned int depth = optixGetPayload_3();
    
    // Max recursion depth
    if (depth >= 10)
    {
        optixSetPayload_0(__float_as_uint(0.0f));
        optixSetPayload_1(__float_as_uint(0.0f));
        optixSetPayload_2(__float_as_uint(0.0f));
        return;
    }
    
    const float3 ray_origin = optixGetWorldRayOrigin();
    const float3 ray_dir = optixGetWorldRayDirection();
    const float t_hit = optixGetRayTmax();
    const float3 hit_point = ray_origin + t_hit * ray_dir;
    
    // Get normal from intersection attributes
    float3 normal = make_float3(
        __uint_as_float(optixGetAttribute_0()),
        __uint_as_float(optixGetAttribute_1()),
        __uint_as_float(optixGetAttribute_2()));
    normal = normalize(normal);
    
    Material mat = params.sphere_materials[prim_idx];
    
    float3 new_origin;
    float3 new_dir;
    
    if (mat.type == MATERIAL_SPECULAR)
    {
        // Perfect mirror reflection
        new_dir = reflect(ray_dir, normal);
        new_origin = hit_point + normal * 0.001f;
    }
    else if (mat.type == MATERIAL_TRANSMISSIVE)
    {
        // Refraction (glass)
        float ior = 1.5f;  // Index of refraction for glass
        
        // Determine if entering or exiting
        bool entering = dot(ray_dir, normal) < 0.0f;
        float3 n = entering ? normal : -normal;
        float eta = entering ? (1.0f / ior) : ior;
        
        float3 refracted;
        if (refract(ray_dir, n, eta, refracted))
        {
            new_dir = normalize(refracted);
            new_origin = hit_point - n * 0.001f;  // Move slightly inside
        }
        else
        {
            // Total internal reflection
            new_dir = reflect(ray_dir, n);
            new_origin = hit_point + n * 0.001f;
        }
    }
    else
    {
        // Unknown material - black
        optixSetPayload_0(__float_as_uint(0.0f));
        optixSetPayload_1(__float_as_uint(0.0f));
        optixSetPayload_2(__float_as_uint(0.0f));
        return;
    }
    
    // Trace secondary ray
    unsigned int p0 = __float_as_uint(0.0f);
    unsigned int p1 = __float_as_uint(0.0f);
    unsigned int p2 = __float_as_uint(0.0f);
    unsigned int p3 = depth + 1;
    
    optixTrace(
        params.handle,
        new_origin,
        new_dir,
        0.001f,
        1e16f,
        0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        0, 1, 0,
        p0, p1, p2, p3
    );
    
    // Get the reflected/refracted color
    float3 color = make_float3(
        __uint_as_float(p0),
        __uint_as_float(p1),
        __uint_as_float(p2)
    );
    
    if (mat.type == MATERIAL_TRANSMISSIVE)
    {
        // Glass: slight blue tint, minimal attenuation
        color *= make_float3(0.98f, 0.99f, 1.0f);
        // Add slight Fresnel effect (brighter at edges)
        float fresnel = 0.1f + 0.9f * powf(1.0f - fabsf(dot(ray_dir, normal)), 3.0f);
        color += make_float3(0.1f, 0.1f, 0.12f) * fresnel;
    }
    else if (mat.type == MATERIAL_SPECULAR)
    {
        // Mirror: high reflectivity
        color *= 0.95f;
        // Add slight specular highlight
        color += make_float3(0.05f, 0.05f, 0.05f);
    }
    
    optixSetPayload_0(__float_as_uint(color.x));
    optixSetPayload_1(__float_as_uint(color.y));
    optixSetPayload_2(__float_as_uint(color.z));
}

