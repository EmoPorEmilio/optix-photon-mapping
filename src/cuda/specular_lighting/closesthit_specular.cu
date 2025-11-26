#include <optix.h>
#include <sutil/vec_math.h>
#include "specular_launch_params.h"
#include "../photon_gather.h"

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

// Schlick's approximation for Fresnel reflectance (Jensen's algorithm)
// R(θ) = R0 + (1 - R0) * (1 - cos(θ))^5
// where R0 = ((n1 - n2) / (n1 + n2))^2
__device__ float schlickFresnel(float cos_i, float n1, float n2)
{
    float r0 = (n1 - n2) / (n1 + n2);
    r0 = r0 * r0;
    float one_minus_cos = 1.0f - fabsf(cos_i);
    float one_minus_cos5 = one_minus_cos * one_minus_cos * one_minus_cos * one_minus_cos * one_minus_cos;
    return r0 + (1.0f - r0) * one_minus_cos5;
}

// Helper to trace a secondary ray and get the color
__device__ float3 traceSecondaryRay(const float3& origin, const float3& direction, unsigned int depth)
{
    unsigned int p0 = __float_as_uint(0.0f);
    unsigned int p1 = __float_as_uint(0.0f);
    unsigned int p2 = __float_as_uint(0.0f);
    unsigned int p3 = depth + 1;
    
    optixTrace(
        params.handle,
        origin,
        direction,
        0.001f,
        1e16f,
        0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        0, 1, 0,
        p0, p1, p2, p3
    );
    
    return make_float3(__uint_as_float(p0), __uint_as_float(p1), __uint_as_float(p2));
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
    float3 normal = getTriangleNormal(ray_dir);
    float3 albedo = make_float3(0.8f, 0.8f, 0.8f);  // Default gray
    
    if (params.triangle_materials)
        albedo = params.triangle_materials[prim_idx].albedo;
    
    // Compute full lighting: direct + indirect + caustics
    float3 color = make_float3(0.0f);
    
    // Direct lighting
    color += computeDirectLight(hit_point, normal, albedo) * 2.0f;
    
    // Indirect (global photon map) - configurable brightness
    float3 indirect = gatherPhotonsRadiance(
        hit_point,
        normal,
        params.global_photon_map,
        params.global_photon_count,
        params.global_kdtree,
        params.gather_radius);
    color += indirect * albedo * params.indirect_brightness;
    
    // Caustics - configurable brightness
    float3 caustics = gatherPhotonsRadiance(
        hit_point,
        normal,
        params.caustic_photon_map,
        params.caustic_photon_count,
        params.caustic_kdtree,
        params.gather_radius * 0.5f);
    color += caustics * params.caustic_brightness;
    
    // Configurable ambient for base visibility
    color += albedo * params.specular_ambient;
    
    // Clamp
    color.x = fminf(1.0f, color.x);
    color.y = fminf(1.0f, color.y);
    color.z = fminf(1.0f, color.z);
    
    optixSetPayload_0(__float_as_uint(color.x));
    optixSetPayload_1(__float_as_uint(color.y));
    optixSetPayload_2(__float_as_uint(color.z));
}

// Spheres show reflection/refraction with proper Fresnel handling (Jensen's algorithm)
extern "C" __global__ void __closesthit__specular_sphere()
{
    const unsigned int prim_idx = optixGetPrimitiveIndex();
    unsigned int depth = optixGetPayload_3();
    
    // Max recursion depth - configurable
    if (depth >= params.max_recursion_depth)
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
    float3 color = make_float3(0.0f);
    
    if (mat.type == MATERIAL_SPECULAR)
    {
        // Perfect mirror reflection
        float3 reflect_dir = reflect(ray_dir, normal);
        float3 reflect_origin = hit_point + normal * 0.001f;
        
        color = traceSecondaryRay(reflect_origin, reflect_dir, depth);
        color *= params.mirror_reflectivity;
    }
    else if (mat.type == MATERIAL_TRANSMISSIVE)
    {
        // Glass with proper Fresnel ray splitting (Jensen's algorithm)
        float ior = params.glass_ior;
        
        // Determine if entering or exiting
        bool entering = dot(ray_dir, normal) < 0.0f;
        float3 n = entering ? normal : -normal;
        float cos_i = -dot(ray_dir, n);
        
        float n1 = entering ? 1.0f : ior;
        float n2 = entering ? ior : 1.0f;
        float eta = n1 / n2;
        
        // Compute Fresnel reflectance using Schlick's approximation
        float fresnel_r = schlickFresnel(cos_i, n1, n2);
        
        // Clamp Fresnel to configured range
        fresnel_r = fmaxf(params.fresnel_min, fminf(1.0f, fresnel_r));
        
        // Compute reflection direction
        float3 reflect_dir = reflect(ray_dir, n);
        float3 reflect_origin = hit_point + n * 0.001f;
        
        // Compute refraction direction
        float3 refracted;
        bool can_refract = refract(ray_dir, n, eta, refracted);
        
        if (!can_refract)
        {
            // Total internal reflection - all light is reflected
            color = traceSecondaryRay(reflect_origin, reflect_dir, depth);
        }
        else
        {
            // Proper Fresnel ray splitting: trace both reflection and refraction
            float3 refract_dir = normalize(refracted);
            float3 refract_origin = hit_point - n * 0.001f;
            
            // Trace reflection ray
            float3 reflect_color = traceSecondaryRay(reflect_origin, reflect_dir, depth);
            
            // Trace refraction ray
            float3 refract_color = traceSecondaryRay(refract_origin, refract_dir, depth);
            
            // Blend by Fresnel coefficient (Jensen's correct weighting)
            color = reflect_color * fresnel_r + refract_color * (1.0f - fresnel_r);
        }
        
        // Apply glass tint
        color *= params.glass_tint;
    }
    else
    {
        // Unknown material - black
        optixSetPayload_0(__float_as_uint(0.0f));
        optixSetPayload_1(__float_as_uint(0.0f));
        optixSetPayload_2(__float_as_uint(0.0f));
        return;
    }
    
    optixSetPayload_0(__float_as_uint(color.x));
    optixSetPayload_1(__float_as_uint(color.y));
    optixSetPayload_2(__float_as_uint(color.z));
}

