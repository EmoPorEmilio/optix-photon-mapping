#include <optix.h>
#include <sutil/vec_math.h>
#include "photon_launch_params.h"
#include "photon_rng.h"

#ifndef PHOTON_PARAMS_DEFINED
extern "C" __constant__ PhotonLaunchParams params;
#endif

// Forward declaration - defined in combined file
__device__ __forceinline__ void recordTrajectoryEvent(
    unsigned int photon_idx, int event_type, const float3 &position,
    const float3 &direction, const float3 &power, int material_type);

//=============================================================================
// Helper Functions
//=============================================================================

// Reflection helper for specular materials
static __forceinline__ __device__ float3 reflect_dir(const float3 &I, const float3 &N)
{
    return I - 2.0f * dot(I, N) * N;
}

// Refraction helper (Snell's law). Returns false for total internal reflection.
static __forceinline__ __device__ bool refract_dir(const float3 &I, const float3 &N, float eta, float3 &refracted)
{
    float cosi = -dot(N, I);
    float sint2 = max(0.0f, 1.0f - cosi * cosi);
    float cost2 = 1.0f - eta * eta * sint2;
    if (cost2 <= 0.0f)
        return false;  // total internal reflection
    float cost = sqrtf(cost2);
    refracted = eta * I + (eta * cosi - cost) * N;
    return true;
}

// Schlick's approximation for Fresnel reflectance
static __forceinline__ __device__ float schlickFresnelPhoton(float cos_i, float n1, float n2)
{
    float r0 = (n1 - n2) / (n1 + n2);
    r0 = r0 * r0;
    float one_minus_cos = 1.0f - fabsf(cos_i);
    float one_minus_cos5 = one_minus_cos * one_minus_cos * one_minus_cos * one_minus_cos * one_minus_cos;
    return r0 + (1.0f - r0) * one_minus_cos5;
}

//=============================================================================
// Photon Storage Helpers
//=============================================================================

// Store photon in global photon map
__device__ void store_photon(const float3 &hit_point, const float3 &incident_dir, const float3 &throughput)
{
    unsigned int stored_idx = atomicAdd((unsigned int *)params.photon_counter, 1u);
    if (stored_idx < params.num_photons)
    {
        Photon photon;
        photon.position = hit_point;
        photon.power = throughput;
        photon.incidentDir = normalize(incident_dir);
        photon.flag = 0;
        params.photons_out[stored_idx] = photon;
    }
}

// Store caustic photon (previous hit was specular/transmissive)
__device__ void store_caustic_photon(const float3 &hit_point, const float3 &incident_dir, const float3 &throughput)
{
    unsigned int stored_idx = atomicAdd((unsigned int *)params.caustic_photon_counter, 1u);
    if (stored_idx < params.num_photons)
    {
        Photon photon;
        photon.position = hit_point;
        photon.power = throughput;
        photon.incidentDir = normalize(incident_dir);
        photon.flag = 1;  // Mark as caustic
        params.caustic_photons_out[stored_idx] = photon;
    }
}

//=============================================================================
// Triangle Closest Hit (Diffuse Surfaces)
//=============================================================================
extern "C" __global__ void __closesthit__photon_hit()
{
    const unsigned int prim_idx = optixGetPrimitiveIndex();

    // Skip photons that hit light sources
    if (prim_idx >= params.quadLightStartIndex)
    {
        unsigned int photon_idx = optixGetPayload_10();
        float3 hit_point = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();
        float3 throughput = make_float3(__uint_as_float(optixGetPayload_0()),
                                        __uint_as_float(optixGetPayload_1()),
                                        __uint_as_float(optixGetPayload_2()));
        
        recordTrajectoryEvent(photon_idx, EVENT_HIT_LIGHT, hit_point, 
                              optixGetWorldRayDirection(), throughput);
        optixSetPayload_11(0u);
        return;
    }

    const float3 hit_point = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();
    const float3 incident_dir = optixGetWorldRayDirection();

    // Compute geometric normal from triangle vertices
    float3 vertices[3];
    optixGetTriangleVertexData(
        optixGetGASTraversableHandle(),
        prim_idx,
        optixGetSbtGASIndex(),
        0.0f,
        vertices);
    float3 edge1 = vertices[1] - vertices[0];
    float3 edge2 = vertices[2] - vertices[0];
    float3 normal = normalize(cross(edge1, edge2));
    if (dot(normal, incident_dir) > 0.0f)
        normal = -normal;

    float3 throughput = make_float3(__uint_as_float(optixGetPayload_0()),
                                    __uint_as_float(optixGetPayload_1()),
                                    __uint_as_float(optixGetPayload_2()));
    unsigned int packedState = optixGetPayload_9();
    unsigned int depth = packedState & 0x3fffffffu;
    unsigned int insideFlag = (packedState >> 30) & 0x1u;
    unsigned int prevWasSpecular = (packedState >> 31) & 0x1u;
    unsigned int photon_idx = optixGetPayload_10();

    // Look up material for this triangle
    Material mat;
    if (params.triangle_materials)
        mat = params.triangle_materials[prim_idx];
    else
    {
        mat.type = MATERIAL_DIFFUSE;
        mat.albedo = make_float3(0.5f, 0.5f, 0.5f);
        mat.diffuseProb = 0.5f;
    }

    // RNG state seeded from photon index and depth
    unsigned int rngState = photon_idx * 747796405u + depth * 2891336453u;

    // Diffuse surface handling
    if (mat.type == MATERIAL_DIFFUSE)
    {
        float xi = ph_rand(rngState);

        // Store photon if it has bounced at least once (skip direct illumination)
        if (depth > 0)
        {
            // Record DIFFUSE_STORED event
            recordTrajectoryEvent(photon_idx, EVENT_DIFFUSE_STORED, hit_point,
                                  incident_dir, throughput, TRAJ_MAT_DIFFUSE);

            if (prevWasSpecular)
                store_caustic_photon(hit_point, incident_dir, throughput);
            else
                store_photon(hit_point, incident_dir, throughput);
        }

        // Russian Roulette
        if (xi >= mat.diffuseProb)
        {
            // Absorbed
            recordTrajectoryEvent(photon_idx, EVENT_DIFFUSE_ABSORBED, hit_point,
                                  incident_dir, throughput, TRAJ_MAT_DIFFUSE);
            optixSetPayload_11(0u);
            return;
        }

        // Survived - bounce continues
        throughput *= mat.albedo / mat.diffuseProb;

        // Sample new diffuse direction
        float u1 = ph_rand(rngState);
        float u2 = ph_rand(rngState);
        float phi = 2.0f * M_PI * u1;
        float cos_theta = sqrtf(u2);
        float sin_theta = sqrtf(1.0f - u2);
        float3 local_dir = make_float3(cosf(phi) * sin_theta, cos_theta, sinf(phi) * sin_theta);

        // Build orthonormal basis around surface normal
        float3 w = normal;
        float3 u = normalize(fabsf(w.x) > 0.1f ? make_float3(0.0f, 1.0f, 0.0f)
                                               : make_float3(1.0f, 0.0f, 0.0f));
        float3 v = normalize(cross(w, u));
        u = normalize(cross(v, w));
        float3 new_dir = normalize(u * local_dir.x + w * local_dir.y + v * local_dir.z);

        // Record DIFFUSE_BOUNCE event
        recordTrajectoryEvent(photon_idx, EVENT_DIFFUSE_BOUNCE, hit_point,
                              new_dir, throughput, TRAJ_MAT_DIFFUSE);

        const float eps = 1e-3f;
        float3 new_origin = hit_point + new_dir * eps;

        depth++;
        prevWasSpecular = 0u;

        // Write updated state back to payload
        optixSetPayload_0(__float_as_uint(throughput.x));
        optixSetPayload_1(__float_as_uint(throughput.y));
        optixSetPayload_2(__float_as_uint(throughput.z));
        optixSetPayload_3(__float_as_uint(new_origin.x));
        optixSetPayload_4(__float_as_uint(new_origin.y));
        optixSetPayload_5(__float_as_uint(new_origin.z));
        optixSetPayload_6(__float_as_uint(new_dir.x));
        optixSetPayload_7(__float_as_uint(new_dir.y));
        optixSetPayload_8(__float_as_uint(new_dir.z));
        unsigned int newPackedState = (prevWasSpecular << 31) | (insideFlag << 30) | (depth & 0x3fffffffu);
        optixSetPayload_9(newPackedState);
        optixSetPayload_10(photon_idx);
        optixSetPayload_11(1u);  // continue
        return;
    }

    // Non-diffuse triangle - terminate
    optixSetPayload_11(0u);
}

//=============================================================================
// Sphere Closest Hit (Specular/Transmissive Materials)
//=============================================================================
extern "C" __global__ void __closesthit__photon_sphere_hit()
{
    const unsigned int prim_idx = optixGetPrimitiveIndex();

    const float3 hit_point = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();
    const float3 incident_dir = optixGetWorldRayDirection();

    // Get normal from custom sphere intersection program
    float3 normal = make_float3(
        __uint_as_float(optixGetAttribute_0()),
        __uint_as_float(optixGetAttribute_1()),
        __uint_as_float(optixGetAttribute_2()));
    normal = normalize(normal);

    float3 throughput = make_float3(__uint_as_float(optixGetPayload_0()),
                                    __uint_as_float(optixGetPayload_1()),
                                    __uint_as_float(optixGetPayload_2()));
    unsigned int packedState = optixGetPayload_9();
    unsigned int depth = packedState & 0x3fffffffu;
    unsigned int insideFlag = (packedState >> 30) & 0x1u;
    unsigned int photon_idx = optixGetPayload_10();

    // Look up material for this sphere
    Material mat = params.sphere_materials[prim_idx];

    // RNG state for Fresnel stochastic selection
    unsigned int rngState = photon_idx * 747796405u + depth * 2891336453u + 12345u;

    depth++;
    if (depth >= params.max_depth)
    {
        recordTrajectoryEvent(photon_idx, EVENT_MAX_DEPTH, hit_point,
                              incident_dir, throughput, mat.type);
        optixSetPayload_11(0u);
        return;
    }

    float3 new_dir;
    unsigned int prevWasSpecular = 1u;
    int event_type = EVENT_NONE;
    int material_type = mat.type;

    if (mat.type == MATERIAL_TRANSMISSIVE)
    {
        // Glass with Fresnel-weighted stochastic selection
        const float n_air = 1.0f;
        const float n_glass = 1.5f;

        float3 N = normal;
        float n1 = n_air;
        float n2 = n_glass;

        if (insideFlag)
        {
            N = -normal;
            n1 = n_glass;
            n2 = n_air;
        }

        float cos_i = -dot(incident_dir, N);
        float eta = n1 / n2;
        float fresnel_r = schlickFresnelPhoton(cos_i, n1, n2);
        float xi = ph_rand(rngState);

        float3 refr;
        bool can_refract = refract_dir(incident_dir, N, eta, refr);

        if (!can_refract || xi < fresnel_r)
        {
            // Reflect (TIR or Fresnel-selected)
            new_dir = normalize(reflect_dir(incident_dir, N));
            event_type = EVENT_GLASS_REFLECT;
        }
        else
        {
            // Refract
            new_dir = normalize(refr);
            insideFlag = 1u - insideFlag;
            event_type = EVENT_GLASS_REFRACT;
        }
    }
    else if (mat.type == MATERIAL_SPECULAR)
    {
        // Perfect mirror reflection
        new_dir = normalize(reflect_dir(incident_dir, normal));
        event_type = EVENT_SPECULAR_REFLECT;
    }
    else
    {
        // Unknown material - absorb
        recordTrajectoryEvent(photon_idx, EVENT_DIFFUSE_ABSORBED, hit_point,
                              incident_dir, throughput, material_type);
        optixSetPayload_11(0u);
        return;
    }

    // Record the bounce event
    recordTrajectoryEvent(photon_idx, event_type, hit_point, new_dir, throughput, material_type);

    const float eps = 1e-3f;
    float3 new_origin = hit_point + new_dir * eps;

    optixSetPayload_0(__float_as_uint(throughput.x));
    optixSetPayload_1(__float_as_uint(throughput.y));
    optixSetPayload_2(__float_as_uint(throughput.z));
    optixSetPayload_3(__float_as_uint(new_origin.x));
    optixSetPayload_4(__float_as_uint(new_origin.y));
    optixSetPayload_5(__float_as_uint(new_origin.z));
    optixSetPayload_6(__float_as_uint(new_dir.x));
    optixSetPayload_7(__float_as_uint(new_dir.y));
    optixSetPayload_8(__float_as_uint(new_dir.z));
    unsigned int newPackedState = (prevWasSpecular << 31) | (insideFlag << 30) | (depth & 0x3fffffffu);
    optixSetPayload_9(newPackedState);
    optixSetPayload_10(photon_idx);
    optixSetPayload_11(1u);
}
