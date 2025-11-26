
#include <optix.h>
#include <sutil/vec_math.h>
#include "photon_launch_params.h"
#include "photon_rng.h"

extern "C" __constant__ PhotonLaunchParams params;

// Simple reflection helper for specular materials.
static __forceinline__ __device__ float3 reflect_dir(const float3 &I, const float3 &N)
{
    return I - 2.0f * dot(I, N) * N;
}

// Simple refraction helper for transmissive materials (Snell's law).
// Returns false if there is total internal reflection.
static __forceinline__ __device__ bool refract_dir(const float3 &I, const float3 &N, float eta, float3 &refracted)
{
    float cosi = -dot(N, I);
    float sint2 = max(0.0f, 1.0f - cosi * cosi);
    float cost2 = 1.0f - eta * eta * sint2;
    if (cost2 <= 0.0f)
        return false; // total internal reflection
    float cost = sqrtf(cost2);
    refracted = eta * I + (eta * cosi - cost) * N;
    return true;
}

// Helper to store photon in global photon map
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

// Helper to store caustic photon (previous hit was specular/transmissive)
__device__ void store_caustic_photon(const float3 &hit_point, const float3 &incident_dir, const float3 &throughput)
{
    unsigned int stored_idx = atomicAdd((unsigned int *)params.caustic_photon_counter, 1u);

    if (stored_idx < params.num_photons)
    {
        Photon photon;
        photon.position = hit_point;
        photon.power = throughput;
        photon.incidentDir = normalize(incident_dir);
        photon.flag = 1; // Mark as caustic

        params.caustic_photons_out[stored_idx] = photon;
    }
}

// Payload bit layout for tracking:
// Payload 9: bits 0-29 = depth, bit 30 = insideFlag, bit 31 = prevWasSpecular (S or T hit)

extern "C" __global__ void __closesthit__photon_hit()
{
    const unsigned int prim_idx = optixGetPrimitiveIndex();

    // Skip photons that hit light sources (if they are part of geometry)
    if (prim_idx >= params.quadLightStartIndex)
    {
        optixSetPayload_11(0u);
        return;
    }

    const float3 hit_point = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();
    const float3 incident_dir = optixGetWorldRayDirection();
    // Approximate surface normal by flipping the incident direction.
    const float3 normal = normalize(-incident_dir);

    float3 throughput = make_float3(__uint_as_float(optixGetPayload_0()),
                                    __uint_as_float(optixGetPayload_1()),
                                    __uint_as_float(optixGetPayload_2()));
    unsigned int packedState = optixGetPayload_9();
    unsigned int depth = packedState & 0x3fffffffu;           // bits 0-29
    unsigned int insideFlag = (packedState >> 30) & 0x1u;      // bit 30
    unsigned int prevWasSpecular = (packedState >> 31) & 0x1u; // bit 31: was previous hit S or T?
    unsigned int photon_idx = optixGetPayload_10();

    // Look up material for this triangle.
    Material mat;
    if (params.triangle_materials)
        mat = params.triangle_materials[prim_idx];
    else
    {
        mat.type = MATERIAL_DIFFUSE;
        mat.albedo = make_float3(0.5f, 0.5f, 0.5f);
        mat.diffuseProb = 0.5f;
    }

    // RNG state seeded from photon index and depth.
    unsigned int rngState = photon_idx * 747796405u + depth * 2891336453u;

    // For diffuse surfaces, apply Russian roulette and potentially store photon
    if (mat.type == MATERIAL_DIFFUSE)
    {
        float xi = ph_rand(rngState);

        // Store photon BEFORE Russian roulette decision if it has bounced at least once
        // This captures the incoming flux (Jensen's algorithm - skip direct illumination)
        if (depth > 0)
        {
            // If previous hit was specular/transmissive, this is a CAUSTIC photon
            if (prevWasSpecular)
            {
                store_caustic_photon(hit_point, incident_dir, throughput);
            }
            else
            {
                store_photon(hit_point, incident_dir, throughput);
            }
        }

        if (xi >= mat.diffuseProb)
        {
            optixSetPayload_11(0u); // absorbed
            return;
        }

        // Update throughput with diffuse albedo for the NEXT bounce.
        throughput *= mat.albedo;

        // Sample new diffuse direction over hemisphere around the normal.
        float u1 = ph_rand(rngState);
        float u2 = ph_rand(rngState);

        float phi = 2.0f * M_PI * u1;
        float cos_theta = sqrtf(u2);
        float sin_theta = sqrtf(1.0f - u2);
        float3 local_dir = make_float3(cosf(phi) * sin_theta, cos_theta, sinf(phi) * sin_theta);

        // Build orthonormal basis around the surface normal.
        float3 w = normal;
        float3 u = normalize(fabsf(w.x) > 0.1f ? make_float3(0.0f, 1.0f, 0.0f)
                                               : make_float3(1.0f, 0.0f, 0.0f));
        float3 v = normalize(cross(w, u));
        u = normalize(cross(v, w));

        float3 new_dir = normalize(u * local_dir.x + w * local_dir.y + v * local_dir.z);

        // Offset origin to avoid self-intersection.
        const float eps = 1e-3f;
        float3 new_origin = hit_point + new_dir * eps;

        // Increment depth, clear prevWasSpecular (we just hit diffuse)
        depth++;
        prevWasSpecular = 0u;

        // Write updated state back to payload.
        optixSetPayload_0(__float_as_uint(throughput.x));
        optixSetPayload_1(__float_as_uint(throughput.y));
        optixSetPayload_2(__float_as_uint(throughput.z));
        optixSetPayload_3(__float_as_uint(new_origin.x));
        optixSetPayload_4(__float_as_uint(new_origin.y));
        optixSetPayload_5(__float_as_uint(new_origin.z));
        optixSetPayload_6(__float_as_uint(new_dir.x));
        optixSetPayload_7(__float_as_uint(new_dir.y));
        optixSetPayload_8(__float_as_uint(new_dir.z));
        // Pack state back: depth | insideFlag | prevWasSpecular
        unsigned int newPackedState = (prevWasSpecular << 31) | (insideFlag << 30) | (depth & 0x3fffffffu);
        optixSetPayload_9(newPackedState);
        optixSetPayload_10(photon_idx);
        optixSetPayload_11(1u); // continue
        return;
    }

    // Fallback: if we somehow hit a non-diffuse triangle, just terminate.
    optixSetPayload_11(0u);
}

extern "C" __global__ void __closesthit__photon_sphere_hit()
{
    const unsigned int prim_idx = optixGetPrimitiveIndex();

    const float3 hit_point = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();
    const float3 incident_dir = optixGetWorldRayDirection();

    // Use the true geometric normal from the custom sphere intersection program.
    float3 normal = make_float3(
        __uint_as_float(optixGetAttribute_0()),
        __uint_as_float(optixGetAttribute_1()),
        __uint_as_float(optixGetAttribute_2()));
    normal = normalize(normal);

    float3 throughput = make_float3(__uint_as_float(optixGetPayload_0()),
                                    __uint_as_float(optixGetPayload_1()),
                                    __uint_as_float(optixGetPayload_2()));

    unsigned int packedState = optixGetPayload_9();
    unsigned int depth = packedState & 0x3fffffffu;           // bits 0-29
    unsigned int insideFlag = (packedState >> 30) & 0x1u;      // bit 30
    // prevWasSpecular not used here - we're hitting a sphere
    unsigned int photon_idx = optixGetPayload_10();

    // Look up material for this sphere.
    Material mat = params.sphere_materials[prim_idx];

    // Spheres are specular/transmissive - NO storage on them (photons pass through or reflect)

    // Prepare next bounce.
    depth++;
    if (depth >= params.max_depth)
    {
        optixSetPayload_11(0u);
        return;
    }

    float3 new_dir;
    unsigned int prevWasSpecular = 1u; // This hit IS specular/transmissive

    if (mat.type == MATERIAL_TRANSMISSIVE)
    {
        // Refractive behavior with simple inside/outside tracking using insideFlag.
        const float n_air = 1.0f;
        const float n_glass = 1.5f;

        float3 N = normal;
        float n1 = n_air;
        float n2 = n_glass;

        // If we are already inside the sphere, swap indices and flip normal.
        if (insideFlag)
        {
            N = -normal;
            n1 = n_glass;
            n2 = n_air;
        }

        float eta = n1 / n2;
        float3 refr;
        if (!refract_dir(incident_dir, N, eta, refr))
        {
            // Total internal reflection fallback.
            refr = reflect_dir(incident_dir, N);
        }
        else
        {
            // Toggle medium only when we successfully refract.
            insideFlag = 1u - insideFlag;
        }

        // NO color modulation for transmissive - light passes through unchanged
        new_dir = normalize(refr);
    }
    else if (mat.type == MATERIAL_SPECULAR)
    {
        // Perfect specular reflection: NO color modulation
        new_dir = normalize(reflect_dir(incident_dir, normal));
        // Reflection does not change medium, so insideFlag stays the same.
    }
    else
    {
        // Default/Other materials absorb for now.
        optixSetPayload_11(0u);
        return;
    }

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
    // Pack state back: prevWasSpecular | insideFlag | depth
    unsigned int newPackedState = (prevWasSpecular << 31) | (insideFlag << 30) | (depth & 0x3fffffffu);
    optixSetPayload_9(newPackedState);
    optixSetPayload_10(photon_idx);
    optixSetPayload_11(1u);
}
