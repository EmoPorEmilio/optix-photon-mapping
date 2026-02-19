#include <optix.h>
#include <sutil/vec_math.h>
#include "photon_launch_params.h"
#include "photon_rng.h"

#ifndef PHOTON_PARAMS_DEFINED
extern "C" __constant__ PhotonLaunchParams params;
#endif

// Forward declaration - defined in combined file (has default for material_type)
__device__ __forceinline__ void recordTrajectoryEvent(
    unsigned int photon_idx, int event_type, const float3 &position,
    const float3 &direction, const float3 &power, int material_type);

//=============================================================================
// Volume Scattering Helpers (Jensen's PDF §1.4, §3.3)
//=============================================================================

// Sample isotropic direction (uniform sphere)
static __forceinline__ __device__ float3 sampleIsotropicDirection(unsigned int &rngState)
{
    float u1 = ph_rand(rngState);
    float u2 = ph_rand(rngState);
    float cos_theta = 1.0f - 2.0f * u1;
    float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
    float phi = 2.0f * M_PIf * u2;
    return make_float3(sin_theta * cosf(phi), cos_theta, sin_theta * sinf(phi));
}

// Store volume photon at scattering event in participating media
static __forceinline__ __device__ void storeVolumePhoton(
    const float3 &position, const float3 &incident_dir, const float3 &power)
{
    if (!params.volume_photons_out)
        return;
    unsigned int idx = atomicAdd((unsigned int *)params.volume_photon_counter, 1u);
    if (idx < params.num_photons)
    {
        params.volume_photons_out[idx].position = position;
        params.volume_photons_out[idx].power = power;
        params.volume_photons_out[idx].direction = normalize(incident_dir);
    }
}

//=============================================================================
// Photon Emission Raygen Program
//=============================================================================
extern "C" __global__ void __raygen__photon_emitter()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    const unsigned int photon_idx = idx.x + idx.y * dim.x + idx.z * dim.x * dim.y;

    if (photon_idx >= params.num_photons)
        return;

    // Initialize trajectory for this photon (if recording)
    if (params.record_trajectories && params.trajectories_out)
    {
        params.trajectories_out[photon_idx].photon_id = photon_idx;
        params.trajectories_out[photon_idx].event_count = 0;
    }

    const QuadLight light = params.light;

    // Initial emission from the light
    unsigned int rngState = photon_idx * 747796405u + 2891336453u;
    float u0 = ph_rand(rngState);
    float u1 = ph_rand(rngState);
    float u2 = ph_rand(rngState);
    float u3 = ph_rand(rngState);

    float3 origin, direction;
    light.samplePhotonEmission(u0, u1, u2, u3, origin, direction);

    // Path throughput starts as per-photon power
    float3 throughput = params.photon_power;

    // Record EMITTED event
    recordTrajectoryEvent(photon_idx, EVENT_EMITTED, origin, direction, throughput);

    unsigned int p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11;

    // Bounce loop driven from raygen
    // Payload 9 bit layout: bit 31 = prevWasSpecular, bit 30 = insideFlag, bits 0-29 = depth
    unsigned int depth = 0u;
    unsigned int insideFlag = 0u;
    unsigned int prevWasSpecular = 0u;

    // Local RNG for volume scattering (separate from closest-hit RNG)
    unsigned int volumeRngState = photon_idx * 1234567u + 9876543u;

    for (;;)
    {
        //=====================================================================
        // Volume Scattering Check (Jensen's PDF §1.4)
        // Sample free path BEFORE surface trace, then compare with hit distance
        //=====================================================================
        float volume_scatter_dist = 1e16f;  // No scatter by default
        bool will_scatter_in_volume = false;

        if (params.enable_volume_scattering && params.volume.sigma_t > 0.0f)
        {
            // Check if ray origin is in volume region
            if (params.volume.contains(origin))
            {
                // Sample free path distance: t = -ln(ξ) / σ_t
                float xi_free = ph_rand(volumeRngState);
                volume_scatter_dist = -logf(fmaxf(xi_free, 1e-8f)) / params.volume.sigma_t;
                will_scatter_in_volume = true;
            }
        }

        // Save state before trace (needed for volume scatter override)
        float3 origin_before = origin;
        float3 direction_before = direction;
        float3 throughput_before = throughput;

        // Pack state into payload 9
        unsigned int packedState = (prevWasSpecular << 31) | (insideFlag << 30) | (depth & 0x3fffffffu);

        // Pack payload for this segment
        p0  = __float_as_uint(throughput.x);
        p1  = __float_as_uint(throughput.y);
        p2  = __float_as_uint(throughput.z);
        p3  = __float_as_uint(origin.x);
        p4  = __float_as_uint(origin.y);
        p5  = __float_as_uint(origin.z);
        p6  = __float_as_uint(direction.x);
        p7  = __float_as_uint(direction.y);
        p8  = __float_as_uint(direction.z);
        p9  = packedState;
        p10 = photon_idx;  // for RNG and trajectory access in closest hit
        p11 = 1u;          // continue flag (1 = continue, 0 = terminate)

        unsigned int flags = OPTIX_RAY_FLAG_DISABLE_ANYHIT;
        optixTrace(params.handle,
                   origin,
                   direction,
                   0.0f,
                   1e16f,
                   0.0f,
                   OptixVisibilityMask(255),
                   flags,
                   0,
                   1,
                   0,
                   p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11);

        // Unpack updated state from payload
        throughput.x = __uint_as_float(p0);
        throughput.y = __uint_as_float(p1);
        throughput.z = __uint_as_float(p2);
        origin.x     = __uint_as_float(p3);
        origin.y     = __uint_as_float(p4);
        origin.z     = __uint_as_float(p5);
        direction.x  = __uint_as_float(p6);
        direction.y  = __uint_as_float(p7);
        direction.z  = __uint_as_float(p8);

        // Unpack state from payload 9
        packedState = p9;
        depth = packedState & 0x3fffffffu;
        insideFlag = (packedState >> 30) & 0x1u;
        prevWasSpecular = (packedState >> 31) & 0x1u;

        unsigned int cont = p11;

        //=====================================================================
        // Volume Scattering Override (Jensen's PDF §1.4, §3.3)
        // If scatter distance < surface hit distance, scatter in volume
        //=====================================================================
        if (will_scatter_in_volume)
        {
            // Compute surface hit distance from origin change
            float3 segment_vec = origin - origin_before;
            float surface_hit_dist = length(segment_vec);

            // If volume scatter happens BEFORE surface hit
            if (volume_scatter_dist < surface_hit_dist)
            {
                // Compute scatter position
                float3 scatter_pos = origin_before + volume_scatter_dist * direction_before;

                // Verify scatter point is still in volume
                if (params.volume.contains(scatter_pos))
                {
                    // Russian Roulette for scatter vs absorb (albedo = σ_s / σ_t)
                    float albedo = params.volume.albedo();
                    float xi_rr = ph_rand(volumeRngState);

                    if (xi_rr < albedo)
                    {
                        // SCATTER: Store volume photon and continue
                        storeVolumePhoton(scatter_pos, direction_before, throughput_before);

                        // Record trajectory event
                        recordTrajectoryEvent(photon_idx, EVENT_VOLUME_SCATTER, scatter_pos,
                                              direction_before, throughput_before, TRAJ_MAT_VOLUME);

                        // Sample new isotropic direction
                        float3 new_dir = sampleIsotropicDirection(volumeRngState);

                        // Update state for next bounce (override closest hit result)
                        origin = scatter_pos + new_dir * 1e-3f;  // Small offset
                        direction = new_dir;
                        throughput = throughput_before;  // No throughput change for volume scatter

                        // Continue bouncing (don't count as depth, it's volume scatter)
                        cont = 1u;
                    }
                    else
                    {
                        // ABSORB in volume
                        recordTrajectoryEvent(photon_idx, EVENT_VOLUME_ABSORBED, scatter_pos,
                                              direction_before, throughput_before, TRAJ_MAT_VOLUME);
                        cont = 0u;
                    }
                }
            }
        }

        if (!cont)
            break;

        if (depth >= params.max_depth)
        {
            // Record MAX_DEPTH termination event
            recordTrajectoryEvent(photon_idx, EVENT_MAX_DEPTH, origin, direction, throughput);
            break;
        }
    }
}
