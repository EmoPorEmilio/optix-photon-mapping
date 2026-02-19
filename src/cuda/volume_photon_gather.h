#pragma once

//=============================================================================
// Volume Photon Gathering (Jensen's PDF §3.3, Eq. 14)
//
// Volume radiance estimate for participating media (fog):
//   L_i(x,ω) ≈ (1/σ_t) Σ f(x,ω'_p,ω) * ΔΦ_p / (4/3 πr³)
//
// For isotropic phase function f(θ) = 1/(4π), this simplifies to:
//   L_i(x,ω) ≈ Σ ΔΦ_p / (σ_t * (4/3 πr³) * 4π)
//            = Σ ΔΦ_p / (σ_t * (16/3) π² r³)
//=============================================================================

#include <sutil/vec_math.h>
#include "../rendering/photon/VolumePhoton.h"

#define VOL_PI 3.14159265358979323846f
#define VOL_INV_4PI 0.07957747154594767f    // 1/(4π)

//=============================================================================
// Linear O(n) Volume Photon Gathering (no KD-tree)
//=============================================================================
static __forceinline__ __device__ float3 gatherVolumePhotonsLinear(
    const float3 &position,
    const VolumePhoton *volume_photons,
    unsigned int photon_count,
    float gather_radius,
    float sigma_t)
{
    if (photon_count == 0 || volume_photons == nullptr || sigma_t <= 0.0f)
        return make_float3(0.0f);

    float3 flux_sum = make_float3(0.0f);
    const float radius_sq = gather_radius * gather_radius;
    unsigned int found = 0;

    for (unsigned int i = 0; i < photon_count; i++)
    {
        const VolumePhoton &vp = volume_photons[i];

        float3 diff = position - vp.position;
        float dist_sq = dot(diff, diff);

        if (dist_sq < radius_sq)
        {
            // Cone filter weight (same as surface, Jensen §3.2.1)
            float weight = 1.0f - sqrtf(dist_sq) / gather_radius;
            flux_sum += vp.power * weight;
            found++;
        }
    }

    if (found == 0)
        return make_float3(0.0f);

    // Jensen's Eq. 14: divide by σ_t and spherical volume (4/3)πr³
    // With isotropic phase function: f = 1/(4π)
    float r3 = gather_radius * gather_radius * gather_radius;
    float volume = (4.0f / 3.0f) * VOL_PI * r3;

    // Cone filter normalization: k / (1 - 2/(3k)) where k = 3 → factor = 3
    const float cone_k = 3.0f;
    float cone_norm = cone_k;

    // L_i = (1/σ_t) * (1/volume) * (1/4π) * Σ(power * weight) * cone_norm
    float3 radiance = flux_sum * cone_norm * VOL_INV_4PI / (sigma_t * volume);

    return radiance;
}

//=============================================================================
// Volume Radiance with Ray Marching
// Accumulates volume radiance along a ray segment through participating media
//
// Jensen's approach: integrate in-scattered radiance along the ray
// For discrete samples: Σ L_i(x_t) * transmittance(t) * dt
//=============================================================================
static __forceinline__ __device__ float3 gatherVolumeRadianceAlongRay(
    const float3 &ray_origin,
    const float3 &ray_dir,
    float t_near,
    float t_far,
    const VolumePhoton *volume_photons,
    unsigned int photon_count,
    const VolumeProperties &volume,
    float gather_radius,
    float step_size)
{
    if (t_far <= t_near || photon_count == 0)
        return make_float3(0.0f);

    float3 accumulated = make_float3(0.0f);
    float t = t_near;

    // Ray march through volume region
    while (t < t_far)
    {
        float3 sample_pos = ray_origin + t * ray_dir;

        // Only gather if inside volume bounds
        if (volume.contains(sample_pos))
        {
            // Gather volume photons at this sample point
            float3 Li = gatherVolumePhotonsLinear(
                sample_pos, volume_photons, photon_count,
                gather_radius, volume.sigma_t);

            // Transmittance from ray origin to this point: exp(-σ_t * t)
            float transmittance = expf(-volume.sigma_t * t);

            // Accumulate with transmittance weighting
            accumulated += Li * transmittance * step_size;
        }

        t += step_size;
    }

    return accumulated;
}

//=============================================================================
// Compute ray intersection with volume bounds (AABB)
// Returns t_near and t_far for the ray segment inside the volume
//=============================================================================
static __forceinline__ __device__ bool intersectVolumeBounds(
    const float3 &ray_origin,
    const float3 &ray_dir,
    const VolumeProperties &volume,
    float t_min,
    float t_max,
    float &t_near,
    float &t_far)
{
    float3 invDir = make_float3(
        ray_dir.x != 0.0f ? 1.0f / ray_dir.x : 1e16f,
        ray_dir.y != 0.0f ? 1.0f / ray_dir.y : 1e16f,
        ray_dir.z != 0.0f ? 1.0f / ray_dir.z : 1e16f);

    float3 t0 = (volume.bounds_min - ray_origin) * invDir;
    float3 t1 = (volume.bounds_max - ray_origin) * invDir;

    // Handle negative direction
    float3 tmin = make_float3(
        fminf(t0.x, t1.x),
        fminf(t0.y, t1.y),
        fminf(t0.z, t1.z));
    float3 tmax = make_float3(
        fmaxf(t0.x, t1.x),
        fmaxf(t0.y, t1.y),
        fmaxf(t0.z, t1.z));

    t_near = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.z, t_min));
    t_far = fminf(fminf(tmax.x, tmax.y), fminf(tmax.z, t_max));

    return t_near < t_far && t_far > 0.0f;
}

//=============================================================================
// Compute transmittance through volume (Beer-Lambert law)
//=============================================================================
static __forceinline__ __device__ float3 computeVolumeTransmittance(
    const float3 &ray_origin,
    const float3 &ray_dir,
    float t_surface,
    const VolumeProperties &volume)
{
    float t_near, t_far;
    if (!intersectVolumeBounds(ray_origin, ray_dir, volume, 0.0f, t_surface, t_near, t_far))
        return make_float3(1.0f);  // No intersection, full transmittance

    // Distance traveled through volume
    float dist_in_volume = t_far - t_near;
    if (dist_in_volume <= 0.0f)
        return make_float3(1.0f);

    // Beer-Lambert: T = exp(-σ_t * d)
    float T = expf(-volume.sigma_t * dist_in_volume);
    return make_float3(T, T, T);
}
