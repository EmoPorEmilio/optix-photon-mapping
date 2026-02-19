#pragma once

#include <sutil/vec_math.h>

// Macro for CUDA host/device compatibility
#ifdef __CUDACC__
#define CUDA_HOSTDEVICE __host__ __device__
#else
#define CUDA_HOSTDEVICE
#endif

// Volume photon for participating media (Jensen's algorithm §1.4, §3.3)
// Stored at scattering events inside the volume, not on surfaces
struct VolumePhoton
{
    float3 position;    // Position in volume (3D space, not on surface)
    float3 power;       // Photon power (RGB) at scatter point
    float3 direction;   // Incident direction (for phase function evaluation)

    CUDA_HOSTDEVICE VolumePhoton()
        : position(make_float3(0.0f, 0.0f, 0.0f)),
          power(make_float3(0.0f, 0.0f, 0.0f)),
          direction(make_float3(0.0f, -1.0f, 0.0f)) {}

    CUDA_HOSTDEVICE VolumePhoton(const float3& pos, const float3& pow, const float3& dir)
        : position(pos), power(pow), direction(dir) {}
};

// Volume properties for participating media
struct VolumeProperties
{
    float sigma_t;          // Extinction coefficient (absorption + scattering)
    float sigma_s;          // Scattering coefficient (must be <= sigma_t)
    float3 bounds_min;      // Volume region minimum
    float3 bounds_max;      // Volume region maximum
    float falloff_distance; // Distance over which density fades to zero at boundary

    CUDA_HOSTDEVICE VolumeProperties()
        : sigma_t(0.01f),
          sigma_s(0.008f),
          bounds_min(make_float3(0.0f, 0.0f, 0.0f)),
          bounds_max(make_float3(556.0f, 548.0f, 559.0f)),  // Full Cornell box - density handles height falloff
          falloff_distance(80.0f)  // Not used for height falloff (uses exponential instead)
    {}

    // Single-scattering albedo: probability of scattering vs absorption
    CUDA_HOSTDEVICE float albedo() const
    {
        return sigma_t > 0.0f ? sigma_s / sigma_t : 0.0f;
    }

    // Check if point is inside volume (with soft boundary)
    CUDA_HOSTDEVICE bool contains(const float3& p) const
    {
        return p.x >= bounds_min.x && p.x <= bounds_max.x &&
               p.y >= bounds_min.y && p.y <= bounds_max.y &&
               p.z >= bounds_min.z && p.z <= bounds_max.z;
    }

    // Smoothstep function for smooth interpolation
    CUDA_HOSTDEVICE float smoothstep(float edge0, float edge1, float x) const
    {
        float t = fminf(fmaxf((x - edge0) / (edge1 - edge0), 0.0f), 1.0f);
        return t * t * (3.0f - 2.0f * t);
    }

    // Get fog density at a point with smooth HEIGHT-ONLY falloff
    // Ground fog: full density at floor, fades to zero at fog ceiling
    // No X/Z boundaries - fog fills horizontally within the Cornell box
    CUDA_HOSTDEVICE float densityAt(const float3& p) const
    {
        // Only check if inside the Cornell box horizontally (no density falloff on X/Z)
        if (p.x < bounds_min.x || p.x > bounds_max.x ||
            p.z < bounds_min.z || p.z > bounds_max.z)
            return 0.0f;

        // Below floor - no fog
        if (p.y < bounds_min.y)
            return 0.0f;

        // Above fog ceiling - no fog
        if (p.y > bounds_max.y)
            return 0.0f;

        // Height-based density: exponential falloff from floor to ceiling
        // Density is maximum at floor, smoothly decreases with height
        float height_ratio = (p.y - bounds_min.y) / (bounds_max.y - bounds_min.y);

        // Exponential falloff: denser near floor, thins out with height
        // exp(-k * h) where h is normalized height [0,1]
        float k = 3.0f;  // Controls how fast fog thins with height
        float density = expf(-k * height_ratio);

        return density;
    }
};
