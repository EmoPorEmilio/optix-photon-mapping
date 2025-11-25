#pragma once

#include <sutil/vec_math.h>

// Macro for CUDA host/device compatibility
#ifdef __CUDACC__
#define CUDA_HOSTDEVICE __host__ __device__
#else
#define CUDA_HOSTDEVICE
#endif

// Stored photon for the photon map (Jensen's algorithm)
// Uses float3 for power and direction for simplicity and debugging
struct Photon
{
    float3 position;    // Hit position on surface
    float3 power;       // Photon power (RGB) - modulated by surface interactions
    float3 incidentDir; // Incident direction (normalized) - direction photon was traveling
    short flag;         // Flag used in kd-tree (0 = leaf, 1/2/3 = split axis)

    CUDA_HOSTDEVICE Photon()
        : position(make_float3(0.0f, 0.0f, 0.0f)),
          power(make_float3(0.0f, 0.0f, 0.0f)),
          incidentDir(make_float3(0.0f, -1.0f, 0.0f)),
          flag(0) {}

    CUDA_HOSTDEVICE Photon(const float3 &pos, const float3 &pow, const float3 &dir)
        : position(pos), power(pow), incidentDir(dir), flag(0) {}
};
