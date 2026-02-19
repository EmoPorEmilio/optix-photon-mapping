#pragma once

#include <optix.h>
#include <sutil/vec_math.h>
#include "../../scene/Material.h"
#include "../../rendering/photon/VolumePhoton.h"

// Light data for direct illumination
struct DirectLightData
{
    float3 position;    // Center of area light
    float3 normal;      // Light facing direction
    float3 u;           // Light local U axis
    float3 v;           // Light local V axis (computed as cross(normal, u))
    float halfWidth;    // Half width of light
    float halfHeight;   // Half height of light
    float3 intensity;   // Light emission (RGB)
};

struct DirectLaunchParams
{
    // Output buffer
    float4* frame_buffer;
    unsigned int width;
    unsigned int height;

    // Camera
    float3 eye;
    float3 U, V, W;  // Camera basis vectors

    // Scene
    OptixTraversableHandle handle;

    // Materials
    Material* triangle_materials;
    Material sphere_materials[2];

    // Light (single area light for now)
    DirectLightData light;
    unsigned int quadLightStartIndex;  // Triangle index where light geometry starts

    // Sphere geometry for shadow rays
    float3 sphere1_center;
    float sphere1_radius;
    float3 sphere2_center;
    float sphere2_radius;

    // Configurable lighting parameters
    float ambient;              // Base ambient light
    float shadow_ambient;       // Ambient in shadowed areas
    float intensity_multiplier; // Direct lighting intensity
    float attenuation_factor;   // Light falloff factor

    //=========================================================================
    // Volume/Fog Parameters (Jensen's PDF §1.4, §3.3)
    //=========================================================================
    bool enable_fog;                    // Enable fog rendering
    VolumeProperties volume;            // Fog volume properties
    VolumePhoton* volume_photons;       // Volume photon map (optional, for proper scattering)
    unsigned int volume_photon_count;   // Number of volume photons
    float fog_gather_radius;            // Radius for volume photon gathering
    float3 fog_color;                   // Base fog color (for simple atmospheric fog)

    __host__ __device__ DirectLaunchParams()
        : enable_fog(false), volume_photons(nullptr), volume_photon_count(0),
          fog_gather_radius(20.0f), fog_color(make_float3(0.8f, 0.85f, 0.9f)) {}
};

