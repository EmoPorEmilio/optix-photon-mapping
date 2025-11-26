#pragma once

#include <optix.h>
#include <sutil/vec_math.h>
#include "../../scene/Material.h"

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

    __host__ __device__ DirectLaunchParams() {}
};

