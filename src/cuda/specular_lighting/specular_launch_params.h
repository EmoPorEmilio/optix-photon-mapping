#pragma once

#include <optix.h>
#include <sutil/vec_math.h>
#include "../../scene/Material.h"
#include "../../rendering/photon/Photon.h"

struct SpecularLaunchParams
{
    // Output
    float4* frame_buffer;
    unsigned int width;
    unsigned int height;

    // Camera
    float3 eye;
    float3 U, V, W;

    // Scene
    OptixTraversableHandle handle;

    // Materials
    Material* triangle_materials;
    Material sphere_materials[2];

    // Photon maps for full lighting
    Photon* global_photon_map;
    unsigned int global_photon_count;
    Photon* caustic_photon_map;
    unsigned int caustic_photon_count;
    float gather_radius;

    // Light
    float3 light_position;
    float3 light_intensity;

    // For detecting what we hit
    unsigned int quadLightStartIndex;
};

