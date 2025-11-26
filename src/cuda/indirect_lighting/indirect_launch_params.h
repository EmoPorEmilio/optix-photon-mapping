#pragma once

#include <optix.h>
#include <sutil/vec_math.h>
#include "../../scene/Material.h"
#include "../../rendering/photon/Photon.h"

struct IndirectLaunchParams
{
    // Output
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

    // Photon map for indirect illumination
    Photon* photon_map;
    unsigned int photon_count;
    float gather_radius;  // Search radius for photon gathering
    float brightness_multiplier;  // Configurable visibility multiplier

    // Scene bounds (for normalization)
    unsigned int quadLightStartIndex;
};

