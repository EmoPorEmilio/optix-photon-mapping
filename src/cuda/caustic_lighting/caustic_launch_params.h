#pragma once

#include <optix.h>
#include <sutil/vec_math.h>
#include "../../scene/Material.h"
#include "../../rendering/photon/Photon.h"

struct CausticLaunchParams
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
    Material sphere_materials[2];

    // Caustic photon map
    Photon* caustic_photon_map;
    unsigned int caustic_photon_count;
    float gather_radius;

    // For light source detection
    unsigned int quadLightStartIndex;
};

