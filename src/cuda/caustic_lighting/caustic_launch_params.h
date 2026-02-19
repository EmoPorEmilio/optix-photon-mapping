#pragma once

#include <optix.h>
#include <sutil/vec_math.h>
#include "../../scene/Material.h"
#include "../../rendering/photon/Photon.h"
#include "../../rendering/photon/PhotonKDTreeDevice.h"

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

    // Materials - needed for BRDF (Jensen's Eq. 8: f_r term)
    Material* triangle_materials;
    Material sphere_materials[2];

    // Caustic photon map (linear array for fallback)
    Photon* caustic_photon_map;
    unsigned int caustic_photon_count;
    float gather_radius;
    float brightness_multiplier;  // Configurable visibility multiplier

    // kd-tree for O(log n) caustic photon queries
    PhotonKDTreeDevice caustic_kdtree;

    // For light source detection
    unsigned int quadLightStartIndex;
};

