#pragma once

#include <optix.h>
#include <sutil/vec_math.h>
#include "../../scene/Material.h"
#include "../../rendering/photon/Photon.h"
#include "../../rendering/photon/PhotonKDTreeDevice.h"

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

    // Photon maps for full lighting (linear arrays for fallback)
    Photon* global_photon_map;
    unsigned int global_photon_count;
    Photon* caustic_photon_map;
    unsigned int caustic_photon_count;
    float gather_radius;

    // kd-trees for O(log n) photon queries
    PhotonKDTreeDevice global_kdtree;
    PhotonKDTreeDevice caustic_kdtree;

    // Light
    float3 light_position;
    float3 light_intensity;

    // For detecting what we hit
    unsigned int quadLightStartIndex;

    // Configurable specular parameters
    unsigned int max_recursion_depth;
    float glass_ior;
    float3 glass_tint;
    float mirror_reflectivity;
    float fresnel_min;
    float specular_ambient;
    float indirect_brightness;
    float caustic_brightness;
};

