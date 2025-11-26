#pragma once

#include <optix.h>
#include <sutil/vec_math.h>
#include "../../scene/Material.h"
#include "../../rendering/photon/Photon.h"

// Forward declaration of kd-tree node for host/device compatibility
struct PhotonKDNode;

// kd-tree structure (host/device compatible)
struct PhotonKDTree
{
    PhotonKDNode* nodes;
    unsigned int num_nodes;
    unsigned int max_depth;
    bool valid;
};

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

    // Caustic photon map (linear array for fallback)
    Photon* caustic_photon_map;
    unsigned int caustic_photon_count;
    float gather_radius;
    float brightness_multiplier;  // Configurable visibility multiplier

    // kd-tree for O(log n) caustic photon queries
    PhotonKDTree caustic_kdtree;

    // For light source detection
    unsigned int quadLightStartIndex;
};

