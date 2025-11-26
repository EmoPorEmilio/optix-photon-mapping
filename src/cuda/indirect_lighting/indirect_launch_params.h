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

    // Photon map for indirect illumination (linear array for fallback)
    Photon* photon_map;
    unsigned int photon_count;
    float gather_radius;  // Search radius for photon gathering
    float brightness_multiplier;  // Configurable visibility multiplier

    // kd-tree for O(log n) photon queries (Jensen's algorithm)
    PhotonKDTree kdtree;

    // Scene bounds (for normalization)
    unsigned int quadLightStartIndex;
};

