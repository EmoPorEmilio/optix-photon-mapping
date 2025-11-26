#pragma once

#include "Photon.h"

// GPU-side kd-tree node used for photon gathering
struct PhotonKDNodeDevice
{
    Photon photon;   // Stored photon
    int left;        // Index of left child (-1 if none)
    int right;       // Index of right child (-1 if none)
    int axis;        // Split axis: 0=x, 1=y, 2=z
    float split;     // Split plane value along axis
};

// GPU kd-tree descriptor passed through OptiX launch params
struct PhotonKDTreeDevice
{
    PhotonKDNodeDevice *nodes = nullptr;
    unsigned int node_count = 0;
    unsigned int num_nodes = 0; // legacy alias
    unsigned int max_depth = 0; // legacy alias (unused, kept for compatibility)
    bool valid = false;
};


