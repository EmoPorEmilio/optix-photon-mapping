

#pragma once

#include "Photon.h"
#include <vector>

// CPU-side photon kd-tree based on Jensen's photon mapping algorithm.
// The OptiX photon pass still writes into a flat array on the GPU; once
// photons are copied back to the host we build this kd-tree for queries.
class PhotonMapper
{
public:
    struct KDNode
    {
        Photon photon; // Stored photon (position + power + direction encoding).
        int axis;      // Split axis: 0 = x, 1 = y, 2 = z.
        int left;      // Index of left child in nodes vector, or -1.
        int right;     // Index of right child in nodes vector, or -1.
    };

private:
    std::vector<KDNode> nodes;
    int rootIndex = -1;
    unsigned int max_photons = 0;

public:
    // Construct an empty mapper with a maximum photon capacity hint.
    explicit PhotonMapper(unsigned int amount_of_photons);

    // Build a balanced kd-tree from an array of photons (median splits).
    // This replaces any previous tree contents. Only the first max_photons
    // photons are used.
    void buildFromArray(const Photon *photons, size_t count);

    // Clear the kd-tree.
    void clear();

    // Number of photons currently stored in the kd-tree.
    size_t size() const;

    // Maximum photons requested at construction.
    unsigned int getMaxPhotons() const;

    // Whether the kd-tree currently has no photons.
    bool empty() const;

    // Access to underlying nodes for debugging / visualization.
    const std::vector<KDNode> &getNodes() const { return nodes; }
    int getRootIndex() const { return rootIndex; }
};


