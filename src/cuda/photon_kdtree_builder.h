#pragma once

#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "photon_kdtree.h"

// Host-side kd-tree builder for photon mapping
// Builds a balanced kd-tree on CPU, then uploads to GPU for queries

class PhotonKDTreeBuilder
{
private:
    std::vector<PhotonKDNode> h_nodes;
    PhotonKDNode* d_nodes = nullptr;
    unsigned int num_nodes = 0;
    unsigned int allocated_nodes = 0;
    
    // Helper to get photon coordinate by axis
    static float getAxisValue(const Photon& p, int axis)
    {
        switch (axis)
        {
            case 0: return p.position.x;
            case 1: return p.position.y;
            default: return p.position.z;
        }
    }
    
    // Recursive build function - creates a balanced kd-tree
    void buildRecursive(std::vector<Photon>& photons, 
                        unsigned int start, unsigned int end,
                        unsigned int node_idx, int depth)
    {
        // Ensure we have enough space in the node array
        if (node_idx >= h_nodes.size())
            return;
        
        if (start >= end)
        {
            // Empty node
            h_nodes[node_idx].split_axis = -2;  // Mark as invalid/empty
            return;
        }
        
        if (end - start == 1)
        {
            // Leaf node with single photon
            h_nodes[node_idx].photon = photons[start];
            h_nodes[node_idx].split_axis = -1;  // Leaf marker
            h_nodes[node_idx].split_value = 0.0f;
            return;
        }
        
        // Choose split axis (cycle through x, y, z)
        int axis = depth % 3;
        
        // Find median using nth_element (O(n) average)
        unsigned int mid = (start + end) / 2;
        std::nth_element(photons.begin() + start, 
                         photons.begin() + mid,
                         photons.begin() + end,
                         [axis](const Photon& a, const Photon& b) {
                             return getAxisValue(a, axis) < getAxisValue(b, axis);
                         });
        
        // Store median photon at this node
        h_nodes[node_idx].photon = photons[mid];
        h_nodes[node_idx].split_axis = axis;
        h_nodes[node_idx].split_value = getAxisValue(photons[mid], axis);
        
        // Recursively build children
        unsigned int left_child = 2 * node_idx + 1;
        unsigned int right_child = 2 * node_idx + 2;
        
        buildRecursive(photons, start, mid, left_child, depth + 1);
        buildRecursive(photons, mid + 1, end, right_child, depth + 1);
    }
    
public:
    PhotonKDTreeBuilder() = default;
    
    ~PhotonKDTreeBuilder()
    {
        freeGPU();
    }
    
    // Build kd-tree from photon array
    // photons: vector of photons (will be modified for median finding)
    void build(std::vector<Photon>& photons)
    {
        num_nodes = photons.size();
        if (num_nodes == 0)
            return;
        
        // Calculate size needed for complete binary tree representation
        // We need space for a complete binary tree that can hold all photons
        unsigned int tree_depth = 0;
        unsigned int temp = num_nodes;
        while (temp > 0)
        {
            tree_depth++;
            temp >>= 1;
        }
        
        // Allocate for complete binary tree (2^depth - 1 nodes max)
        unsigned int max_nodes = (1u << tree_depth) - 1;
        // But we might need one more level
        max_nodes = (1u << (tree_depth + 1)) - 1;
        
        h_nodes.resize(max_nodes);
        
        // Initialize all nodes as invalid
        for (auto& node : h_nodes)
        {
            node.split_axis = -2;  // Invalid/empty marker
        }
        
        // Build the tree
        buildRecursive(photons, 0, num_nodes, 0, 0);
        
        // Update num_nodes to actual allocated size
        num_nodes = static_cast<unsigned int>(h_nodes.size());
    }
    
    // Upload tree to GPU
    void uploadToGPU(cudaStream_t stream = 0)
    {
        if (num_nodes == 0)
            return;
        
        // Reallocate if needed
        if (num_nodes > allocated_nodes)
        {
            freeGPU();
            cudaMalloc(&d_nodes, num_nodes * sizeof(PhotonKDNode));
            allocated_nodes = num_nodes;
        }
        
        cudaMemcpyAsync(d_nodes, h_nodes.data(), 
                        num_nodes * sizeof(PhotonKDNode),
                        cudaMemcpyHostToDevice, stream);
    }
    
    // Get tree structure for GPU
    PhotonKDTree getTree() const
    {
        PhotonKDTree tree;
        tree.nodes = d_nodes;
        tree.num_nodes = num_nodes;
        tree.max_depth = 32;  // Max stack depth
        tree.valid = (num_nodes > 0 && d_nodes != nullptr);
        return tree;
    }
    
    // Free GPU memory
    void freeGPU()
    {
        if (d_nodes)
        {
            cudaFree(d_nodes);
            d_nodes = nullptr;
        }
        allocated_nodes = 0;
    }
    
    // Clear all data
    void clear()
    {
        h_nodes.clear();
        num_nodes = 0;
    }
    
    // Get number of nodes
    unsigned int getNumNodes() const { return num_nodes; }
};

