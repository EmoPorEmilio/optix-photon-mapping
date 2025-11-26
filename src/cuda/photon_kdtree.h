#pragma once

#include <sutil/vec_math.h>
#include "../rendering/photon/Photon.h"

// GPU-compatible kd-tree for photon mapping (Jensen's algorithm)
// Uses a linearized array representation for efficient GPU traversal
// The tree is built on CPU and uploaded to GPU for queries

// kd-tree node stored in linearized array format
// For a balanced tree, left child of node i is at 2*i+1, right at 2*i+2
struct PhotonKDNode
{
    Photon photon;      // The photon stored at this node
    int split_axis;     // 0=x, 1=y, 2=z, -1=leaf/empty
    float split_value;  // Split plane position (photon coordinate on split axis)
};

// kd-tree parameters passed to GPU
struct PhotonKDTree
{
    PhotonKDNode* nodes;    // Array of kd-tree nodes (linearized)
    unsigned int num_nodes; // Total number of nodes in the tree
    unsigned int max_depth; // Maximum depth of the tree
    bool valid;             // Whether the tree is valid/built
};

// Device function to get coordinate value by axis
static __forceinline__ __device__ float getAxisValue(const float3& pos, int axis)
{
    return (axis == 0) ? pos.x : ((axis == 1) ? pos.y : pos.z);
}

// Device function for kd-tree radius query
// Uses iterative stack-based traversal for GPU efficiency
static __forceinline__ __device__ float3 gatherPhotonsKDTree(
    const float3& hit_point,
    const float3& normal,
    const PhotonKDTree& tree,
    float gather_radius)
{
    if (!tree.valid || tree.num_nodes == 0 || tree.nodes == nullptr)
        return make_float3(0.0f);
    
    float3 flux_sum = make_float3(0.0f);
    const float radius_sq = gather_radius * gather_radius;
    const float inv_pi = 0.31830988618f;  // 1/π
    
    // Stack for iterative traversal (32 levels is plenty for millions of photons)
    unsigned int stack[32];
    int stack_ptr = 0;
    
    // Start at root
    stack[stack_ptr++] = 0;
    
    while (stack_ptr > 0)
    {
        unsigned int node_idx = stack[--stack_ptr];
        
        // Bounds check
        if (node_idx >= tree.num_nodes)
            continue;
        
        const PhotonKDNode& node = tree.nodes[node_idx];
        
        // Skip empty nodes
        if (node.split_axis < -1)
            continue;
        
        // Check distance to this photon
        float3 diff = hit_point - node.photon.position;
        float dist_sq = dot(diff, diff);
        
        if (dist_sq < radius_sq)
        {
            // Check if photon is on the correct side of the surface
            float incidentDot = dot(node.photon.incidentDir, normal);
            if (incidentDot < 0.0f)
            {
                // Cone filter weight
                float weight = 1.0f - sqrtf(dist_sq) / gather_radius;
                flux_sum += node.photon.power * weight;
            }
        }
        
        // Determine which children to visit
        if (node.split_axis >= 0)  // Not a leaf
        {
            float query_val = getAxisValue(hit_point, node.split_axis);
            float split_val = node.split_value;
            float dist_to_plane = query_val - split_val;
            
            unsigned int left_child = 2 * node_idx + 1;
            unsigned int right_child = 2 * node_idx + 2;
            
            // Visit the closer child first (for potential early termination)
            if (dist_to_plane < 0)
            {
                // Query point is on left side
                // Always check left child
                if (left_child < tree.num_nodes && stack_ptr < 31)
                    stack[stack_ptr++] = left_child;
                
                // Check right child only if it might contain photons within radius
                if (right_child < tree.num_nodes && 
                    dist_to_plane * dist_to_plane < radius_sq &&
                    stack_ptr < 31)
                    stack[stack_ptr++] = right_child;
            }
            else
            {
                // Query point is on right side
                // Always check right child
                if (right_child < tree.num_nodes && stack_ptr < 31)
                    stack[stack_ptr++] = right_child;
                
                // Check left child only if it might contain photons within radius
                if (left_child < tree.num_nodes && 
                    dist_to_plane * dist_to_plane < radius_sq &&
                    stack_ptr < 31)
                    stack[stack_ptr++] = left_child;
            }
        }
    }
    
    // Jensen's radiance estimation with cone filter normalization (factor of 3)
    // and 1/π for diffuse BRDF
    float cone_normalization = 3.0f;
    float area = 3.14159265f * radius_sq;
    float3 radiance = flux_sum * (cone_normalization / area) * inv_pi;
    
    return radiance;
}

// Alternative: gather with albedo modulation for indirect lighting
static __forceinline__ __device__ float3 gatherPhotonsKDTreeWithAlbedo(
    const float3& hit_point,
    const float3& normal,
    const float3& albedo,
    const PhotonKDTree& tree,
    float gather_radius,
    float brightness_multiplier)
{
    float3 radiance = gatherPhotonsKDTree(hit_point, normal, tree, gather_radius);
    return radiance * albedo * brightness_multiplier;
}

