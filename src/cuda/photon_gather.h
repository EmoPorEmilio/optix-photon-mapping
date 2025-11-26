#pragma once

// Common photon gathering functions for all lighting passes
// Shared device code to eliminate duplication across shaders

#include <sutil/vec_math.h>
#include "../rendering/photon/Photon.h"
#include "../rendering/photon/PhotonKDTreeDevice.h"

// Constants for radiance estimation
#define PM_INV_PI 0.31830988618f   // 1/π
#define PM_CONE_NORMALIZATION 3.0f // Cone filter normalization factor

// Compute geometric triangle normal from vertex positions (shared utility)
static __forceinline__ __device__ float3 computeTriangleNormal(const float3 &ray_dir)
{
    float3 vertices[3];
    optixGetTriangleVertexData(
        optixGetGASTraversableHandle(),
        optixGetPrimitiveIndex(),
        optixGetSbtGASIndex(),
        0.0f,
        vertices);

    float3 edge1 = vertices[1] - vertices[0];
    float3 edge2 = vertices[2] - vertices[0];
    float3 normal = normalize(cross(edge1, edge2));

    // Ensure normal faces toward the ray origin
    if (dot(normal, ray_dir) > 0.0f)
        normal = -normal;

    return normal;
}

// Linear photon gathering - O(n) fallback when no acceleration structure is available
// Jensen's radiance estimation: L = (albedo/π) * (3/(π*r²)) * Σ(Φ_p * K)
static __forceinline__ __device__ float3 gatherPhotonsLinear(
    const float3 &hit_point,
    const float3 &normal,
    const Photon *photon_map,
    unsigned int photon_count,
    float gather_radius)
{
    if (photon_count == 0 || photon_map == nullptr)
        return make_float3(0.0f);

    float3 flux_sum = make_float3(0.0f);
    const float radius_sq = gather_radius * gather_radius;

    for (unsigned int i = 0; i < photon_count; i++)
    {
        const Photon &photon = photon_map[i];

        float3 diff = hit_point - photon.position;
        float dist_sq = dot(diff, diff);

        if (dist_sq < radius_sq)
        {
            // Check if photon is on the correct side of the surface
            float incidentDot = dot(photon.incidentDir, normal);
            if (incidentDot < 0.0f)
            {
                // Cone filter weight (gives more weight to closer photons)
                float weight = 1.0f - sqrtf(dist_sq) / gather_radius;
                flux_sum += photon.power * weight;
            }
        }
    }

    // Apply Jensen's radiance estimation formula
    // Cone filter normalization (factor of 3) and 1/π for diffuse BRDF
    float area = 3.14159265f * radius_sq;
    float3 radiance = flux_sum * (PM_CONE_NORMALIZATION / area) * PM_INV_PI;

    return radiance;
}

// Linear gathering with albedo modulation for indirect illumination
static __forceinline__ __device__ float axisValue(const float3 &p, int axis)
{
    return axis == 0 ? p.x : (axis == 1 ? p.y : p.z);
}

// kd-tree photon gathering (O(log n))
static __forceinline__ __device__ float3 gatherPhotonsKDTree(
    const float3 &hit_point,
    const float3 &normal,
    const PhotonKDTreeDevice &tree,
    float gather_radius)
{
    if (!tree.valid || !tree.nodes || tree.node_count == 0)
        return make_float3(0.0f);

    float3 flux_sum = make_float3(0.0f);
    const float radius_sq = gather_radius * gather_radius;

    unsigned int stack[64];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0;

    while (stack_ptr > 0)
    {
        unsigned int idx = stack[--stack_ptr];
        if (idx >= tree.node_count)
            continue;

        const PhotonKDNodeDevice &node = tree.nodes[idx];

        float3 diff = hit_point - node.photon.position;
        float dist_sq = dot(diff, diff);
        if (dist_sq < radius_sq)
        {
            float incidentDot = dot(node.photon.incidentDir, normal);
            if (incidentDot < 0.0f)
            {
                float weight = 1.0f - sqrtf(dist_sq) / gather_radius;
                flux_sum += node.photon.power * weight;
            }
        }

        float dist_to_plane = axisValue(hit_point, node.axis) - node.split;
        int nearChild = dist_to_plane < 0.0f ? node.left : node.right;
        int farChild = dist_to_plane < 0.0f ? node.right : node.left;

        if (nearChild >= 0 && stack_ptr < 63)
            stack[stack_ptr++] = static_cast<unsigned int>(nearChild);

        if (farChild >= 0 && stack_ptr < 63 && dist_to_plane * dist_to_plane < radius_sq)
            stack[stack_ptr++] = static_cast<unsigned int>(farChild);
    }

    float area = 3.14159265f * radius_sq;
    float3 radiance = flux_sum * (PM_CONE_NORMALIZATION / area) * PM_INV_PI;
    return radiance;
}

// Gather helper that prefers kd-tree but falls back to linear search.
static __forceinline__ __device__ float3 gatherPhotonsRadiance(
    const float3 &hit_point,
    const float3 &normal,
    const Photon *photon_map,
    unsigned int photon_count,
    const PhotonKDTreeDevice &tree,
    float gather_radius)
{
    if (tree.valid)
        return gatherPhotonsKDTree(hit_point, normal, tree, gather_radius);
    return gatherPhotonsLinear(hit_point, normal, photon_map, photon_count, gather_radius);
}

static __forceinline__ __device__ float3 gatherPhotonsRadianceWithAlbedo(
    const float3 &hit_point,
    const float3 &normal,
    const float3 &albedo,
    const Photon *photon_map,
    unsigned int photon_count,
    const PhotonKDTreeDevice &tree,
    float gather_radius,
    float brightness_multiplier)
{
    float3 radiance = gatherPhotonsRadiance(hit_point, normal, photon_map, photon_count, tree, gather_radius);
    return radiance * albedo * brightness_multiplier;
}
