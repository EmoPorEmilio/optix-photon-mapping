#pragma once

#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#include "Photon.h"
#include "PhotonKDTreeDevice.h"

// CPU-side kd-tree builder that mirrors Jensen's balanced kd-tree construction.
// Builds a balanced kd-tree from the provided photon list and uploads it to the GPU.
class PhotonKDTree
{
public:
    PhotonKDTree()
    {
        deviceTree.nodes = nullptr;
        deviceTree.node_count = 0;
        deviceTree.valid = false;
    }

    ~PhotonKDTree()
    {
        freeDevice();
    }

    // Disallow copy to avoid double frees
    PhotonKDTree(const PhotonKDTree &) = delete;
    PhotonKDTree &operator=(const PhotonKDTree &) = delete;

    // Allow move
    PhotonKDTree(PhotonKDTree &&other) noexcept
    {
        *this = std::move(other);
    }

    PhotonKDTree &operator=(PhotonKDTree &&other) noexcept
    {
        if (this != &other)
        {
            freeDevice();
            hostNodes = std::move(other.hostNodes);
            d_nodes = other.d_nodes;
            deviceTree = other.deviceTree;

            other.d_nodes = nullptr;
            other.deviceTree.nodes = nullptr;
            other.deviceTree.node_count = 0;
            other.deviceTree.valid = false;
        }
        return *this;
    }

    // Build kd-tree from photon list. If photons is empty, the kd-tree becomes invalid.
    void build(const std::vector<Photon> &photons)
    {
        freeDevice();
        hostNodes.clear();
        deviceTree.nodes = nullptr;
        deviceTree.node_count = 0;
        deviceTree.valid = false;

        if (photons.empty())
            return;

        std::vector<Photon> points = photons;
        hostNodes.reserve(points.size());
        buildRecursive(points, 0, static_cast<int>(points.size()), 0);
        uploadToGPU();
    }

    void clear()
    {
        freeDevice();
        hostNodes.clear();
        deviceTree.nodes = nullptr;
        deviceTree.node_count = 0;
        deviceTree.valid = false;
    }

    const PhotonKDTreeDevice &getDeviceTree() const { return deviceTree; }

private:
    struct HostNode
    {
        Photon photon;
        int axis;
        float split;
        int left;
        int right;
    };

    std::vector<HostNode> hostNodes;
    PhotonKDNodeDevice *d_nodes = nullptr;
    PhotonKDTreeDevice deviceTree{};

    static float coord(const Photon &p, int axis)
    {
        switch (axis)
        {
        case 0:
            return p.position.x;
        case 1:
            return p.position.y;
        default:
            return p.position.z;
        }
    }

    int buildRecursive(std::vector<Photon> &points, int begin, int end, int depth)
    {
        if (begin >= end)
            return -1;

        int axis = depth % 3;
        int mid = (begin + end) / 2;

        std::nth_element(points.begin() + begin, points.begin() + mid, points.begin() + end,
                         [axis](const Photon &a, const Photon &b)
                         {
                             return coord(a, axis) < coord(b, axis);
                         });

        int nodeIndex = static_cast<int>(hostNodes.size());
        hostNodes.push_back({});

        int left = buildRecursive(points, begin, mid, depth + 1);
        int right = buildRecursive(points, mid + 1, end, depth + 1);

        HostNode &node = hostNodes[nodeIndex];
        node.photon = points[mid];
        node.axis = axis;
        node.split = coord(points[mid], axis);
        node.left = left;
        node.right = right;

        return nodeIndex;
    }

    void uploadToGPU()
    {
        if (hostNodes.empty())
            return;

        size_t bytes = hostNodes.size() * sizeof(PhotonKDNodeDevice);
        cudaMalloc(reinterpret_cast<void **>(&d_nodes), bytes);

        std::vector<PhotonKDNodeDevice> deviceData(hostNodes.size());
        for (size_t i = 0; i < hostNodes.size(); ++i)
        {
            deviceData[i].photon = hostNodes[i].photon;
            deviceData[i].axis = hostNodes[i].axis;
            deviceData[i].split = hostNodes[i].split;
            deviceData[i].left = hostNodes[i].left;
            deviceData[i].right = hostNodes[i].right;
        }

        cudaMemcpy(d_nodes, deviceData.data(), bytes, cudaMemcpyHostToDevice);

        deviceTree.nodes = d_nodes;
        deviceTree.node_count = static_cast<unsigned int>(hostNodes.size());
        deviceTree.num_nodes = deviceTree.node_count;
        deviceTree.max_depth = 0;
        deviceTree.valid = deviceTree.node_count > 0;
    }

    void freeDevice()
    {
        if (d_nodes)
        {
            cudaFree(d_nodes);
            d_nodes = nullptr;
        }
    }
};

