

#include "PhotonMapper.h"
#include <sutil/vec_math.h>
#include <algorithm>
#include <cmath>

PhotonMapper::PhotonMapper(unsigned int amount_of_photons)
    : max_photons(amount_of_photons)
{
}

// Helper to get photon coordinate along axis.
static inline float photon_coord(const Photon &p, int axis)
{
    switch (axis)
    {
    case 0: return p.position.x;
    case 1: return p.position.y;
    default: return p.position.z;
    }
}

// Recursive builder: partitions [begin, end) in 'points' and writes nodes into 'nodes'.
static int buildRecursive(std::vector<PhotonMapper::KDNode> &nodes,
                          std::vector<Photon> &points,
                          int begin,
                          int end,
                          int depth)
{
    if (begin >= end)
        return -1;

    const int axis = depth % 3;
    const int mid = (begin + end) / 2;

    std::nth_element(points.begin() + begin, points.begin() + mid, points.begin() + end,
                     [axis](const Photon &a, const Photon &b)
                     {
                         return photon_coord(a, axis) < photon_coord(b, axis);
                     });

    PhotonMapper::KDNode node;
    node.photon = points[mid];
    node.axis = axis;
    node.left = -1;
    node.right = -1;

    const int nodeIndex = static_cast<int>(nodes.size());
    nodes.push_back(node);

    const int leftChild = buildRecursive(nodes, points, begin, mid, depth + 1);
    const int rightChild = buildRecursive(nodes, points, mid + 1, end, depth + 1);

    nodes[nodeIndex].left = leftChild;
    nodes[nodeIndex].right = rightChild;

    return nodeIndex;
}

void PhotonMapper::buildFromArray(const Photon *photons, size_t count)
{
    clear();

    if (!photons || count == 0)
        return;

    const size_t usedCount = std::min(count, static_cast<size_t>(max_photons));

    std::vector<Photon> points(usedCount);
    for (size_t i = 0; i < usedCount; ++i)
    {
        points[i] = photons[i];
    }

    nodes.reserve(usedCount);
    rootIndex = buildRecursive(nodes, points, 0, static_cast<int>(usedCount), 0);
}

void PhotonMapper::clear()
{
    nodes.clear();
    rootIndex = -1;
}

size_t PhotonMapper::size() const
{
    return nodes.size();
}

unsigned int PhotonMapper::getMaxPhotons() const
{
    return max_photons;
}

bool PhotonMapper::empty() const
{
    return nodes.empty();
}
