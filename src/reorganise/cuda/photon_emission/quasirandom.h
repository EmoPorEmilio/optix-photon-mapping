#pragma once

#include <optix.h>

// Simple scrambled base-2 low-discrepancy sequence (Van der Corput style).
// This gives you quasi-random samples in [0,1) indexed by (index, dim).
// It is not a full multi-dimensional Sobol implementation, but plays a
// similar role and is lightweight enough to keep inline in this project.

static __forceinline__ __device__ unsigned int qr_hash(unsigned int v)
{
    // PCG-inspired integer hash
    v ^= v >> 17;
    v *= 0xed5ad4bbU;
    v ^= v >> 11;
    v *= 0xac4c1b51U;
    v ^= v >> 15;
    v *= 0x31848babU;
    v ^= v >> 14;
    return v;
}

static __forceinline__ __device__ float qr_radicalInverseVdC(unsigned int bits)
{
    bits = (bits << 16) | (bits >> 16);
    bits = ((bits & 0x55555555u) << 1) | ((bits & 0xAAAAAAAAu) >> 1);
    bits = ((bits & 0x33333333u) << 2) | ((bits & 0xCCCCCCCCu) >> 2);
    bits = ((bits & 0x0F0F0F0Fu) << 4) | ((bits & 0xF0F0F0F0u) >> 4);
    bits = ((bits & 0x00FF00FFu) << 8) | ((bits & 0xFF00FF00u) >> 8);
    return static_cast<float>(bits) * 2.3283064365386963e-10f; // 1 / 2^32
}

// Quasi-random float in [0,1) for the given photon index and dimension.
static __forceinline__ __device__ float quasiRandom01(unsigned int index, unsigned int dim)
{
    // Different scramble per dimension
    unsigned int dimSeed = qr_hash(dim + 1u);
    unsigned int bits = index ^ dimSeed;
    return qr_radicalInverseVdC(bits);
}
