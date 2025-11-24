#pragma once

#include <optix.h>

// Shared hash-based RNG utilities for the photon emission pass.

static __forceinline__ __device__ unsigned int ph_hash(unsigned int x)
{
    x ^= x >> 17;
    x *= 0xed5ad4bbU;
    x ^= x >> 11;
    x *= 0xac4c1b51U;
    x ^= x >> 15;
    x *= 0x31848babU;
    x ^= x >> 14;
    return x;
}

static __forceinline__ __device__ float ph_rand(unsigned int &state)
{
    state = ph_hash(state);
    return static_cast<float>(state) * 2.3283064365386963e-10f; // 1 / 2^32
}


