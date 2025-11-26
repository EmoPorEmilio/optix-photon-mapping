

#include <optix.h>
#include <sutil/vec_math.h>
#include "photon_launch_params.h"
#include "photon_rng.h"

extern "C" __constant__ PhotonLaunchParams params;

extern "C" __global__ void __raygen__photon_emitter()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    const unsigned int photon_idx = idx.x + idx.y * dim.x + idx.z * dim.x * dim.y;

    if (photon_idx >= params.num_photons)
        return;

    const QuadLight light = params.light;

    // Initial emission from the light.
    unsigned int rngState = photon_idx * 747796405u + 2891336453u;
    float u0 = ph_rand(rngState);
    float u1 = ph_rand(rngState);
    float u2 = ph_rand(rngState);
    float u3 = ph_rand(rngState);

    float3 origin, direction;
    light.samplePhotonEmission(u0, u1, u2, u3, origin, direction);

    // Path throughput starts as per-photon power.
    float3 throughput = params.photon_power;

    unsigned int p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11;

    // Bounce loop driven from raygen.
    // Payload 9 bit layout: bit 31 = prevWasSpecular, bit 30 = insideFlag, bits 0-29 = depth
    unsigned int depth = 0u;
    unsigned int insideFlag = 0u;
    unsigned int prevWasSpecular = 0u;
    
    for (;;)
    {
        // Pack state into payload 9
        unsigned int packedState = (prevWasSpecular << 31) | (insideFlag << 30) | (depth & 0x3fffffffu);
        
        // Pack payload for this segment.
        p0  = __float_as_uint(throughput.x);
        p1  = __float_as_uint(throughput.y);
        p2  = __float_as_uint(throughput.z);
        p3  = __float_as_uint(origin.x);
        p4  = __float_as_uint(origin.y);
        p5  = __float_as_uint(origin.z);
        p6  = __float_as_uint(direction.x);
        p7  = __float_as_uint(direction.y);
        p8  = __float_as_uint(direction.z);
        p9  = packedState;
        p10 = photon_idx;       // for RNG in closest hit
        p11 = 1u;               // continue flag (1 = continue, 0 = terminate)

        unsigned int flags = OPTIX_RAY_FLAG_DISABLE_ANYHIT;
        optixTrace(params.handle,
                   origin,
                   direction,
                   0.0f,
                   1e16f,
                   0.0f,
                   OptixVisibilityMask(255),
                   flags,
                   0,
                   1,
                   0,
                   p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11);

        // Unpack updated state from payload.
        throughput.x = __uint_as_float(p0);
        throughput.y = __uint_as_float(p1);
        throughput.z = __uint_as_float(p2);
        origin.x     = __uint_as_float(p3);
        origin.y     = __uint_as_float(p4);
        origin.z     = __uint_as_float(p5);
        direction.x  = __uint_as_float(p6);
        direction.y  = __uint_as_float(p7);
        direction.z  = __uint_as_float(p8);
        
        // Unpack state from payload 9
        packedState = p9;
        depth = packedState & 0x3fffffffu;
        insideFlag = (packedState >> 30) & 0x1u;
        prevWasSpecular = (packedState >> 31) & 0x1u;
        
        unsigned int cont = p11;

        if (!cont || depth >= params.max_depth)
            break;
    }
}
