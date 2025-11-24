

#include <optix.h>
#include "photon_launch_params.h"


extern "C" __global__ void __miss__photon_miss()
{
    // Signal termination to the raygen bounce loop.
    optixSetPayload_11(0u);
}



