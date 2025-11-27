#include <optix.h>
#include <sutil/vec_math.h>
#include "photon_launch_params.h"

#ifndef PHOTON_PARAMS_DEFINED
extern "C" __constant__ PhotonLaunchParams params;
#endif

extern "C" __global__ void __miss__photon_miss()
{
    // Record MISS event (ray escaped the scene)
    if (params.record_trajectories && params.trajectories_out)
    {
        unsigned int photon_idx = optixGetPayload_10();
        
        // Get ray info from payload
        float3 origin = make_float3(
            __uint_as_float(optixGetPayload_3()),
            __uint_as_float(optixGetPayload_4()),
            __uint_as_float(optixGetPayload_5()));
        float3 direction = make_float3(
            __uint_as_float(optixGetPayload_6()),
            __uint_as_float(optixGetPayload_7()),
            __uint_as_float(optixGetPayload_8()));
        float3 throughput = make_float3(
            __uint_as_float(optixGetPayload_0()),
            __uint_as_float(optixGetPayload_1()),
            __uint_as_float(optixGetPayload_2()));

        // Record miss event at a far point along the ray direction
        float3 miss_pos = origin + direction * 1000.0f;
        
        PhotonTrajectory &traj = params.trajectories_out[photon_idx];
        traj.addEvent(EVENT_MISS, miss_pos, direction, throughput, TRAJ_MAT_NONE);
    }

    // Signal termination to the raygen bounce loop
    optixSetPayload_11(0u);
}
