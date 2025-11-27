// Combined CUDA file for photon emission pass
// Individual .cu files are included to create a single compilation unit

#include <optix.h>
#include <sutil/vec_math.h>
#include "photon_emission/photon_launch_params.h"

// Single definition of params for the combined compilation unit
extern "C" __constant__ PhotonLaunchParams params;

// Shared helper function for trajectory recording
// Defined here once to avoid duplication across included .cu files
__device__ __forceinline__ void recordTrajectoryEvent(
    unsigned int photon_idx,
    int event_type,
    const float3 &position,
    const float3 &direction,
    const float3 &power,
    int material_type = TRAJ_MAT_NONE)
{
    if (!params.record_trajectories || !params.trajectories_out)
        return;

    PhotonTrajectory &traj = params.trajectories_out[photon_idx];
    traj.addEvent(event_type, position, direction, power, material_type);
}

// Prevent redefinition of params in included files
#define PHOTON_PARAMS_DEFINED

#include "photon_emission/raygen_photons.cu"
#include "photon_emission/closesthit_store.cu"
#include "photon_emission/miss_photons.cu"
#include "photon_emission/intersection_photon_sphere.cu"

