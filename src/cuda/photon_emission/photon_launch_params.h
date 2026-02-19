#pragma once

#include <optix.h>
#include <sutil/vec_math.h>
#include "../../lighting/QuadLight.h"
#include "../../rendering/photon/Photon.h"
#include "../../rendering/photon/VolumePhoton.h"
#include "../../rendering/photon/PhotonTrajectory.h"
#include "../../scene/Material.h"

// Simple sphere geometry description for the photon pass.
struct PhotonSphereData
{
    float3 center;
    float  radius;
};

struct PhotonLaunchParams
{
    OptixTraversableHandle handle;
    QuadLight light;

    // Per-photon launch configuration
    unsigned int num_photons;
    float3 photon_power;  // power per emitted photon (before surface interactions)

    // Scene data needed in the photon pass
    unsigned int quadLightStartIndex;

    // Original per-triangle colors (still available if needed for debugging/visualization)
    float3* triangle_colors;  // same layout as Scene::exportTriangleColors()

    // Material data for photon transport
    Material* triangle_materials;  // one material per triangle
    Material sphere_materials[2];  // two spheres in this scene

    // Analytic sphere geometry for photon pass
    PhotonSphereData sphere1;
    PhotonSphereData sphere2;

    unsigned int max_depth;

    // Output buffers - Global photon map (indirect illumination)
    Photon* photons_out;
    CUdeviceptr photon_counter;

    // Caustic photon map (specular/transmissive -> diffuse)
    Photon* caustic_photons_out;
    CUdeviceptr caustic_photon_counter;

    //=========================================================================
    // Trajectory Recording (for debugging/visualization)
    //=========================================================================
    bool record_trajectories;           // Enable full trajectory recording
    PhotonTrajectory* trajectories_out; // Output buffer (one per photon)

    //=========================================================================
    // Volume/Participating Media (Jensen's PDF §1.4, §3.3)
    //=========================================================================
    bool enable_volume_scattering;      // Enable fog/participating media
    VolumeProperties volume;            // Volume extinction/scattering coefficients
    VolumePhoton* volume_photons_out;   // Output buffer for volume photons
    CUdeviceptr volume_photon_counter;  // Atomic counter for volume photons

    __host__ __device__ PhotonLaunchParams()
        : record_trajectories(false), trajectories_out(nullptr),
          enable_volume_scattering(false), volume_photons_out(nullptr),
          volume_photon_counter(0) {}
};



