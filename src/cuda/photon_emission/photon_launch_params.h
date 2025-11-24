

#pragma once

#include <optix.h>
#include <sutil/vec_math.h>
#include "../../lighting/QuadLight.h"
#include "../../rendering/photon/Photon.h"
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
    float3 photon_power; // power per emitted photon (before surface interactions)

    // Scene data needed in the photon pass
    unsigned int quadLightStartIndex;

    // Original per-triangle colors (still available if needed for debugging/visualization)
    float3* triangle_colors;   // same layout as Scene::exportTriangleColors()

    // New material data for photon transport.
    Material* triangle_materials; // one material per triangle
    Material  sphere_materials[2]; // two spheres in this scene

    // Analytic sphere geometry for photon pass (mirrors what's used in the render pass).
    PhotonSphereData sphere1;
    PhotonSphereData sphere2;

    unsigned int max_depth;

    // Output buffers
    Photon* photons_out;
    CUdeviceptr photon_counter;


    __host__ __device__ PhotonLaunchParams() {}
};



