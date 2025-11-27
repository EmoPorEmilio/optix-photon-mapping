
#pragma once

#include <optix.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>
#include <sutil/vec_math.h>
#include <sutil/sutil.h>
#include <sutil/Exception.h>
#include <cuda_runtime.h>

#include "OptixContext.h"
#include "OptixParams.h"
#include "OptixModule.h"
#include "OptixProgramGroups.h"
#include "OptixPipeline.h"
#include "OptixSBT.h"
#include "OptixAccelerationStructuresBuilder.h"
#include "OptixLaunch.h"
#include "../scene/Camera.h"
#include "../scene/Scene.h"
#include "../lighting/QuadLight.h"
#include "../rendering/photon/Photon.h"
#include "../rendering/photon/PhotonKDTreeDevice.h"

#include <iostream>
#include <fstream>
#include <stdexcept>

struct OptixVertex;
class QuadLight;

class OptixManager
{
private:
    OptixDeviceContext context = 0;
    OptixModule module = 0;
    OptixPipeline pipeline = 0;
    OptixPipelineCompileOptions pipeline_compile_options = {};
    OptixProgramGroup raygen_group = 0;
    OptixProgramGroup miss_group = 0;
    OptixProgramGroup triangle_hit_group = 0;
    OptixProgramGroup sphere_hit_group = 0;
    OptixProgramGroup sphere_intersection_group = 0;
    OptixShaderBindingTable sbt = {};

    OptixTraversableHandle triangle_gas_handle = 0;
    CUdeviceptr d_triangle_vertices = 0;
    CUdeviceptr d_triangle_gas_buffer = 0;
    CUdeviceptr d_triangle_colors = 0;
    CUdeviceptr d_triangle_materials = 0;

    OptixTraversableHandle sphere_gas_handle = 0;
    CUdeviceptr d_sphere_aabb_buffer = 0;
    CUdeviceptr d_sphere_gas_buffer = 0;

    OptixTraversableHandle ias_handle = 0;
    CUdeviceptr d_ias_buffer = 0;
    CUdeviceptr d_ias_instances = 0;

    CUstream stream = 0;
    OptixContext contextOwner;
    OptixLaunch launcher;

    OptixModule photon_module = 0;
    OptixPipeline photon_pipeline = 0;
    OptixProgramGroup photon_raygen_group = 0;
    OptixProgramGroup photon_miss_group = 0;
    OptixProgramGroup photon_triangle_hit_group = 0;
    OptixProgramGroup photon_sphere_hit_group = 0;
    OptixShaderBindingTable photon_sbt = {};

    CUdeviceptr d_photon_buffer = 0;
    CUdeviceptr d_photon_counter = 0;

    // Caustic photon map buffers
    CUdeviceptr d_caustic_photon_buffer = 0;
    CUdeviceptr d_caustic_photon_counter = 0;

    // Direct lighting pipeline
    OptixModule direct_module = 0;
    OptixPipeline direct_pipeline = 0;
    OptixProgramGroup direct_raygen_group = 0;
    OptixProgramGroup direct_miss_group = 0;
    OptixProgramGroup direct_shadow_miss_group = 0;
    OptixProgramGroup direct_triangle_hit_group = 0;
    OptixProgramGroup direct_sphere_hit_group = 0;
    OptixProgramGroup direct_shadow_triangle_hit_group = 0;
    OptixProgramGroup direct_shadow_sphere_hit_group = 0;
    OptixShaderBindingTable direct_sbt = {};

    // Indirect lighting pipeline
    OptixModule indirect_module = 0;
    OptixPipeline indirect_pipeline = 0;
    OptixProgramGroup indirect_raygen_group = 0;
    OptixProgramGroup indirect_miss_group = 0;
    OptixProgramGroup indirect_triangle_hit_group = 0;
    OptixProgramGroup indirect_sphere_hit_group = 0;
    OptixShaderBindingTable indirect_sbt = {};

    // Caustic lighting pipeline
    OptixModule caustic_module = 0;
    OptixPipeline caustic_pipeline = 0;
    OptixProgramGroup caustic_raygen_group = 0;
    OptixProgramGroup caustic_miss_group = 0;
    OptixProgramGroup caustic_triangle_hit_group = 0;
    OptixProgramGroup caustic_sphere_hit_group = 0;
    OptixShaderBindingTable caustic_sbt = {};

    // Specular lighting pipeline (reflection/refraction)
    OptixModule specular_module = 0;
    OptixPipeline specular_pipeline = 0;
    OptixProgramGroup specular_raygen_group = 0;
    OptixProgramGroup specular_miss_group = 0;
    OptixProgramGroup specular_triangle_hit_group = 0;
    OptixProgramGroup specular_sphere_hit_group = 0;
    OptixShaderBindingTable specular_sbt = {};

    // Sphere geometry data
    float3 sphere1_center = make_float3(185.0f, 82.5f, 169.0f);
    float sphere1_radius = 82.5f;
    float3 sphere2_center = make_float3(368.0f, 82.5f, 351.0f);
    float sphere2_radius = 82.5f;

    // Light geometry info
    unsigned int quadLightStartIndex = 10; // Default, should be updated from scene

    bool initialized = false;

public:
    OptixManager() = default;
    ~OptixManager() { cleanup(); }

    bool initialize();

    bool createPipeline();

    bool isInitialized() const { return initialized; }
    OptixDeviceContext getContext() const { return context; }
    OptixPipeline getPipeline() const { return pipeline; }
    OptixTraversableHandle getIASHandle() const { return ias_handle; }

    bool buildTriangleGAS(const std::vector<OptixVertex> &vertices, const std::vector<float3> &colors,
                          const std::vector<int> &materialTypes = {});
    bool buildSphereGAS(float3 center1, float radius1, float3 center2, float radius2);
    bool buildIAS();

    void setQuadLightStartIndex(unsigned int index) { quadLightStartIndex = index; }

    bool createPhotonPipeline();
    void launchPhotonPass(unsigned int num_photons, const QuadLight &light,
                          unsigned int quadLightStartIndex,
                          CUdeviceptr &out_photons, unsigned int &out_count,
                          CUdeviceptr &out_caustic_photons, unsigned int &out_caustic_count);

    // Direct lighting pipeline
    bool createDirectLightingPipeline();
    void launchDirectLighting(unsigned int width, unsigned int height, const Camera &camera,
                              float ambient, float shadow_ambient, float intensity, float attenuation,
                              float4 *d_output);

    // Indirect lighting pipeline (color bleeding)
    bool createIndirectLightingPipeline();
    void launchIndirectLighting(unsigned int width, unsigned int height, const Camera &camera,
                                const Photon *d_photon_map, unsigned int photon_count,
                                float gather_radius, float brightness_multiplier,
                                const PhotonKDTreeDevice &kdtree,
                                float4 *d_output);

    // Caustic lighting pipeline (specular/glossy spheres)
    bool createCausticLightingPipeline();
    void launchCausticLighting(unsigned int width, unsigned int height, const Camera &camera,
                               const Photon *d_caustic_map, unsigned int caustic_count,
                               float gather_radius, float brightness_multiplier,
                               const PhotonKDTreeDevice &kdtree,
                               float4 *d_output);

    // Specular lighting pipeline (reflection/refraction on spheres)
    bool createSpecularLightingPipeline();
    struct SpecularParams
    {
        float gather_radius = 100.0f;
        unsigned int max_recursion_depth = 10;
        float glass_ior = 1.5f;
        float3 glass_tint = make_float3(0.98f, 0.99f, 1.0f);
        float mirror_reflectivity = 0.95f;
        float fresnel_min = 0.1f;
        float specular_ambient = 0.15f;
        float indirect_brightness = 100000.0f;
        float caustic_brightness = 200000.0f;
    };
    void launchSpecularLighting(unsigned int width, unsigned int height, const Camera &camera,
                                const Photon *d_global_map, unsigned int global_count,
                                const PhotonKDTreeDevice &global_kdtree,
                                const Photon *d_caustic_map, unsigned int caustic_count,
                                const PhotonKDTreeDevice &caustic_kdtree,
                                const SpecularParams &spec_params, float4 *d_output);

    void render(unsigned int width, unsigned int height, const Camera &camera, unsigned char *output_buffer);

private:
    static void contextLogCallback(unsigned int level, const char *tag, const char *message, void *)
    {
        std::cerr << "[" << level << "][" << tag << "]: " << message << std::endl;
    }

    bool loadModule();

    bool createProgramGroups();

    bool linkPipeline();

    void buildSBT();

    bool loadPhotonModule();
    bool createPhotonProgramGroups();
    bool linkPhotonPipeline();
    void buildPhotonSBT();

    // Direct lighting helpers
    bool loadDirectModule();
    bool createDirectProgramGroups();
    bool linkDirectPipeline();
    void buildDirectSBT();

    // Indirect lighting helpers
    bool loadIndirectModule();
    bool createIndirectProgramGroups();
    bool linkIndirectPipeline();
    void buildIndirectSBT();

    // Caustic lighting helpers
    bool loadCausticModule();
    bool createCausticProgramGroups();
    bool linkCausticPipeline();
    void buildCausticSBT();

    // Specular lighting helpers
    bool loadSpecularModule();
    bool createSpecularProgramGroups();
    bool linkSpecularPipeline();
    void buildSpecularSBT();

    void cleanup();
};

struct OptixVertex;

bool buildTriangleGAS(const std::vector<OptixVertex> &vertices);

bool buildSphereGAS(float3 center1, float radius1, float3 center2, float radius2);

bool buildIAS();

void render(unsigned int width, unsigned int height, const Camera &camera, unsigned char *output_buffer);
