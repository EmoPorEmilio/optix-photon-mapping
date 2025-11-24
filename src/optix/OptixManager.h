
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

    bool buildTriangleGAS(const std::vector<OptixVertex> &vertices, const std::vector<float3> &colors);
    bool buildSphereGAS(float3 center1, float radius1, float3 center2, float radius2);
    bool buildIAS();

    
    bool createPhotonPipeline();
    void launchPhotonPass(unsigned int num_photons, const QuadLight& light,
                         unsigned int quadLightStartIndex, CUdeviceptr& out_photons, unsigned int& out_count);

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

    void cleanup();
};

struct OptixVertex; 

bool buildTriangleGAS(const std::vector<OptixVertex> &vertices);

bool buildSphereGAS(float3 center1, float radius1, float3 center2, float radius2);

bool buildIAS();

void render(unsigned int width, unsigned int height, const Camera &camera, unsigned char *output_buffer);



