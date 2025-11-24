#pragma once
#include <optix.h>
#include <cuda_runtime.h>
#include <sutil/Exception.h>
#include <iostream>
#include <stdexcept>

#include "OptixParams.h" 
#include "../scene/Camera.h"   

class OptixLaunch
{
public:
    OptixLaunch() = default;
    ~OptixLaunch();

    void allocateBuffers(unsigned int width, unsigned int height);
    void freeBuffers();

    void launch(
        OptixPipeline pipeline,
        CUstream stream,
        const OptixShaderBindingTable& sbt,
        unsigned int width,
        unsigned int height,
        const Camera& camera,
        OptixTraversableHandle iasHandle,
        unsigned char* outputBuffer,
        CUdeviceptr triangleColors,
        float3 sphere1Color,
        float3 sphere2Color);

private:
    CUdeviceptr d_frame_buffer = 0;
    CUdeviceptr d_params = 0;
    unsigned int allocated_width = 0;
    unsigned int allocated_height = 0;
    bool buffers_allocated = false;
};



