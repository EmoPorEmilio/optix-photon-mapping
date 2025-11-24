#include "OptixLaunch.h"
#include <sutil/vec_math.h>

OptixLaunch::~OptixLaunch()
{
    freeBuffers();
}

void OptixLaunch::allocateBuffers(unsigned int width, unsigned int height)
{
    
    if (buffers_allocated && allocated_width == width && allocated_height == height)
        return; 

    
    if (buffers_allocated)
    {
        freeBuffers();
    }

    const size_t buffer_size = width * height * sizeof(uchar4);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_frame_buffer), buffer_size));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_params), sizeof(Params)));

    allocated_width = width;
    allocated_height = height;
    buffers_allocated = true;
}

void OptixLaunch::freeBuffers()
{
    if (!buffers_allocated)
        return;

    if (d_frame_buffer)
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_frame_buffer)));
    if (d_params)
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_params)));
    d_frame_buffer = 0;
    d_params = 0;
    allocated_width = 0;
    allocated_height = 0;
    buffers_allocated = false;
}

void OptixLaunch::launch(
    OptixPipeline pipeline,
    CUstream stream,
    const OptixShaderBindingTable &sbt,
    unsigned int width,
    unsigned int height,
    const Camera &camera,
    OptixTraversableHandle iasHandle,
    unsigned char *outputBuffer,
    CUdeviceptr triangleColors,
    float3 sphere1Color,
    float3 sphere2Color)
{
    if (!buffers_allocated)
    {
        throw std::runtime_error("Buffers not allocated. Call allocateBuffers() first.");
    }

    const size_t buffer_size = width * height * sizeof(uchar4);

    
    float3 U = camera.getU();
    float3 V = camera.getV();
    float3 W = camera.getW();

    Params params;
    params.frame_buffer = reinterpret_cast<uchar4 *>(d_frame_buffer);
    params.width = width;
    params.height = height;
    params.eye = camera.position;
    params.U = U;
    params.V = V;
    params.W = W;
    params.handle = iasHandle;

    
    params.sphere1.center = make_float3(185.0f, 82.5f, 169.0f);
    params.sphere1.radius = 82.5f;
    params.sphere2.center = make_float3(368.0f, 103.5f, 351.0f);
    params.sphere2.radius = 103.5f;

    
    params.triangle_colors = reinterpret_cast<float3 *>(triangleColors);
    params.sphere1_color = sphere1Color;
    params.sphere2_color = sphere2Color;

    
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_params), &params, sizeof(Params), cudaMemcpyHostToDevice));

    
    OPTIX_CHECK(optixLaunch(pipeline, stream, d_params, sizeof(Params), &sbt, width, height, 1));

    
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(outputBuffer, reinterpret_cast<void *>(d_frame_buffer), buffer_size, cudaMemcpyDeviceToHost));
}

