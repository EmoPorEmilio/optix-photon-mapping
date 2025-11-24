#pragma once

#include <optix.h>
#include <sutil/Exception.h>
#include <cuda_runtime.h>
#include <vector>
#include "../Scene.h" 



class OptixAccelerationStructuresBuilder
{
public:
    static void buildTriangleGAS(
        OptixDeviceContext context,
        CUstream stream,
        const std::vector<OptixVertex> &vertices,
        CUdeviceptr &d_vertices,
        CUdeviceptr &d_output,
        OptixTraversableHandle &outHandle)
    {
        const size_t size = vertices.size() * sizeof(OptixVertex);
        CUDA_CHECK(cudaMalloc((void **)&d_vertices, size));
        CUDA_CHECK(cudaMemcpy((void *)d_vertices, vertices.data(), size, cudaMemcpyHostToDevice));

        OptixBuildInput input = {};
        input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        input.triangleArray.vertexStrideInBytes = sizeof(OptixVertex);
        input.triangleArray.numVertices = static_cast<uint32_t>(vertices.size());
        input.triangleArray.vertexBuffers = &d_vertices;
        uint32_t flags[] = {OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT};
        input.triangleArray.flags = flags;
        input.triangleArray.numSbtRecords = 1;

        OptixAccelBuildOptions opts = {};
        opts.buildFlags = OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
        opts.operation = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &opts, &input, 1, &sizes));
        CUdeviceptr d_temp;
        CUDA_CHECK(cudaMalloc((void **)&d_temp, sizes.tempSizeInBytes));
        CUDA_CHECK(cudaMalloc((void **)&d_output, sizes.outputSizeInBytes));
        OPTIX_CHECK(optixAccelBuild(context, stream, &opts, &input, 1, d_temp, sizes.tempSizeInBytes, d_output, sizes.outputSizeInBytes, &outHandle, nullptr, 0));
        CUDA_CHECK(cudaFree((void *)d_temp));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    static void buildSphereGAS(
        OptixDeviceContext context,
        CUstream stream,
        const OptixAabb *aabbsHost,
        uint32_t count,
        CUdeviceptr &d_aabbs,
        CUdeviceptr &d_output,
        OptixTraversableHandle &outHandle)
    {
        CUDA_CHECK(cudaMalloc((void **)&d_aabbs, sizeof(OptixAabb) * count));
        CUDA_CHECK(cudaMemcpy((void *)d_aabbs, aabbsHost, sizeof(OptixAabb) * count, cudaMemcpyHostToDevice));

        uint32_t flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
        OptixBuildInput input = {};
        input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        input.customPrimitiveArray.aabbBuffers = &d_aabbs;
        input.customPrimitiveArray.numPrimitives = count;
        input.customPrimitiveArray.flags = flags;
        input.customPrimitiveArray.numSbtRecords = 1;

        OptixAccelBuildOptions opts = {};
        opts.buildFlags = OPTIX_BUILD_FLAG_NONE;
        opts.operation = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &opts, &input, 1, &sizes));
        CUdeviceptr d_temp;
        CUDA_CHECK(cudaMalloc((void **)&d_temp, sizes.tempSizeInBytes));
        CUDA_CHECK(cudaMalloc((void **)&d_output, sizes.outputSizeInBytes));
        OPTIX_CHECK(optixAccelBuild(context, stream, &opts, &input, 1, d_temp, sizes.tempSizeInBytes, d_output, sizes.outputSizeInBytes, &outHandle, nullptr, 0));
        CUDA_CHECK(cudaFree((void *)d_temp));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    static void buildIAS(
        OptixDeviceContext context,
        CUstream stream,
        const std::vector<OptixInstance> &hostInstances,
        CUdeviceptr &d_instances,
        CUdeviceptr &d_output,
        OptixTraversableHandle &outHandle)
    {
        CUDA_CHECK(cudaMalloc((void **)&d_instances, sizeof(OptixInstance) * hostInstances.size()));
        CUDA_CHECK(cudaMemcpy((void *)d_instances, hostInstances.data(), sizeof(OptixInstance) * hostInstances.size(), cudaMemcpyHostToDevice));

        OptixBuildInput input = {};
        input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        input.instanceArray.instances = d_instances;
        input.instanceArray.numInstances = static_cast<uint32_t>(hostInstances.size());

        OptixAccelBuildOptions opts = {};
        opts.buildFlags = OPTIX_BUILD_FLAG_NONE;
        opts.operation = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &opts, &input, 1, &sizes));
        CUdeviceptr d_temp;
        CUDA_CHECK(cudaMalloc((void **)&d_temp, sizes.tempSizeInBytes));
        CUDA_CHECK(cudaMalloc((void **)&d_output, sizes.outputSizeInBytes));
        OPTIX_CHECK(optixAccelBuild(context, stream, &opts, &input, 1, d_temp, sizes.tempSizeInBytes, d_output, sizes.outputSizeInBytes, &outHandle, nullptr, 0));
        CUDA_CHECK(cudaFree((void *)d_temp));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
};



