#pragma once

#include <optix.h>
#include <sutil/Exception.h>
#include <cuda_runtime.h>
#include <cstring>

template <typename T>
struct SbtRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

struct RaygenData { };
struct MissData { float4 bg; };
struct HitData { };

class OptixSBTBuilder
{
public:
    static void build(
        OptixProgramGroup raygen,
        OptixProgramGroup miss,
        OptixProgramGroup triHit,
        OptixProgramGroup sphHit,
        OptixShaderBindingTable& sbt)
    {
        
        SbtRecord<RaygenData> rg = {};
        OPTIX_CHECK(optixSbtRecordPackHeader(raygen, &rg));
        CUdeviceptr d_rg;
        CUDA_CHECK(cudaMalloc((void**)&d_rg, sizeof(rg)));
        CUDA_CHECK(cudaMemcpy((void*)d_rg, &rg, sizeof(rg), cudaMemcpyHostToDevice));
        sbt.raygenRecord = d_rg;

        
        SbtRecord<MissData> ms = {};
        ms.data.bg = make_float4(0.f, 0.f, 0.f, 1.f);
        OPTIX_CHECK(optixSbtRecordPackHeader(miss, &ms));
        CUdeviceptr d_ms;
        CUDA_CHECK(cudaMalloc((void**)&d_ms, sizeof(ms)));
        CUDA_CHECK(cudaMemcpy((void*)d_ms, &ms, sizeof(ms), cudaMemcpyHostToDevice));
        sbt.missRecordBase = d_ms;
        sbt.missRecordStrideInBytes = sizeof(SbtRecord<MissData>);
        sbt.missRecordCount = 1;

        
        SbtRecord<HitData> hg[2] = {};
        OPTIX_CHECK(optixSbtRecordPackHeader(triHit, &hg[0]));
        OPTIX_CHECK(optixSbtRecordPackHeader(sphHit, &hg[1]));
        CUdeviceptr d_hg;
        CUDA_CHECK(cudaMalloc((void**)&d_hg, sizeof(hg)));
        CUDA_CHECK(cudaMemcpy((void*)d_hg, hg, sizeof(hg), cudaMemcpyHostToDevice));
        sbt.hitgroupRecordBase = d_hg;
        sbt.hitgroupRecordStrideInBytes = sizeof(SbtRecord<HitData>);
        sbt.hitgroupRecordCount = 2;
    }
};





