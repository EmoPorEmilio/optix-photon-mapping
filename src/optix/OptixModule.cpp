#include "OptixModule.h"
#include <sutil/Exception.h>
#include <fstream>
#include <vector>

OptixModule createOptixModule(
    OptixDeviceContext context,
    const OptixModuleCompileOptions &moduleOptions,
    OptixPipelineCompileOptions *pipelineOptions,
    const char *optixirPath)
{
    std::ifstream file(optixirPath, std::ios::binary);
    if (!file)
        throw std::runtime_error("Failed to open OptiX IR file");
    file.seekg(0, std::ios::end);
    const size_t size = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);
    std::vector<char> data(size);
    file.read(data.data(), size);
    file.close();

    char log[2048];
    size_t logSize = sizeof(log);
    OptixModule module = nullptr;
    OPTIX_CHECK_LOG(
        optixModuleCreate(
            context,
            &moduleOptions,
            pipelineOptions,
            data.data(),
            size,
            log, &logSize,
            &module),
        log, logSize);
    return module;
}



