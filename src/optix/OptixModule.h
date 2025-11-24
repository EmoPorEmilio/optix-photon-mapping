

#pragma once

#include <optix.h>

OptixModule createOptixModule(
    OptixDeviceContext context,
    const OptixModuleCompileOptions &moduleOptions,
    OptixPipelineCompileOptions *pipelineOptions,
    const char *optixirPath);





