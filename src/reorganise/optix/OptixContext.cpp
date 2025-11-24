#include "OptixContext.h"
#include <sutil/Exception.h>

OptixContext::OptixContext() = default;

OptixContext::~OptixContext()
{
    destroy();
}

void OptixContext::create(LogCallback logCallback, unsigned int logLevel)
{
    if (m_created)
        return;
    CUDA_CHECK(cudaFree(0));
    OPTIX_CHECK(optixInit());
    CUcontext cuCtx = 0; 
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = logCallback;
    options.logCallbackLevel = logLevel;
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &m_context));
    CUDA_CHECK(cudaStreamCreate(&m_stream));
    m_created = true;
}

void OptixContext::destroy()
{
    if (!m_created)
        return;
    if (m_stream)
        CUDA_CHECK(cudaStreamDestroy(m_stream));
    if (m_context)
        OPTIX_CHECK(optixDeviceContextDestroy(m_context));
    m_stream = nullptr;
    m_context = nullptr;
    m_created = false;
}



