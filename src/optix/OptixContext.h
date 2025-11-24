
#pragma once

#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>

class OptixContext
{
public:
    OptixContext();
    ~OptixContext();

    using LogCallback = void (*)(unsigned int, const char *, const char *, void *);
    void create(LogCallback logCallback, unsigned int logLevel);
    void destroy();

    OptixDeviceContext get() const { return m_context; }
    CUstream stream() const { return m_stream; }

private:
    OptixDeviceContext m_context = nullptr;
    CUstream m_stream = nullptr;
    bool m_created = false;
};



