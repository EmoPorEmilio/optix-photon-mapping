
#include <optix.h>
#include <sutil/vec_math.h>

extern "C" __global__ void __miss__ms()
{
    optixSetPayload_0(__float_as_uint(0.0f));
    optixSetPayload_1(__float_as_uint(0.0f));
    optixSetPayload_2(__float_as_uint(0.0f));
}
