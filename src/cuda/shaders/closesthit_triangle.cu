
#include <optix.h>
#include "optix/OptixParams.h"
#include <sutil/vec_math.h>

extern "C" __global__ void __closesthit__triangle_ch()
{
    
    extern __constant__ Params params;
    
    
    const unsigned int prim_idx = optixGetPrimitiveIndex();
    
    
    const float3 color = params.triangle_colors[prim_idx];

    
    optixSetPayload_0(__float_as_uint(color.x));
    optixSetPayload_1(__float_as_uint(color.y));
    optixSetPayload_2(__float_as_uint(color.z));
}



