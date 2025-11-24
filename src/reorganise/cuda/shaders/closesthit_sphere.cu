
#include <optix.h>
#include "optix/OptixParams.h"
#include <sutil/vec_math.h>

extern "C" __global__ void __closesthit__sphere_ch()
{
    
    extern __constant__ Params params;
    
    
    const unsigned int prim_idx = optixGetPrimitiveIndex();
    
    
    float3 color;
    if (prim_idx == 0) {
        color = params.sphere1_color;
    } else {
        color = params.sphere2_color;
    }

    
    optixSetPayload_0(__float_as_uint(color.x));
    optixSetPayload_1(__float_as_uint(color.y));
    optixSetPayload_2(__float_as_uint(color.z));
}



