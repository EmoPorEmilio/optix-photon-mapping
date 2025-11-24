
#include <optix.h>
#include "optix/OptixParams.h"
#include <sutil/vec_math.h>

extern "C"
{
    __constant__ Params params;
}


extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();      
    const uint3 dim = optixGetLaunchDimensions(); 

    
    const float2 d = 2.0f * make_float2(
                                (static_cast<float>(idx.x) + 0.5f) / static_cast<float>(dim.x),
                                (static_cast<float>(idx.y) + 0.5f) / static_cast<float>(dim.y)) -
                     1.0f;

    
    const float3 ray_direction = normalize(d.x * params.U + d.y * params.V + params.W);
    const float3 ray_origin = params.eye;

    
    unsigned int p0 = __float_as_uint(0.0f);
    unsigned int p1 = __float_as_uint(0.0f);
    unsigned int p2 = __float_as_uint(0.0f);

    
    optixTrace(
        params.handle,
        ray_origin,
        ray_direction,
        0.01f, 
        1e16f, 
        0.0f,  
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_NONE,
        0, 
        1, 
        0, 
        p0, p1, p2);

    
    const float3 color = make_float3(__uint_as_float(p0), __uint_as_float(p1), __uint_as_float(p2));
    const unsigned int image_index = idx.y * dim.x + idx.x;

    
    params.frame_buffer[image_index].x = static_cast<unsigned char>(fminf(255.0f, color.x * 255.0f));
    params.frame_buffer[image_index].y = static_cast<unsigned char>(fminf(255.0f, color.y * 255.0f));
    params.frame_buffer[image_index].z = static_cast<unsigned char>(fminf(255.0f, color.z * 255.0f));
    params.frame_buffer[image_index].w = 255; 
}



