

#pragma once

#include <optix.h>
#include <sutil/vec_math.h> 


struct SphereData
{
    float3 center;
    float radius;
};


struct Params
{
    
    uchar4 *frame_buffer; 

    
    unsigned int width;
    unsigned int height;

    
    float3 eye; 
    float3 U;   
    float3 V;   
    float3 W;   

    
    OptixTraversableHandle handle; 

    
    SphereData sphere1;
    SphereData sphere2;

    
    float3 *triangle_colors;
    float3 sphere1_color;
    float3 sphere2_color;
};


struct RayPayload
{
    float3 color; 
};





