

#pragma once

#include <sutil/vec_math.h>  
#include "Object.h"

class Triangle : public Object
{
public:
    float3 v0, v1, v2; 

    Triangle(const float3& a, const float3& b, const float3& c) 
        : Object(), v0(a), v1(b), v2(c) {}
    
    Triangle(const float3& a, const float3& b, const float3& c, const float3& col) 
        : Object(col), v0(a), v1(b), v2(c) {}

    
};




