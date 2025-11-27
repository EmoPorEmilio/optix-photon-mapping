

#pragma once

#include <sutil/vec_math.h>  
#include "Object.h"
#include "Material.h"

class Triangle : public Object
{
public:
    float3 v0, v1, v2; 
    int materialType = MATERIAL_DIFFUSE;

    Triangle(const float3& a, const float3& b, const float3& c) 
        : Object(), v0(a), v1(b), v2(c), materialType(MATERIAL_DIFFUSE) {}
    
    Triangle(const float3& a, const float3& b, const float3& c, const float3& col) 
        : Object(col), v0(a), v1(b), v2(c), materialType(MATERIAL_DIFFUSE) {}

    Triangle(const float3& a, const float3& b, const float3& c, const float3& col, int matType) 
        : Object(col), v0(a), v1(b), v2(c), materialType(matType) {}

    int getMaterialType() const { return materialType; }
};




