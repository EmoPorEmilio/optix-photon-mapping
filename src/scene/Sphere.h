

#pragma once

#include "Object.h"
#include <sutil/vec_math.h>

class Sphere : public Object
{
private:
    float3 center;
    float radius;

public:
    Sphere(const float3& c, float r) 
        : Object(), center(c), radius(r) {}
    
    Sphere(const float3& c, float r, const float3& col)
        : Object(col), center(c), radius(r) {}

    Sphere(const float3& c, float r, const float3& col, int materialType)
        : Object(col, materialType), center(c), radius(r) {}

    
    float3 getCenter() const { return center; }
    float getRadius() const { return radius; }

    
};




