

#pragma once

#include <sutil/vec_math.h> 

class Light
{
public:
    virtual ~Light() = default;

    
    virtual float3 getPosition() const = 0;

    
    virtual float3 getIntensity() const = 0;

    
    virtual float getArea() const { return 0.0f; }

    
    virtual bool isAreaLight() const { return false; }
};



