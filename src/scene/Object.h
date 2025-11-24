

#pragma once

#include <sutil/vec_math.h>

class Object
{
protected:
    float3 color;

public:
    Object() : color(make_float3(0.8f, 0.8f, 0.8f)) {}
    Object(const float3& col) : color(col) {}
    virtual ~Object() = default;
    
    float3 getColor() const { return color; }
    void setColor(const float3& col) { color = col; }
};




