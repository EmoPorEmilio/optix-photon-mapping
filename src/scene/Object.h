

#pragma once

#include <sutil/vec_math.h>
#include "Material.h"

class Object
{
protected:
    float3 color;
    int materialType;

public:
    Object() : color(make_float3(0.8f, 0.8f, 0.8f)), materialType(MATERIAL_DIFFUSE) {}
    Object(const float3& col) : color(col), materialType(MATERIAL_DIFFUSE) {}
    Object(const float3& col, int matType) : color(col), materialType(matType) {}
    virtual ~Object() = default;

    float3 getColor() const { return color; }
    void setColor(const float3& col) { color = col; }
    int getMaterialType() const { return materialType; }
    void setMaterialType(int type) { materialType = type; }
};




