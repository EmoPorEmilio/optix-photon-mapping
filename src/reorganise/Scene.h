

#pragma once

#include <optix_types.h>  
#include <sutil/vec_math.h>  
#include <vector>
#include <memory>
#include "Object.h"
#include "Triangle.h"
#include "Sphere.h"
#include "Light.h"


struct OptixVertex
{
    float x, y, z, pad;
};

class Scene
{
private:
    std::vector<std::unique_ptr<Object>> objects;
    std::vector<std::unique_ptr<Light>> lights;

public:
    Scene();

    void addObject(std::unique_ptr<Object> obj);
    void addLight(std::unique_ptr<Light> light);
    void clear();

    
    
    std::vector<OptixVertex> exportTriangleVertices() const;
    
    
    std::vector<float3> exportTriangleColors() const;
    
    
    unsigned int getQuadLightStartIndex() const { return quadLightStartIndex; }
    unsigned int getQuadLightTriangleCount() const { return quadLightTriangleCount; }

    
    const std::vector<std::unique_ptr<Light>>& getLights() const { return lights; }

    // Access to all objects for debug/visualization (e.g., photon map rendering).
    const std::vector<std::unique_ptr<Object>>& getObjects() const { return objects; }

private:
    
    mutable unsigned int quadLightStartIndex = 0;
    mutable unsigned int quadLightTriangleCount = 0;
};




