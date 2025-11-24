

#include "QuadLight.h"
#include <sutil/vec_math.h>
#include <algorithm> 

QuadLight::QuadLight(const float3 &centerPos, const float3 &normalVec,
                     const float3 &uVec, float width, float height,
                     const float3 &lightIntensity)
    : center(centerPos), normal(normalize(normalVec)), intensity(lightIntensity)
{
    
    halfWidth = width * 0.5f;
    halfHeight = height * 0.5f;

    
    
    u = normalize(uVec);
    v = normalize(cross(normal, u));

    
    if (dot(cross(normal, u), v) < 0.0f)
    {
        v = -v;
    }
}

void QuadLight::getVertices(float3 &v0, float3 &v1, float3 &v2, float3 &v3) const
{
    
    float3 halfU = u * halfWidth;
    float3 halfV = v * halfHeight;

    v0 = center - halfU - halfV; 
    v1 = center + halfU - halfV; 
    v2 = center + halfU + halfV; 
    v3 = center - halfU + halfV; 
}



