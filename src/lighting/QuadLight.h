

#pragma once

#include "Light.h"
#include <sutil/vec_math.h>
#include <cuda_runtime.h>

class QuadLight : public Light
{
private:
    float3 center;    
    float3 normal;    
    float3 u;         
    float3 v;         
    float halfWidth;  
    float halfHeight; 
    float3 intensity; 

public:
    
    QuadLight() = default;

    
    QuadLight(const float3 &centerPos, const float3 &normalVec,
              const float3 &uVec, float width, float height,
              const float3 &lightIntensity);

    
    float3 getPosition() const override { return center; }
    float3 getIntensity() const override { return intensity; }
    float getArea() const override { return (halfWidth * 2.0f) * (halfHeight * 2.0f); }
    bool isAreaLight() const override { return true; }

    
    void getVertices(float3 &v0, float3 &v1, float3 &v2, float3 &v3) const;

    
    float3 getNormal() const { return normal; }

    
    float3 getU() const { return u; }
    float3 getV() const { return v; }

    
    float getWidth() const { return halfWidth * 2.0f; }
    float getHeight() const { return halfHeight * 2.0f; }


    // New interface: use four independent random samples in [0,1) for
    //  - (u_pos, v_pos): position on the quad
    //  - (u_dir_1, u_dir_2): hemisphere direction sampling
    __host__ __device__
    void samplePhotonEmission(float u_pos, float v_pos,
                              float u_dir_1, float u_dir_2,
                              float3 &position, float3 &direction) const
    {
        // Map [0,1) -> [-1,1] on the local quad axes.
        float u_sample = u_pos * 2.0f - 1.0f;
        float v_sample = v_pos * 2.0f - 1.0f;

        position = center + u * u_sample * halfWidth + v * v_sample * halfHeight;

        // Cosine-weighted hemisphere around the light normal.
        float phi = 2.0f * M_PI * u_dir_1;
        float cos_theta = sqrtf(u_dir_2);
        float sin_theta = sqrtf(1.0f - u_dir_2);

        float3 local_dir = make_float3(cosf(phi) * sin_theta, cos_theta, sinf(phi) * sin_theta);
        direction = normalize(u * local_dir.x + normal * local_dir.y + v * local_dir.z);
    }

    // Backwards-compatible wrapper for legacy uint2-based RNG usage.
    __host__ __device__
    void samplePhotonEmission(uint2 rng, float3 &position, float3 &direction) const
    {
        float u_pos = static_cast<float>(rng.x) / 4294967296.0f;
        float v_pos = static_cast<float>(rng.y) / 4294967296.0f;

        // Derive two additional samples from a simple hash to decorrelate.
        unsigned int hx = rng.x * 1664525u + 1013904223u;
        unsigned int hy = rng.y * 1103515245u + 12345u;
        float u_dir_1 = static_cast<float>(hx) / 4294967296.0f;
        float u_dir_2 = static_cast<float>(hy) / 4294967296.0f;

        samplePhotonEmission(u_pos, v_pos, u_dir_1, u_dir_2, position, direction);
    }
};



