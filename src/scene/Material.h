#pragma once

#include <sutil/vec_math.h>

// Simple material model shared between host and device.
// For now we distinguish diffuse, specular (reflective) and transmissive and a single RGB albedo.

enum MaterialType
{
    MATERIAL_DIFFUSE = 0,
    MATERIAL_SPECULAR = 1,
    MATERIAL_TRANSMISSIVE = 2,
    MATERIAL_GLASS = 2  // Alias for transmissive
};

struct Material
{
    int   type;              // One of MaterialType
    float3 albedo;           // RGB albedo / reflectance / transmittance tint
    float  diffuseProb;      // Probability of diffuse reflection (used for Russian roulette on walls)
    float  transmissiveCoeff; // Strength of transmission (used on transmissive materials)
};


