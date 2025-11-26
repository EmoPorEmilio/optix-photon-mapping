#pragma once

#include <string>
#include <vector>
#include <memory>
#include <array>
#include <sutil/vec_math.h>
#include "Triangle.h"

// Configuration for loading an OBJ mesh
struct MeshConfig
{
    std::string path;
    float3 position = make_float3(0.0f, 0.0f, 0.0f);
    float scale = 1.0f;
    float3 color = make_float3(0.8f, 0.8f, 0.8f);
    std::string materialType = "diffuse";  // diffuse, specular, transmissive
    float ior = 1.5f;  // For transmissive materials
};

// Simple OBJ file loader
class ObjLoader
{
public:
    // Load an OBJ file and return a vector of triangles
    // Applies position offset and uniform scale
    static std::vector<std::unique_ptr<Triangle>> load(const MeshConfig& config);

    // Load with explicit parameters
    static std::vector<std::unique_ptr<Triangle>> load(
        const std::string& path,
        const float3& position,
        float scale,
        const float3& color);

private:
    struct ObjData
    {
        std::vector<float3> vertices;
        std::vector<std::array<int, 3>> faces;  // Triangle indices (0-based)
    };

    static ObjData parseObjFile(const std::string& path);
};
