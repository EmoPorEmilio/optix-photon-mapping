#include "ObjLoader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <array>

ObjLoader::ObjData ObjLoader::parseObjFile(const std::string& path)
{
    ObjData data;
    std::ifstream file(path);
    
    if (!file.is_open())
    {
        std::cerr << "ObjLoader: Failed to open file: " << path << std::endl;
        return data;
    }

    std::string line;
    while (std::getline(file, line))
    {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#')
            continue;

        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;

        if (prefix == "v")
        {
            // Vertex position
            float x, y, z;
            iss >> x >> y >> z;
            data.vertices.push_back(make_float3(x, y, z));
        }
        else if (prefix == "f")
        {
            // Face - can be "f v1 v2 v3" or "f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3"
            std::vector<int> faceIndices;
            std::string token;
            
            while (iss >> token)
            {
                // Parse the vertex index (first number before any /)
                int vertexIndex = 0;
                size_t slashPos = token.find('/');
                if (slashPos != std::string::npos)
                {
                    vertexIndex = std::stoi(token.substr(0, slashPos));
                }
                else
                {
                    vertexIndex = std::stoi(token);
                }
                
                // OBJ indices are 1-based, convert to 0-based
                faceIndices.push_back(vertexIndex - 1);
            }

            // Triangulate the face (fan triangulation for polygons)
            for (size_t i = 1; i + 1 < faceIndices.size(); ++i)
            {
                std::array<int, 3> tri = {faceIndices[0], faceIndices[i], faceIndices[i + 1]};
                data.faces.push_back(tri);
            }
        }
    }

    std::cout << "ObjLoader: Loaded " << path << " - " 
              << data.vertices.size() << " vertices, " 
              << data.faces.size() << " triangles" << std::endl;

    return data;
}

std::vector<std::unique_ptr<Triangle>> ObjLoader::load(const MeshConfig& config)
{
    return load(config.path, config.position, config.scale, config.color);
}

std::vector<std::unique_ptr<Triangle>> ObjLoader::load(
    const std::string& path,
    const float3& position,
    float scale,
    const float3& color)
{
    std::vector<std::unique_ptr<Triangle>> triangles;
    
    ObjData data = parseObjFile(path);
    
    if (data.vertices.empty() || data.faces.empty())
    {
        std::cerr << "ObjLoader: No geometry loaded from " << path << std::endl;
        return triangles;
    }

    // Calculate bounding box to center the mesh
    float3 minBounds = data.vertices[0];
    float3 maxBounds = data.vertices[0];
    
    for (const auto& v : data.vertices)
    {
        minBounds.x = std::min(minBounds.x, v.x);
        minBounds.y = std::min(minBounds.y, v.y);
        minBounds.z = std::min(minBounds.z, v.z);
        maxBounds.x = std::max(maxBounds.x, v.x);
        maxBounds.y = std::max(maxBounds.y, v.y);
        maxBounds.z = std::max(maxBounds.z, v.z);
    }

    float3 center = (minBounds + maxBounds) * 0.5f;
    float3 size = maxBounds - minBounds;
    
    std::cout << "ObjLoader: Mesh bounds: min(" << minBounds.x << "," << minBounds.y << "," << minBounds.z 
              << ") max(" << maxBounds.x << "," << maxBounds.y << "," << maxBounds.z << ")" << std::endl;
    std::cout << "ObjLoader: Mesh size: (" << size.x << "," << size.y << "," << size.z << ")" << std::endl;

    // Transform vertices: center at origin, scale, then translate to position
    // Position.y = 0 means the bottom of the mesh sits on y=0
    std::vector<float3> transformedVertices;
    transformedVertices.reserve(data.vertices.size());
    
    for (const auto& v : data.vertices)
    {
        float3 transformed;
        // Center horizontally, but keep bottom at y=0
        transformed.x = (v.x - center.x) * scale + position.x;
        transformed.y = (v.y - minBounds.y) * scale + position.y;  // Bottom at position.y
        transformed.z = (v.z - center.z) * scale + position.z;
        transformedVertices.push_back(transformed);
    }

    // Create triangles
    triangles.reserve(data.faces.size());
    for (const auto& face : data.faces)
    {
        if (face[0] >= 0 && face[0] < static_cast<int>(transformedVertices.size()) &&
            face[1] >= 0 && face[1] < static_cast<int>(transformedVertices.size()) &&
            face[2] >= 0 && face[2] < static_cast<int>(transformedVertices.size()))
        {
            triangles.push_back(std::make_unique<Triangle>(
                transformedVertices[face[0]],
                transformedVertices[face[1]],
                transformedVertices[face[2]],
                color
            ));
        }
    }

    std::cout << "ObjLoader: Created " << triangles.size() << " triangles at position ("
              << position.x << "," << position.y << "," << position.z << ") with scale " << scale << std::endl;

    return triangles;
}
