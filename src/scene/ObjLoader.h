#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <sutil/vec_math.h>
#include "Scene.h"
#include "Triangle.h"

// Simple Wavefront OBJ loader for triangle meshes.
// For this project we only need vertex positions and triangular faces.
class ObjLoader
{
public:
    // Loads an OBJ file and adds its triangles to the Scene as Triangle objects.
    // - path: relative or absolute path to the OBJ file.
    // - position: world-space offset applied to all vertices.
    // - scale: uniform scale applied to all vertices.
    // - color: per-triangle diffuse color.
    static bool addModelToScene(
        Scene &scene,
        const std::string &path,
        const float3 &position,
        float scale,
        const float3 &color)
    {
        struct ObjVertex
        {
            float3 pos;
        };

        auto parseFaceIndex = [](const std::string &token, int &vertexIndex) -> bool
        {
            if (token.empty())
                return false;
            std::stringstream ss(token);
            ss >> vertexIndex;
            if (ss.fail())
                return false;
            return true;
        };

        std::ifstream file(path);
        if (!file)
        {
            std::cerr << "ObjLoader: Failed to open OBJ file: " << path << std::endl;
            return false;
        }

        std::vector<ObjVertex> vertices;
        vertices.reserve(1000);

        std::string line;
        while (std::getline(file, line))
        {
            if (line.empty())
                continue;

            std::stringstream ss(line);
            std::string prefix;
            ss >> prefix;

            if (prefix == "v")
            {
                float x, y, z;
                ss >> x >> y >> z;
                vertices.push_back({make_float3(x, y, z)});
            }
            else if (prefix == "f")
            {
                std::string t0, t1, t2;
                ss >> t0 >> t1 >> t2;
                if (t0.empty() || t1.empty() || t2.empty())
                    continue;

                int i0 = 0, i1 = 0, i2 = 0;
                if (!parseFaceIndex(t0, i0) ||
                    !parseFaceIndex(t1, i1) ||
                    !parseFaceIndex(t2, i2))
                {
                    continue;
                }

                // OBJ indices are 1-based.
                i0 -= 1;
                i1 -= 1;
                i2 -= 1;

                if (i0 < 0 || i1 < 0 || i2 < 0 ||
                    i0 >= static_cast<int>(vertices.size()) ||
                    i1 >= static_cast<int>(vertices.size()) ||
                    i2 >= static_cast<int>(vertices.size()))
                {
                    continue;
                }

                float3 v0 = position + scale * vertices[i0].pos;
                float3 v1 = position + scale * vertices[i1].pos;
                float3 v2 = position + scale * vertices[i2].pos;

                scene.addObject(std::make_unique<Triangle>(v0, v1, v2, color));
            }
        }

        std::cout << "ObjLoader: loaded " << vertices.size()
                  << " vertices from " << path << " and added triangles to scene" << std::endl;

        return true;
    }
};



