

#include "Scene.h"
#include "Triangle.h" 
#include "Sphere.h"
#include "QuadLight.h"
#include <optix_types.h> 

Scene::Scene() {}

void Scene::addObject(std::unique_ptr<Object> obj)
{
    objects.push_back(std::move(obj));
}

void Scene::addLight(std::unique_ptr<Light> light)
{
    lights.push_back(std::move(light));
}

void Scene::clear()
{
    objects.clear();
    lights.clear();
}



std::vector<OptixVertex> Scene::exportTriangleVertices() const
{
    std::vector<OptixVertex> vertices;

    
    for (const auto& obj : objects)
    {
        Triangle* tri = dynamic_cast<Triangle*>(obj.get());
        if (tri)
        {
            
            vertices.push_back({tri->v0.x, tri->v0.y, tri->v0.z, 0.0f});
            vertices.push_back({tri->v1.x, tri->v1.y, tri->v1.z, 0.0f});
            vertices.push_back({tri->v2.x, tri->v2.y, tri->v2.z, 0.0f});
        }
    }

    
    quadLightStartIndex = static_cast<unsigned int>(vertices.size() / 3);
    quadLightTriangleCount = 0;

    
    for (const auto& light : lights)
    {
        if (light->isAreaLight())
        {
            const QuadLight* quadLight = dynamic_cast<const QuadLight*>(light.get());
            if (quadLight)
            {
                
                float3 v0, v1, v2, v3;
                quadLight->getVertices(v0, v1, v2, v3);

                
                vertices.push_back({v0.x, v0.y, v0.z, 0.0f});
                vertices.push_back({v1.x, v1.y, v1.z, 0.0f});
                vertices.push_back({v2.x, v2.y, v2.z, 0.0f});

                
                vertices.push_back({v0.x, v0.y, v0.z, 0.0f});
                vertices.push_back({v2.x, v2.y, v2.z, 0.0f});
                vertices.push_back({v3.x, v3.y, v3.z, 0.0f});

                quadLightTriangleCount += 2;
            }
        }
    }

    return vertices;
}

std::vector<float3> Scene::exportTriangleColors() const
{
    std::vector<float3> colors;

    
    for (const auto& obj : objects)
    {
        Triangle* tri = dynamic_cast<Triangle*>(obj.get());
        if (tri)
        {
            
            float3 color = tri->getColor();
            colors.push_back(color);
        }
    }

    
    for (const auto& light : lights)
    {
        if (light->isAreaLight())
        {
            const QuadLight* quadLight = dynamic_cast<const QuadLight*>(light.get());
            if (quadLight)
            {
                
                float3 yellow = make_float3(1.0f, 1.0f, 0.0f);
                colors.push_back(yellow); 
                colors.push_back(yellow); 
            }
        }
    }

    return colors;
}



