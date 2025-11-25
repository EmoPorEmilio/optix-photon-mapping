#pragma once

#include "../scene/Scene.h"
#include "../scene/Sphere.h"
#include "../scene/Triangle.h"
#include "Constants.h"
#include <sutil/vec_math.h>
#include <cmath>

// Result of a collision check with detailed hit information
struct CollisionResult
{
    bool hit;
    float3 hitPoint;
    float3 normal;
    float3 surfaceColor;
    float distance;

    // Material properties for Russian Roulette
    float diffuseProb; // Probability of diffuse reflection (vs absorption)

    CollisionResult() : hit(false),
                        hitPoint(make_float3(0.0f)),
                        normal(make_float3(0.0f, 1.0f, 0.0f)),
                        surfaceColor(make_float3(0.8f, 0.8f, 0.8f)),
                        distance(1e30f),
                        diffuseProb(0.5f) {}
};

class CollisionDetector
{
private:
    const Scene *scene;
    float collisionRadius;
    float cornellWidth;
    float cornellHeight;
    float cornellDepth;

    // Cornell box wall colors
    float3 floorColor;
    float3 ceilingColor;
    float3 leftWallColor;  // Red
    float3 rightWallColor; // Blue
    float3 backWallColor;
    float3 frontWallColor;

public:
    CollisionDetector(const Scene *s, float radius, float width, float height, float depth)
        : scene(s), collisionRadius(radius), cornellWidth(width), cornellHeight(height), cornellDepth(depth)
    {
        // Standard Cornell box colors
        floorColor = make_float3(Constants::Cornell::WHITE_R, Constants::Cornell::WHITE_G, Constants::Cornell::WHITE_B);
        ceilingColor = make_float3(Constants::Cornell::WHITE_R, Constants::Cornell::WHITE_G, Constants::Cornell::WHITE_B);
        leftWallColor = make_float3(Constants::Cornell::RED_R, Constants::Cornell::RED_G, Constants::Cornell::RED_B);
        rightWallColor = make_float3(Constants::Cornell::BLUE_R, Constants::Cornell::BLUE_G, Constants::Cornell::BLUE_B);
        backWallColor = make_float3(Constants::Cornell::WHITE_R, Constants::Cornell::WHITE_G, Constants::Cornell::WHITE_B);
        frontWallColor = make_float3(Constants::Cornell::WHITE_R, Constants::Cornell::WHITE_G, Constants::Cornell::WHITE_B);
    }

    // Simple collision check (legacy)
    bool checkCollision(const float3 &position) const
    {
        CollisionResult result = raycast(position - make_float3(0.0f, 1.0f, 0.0f), make_float3(0.0f, 1.0f, 0.0f), 2.0f);
        return result.hit;
    }

    // Full raycast with hit information for bouncing
    CollisionResult raycast(const float3 &origin, const float3 &direction, float maxDist = 1e30f) const
    {
        CollisionResult result;
        result.hit = false;
        result.distance = maxDist;

        float3 dir = normalize(direction);

        // Check Cornell Box walls
        checkWallRaycast(origin, dir, result);

        // Check scene objects (spheres and triangles)
        if (scene)
        {
            for (const auto &obj : scene->getObjects())
            {
                if (auto sphere = dynamic_cast<Sphere *>(obj.get()))
                {
                    checkSphereRaycast(origin, dir, sphere, result);
                }
                else if (auto tri = dynamic_cast<Triangle *>(obj.get()))
                {
                    checkTriangleRaycast(origin, dir, tri, result);
                }
            }
        }

        return result;
    }

    // Check if a position is inside any surface (for step-based collision)
    CollisionResult checkPositionCollision(const float3 &position) const
    {
        CollisionResult result;
        result.hit = false;
        const float epsilon = collisionRadius;

        // Floor (y = 0)
        if (position.y <= epsilon)
        {
            result.hit = true;
            result.hitPoint = make_float3(position.x, 0.0f, position.z);
            result.normal = make_float3(0.0f, 1.0f, 0.0f);
            result.surfaceColor = floorColor;
            result.diffuseProb = 0.7f;
            return result;
        }

        // Left wall (x = 0) - RED
        if (position.x <= epsilon)
        {
            result.hit = true;
            result.hitPoint = make_float3(0.0f, position.y, position.z);
            result.normal = make_float3(1.0f, 0.0f, 0.0f);
            result.surfaceColor = leftWallColor;
            result.diffuseProb = 0.7f;
            return result;
        }

        // Right wall (x = cornellWidth) - BLUE
        if (position.x >= cornellWidth - epsilon)
        {
            result.hit = true;
            result.hitPoint = make_float3(cornellWidth, position.y, position.z);
            result.normal = make_float3(-1.0f, 0.0f, 0.0f);
            result.surfaceColor = rightWallColor;
            result.diffuseProb = 0.7f;
            return result;
        }

        // Front wall (z = 0)
        if (position.z <= epsilon)
        {
            result.hit = true;
            result.hitPoint = make_float3(position.x, position.y, 0.0f);
            result.normal = make_float3(0.0f, 0.0f, 1.0f);
            result.surfaceColor = frontWallColor;
            result.diffuseProb = 0.7f;
            return result;
        }

        // Back wall (z = cornellDepth)
        if (position.z >= cornellDepth - epsilon)
        {
            result.hit = true;
            result.hitPoint = make_float3(position.x, position.y, cornellDepth);
            result.normal = make_float3(0.0f, 0.0f, -1.0f);
            result.surfaceColor = backWallColor;
            result.diffuseProb = 0.7f;
            return result;
        }

        // Check spheres
        if (scene)
        {
            for (const auto &obj : scene->getObjects())
            {
                if (auto sphere = dynamic_cast<Sphere *>(obj.get()))
                {
                    float3 center = sphere->getCenter();
                    float radius = sphere->getRadius();
                    float dist = length(position - center);
                    if (dist <= radius + collisionRadius)
                    {
                        result.hit = true;
                        result.normal = normalize(position - center);
                        result.hitPoint = center + result.normal * radius;
                        result.surfaceColor = sphere->getColor();
                        result.diffuseProb = 0.5f;
                        return result;
                    }
                }
            }
        }

        return result;
    }

private:
    void checkWallRaycast(const float3 &origin, const float3 &dir, CollisionResult &result) const
    {
        const float eps = 0.001f;

        // Floor (y = 0)
        if (dir.y < -eps)
        {
            float t = -origin.y / dir.y;
            if (t > eps && t < result.distance)
            {
                float3 hitP = origin + dir * t;
                if (hitP.x >= 0 && hitP.x <= cornellWidth && hitP.z >= 0 && hitP.z <= cornellDepth)
                {
                    result.hit = true;
                    result.distance = t;
                    result.hitPoint = hitP;
                    result.normal = make_float3(0.0f, 1.0f, 0.0f);
                    result.surfaceColor = floorColor;
                    result.diffuseProb = 0.7f;
                }
            }
        }

        // Ceiling (y = cornellHeight)
        if (dir.y > eps)
        {
            float t = (cornellHeight - origin.y) / dir.y;
            if (t > eps && t < result.distance)
            {
                float3 hitP = origin + dir * t;
                if (hitP.x >= 0 && hitP.x <= cornellWidth && hitP.z >= 0 && hitP.z <= cornellDepth)
                {
                    result.hit = true;
                    result.distance = t;
                    result.hitPoint = hitP;
                    result.normal = make_float3(0.0f, -1.0f, 0.0f);
                    result.surfaceColor = ceilingColor;
                    result.diffuseProb = 0.7f;
                }
            }
        }

        // Left wall (x = 0) - RED
        if (dir.x < -eps)
        {
            float t = -origin.x / dir.x;
            if (t > eps && t < result.distance)
            {
                float3 hitP = origin + dir * t;
                if (hitP.y >= 0 && hitP.y <= cornellHeight && hitP.z >= 0 && hitP.z <= cornellDepth)
                {
                    result.hit = true;
                    result.distance = t;
                    result.hitPoint = hitP;
                    result.normal = make_float3(1.0f, 0.0f, 0.0f);
                    result.surfaceColor = leftWallColor;
                    result.diffuseProb = 0.7f;
                }
            }
        }

        // Right wall (x = cornellWidth) - BLUE
        if (dir.x > eps)
        {
            float t = (cornellWidth - origin.x) / dir.x;
            if (t > eps && t < result.distance)
            {
                float3 hitP = origin + dir * t;
                if (hitP.y >= 0 && hitP.y <= cornellHeight && hitP.z >= 0 && hitP.z <= cornellDepth)
                {
                    result.hit = true;
                    result.distance = t;
                    result.hitPoint = hitP;
                    result.normal = make_float3(-1.0f, 0.0f, 0.0f);
                    result.surfaceColor = rightWallColor;
                    result.diffuseProb = 0.7f;
                }
            }
        }

        // Front wall (z = 0)
        if (dir.z < -eps)
        {
            float t = -origin.z / dir.z;
            if (t > eps && t < result.distance)
            {
                float3 hitP = origin + dir * t;
                if (hitP.x >= 0 && hitP.x <= cornellWidth && hitP.y >= 0 && hitP.y <= cornellHeight)
                {
                    result.hit = true;
                    result.distance = t;
                    result.hitPoint = hitP;
                    result.normal = make_float3(0.0f, 0.0f, 1.0f);
                    result.surfaceColor = frontWallColor;
                    result.diffuseProb = 0.7f;
                }
            }
        }

        // Back wall (z = cornellDepth)
        if (dir.z > eps)
        {
            float t = (cornellDepth - origin.z) / dir.z;
            if (t > eps && t < result.distance)
            {
                float3 hitP = origin + dir * t;
                if (hitP.x >= 0 && hitP.x <= cornellWidth && hitP.y >= 0 && hitP.y <= cornellHeight)
                {
                    result.hit = true;
                    result.distance = t;
                    result.hitPoint = hitP;
                    result.normal = make_float3(0.0f, 0.0f, -1.0f);
                    result.surfaceColor = backWallColor;
                    result.diffuseProb = 0.7f;
                }
            }
        }
    }

    void checkSphereRaycast(const float3 &origin, const float3 &dir, const Sphere *sphere, CollisionResult &result) const
    {
        float3 center = sphere->getCenter();
        float radius = sphere->getRadius();

        float3 oc = origin - center;
        float a = dot(dir, dir);
        float b = 2.0f * dot(oc, dir);
        float c = dot(oc, oc) - radius * radius;
        float discriminant = b * b - 4.0f * a * c;

        if (discriminant >= 0.0f)
        {
            float t = (-b - sqrtf(discriminant)) / (2.0f * a);
            if (t > 0.001f && t < result.distance)
            {
                result.hit = true;
                result.distance = t;
                result.hitPoint = origin + dir * t;
                result.normal = normalize(result.hitPoint - center);
                result.surfaceColor = sphere->getColor();
                result.diffuseProb = 0.5f;
            }
        }
    }

    void checkTriangleRaycast(const float3 &origin, const float3 &dir, const Triangle *tri, CollisionResult &result) const
    {
        // Möller–Trumbore intersection algorithm
        const float eps = 1e-6f;

        float3 edge1 = tri->v1 - tri->v0;
        float3 edge2 = tri->v2 - tri->v0;
        float3 h = cross(dir, edge2);
        float a = dot(edge1, h);

        if (fabsf(a) < eps)
            return; // Parallel

        float f = 1.0f / a;
        float3 s = origin - tri->v0;
        float u = f * dot(s, h);
        if (u < 0.0f || u > 1.0f)
            return;

        float3 q = cross(s, edge1);
        float v = f * dot(dir, q);
        if (v < 0.0f || u + v > 1.0f)
            return;

        float t = f * dot(edge2, q);
        if (t > eps && t < result.distance)
        {
            result.hit = true;
            result.distance = t;
            result.hitPoint = origin + dir * t;
            result.normal = normalize(cross(edge1, edge2));
            // Flip normal to face the ray
            if (dot(result.normal, dir) > 0)
            {
                result.normal = -result.normal;
            }
            result.surfaceColor = tri->getColor();
            result.diffuseProb = 0.7f;
        }
    }
};
