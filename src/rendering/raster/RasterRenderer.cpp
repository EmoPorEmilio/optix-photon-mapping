#include "RasterRenderer.h"
#include "../../core/Application.h"
#include "../../scene/Triangle.h"
#include "../../scene/Sphere.h"
#include "../../lighting/QuadLight.h"
#include <iostream>
#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

RasterRenderer::RasterRenderer(Window *w, const ViewportRect &vp)
    : window(w), viewport(vp)
{
}

RasterRenderer::~RasterRenderer()
{
    if (shaderProgram) glDeleteProgram(shaderProgram);
    if (VAO) glDeleteVertexArrays(1, &VAO);
    if (VBO) glDeleteBuffers(1, &VBO);
    if (photonVAO) glDeleteVertexArrays(1, &photonVAO);
    if (photonVBO) glDeleteBuffers(1, &photonVBO);
}

void RasterRenderer::setCamera(Camera *cam)
{
    camera = cam;
}

void RasterRenderer::setScene(Scene *s)
{
    scene = s;
    buildSceneGeometry();
}

void RasterRenderer::setAnimatedPhotons(const std::vector<AnimatedPhoton>& p)
{
    photons = p;
}

void RasterRenderer::createShaderProgram()
{
    const char *vertexShaderSource = R"(
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec3 aColor;

        out vec3 FragPos;
        out vec3 Color;

        uniform mat4 view;
        uniform mat4 projection;

        void main()
        {
            gl_Position = projection * view * vec4(aPos, 1.0);
            FragPos = aPos;
            Color = aColor;
        }
    )";

    const char *fragmentShaderSource = R"(
        #version 330 core
        out vec4 FragColor;
        in vec3 Color;

        void main()
        {
            FragColor = vec4(Color, 1.0);
        }
    )";

    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
    glCompileShader(vertexShader);
    // Check errors...

    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fragmentShader);
    // Check errors...

    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

void RasterRenderer::buildSceneGeometry()
{
    if (!scene) return;

    std::vector<float> bufferData; // x, y, z, r, g, b

    auto addTriangle = [&](const float3& v0, const float3& v1, const float3& v2, const float3& color) {
        bufferData.push_back(v0.x); bufferData.push_back(v0.y); bufferData.push_back(v0.z);
        bufferData.push_back(color.x); bufferData.push_back(color.y); bufferData.push_back(color.z);

        bufferData.push_back(v1.x); bufferData.push_back(v1.y); bufferData.push_back(v1.z);
        bufferData.push_back(color.x); bufferData.push_back(color.y); bufferData.push_back(color.z);

        bufferData.push_back(v2.x); bufferData.push_back(v2.y); bufferData.push_back(v2.z);
        bufferData.push_back(color.x); bufferData.push_back(color.y); bufferData.push_back(color.z);
    };

    std::cout << "Building Raster Geometry..." << std::endl;
    int objCount = 0;

    for (const auto& obj : scene->getObjects())
    {
        objCount++;
        if (auto tri = dynamic_cast<Triangle*>(obj.get()))
        {
            addTriangle(tri->v0, tri->v1, tri->v2, tri->getColor());
        }
        else if (auto sph = dynamic_cast<Sphere*>(obj.get()))
        {
            float3 center = sph->getCenter();
            float radius = sph->getRadius();
            float3 color = sph->getColor();

            const int stacks = 20;
            const int slices = 20;

            for (int i = 0; i < stacks; ++i)
            {
                float phi1 = M_PI * float(i) / float(stacks);
                float phi2 = M_PI * float(i + 1) / float(stacks);

                for (int j = 0; j < slices; ++j)
                {
                    float theta1 = 2.0f * M_PI * float(j) / float(slices);
                    float theta2 = 2.0f * M_PI * float(j + 1) / float(slices);

                    auto getPos = [&](float phi, float theta) {
                        return make_float3(
                            center.x + radius * sin(phi) * cos(theta),
                            center.y + radius * cos(phi),
                            center.z + radius * sin(phi) * sin(theta)
                        );
                    };

                    float3 p1 = getPos(phi1, theta1);
                    float3 p2 = getPos(phi2, theta1);
                    float3 p3 = getPos(phi2, theta2);
                    float3 p4 = getPos(phi1, theta2);

                    addTriangle(p1, p2, p3, color);
                    addTriangle(p1, p3, p4, color);
                }
            }
        }
    }

    // Add lights
    for (const auto& light : scene->getLights())
    {
        if (auto ql = dynamic_cast<QuadLight*>(light.get()))
        {
            float3 v0, v1, v2, v3;
            ql->getVertices(v0, v1, v2, v3);
            float3 color = make_float3(1.0f, 1.0f, 1.0f); // White for lights
            addTriangle(v0, v1, v2, color);
            addTriangle(v0, v2, v3, color);
        }
    }

    vertexCount = bufferData.size() / 6;
    std::cout << "RasterRenderer: Generated " << vertexCount << " vertices from " << objCount << " objects." << std::endl;

    if (VAO == 0) glGenVertexArrays(1, &VAO);
    if (VBO == 0) glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, bufferData.size() * sizeof(float), bufferData.data(), GL_STATIC_DRAW);

    // Position
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Color
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
}

void RasterRenderer::renderPhotons()
{
    if (photons.empty())
        return;

    // Generate sphere geometry for each photon
    std::vector<float> photonBufferData; // x, y, z, r, g, b
    const float photonRadius = 8.0f; // Visible yellow sphere
    const float3 photonColor = make_float3(1.0f, 1.0f, 0.0f); // Yellow
    const int stacks = 10;
    const int slices = 10;

    auto addTriangle = [&](const float3& v0, const float3& v1, const float3& v2, const float3& color) {
        photonBufferData.push_back(v0.x); photonBufferData.push_back(v0.y); photonBufferData.push_back(v0.z);
        photonBufferData.push_back(color.x); photonBufferData.push_back(color.y); photonBufferData.push_back(color.z);

        photonBufferData.push_back(v1.x); photonBufferData.push_back(v1.y); photonBufferData.push_back(v1.z);
        photonBufferData.push_back(color.x); photonBufferData.push_back(color.y); photonBufferData.push_back(color.z);

        photonBufferData.push_back(v2.x); photonBufferData.push_back(v2.y); photonBufferData.push_back(v2.z);
        photonBufferData.push_back(color.x); photonBufferData.push_back(color.y); photonBufferData.push_back(color.z);
    };

    for (const auto& photon : photons)
    {
        float3 center = photon.position;
        float radius = photonRadius;

        for (int i = 0; i < stacks; ++i)
        {
            float phi1 = M_PI * float(i) / float(stacks);
            float phi2 = M_PI * float(i + 1) / float(stacks);

            for (int j = 0; j < slices; ++j)
            {
                float theta1 = 2.0f * M_PI * float(j) / float(slices);
                float theta2 = 2.0f * M_PI * float(j + 1) / float(slices);

                auto getPos = [&](float phi, float theta) {
                    return make_float3(
                        center.x + radius * sin(phi) * cos(theta),
                        center.y + radius * cos(phi),
                        center.z + radius * sin(phi) * sin(theta)
                    );
                };

                float3 p1 = getPos(phi1, theta1);
                float3 p2 = getPos(phi2, theta1);
                float3 p3 = getPos(phi2, theta2);
                float3 p4 = getPos(phi1, theta2);

                addTriangle(p1, p2, p3, photonColor);
                addTriangle(p1, p3, p4, photonColor);
            }
        }
    }

    if (photonBufferData.empty())
        return;

    if (photonVAO == 0) glGenVertexArrays(1, &photonVAO);
    if (photonVBO == 0) glGenBuffers(1, &photonVBO);

    glBindVertexArray(photonVAO);
    glBindBuffer(GL_ARRAY_BUFFER, photonVBO);
    glBufferData(GL_ARRAY_BUFFER, photonBufferData.size() * sizeof(float), photonBufferData.data(), GL_DYNAMIC_DRAW);

    // Position
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Color
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    int photonVertexCount = photonBufferData.size() / 6;
    glDrawArrays(GL_TRIANGLES, 0, photonVertexCount);

    glBindVertexArray(0);
}

void RasterRenderer::renderFrame()
{
    if (!window || !camera || !scene) return;

    window->makeCurrent();

    if (shaderProgram == 0) createShaderProgram();

    glViewport(viewport.x, viewport.y, viewport.width, viewport.height);
    glScissor(viewport.x, viewport.y, viewport.width, viewport.height);
    glEnable(GL_SCISSOR_TEST);
    
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE); // Ensure we see both sides of triangles

    glUseProgram(shaderProgram);

    // View Matrix
    float3 eye = camera->getPosition();
    float3 center = camera->getLookAt();
    float3 up = camera->up;

    static int frameCount = 0;
    if (frameCount++ % 100 == 0) {
        std::cout << "Camera Eye: " << eye.x << ", " << eye.y << ", " << eye.z << std::endl;
        std::cout << "Camera LookAt: " << center.x << ", " << center.y << ", " << center.z << std::endl;
    }

    // Manual LookAt Matrix Construction
    // Z axis: direction from target to camera (backwards)
    float3 f = normalize(eye - center); 
    // X axis: right vector
    float3 s = normalize(cross(up, f));
    // Y axis: up vector
    float3 u = cross(f, s);

    // Debug print once
    static bool printed = false;
    if (!printed) {
        std::cout << "RasterRenderer Debug:" << std::endl;
        std::cout << "Eye: " << eye.x << ", " << eye.y << ", " << eye.z << std::endl;
        std::cout << "Center: " << center.x << ", " << center.y << ", " << center.z << std::endl;
        std::cout << "Up: " << up.x << ", " << up.y << ", " << up.z << std::endl;
        std::cout << "F (forward): " << f.x << ", " << f.y << ", " << f.z << std::endl;
        std::cout << "S (right): " << s.x << ", " << s.y << ", " << s.z << std::endl;
        std::cout << "U (up): " << u.x << ", " << u.y << ", " << u.z << std::endl;
        std::cout << "Vertex Count: " << vertexCount << std::endl;
        printed = true;
    }

    // Standard LookAt Matrix (Row-Major for C++, but OpenGL expects Column-Major)
    // We construct it in Row-Major order here, but pass GL_TRUE to transpose it, 
    // OR construct it in Column-Major order and pass GL_FALSE.
    
    // Let's try constructing it in Column-Major order directly for GL_FALSE.
    // [ s.x  u.x  f.x  0 ]
    // [ s.y  u.y  f.y  0 ]
    // [ s.z  u.z  f.z  0 ]
    // [ -dot(s,eye) -dot(u,eye) -dot(f,eye) 1 ]

    float view[16] = {
        s.x, u.x, f.x, 0.0f,
        s.y, u.y, f.y, 0.0f,
        s.z, u.z, f.z, 0.0f,
        -dot(s, eye), -dot(u, eye), -dot(f, eye), 1.0f
    };

    // Projection Matrix
    float fov = camera->fov; // Camera stores FOV in radians
    float aspect = camera->getAspectRatio();
    float nearPlane = 0.1f;
    float farPlane = 5000.0f;
    float tanHalfFov = tan(fov / 2.0f);

    // Standard OpenGL Perspective Matrix (Column-Major)
    // [ 1/(aspect*tan)      0             0           0 ]
    // [       0          1/tan            0           0 ]
    // [       0             0     -(f+n)/(f-n)  -2fn/(f-n) ]
    // [       0             0            -1           0 ]
    
    // BUT: We are passing GL_FALSE to glUniformMatrix4fv, so we need to provide data in COLUMN-MAJOR order.
    // Array layout:
    // 0  4  8  12
    // 1  5  9  13
    // 2  6  10 14
    // 3  7  11 15

    float proj[16] = {
        1.0f / (aspect * tanHalfFov), 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f / tanHalfFov, 0.0f, 0.0f,
        0.0f, 0.0f, -(farPlane + nearPlane) / (farPlane - nearPlane), -1.0f,
        0.0f, 0.0f, -(2.0f * farPlane * nearPlane) / (farPlane - nearPlane), 0.0f
    };

    int viewLoc = glGetUniformLocation(shaderProgram, "view");
    int projLoc = glGetUniformLocation(shaderProgram, "projection");

    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, view);
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, proj);

    glBindVertexArray(VAO);
    glDrawArrays(GL_TRIANGLES, 0, vertexCount);
    glBindVertexArray(0);

    // Render animated photons
    renderPhotons();

    glDisable(GL_SCISSOR_TEST);
}
