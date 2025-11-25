

#pragma once

#include <GLFW/glfw3.h>
#include <glad/glad.h>
#include "../../ui/WindowManager.h"
#include "../../scene/Camera.h"
#include "Photon.h"
#include "../../scene/Scene.h"
#include <vector>
#include <sutil/vec_math.h>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

class PhotonMapRenderer
{
private:
    Window *window = nullptr;
    const ViewportRect *viewport = nullptr;
    const Camera *camera = nullptr;
    const Scene *scene = nullptr;
    // Photon data copied from OptiX photon buffer (positions + power/color).
    std::vector<float3> photonPositions;
    std::vector<float3> photonPowers;
    size_t currentPhotonCount = 0;

    struct DebugTri
    {
        float3 v0, v1, v2;
    };
    struct DebugSphere
    {
        float3 center;
        float radius;
    };
    std::vector<DebugTri> debugTriangles;
    std::vector<DebugSphere> debugSpheres;

    std::vector<unsigned char> pixelBuffer;
    unsigned int textureID = 0;
    unsigned int textureWidth = 0;
    unsigned int textureHeight = 0;

    unsigned int quadVAO = 0;
    unsigned int quadVBO = 0;
    unsigned int quadEBO = 0;
    unsigned int shaderProgram = 0;

    unsigned int compileShader(const char *source, unsigned int type)
    {
        unsigned int shader = glCreateShader(type);
        glShaderSource(shader, 1, &source, nullptr);
        glCompileShader(shader);

        int success;
        char infoLog[512];
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success)
        {
            glGetShaderInfoLog(shader, 512, nullptr, infoLog);
            std::cerr << "Shader compilation error: " << infoLog << std::endl;
        }

        return shader;
    }

    void createShaderProgram()
    {
        const char *vertexShaderSource = R"(
            #version 330 core
            layout (location = 0) in vec2 aPos;
            layout (location = 1) in vec2 aTexCoord;
            out vec2 TexCoord;
            void main()
            {
                TexCoord = aTexCoord;
                gl_Position = vec4(aPos, 0.0, 1.0);
            }
        )";

        const char *fragmentShaderSource = R"(
            #version 330 core
            in vec2 TexCoord;
            out vec4 FragColor;
            uniform sampler2D uTex;
            void main()
            {
                FragColor = texture(uTex, TexCoord);
            }
        )";

        unsigned int vertexShader = compileShader(vertexShaderSource, GL_VERTEX_SHADER);
        unsigned int fragmentShader = compileShader(fragmentShaderSource, GL_FRAGMENT_SHADER);

        shaderProgram = glCreateProgram();
        glAttachShader(shaderProgram, vertexShader);
        glAttachShader(shaderProgram, fragmentShader);
        glLinkProgram(shaderProgram);

        int success;
        char infoLog[512];
        glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
        if (!success)
        {
            glGetProgramInfoLog(shaderProgram, 512, nullptr, infoLog);
            std::cerr << "Shader linking error: " << infoLog << std::endl;
        }

        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
    }

public:
    PhotonMapRenderer() = default;
    ~PhotonMapRenderer()
    {
        if (quadVAO)
            glDeleteVertexArrays(1, &quadVAO);
        if (quadVBO)
            glDeleteBuffers(1, &quadVBO);
        if (quadEBO)
            glDeleteBuffers(1, &quadEBO);
        if (textureID)
            glDeleteTextures(1, &textureID);
        if (shaderProgram)
            glDeleteProgram(shaderProgram);
    }

    bool initialize(Window *win)
    {
        window = win;
        if (!window)
            return false;

        window->makeCurrent();

        createShaderProgram();

        glGenTextures(1, &textureID);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        float quadVertices[] = {

            -1.0f, -1.0f, 0.0f, 0.0f,
            1.0f, -1.0f, 1.0f, 0.0f,
            1.0f, 1.0f, 1.0f, 1.0f,
            -1.0f, 1.0f, 0.0f, 1.0f};
        unsigned int indices[] = {0, 1, 2, 2, 3, 0};

        glGenVertexArrays(1, &quadVAO);
        glGenBuffers(1, &quadVBO);
        glGenBuffers(1, &quadEBO);

        glBindVertexArray(quadVAO);
        glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, quadEBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)(2 * sizeof(float)));
        glEnableVertexAttribArray(1);
        glBindVertexArray(0);

        return true;
    }

    void setViewport(const ViewportRect &vp) { viewport = &vp; }
    void setCamera(const Camera *cam) { camera = cam; }

    void setScene(const Scene *sc)
    {
        scene = sc;
        debugTriangles.clear();
        debugSpheres.clear();

        if (!scene)
            return;

        const auto &objs = scene->getObjects();
        for (const auto &objPtr : objs)
        {
            const Triangle *tri = dynamic_cast<const Triangle *>(objPtr.get());
            if (tri)
            {
                // Triangles are ignored for occlusion in the photon debug view.
                continue;
            }

            const Sphere *sph = dynamic_cast<const Sphere *>(objPtr.get());
            if (sph)
            {
                DebugSphere ds;
                ds.center = sph->getCenter();
                ds.radius = sph->getRadius();
                debugSpheres.push_back(ds);
                continue;
            }
        }
    }

    void uploadFromHost(const Photon *photons, size_t count)
    {
        if (!window || count == 0 || count > 1000000)
            return;

        window->makeCurrent();

        photonPositions.resize(count);
        photonPowers.resize(count);
        for (size_t i = 0; i < count; ++i)
        {
            photonPositions[i] = photons[i].position;
            photonPowers[i] = photons[i].power;
        }

        currentPhotonCount = count;
    }

    void render()
    {
        if (!window || !viewport)
            return;

        static int frameCount = 0;
        if (frameCount++ % 60 == 0 && currentPhotonCount > 0)
        {
            std::cout << "PhotonMapRenderer: rendering " << currentPhotonCount << " photons" << std::endl;
        }

        window->makeCurrent();

        glViewport(viewport->x, viewport->y, viewport->width, viewport->height);
        glScissor(viewport->x, viewport->y, viewport->width, viewport->height);
        glEnable(GL_SCISSOR_TEST);
        glDisable(GL_DEPTH_TEST);
        glClearColor(0.02f, 0.02f, 0.02f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // If no photons yet, just show cleared screen
        if (currentPhotonCount == 0)
        {
            glDisable(GL_SCISSOR_TEST);
            return;
        }

        if (textureWidth != (unsigned int)viewport->width || textureHeight != (unsigned int)viewport->height)
        {
            textureWidth = viewport->width;
            textureHeight = viewport->height;
            pixelBuffer.assign(textureWidth * textureHeight * 3, 0);
            glBindTexture(GL_TEXTURE_2D, textureID);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, textureWidth, textureHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, pixelBuffer.data());
        }
        else
        {
            std::fill(pixelBuffer.begin(), pixelBuffer.end(), 0);
        }

        float3 eye = camera->getPosition();
        float3 lookAt = camera->getLookAt();

        float3 w = normalize(lookAt - eye);
        float3 u_basis = normalize(make_float3(w.z, 0.0f, -w.x));
        float3 v_basis = normalize(cross(w, u_basis));

        float3 f = w;
        float3 s = u_basis;
        float3 u = v_basis;

        float3 U_exact = camera->getU();
        float3 V_exact = camera->getV();
        float3 W_exact = camera->getW();

        float wlen = length(camera->getLookAt() - eye);
        float vlen = length(V_exact);
        float ulen = length(U_exact);

        for (size_t i = 0; i < currentPhotonCount; ++i)
        {
            float3 p = photonPositions[i];
            float3 d = p - eye;

            float dw = dot(d, f);
            if (dw <= 0.0f)
                continue;

            // Optional occlusion test: if the ray from eye to photon hits any scene
            // object before reaching the photon, we skip drawing it.
            if (scene)
            {
                float distPhoton = length(d);
                if (distPhoton > 1e-3f)
                {
                    float3 dir = d / distPhoton;
                    const float eps = 1e-3f;
                    float closest = distPhoton - eps;

                    // Sphere intersection only (triangles are ignored for occlusion).
                    for (const DebugSphere &sph : debugSpheres)
                    {
                        float3 oc = eye - sph.center;
                        float a = dot(dir, dir);
                        float b = 2.0f * dot(oc, dir);
                        float c = dot(oc, oc) - sph.radius * sph.radius;
                        float disc = b * b - 4.0f * a * c;
                        if (disc < 0.0f)
                            continue;
                        float sqrtd = sqrtf(disc);
                        float t1 = (-b - sqrtd) / (2.0f * a);
                        float t2 = (-b + sqrtd) / (2.0f * a);
                        float tHit = t1;
                        if (tHit < eps)
                            tHit = t2;
                        if (tHit > eps && tHit < closest)
                        {
                            closest = tHit;
                        }
                    }

                    if (closest < distPhoton - eps)
                    {
                        // Photon is occluded.
                        continue;
                    }
                }
            }

            float a = dot(d, U_exact) / (ulen * ulen);
            float b = dot(d, V_exact) / (vlen * vlen);
            float c = dot(d, W_exact) / (wlen * wlen);
            if (c <= 0.0f)
                continue;

            float ndcX = a / c;
            float ndcY = b / c;
            if (ndcX < -1.0f || ndcX > 1.0f || ndcY < -1.0f || ndcY > 1.0f)
                continue;

            int px = static_cast<int>((ndcX * 0.5f + 0.5f) * (textureWidth - 1));

            int py = static_cast<int>(((ndcY * 0.5f + 0.5f)) * (textureHeight - 1));
            if (px < 0 || py < 0 || px >= (int)textureWidth || py >= (int)textureHeight)
                continue;

            size_t idx = (static_cast<size_t>(py) * textureWidth + static_cast<size_t>(px)) * 3;

            // Visualize photon color based on stored power
            float3 power = photonPowers[i];

            // Normalize to get the color direction, then boost for visibility
            float maxComp = fmaxf(fmaxf(power.x, power.y), power.z);
            float3 col;
            if (maxComp > 0.0f)
            {
                // Normalize and make bright for visibility
                col = power / maxComp;
            }
            else
            {
                col = make_float3(0.5f, 0.5f, 0.5f); // Gray for zero power
            }

            pixelBuffer[idx + 0] = static_cast<unsigned char>(col.x * 255.0f);
            pixelBuffer[idx + 1] = static_cast<unsigned char>(col.y * 255.0f);
            pixelBuffer[idx + 2] = static_cast<unsigned char>(col.z * 255.0f);
        }

        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, textureWidth, textureHeight, GL_RGB, GL_UNSIGNED_BYTE, pixelBuffer.data());

        glUseProgram(shaderProgram);
        glBindVertexArray(quadVAO);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);

        glDisable(GL_SCISSOR_TEST);
    }
};
