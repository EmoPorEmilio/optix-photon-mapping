#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "WindowManager.h"
#include "Camera.h"
#include "Scene.h"
#include "optix/OptixManager.h"
#include <vector>
#include <cstring>
#include <sutil/vec_math.h> 
#include <iostream>

class Renderer
{
private:
    Window *window;
    const ViewportRect &viewport;
    Camera *camera = nullptr;
    Scene *scene = nullptr;
    OptixManager *optixManager = nullptr; 
    float clearR, clearG, clearB, clearA;

    
    unsigned int textureID = 0;
    unsigned int VAO = 0;
    unsigned int VBO = 0;
    unsigned int EBO = 0;
    unsigned int shaderProgram = 0;
    unsigned int textureWidth = 0;
    unsigned int textureHeight = 0;

    
    unsigned int lightShaderProgram = 0;
    unsigned int lightVAO = 0;
    unsigned int lightVBO = 0;

    
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
            layout (location = 0) in vec3 aPos;
            layout (location = 1) in vec2 aTexCoord;

            out vec2 TexCoord;

            void main()
            {
                gl_Position = vec4(aPos, 1.0);
                TexCoord = aTexCoord;
            }
        )";

        const char *fragmentShaderSource = R"(
            #version 330 core
            out vec4 FragColor;

            in vec2 TexCoord;
            uniform sampler2D ourTexture;

            void main()
            {
                FragColor = texture(ourTexture, TexCoord);
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
    
    Renderer(Window *w, const ViewportRect &vp, float r = 0.2f, float g = 0.2f, float b = 0.2f, float a = 1.0f)
        : window(w), viewport(vp), clearR(r), clearG(g), clearB(b), clearA(a)
    {
    }

    void setCamera(Camera *cam)
    {
        camera = cam;
        
        if (camera)
        {
            float viewportAspect = static_cast<float>(viewport.width) / static_cast<float>(viewport.height);
            camera->setAspectRatio(viewportAspect);
        }
    }
    void setScene(Scene *s) { scene = s; }
    void setOptixManager(OptixManager *optix) { optixManager = optix; }

    
    void drawQuadLights();

    
    void renderFrame()
    {
        if (!window || !camera || !optixManager || !optixManager->isInitialized()) {
            return;
        }

        window->makeCurrent();

        
        size_t rgba_size = viewport.width * viewport.height * sizeof(uchar4);
        std::vector<uchar4> rgba_buffer(viewport.width * viewport.height);

        optixManager->render(viewport.width, viewport.height, *camera, reinterpret_cast<unsigned char *>(rgba_buffer.data()));

        
        glViewport(viewport.x, viewport.y, viewport.width, viewport.height);

        
        if (shaderProgram == 0)
        {
            createShaderProgram();

            
            glGenTextures(1, &textureID);
            glBindTexture(GL_TEXTURE_2D, textureID);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

            float quadVertices[] = {
                
                // position (x,y,z)    texcoord (u,v)
                -1.0f, -1.0f, 0.0f,   0.0f, 0.0f,
                 1.0f, -1.0f, 0.0f,   1.0f, 0.0f,
                 1.0f,  1.0f, 0.0f,   1.0f, 1.0f,
                -1.0f,  1.0f, 0.0f,   0.0f, 1.0f
            };

            unsigned int indices[] = {
                0, 1, 2,
                2, 3, 0};

            glGenVertexArrays(1, &VAO);
            glGenBuffers(1, &VBO);
            glGenBuffers(1, &EBO);

            glBindVertexArray(VAO);
            glBindBuffer(GL_ARRAY_BUFFER, VBO);
            glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

            // position (vec3)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)0);
            glEnableVertexAttribArray(0);
            // texcoord (vec2)
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)(3 * sizeof(float)));
            glEnableVertexAttribArray(1);

            glBindVertexArray(0);
            glBindVertexArray(0);
        }

        
        if (textureWidth != viewport.width || textureHeight != viewport.height)
        {
            glBindTexture(GL_TEXTURE_2D, textureID);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, viewport.width, viewport.height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
            textureWidth = viewport.width;
            textureHeight = viewport.height;
        }

        
        std::vector<unsigned char> rgb_buffer(viewport.width * viewport.height * 3);
        for (size_t i = 0; i < rgba_buffer.size(); ++i)
        {
            rgb_buffer[i * 3 + 0] = rgba_buffer[i].x; 
            rgb_buffer[i * 3 + 1] = rgba_buffer[i].y; 
            rgb_buffer[i * 3 + 2] = rgba_buffer[i].z; 
        }

            glBindTexture(GL_TEXTURE_2D, textureID);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, viewport.width, viewport.height, GL_RGB, GL_UNSIGNED_BYTE, rgb_buffer.data());

            
            

            
            glUseProgram(shaderProgram);
            glBindTexture(GL_TEXTURE_2D, textureID);
            glBindVertexArray(VAO);
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
            glBindVertexArray(0);
    }
};



