

#include "Renderer.h"
#include <sutil/vec_math.h>
#include <cmath>

void Renderer::drawQuadLights()
{
    
    glClearColor(0.0f, 1.0f, 0.0f, 1.0f); 
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    
    glViewport(viewport.x, viewport.y, viewport.width, viewport.height);
    glScissor(viewport.x, viewport.y, viewport.width, viewport.height);
    glEnable(GL_SCISSOR_TEST);
    glClearColor(0.0f, 1.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_SCISSOR_TEST);

    std::cout << "drawQuadLights() called - CHANGED BACKGROUND TO GREEN" << std::endl;
    if (!scene || !camera) {
        std::cout << "drawQuadLights() early return - scene or camera null" << std::endl;
        return;
    }

    GLenum err;

    
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);  
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glClear(GL_DEPTH_BUFFER_BIT);  

    
    const char *vertexShaderSource = R"(
        #version 330 core
        layout (location = 0) in vec3 aPos;
        void main()
        {
            
            if (gl_VertexID == 0) gl_Position = vec4(-0.5, -0.5, 0.0, 1.0);  
            else if (gl_VertexID == 1) gl_Position = vec4(0.5, -0.5, 0.0, 1.0);   
            else if (gl_VertexID == 2) gl_Position = vec4(0.5, 0.5, 0.0, 1.0);    
            else if (gl_VertexID == 3) gl_Position = vec4(-0.5, 0.5, 0.0, 1.0);   
            else if (gl_VertexID == 4) gl_Position = vec4(0.0, 0.0, 0.0, 1.0);    
            else gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
        }
    )";

    const char *fragmentShaderSource = R"(
        #version 330 core
        out vec4 FragColor;
        void main()
        {
            FragColor = vec4(1.0, 0.0, 0.0, 1.0); 
        }
    )";

    
    if (lightShaderProgram == 0)
    {
        unsigned int vertexShader = compileShader(vertexShaderSource, GL_VERTEX_SHADER);
        unsigned int fragmentShader = compileShader(fragmentShaderSource, GL_FRAGMENT_SHADER);

        lightShaderProgram = glCreateProgram();
        glAttachShader(lightShaderProgram, vertexShader);
        glAttachShader(lightShaderProgram, fragmentShader);
        glLinkProgram(lightShaderProgram);

        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);

        
        glGenVertexArrays(1, &lightVAO);
        glGenBuffers(1, &lightVBO);
    }

    glUseProgram(lightShaderProgram);

    
    std::cout << "About to draw QuadLight test triangles" << std::endl;

    
    const auto& lights = scene->getLights();
    std::cout << "Found " << lights.size() << " lights" << std::endl;

    for (const auto& light : lights)
    {
        if (light->isAreaLight())
        {
            std::cout << "Drawing QuadLight test geometry" << std::endl;

            
            glDrawArrays(GL_TRIANGLES, 0, 3);

            
            while ((err = glGetError()) != GL_NO_ERROR) {
                std::cout << "OpenGL error after drawing: " << err << std::endl;
            }

            std::cout << "Drew test triangle" << std::endl;
            break; 
        }
    }

    glBindVertexArray(0);
    glUseProgram(0);
    glDisable(GL_BLEND);

    std::cout << "drawQuadLights() completed" << std::endl;
}



