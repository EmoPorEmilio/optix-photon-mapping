#include "DirectLightRenderer.h"
#include "../../optix/OptixManager.h"
#include <iostream>

// Fullscreen quad vertex shader
static const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoord;
out vec2 TexCoord;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    TexCoord = aTexCoord;
}
)";

// Fullscreen quad fragment shader
static const char* fragmentShaderSource = R"(
#version 330 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D screenTexture;
void main() {
    FragColor = texture(screenTexture, TexCoord);
}
)";

DirectLightRenderer::~DirectLightRenderer()
{
    shutdown();
}

bool DirectLightRenderer::initialize(GLFWwindow* window)
{
    if (initialized)
        return true;

    createShaders();
    createQuad();

    // Create texture for displaying the rendered image
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    initialized = true;
    std::cout << "DirectLightRenderer initialized" << std::endl;
    return true;
}

void DirectLightRenderer::shutdown()
{
    if (d_frameBuffer)
    {
        cudaFree(d_frameBuffer);
        d_frameBuffer = nullptr;
    }

    if (textureID)
    {
        glDeleteTextures(1, &textureID);
        textureID = 0;
    }

    if (vao)
    {
        glDeleteVertexArrays(1, &vao);
        vao = 0;
    }

    if (vbo)
    {
        glDeleteBuffers(1, &vbo);
        vbo = 0;
    }

    if (shaderProgram)
    {
        glDeleteProgram(shaderProgram);
        shaderProgram = 0;
    }

    initialized = false;
}

void DirectLightRenderer::createShaders()
{
    // Compile vertex shader
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
    glCompileShader(vertexShader);

    // Check vertex shader compilation
    GLint success;
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        char infoLog[512];
        glGetShaderInfoLog(vertexShader, 512, nullptr, infoLog);
        std::cerr << "DirectLightRenderer: Vertex shader compilation failed: " << infoLog << std::endl;
    }

    // Compile fragment shader
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fragmentShader);

    // Check fragment shader compilation
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        char infoLog[512];
        glGetShaderInfoLog(fragmentShader, 512, nullptr, infoLog);
        std::cerr << "DirectLightRenderer: Fragment shader compilation failed: " << infoLog << std::endl;
    }

    // Link shaders
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    // Check linking
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success)
    {
        char infoLog[512];
        glGetProgramInfoLog(shaderProgram, 512, nullptr, infoLog);
        std::cerr << "DirectLightRenderer: Shader program linking failed: " << infoLog << std::endl;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

void DirectLightRenderer::createQuad()
{
    // Fullscreen quad vertices (position + texcoord)
    float quadVertices[] = {
        // positions   // texCoords
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,

        -1.0f,  1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f
    };

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
}

void DirectLightRenderer::allocateBuffers(unsigned int width, unsigned int height)
{
    if (width == bufferWidth && height == bufferHeight && d_frameBuffer)
        return;

    if (d_frameBuffer)
    {
        cudaFree(d_frameBuffer);
    }

    bufferWidth = width;
    bufferHeight = height;

    cudaMalloc(&d_frameBuffer, width * height * sizeof(float4));
    h_frameBuffer.resize(width * height);

    std::cout << "DirectLightRenderer: Allocated " << width << "x" << height << " frame buffer" << std::endl;
}

void DirectLightRenderer::render()
{
    if (!initialized || !camera || !optixManager)
    {
        std::cout << "DirectLightRenderer::render() - not ready: init=" << initialized 
                  << " camera=" << (camera ? "yes" : "no") 
                  << " optix=" << (optixManager ? "yes" : "no") << std::endl;
        return;
    }

    unsigned int width = viewport.width;
    unsigned int height = viewport.height;

    if (width == 0 || height == 0)
    {
        std::cout << "DirectLightRenderer::render() - invalid viewport: " << width << "x" << height << std::endl;
        return;
    }

    // Allocate GPU buffer if needed
    allocateBuffers(width, height);

    // Launch OptiX direct lighting pass with configurable parameters
    optixManager->launchDirectLighting(width, height, *camera, ambient, shadowAmbient, intensity, attenuationFactor, d_frameBuffer);

    // Copy result to CPU
    cudaMemcpy(h_frameBuffer.data(), d_frameBuffer, width * height * sizeof(float4), cudaMemcpyDeviceToHost);

    // Debug: check if we got any non-zero pixels
    static int debugCounter = 0;
    if (debugCounter++ % 60 == 0)
    {
        float maxVal = 0.0f;
        for (size_t i = 0; i < h_frameBuffer.size(); i += 1000)
        {
            float4& p = h_frameBuffer[i];
            float m = fmaxf(fmaxf(p.x, p.y), p.z);
            if (m > maxVal) maxVal = m;
        }
        std::cout << "DirectLight: max pixel value sampled = " << maxVal << std::endl;
    }

    // Upload to OpenGL texture
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, h_frameBuffer.data());

    // Save OpenGL state
    GLboolean depthTestEnabled = glIsEnabled(GL_DEPTH_TEST);
    GLboolean blendEnabled = glIsEnabled(GL_BLEND);

    // Disable depth test for fullscreen quad
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);

    // Clear the viewport area
    glViewport(viewport.x, viewport.y, viewport.width, viewport.height);
    glScissor(viewport.x, viewport.y, viewport.width, viewport.height);
    glEnable(GL_SCISSOR_TEST);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_SCISSOR_TEST);

    // Render fullscreen quad
    glUseProgram(shaderProgram);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glUniform1i(glGetUniformLocation(shaderProgram, "screenTexture"), 0);
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
    glUseProgram(0);

    // Restore OpenGL state
    if (depthTestEnabled) glEnable(GL_DEPTH_TEST);
    if (blendEnabled) glEnable(GL_BLEND);
}

