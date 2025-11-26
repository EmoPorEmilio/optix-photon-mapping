#include "SpecularLightRenderer.h"
#include <iostream>

SpecularLightRenderer::SpecularLightRenderer(GLFWwindow* win, const ViewportRect& vp)
    : window(win), viewport(vp), scene(nullptr), camera(nullptr), optixManager(nullptr)
{
}

SpecularLightRenderer::~SpecularLightRenderer()
{
    if (d_frameBuffer)
        cudaFree(d_frameBuffer);
    if (d_globalPhotonMap)
        cudaFree(d_globalPhotonMap);
    if (d_causticPhotonMap)
        cudaFree(d_causticPhotonMap);
    if (textureID)
        glDeleteTextures(1, &textureID);
    if (vao)
        glDeleteVertexArrays(1, &vao);
    if (vbo)
        glDeleteBuffers(1, &vbo);
    if (shaderProgram)
        glDeleteProgram(shaderProgram);
}

void SpecularLightRenderer::uploadGlobalPhotonMap(const std::vector<Photon>& photons)
{
    if (d_globalPhotonMap) { cudaFree(d_globalPhotonMap); d_globalPhotonMap = nullptr; }
    globalPhotonCount = static_cast<unsigned int>(photons.size());
    if (globalPhotonCount > 0) {
        cudaMalloc(&d_globalPhotonMap, globalPhotonCount * sizeof(Photon));
        cudaMemcpy(d_globalPhotonMap, photons.data(), globalPhotonCount * sizeof(Photon), cudaMemcpyHostToDevice);
        std::cout << "SpecularLightRenderer: Uploaded " << globalPhotonCount << " global photons" << std::endl;
    }
}

void SpecularLightRenderer::uploadCausticPhotonMap(const std::vector<Photon>& caustics)
{
    if (d_causticPhotonMap) { cudaFree(d_causticPhotonMap); d_causticPhotonMap = nullptr; }
    causticPhotonCount = static_cast<unsigned int>(caustics.size());
    if (causticPhotonCount > 0) {
        cudaMalloc(&d_causticPhotonMap, causticPhotonCount * sizeof(Photon));
        cudaMemcpy(d_causticPhotonMap, caustics.data(), causticPhotonCount * sizeof(Photon), cudaMemcpyHostToDevice);
        std::cout << "SpecularLightRenderer: Uploaded " << causticPhotonCount << " caustic photons" << std::endl;
    }
}

void SpecularLightRenderer::setupOpenGL()
{
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

void SpecularLightRenderer::createFullscreenQuad()
{
    float vertices[] = {
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f,
        -1.0f,  1.0f,  0.0f, 1.0f
    };

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
}

void SpecularLightRenderer::createShader()
{
    const char* vertexShaderSource = R"(
        #version 330 core
        layout (location = 0) in vec2 aPos;
        layout (location = 1) in vec2 aTexCoord;
        out vec2 TexCoord;
        void main() {
            gl_Position = vec4(aPos, 0.0, 1.0);
            TexCoord = aTexCoord;
        }
    )";

    const char* fragmentShaderSource = R"(
        #version 330 core
        in vec2 TexCoord;
        out vec4 FragColor;
        uniform sampler2D screenTexture;
        void main() {
            FragColor = texture(screenTexture, TexCoord);
        }
    )";

    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
    glCompileShader(vertexShader);

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fragmentShader);

    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    glUseProgram(shaderProgram);
    glUniform1i(glGetUniformLocation(shaderProgram, "screenTexture"), 0);
}

void SpecularLightRenderer::allocateBuffers(unsigned int width, unsigned int height)
{
    if (d_frameBuffer && (bufferWidth != width || bufferHeight != height))
    {
        cudaFree(d_frameBuffer);
        d_frameBuffer = nullptr;
    }

    bufferWidth = width;
    bufferHeight = height;

    if (!d_frameBuffer)
    {
        cudaMalloc(&d_frameBuffer, width * height * sizeof(float4));
        h_frameBuffer.resize(width * height);
        std::cout << "SpecularLightRenderer: Allocated " << width << "x" << height << " frame buffer" << std::endl;
    }
}

bool SpecularLightRenderer::initialize(OptixManager* om)
{
    optixManager = om;
    setupOpenGL();
    createFullscreenQuad();
    createShader();
    initialized = true;
    return true;
}

void SpecularLightRenderer::setViewport(const ViewportRect& vp)
{
    if (vp.width != viewport.width || vp.height != viewport.height)
    {
        allocateBuffers(vp.width, vp.height);
    }
    viewport = vp;
}

void SpecularLightRenderer::render()
{
    if (!initialized || !camera || !optixManager)
        return;

    unsigned int width = viewport.width;
    unsigned int height = viewport.height;

    allocateBuffers(width, height);

    // Launch OptiX specular lighting pass with photon maps for full scene lighting
    optixManager->launchSpecularLighting(width, height, *camera,
                                         d_globalPhotonMap, globalPhotonCount,
                                         d_causticPhotonMap, causticPhotonCount,
                                         specParams, d_frameBuffer);

    // Copy to CPU
    cudaMemcpy(h_frameBuffer.data(), d_frameBuffer, width * height * sizeof(float4), cudaMemcpyDeviceToHost);

    // Upload to OpenGL
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, h_frameBuffer.data());

    // Render fullscreen quad
    glViewport(viewport.x, viewport.y, viewport.width, viewport.height);
    glDisable(GL_DEPTH_TEST);
    glUseProgram(shaderProgram);
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
    glEnable(GL_DEPTH_TEST);
}

