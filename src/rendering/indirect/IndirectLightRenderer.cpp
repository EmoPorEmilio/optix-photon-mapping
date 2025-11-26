#include "IndirectLightRenderer.h"
#include <iostream>

IndirectLightRenderer::IndirectLightRenderer(GLFWwindow* win, const ViewportRect& vp)
    : window(win), viewport(vp), scene(nullptr), camera(nullptr), optixManager(nullptr)
{
}

IndirectLightRenderer::~IndirectLightRenderer()
{
    if (d_frameBuffer)
        cudaFree(d_frameBuffer);
    if (d_photonMap)
        cudaFree(d_photonMap);
    if (textureID)
        glDeleteTextures(1, &textureID);
    if (vao)
        glDeleteVertexArrays(1, &vao);
    if (vbo)
        glDeleteBuffers(1, &vbo);
    if (shaderProgram)
        glDeleteProgram(shaderProgram);
}

void IndirectLightRenderer::setupOpenGL()
{
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

void IndirectLightRenderer::createFullscreenQuad()
{
    float vertices[] = {
        // positions   // texcoords
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

void IndirectLightRenderer::createShader()
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

    // Compile vertex shader
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
    glCompileShader(vertexShader);

    // Compile fragment shader
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fragmentShader);

    // Link program
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // Set sampler uniform
    glUseProgram(shaderProgram);
    glUniform1i(glGetUniformLocation(shaderProgram, "screenTexture"), 0);
}

void IndirectLightRenderer::allocateBuffers(unsigned int width, unsigned int height)
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
        std::cout << "IndirectLightRenderer: Allocated " << width << "x" << height << " frame buffer" << std::endl;
    }
}

bool IndirectLightRenderer::initialize(OptixManager* om)
{
    optixManager = om;
    setupOpenGL();
    createFullscreenQuad();
    createShader();
    initialized = true;
    return true;
}

void IndirectLightRenderer::setViewport(const ViewportRect& vp)
{
    if (vp.width != viewport.width || vp.height != viewport.height)
    {
        allocateBuffers(vp.width, vp.height);
    }
    viewport = vp;
}

void IndirectLightRenderer::uploadPhotonMap(const std::vector<Photon>& photons)
{
    if (d_photonMap)
    {
        cudaFree(d_photonMap);
        d_photonMap = nullptr;
    }

    photonCount = static_cast<unsigned int>(photons.size());
    if (photonCount > 0)
    {
        cudaMalloc(&d_photonMap, photonCount * sizeof(Photon));
        cudaMemcpy(d_photonMap, photons.data(), photonCount * sizeof(Photon), cudaMemcpyHostToDevice);
        std::cout << "IndirectLightRenderer: Uploaded " << photonCount << " photons" << std::endl;
    }
}

void IndirectLightRenderer::render()
{
    if (!initialized || !camera || !optixManager)
        return;

    unsigned int width = viewport.width;
    unsigned int height = viewport.height;

    // Ensure buffers allocated
    allocateBuffers(width, height);

    if (photonCount == 0 || !d_photonMap)
    {
        // No photons - render black
        memset(h_frameBuffer.data(), 0, width * height * sizeof(float4));
    }
    else
    {
        // Launch OptiX indirect lighting pass
        optixManager->launchIndirectLighting(width, height, *camera, d_photonMap, photonCount, gatherRadius, d_frameBuffer);

        // Copy to CPU
        cudaMemcpy(h_frameBuffer.data(), d_frameBuffer, width * height * sizeof(float4), cudaMemcpyDeviceToHost);
    }

    // Upload to OpenGL texture
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

