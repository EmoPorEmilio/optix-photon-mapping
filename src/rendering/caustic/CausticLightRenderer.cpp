#include "CausticLightRenderer.h"
#include <iostream>

CausticLightRenderer::CausticLightRenderer(GLFWwindow* win, const ViewportRect& vp)
    : window(win), viewport(vp), scene(nullptr), camera(nullptr), optixManager(nullptr)
{
}

CausticLightRenderer::~CausticLightRenderer()
{
    if (d_frameBuffer)
        cudaFree(d_frameBuffer);
    if (d_causticMap)
        cudaFree(d_causticMap);
    if (textureID)
        glDeleteTextures(1, &textureID);
    if (vao)
        glDeleteVertexArrays(1, &vao);
    if (vbo)
        glDeleteBuffers(1, &vbo);
    if (shaderProgram)
        glDeleteProgram(shaderProgram);
}

void CausticLightRenderer::setupOpenGL()
{
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

void CausticLightRenderer::createFullscreenQuad()
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

void CausticLightRenderer::createShader()
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

void CausticLightRenderer::allocateBuffers(unsigned int width, unsigned int height)
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
        std::cout << "CausticLightRenderer: Allocated " << width << "x" << height << " frame buffer" << std::endl;
    }
}

bool CausticLightRenderer::initialize(OptixManager* om)
{
    optixManager = om;
    setupOpenGL();
    createFullscreenQuad();
    createShader();
    initialized = true;
    return true;
}

void CausticLightRenderer::setViewport(const ViewportRect& vp)
{
    if (vp.width != viewport.width || vp.height != viewport.height)
    {
        allocateBuffers(vp.width, vp.height);
    }
    viewport = vp;
}

void CausticLightRenderer::uploadCausticMap(const std::vector<Photon>& caustics)
{
    if (d_causticMap)
    {
        cudaFree(d_causticMap);
        d_causticMap = nullptr;
    }

    causticCount = static_cast<unsigned int>(caustics.size());
    if (causticCount > 0)
    {
        cudaMalloc(&d_causticMap, causticCount * sizeof(Photon));
        cudaMemcpy(d_causticMap, caustics.data(), causticCount * sizeof(Photon), cudaMemcpyHostToDevice);
        std::cout << "CausticLightRenderer: Uploaded " << causticCount << " caustic photons" << std::endl;
        causticKDTree.build(caustics);
    }
    else
    {
        causticKDTree.clear();
    }
}

void CausticLightRenderer::render()
{
    if (!initialized || !camera || !optixManager)
        return;

    unsigned int width = viewport.width;
    unsigned int height = viewport.height;

    allocateBuffers(width, height);

    if (causticCount == 0 || !d_causticMap)
    {
        // No caustics - render black with just the spheres base color
        memset(h_frameBuffer.data(), 0, width * height * sizeof(float4));
    }
    else
    {
        optixManager->launchCausticLighting(width, height, *camera, d_causticMap, causticCount, gatherRadius, brightnessMultiplier, causticKDTree.getDeviceTree(), d_frameBuffer);
        cudaMemcpy(h_frameBuffer.data(), d_frameBuffer, width * height * sizeof(float4), cudaMemcpyDeviceToHost);
    }

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, h_frameBuffer.data());

    glViewport(viewport.x, viewport.y, viewport.width, viewport.height);
    glDisable(GL_DEPTH_TEST);
    glUseProgram(shaderProgram);
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
    glEnable(GL_DEPTH_TEST);
}

