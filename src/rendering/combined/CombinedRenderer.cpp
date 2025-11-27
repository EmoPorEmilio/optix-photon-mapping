#include "CombinedRenderer.h"
#include "../../core/Constants.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>

CombinedRenderer::CombinedRenderer(GLFWwindow* win, const ViewportRect& vp)
    : window(win), viewport(vp), scene(nullptr), camera(nullptr), optixManager(nullptr)
{
}

CombinedRenderer::~CombinedRenderer()
{
    if (d_directBuffer) cudaFree(d_directBuffer);
    if (d_indirectBuffer) cudaFree(d_indirectBuffer);
    if (d_causticBuffer) cudaFree(d_causticBuffer);
    if (d_specularBuffer) cudaFree(d_specularBuffer);
    if (d_combinedBuffer) cudaFree(d_combinedBuffer);
    if (d_globalPhotonMap) cudaFree(d_globalPhotonMap);
    if (d_causticPhotonMap) cudaFree(d_causticPhotonMap);
    if (textureID) glDeleteTextures(1, &textureID);
    if (vao) glDeleteVertexArrays(1, &vao);
    if (vbo) glDeleteBuffers(1, &vbo);
    if (shaderProgram) glDeleteProgram(shaderProgram);
}

void CombinedRenderer::setupOpenGL()
{
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

void CombinedRenderer::createFullscreenQuad()
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

void CombinedRenderer::createShader()
{
    const char* vs = R"(
        #version 330 core
        layout (location = 0) in vec2 aPos;
        layout (location = 1) in vec2 aTexCoord;
        out vec2 TexCoord;
        void main() {
            gl_Position = vec4(aPos, 0.0, 1.0);
            TexCoord = aTexCoord;
        }
    )";

    const char* fs = R"(
        #version 330 core
        in vec2 TexCoord;
        out vec4 FragColor;
        uniform sampler2D screenTexture;
        void main() {
            FragColor = texture(screenTexture, TexCoord);
        }
    )";

    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vs, nullptr);
    glCompileShader(vertexShader);

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fs, nullptr);
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

void CombinedRenderer::allocateBuffers(unsigned int width, unsigned int height)
{
    if (bufferWidth == width && bufferHeight == height && d_combinedBuffer)
        return;

    // Free old buffers
    if (d_directBuffer) { cudaFree(d_directBuffer); d_directBuffer = nullptr; }
    if (d_indirectBuffer) { cudaFree(d_indirectBuffer); d_indirectBuffer = nullptr; }
    if (d_causticBuffer) { cudaFree(d_causticBuffer); d_causticBuffer = nullptr; }
    if (d_specularBuffer) { cudaFree(d_specularBuffer); d_specularBuffer = nullptr; }
    if (d_combinedBuffer) { cudaFree(d_combinedBuffer); d_combinedBuffer = nullptr; }

    bufferWidth = width;
    bufferHeight = height;
    size_t bufSize = width * height * sizeof(float4);

    cudaMalloc(&d_directBuffer, bufSize);
    cudaMalloc(&d_indirectBuffer, bufSize);
    cudaMalloc(&d_causticBuffer, bufSize);
    cudaMalloc(&d_specularBuffer, bufSize);
    cudaMalloc(&d_combinedBuffer, bufSize);
    h_combinedBuffer.resize(width * height);

    std::cout << "CombinedRenderer: Allocated 5 buffers at " << width << "x" << height << std::endl;
}

bool CombinedRenderer::initialize(OptixManager* om)
{
    optixManager = om;
    setupOpenGL();
    createFullscreenQuad();
    createShader();
    initialized = true;
    return true;
}

void CombinedRenderer::setViewport(const ViewportRect& vp)
{
    if (vp.width != viewport.width || vp.height != viewport.height)
        allocateBuffers(vp.width, vp.height);
    viewport = vp;
}

void CombinedRenderer::uploadGlobalPhotonMap(const std::vector<Photon>& photons)
{
    if (d_globalPhotonMap) { cudaFree(d_globalPhotonMap); d_globalPhotonMap = nullptr; }
    globalPhotonCount = static_cast<unsigned int>(photons.size());
    if (globalPhotonCount > 0) {
        cudaMalloc(&d_globalPhotonMap, globalPhotonCount * sizeof(Photon));
        cudaMemcpy(d_globalPhotonMap, photons.data(), globalPhotonCount * sizeof(Photon), cudaMemcpyHostToDevice);
        globalKDTree.build(photons);
    } else {
        globalKDTree.clear();
    }
}

void CombinedRenderer::uploadCausticPhotonMap(const std::vector<Photon>& caustics)
{
    if (d_causticPhotonMap) { cudaFree(d_causticPhotonMap); d_causticPhotonMap = nullptr; }
    causticPhotonCount = static_cast<unsigned int>(caustics.size());
    if (causticPhotonCount > 0) {
        cudaMalloc(&d_causticPhotonMap, causticPhotonCount * sizeof(Photon));
        cudaMemcpy(d_causticPhotonMap, caustics.data(), causticPhotonCount * sizeof(Photon), cudaMemcpyHostToDevice);
        causticKDTree.build(caustics);
    } else {
        causticKDTree.clear();
    }
}

void CombinedRenderer::setWeights(float direct, float indirect, float caustic, float specular)
{
    directWeight = direct;
    indirectWeight = indirect;
    causticWeight = caustic;
    specularWeight = specular;
}

void CombinedRenderer::combineBuffers(unsigned int width, unsigned int height)
{
    // Copy all buffers to CPU and combine
    std::vector<float4> h_direct(width * height);
    std::vector<float4> h_indirect(width * height);
    std::vector<float4> h_caustic(width * height);
    std::vector<float4> h_specular(width * height);

    cudaMemcpy(h_direct.data(), d_directBuffer, width * height * sizeof(float4), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_indirect.data(), d_indirectBuffer, width * height * sizeof(float4), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_caustic.data(), d_causticBuffer, width * height * sizeof(float4), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_specular.data(), d_specularBuffer, width * height * sizeof(float4), cudaMemcpyDeviceToHost);

    // Combine with weights - use MAX blending for specular (sphere areas)
    for (unsigned int i = 0; i < width * height; i++)
    {
        // Check if this pixel has specular content (sphere area)
        float specLuminance = h_specular[i].x + h_specular[i].y + h_specular[i].z;
        
        float r, g, b;
        
        if (specLuminance > 0.01f)
        {
            // Sphere pixel: use specular directly (full brightness) 
            // The specular pass already has the full scene reflected/refracted
            r = h_specular[i].x;
            g = h_specular[i].y;
            b = h_specular[i].z;
        }
        else
        {
            // Non-sphere pixel: combine direct + indirect + caustic with weights
            r = h_direct[i].x * directWeight + h_indirect[i].x * indirectWeight + 
                h_caustic[i].x * causticWeight;
            g = h_direct[i].y * directWeight + h_indirect[i].y * indirectWeight + 
                h_caustic[i].y * causticWeight;
            b = h_direct[i].z * directWeight + h_indirect[i].z * indirectWeight + 
                h_caustic[i].z * causticWeight;
        }

        // Clamp to [0, 1] in linear space
        r = std::min(1.0f, std::max(0.0f, r));
        g = std::min(1.0f, std::max(0.0f, g));
        b = std::min(1.0f, std::max(0.0f, b));

        // Apply gamma correction (sRGB) at final output stage
        r = std::pow(r, Constants::Render::INV_GAMMA);
        g = std::pow(g, Constants::Render::INV_GAMMA);
        b = std::pow(b, Constants::Render::INV_GAMMA);

        h_combinedBuffer[i] = make_float4(r, g, b, 1.0f);
    }
}

void CombinedRenderer::render()
{
    if (!initialized || !camera || !optixManager)
        return;

    unsigned int width = viewport.width;
    unsigned int height = viewport.height;
    allocateBuffers(width, height);

    // Run all 4 pipelines
    optixManager->launchDirectLighting(width, height, *camera, directAmbient, directShadowAmbient, directIntensity, directAttenuation, d_directBuffer);
    
    if (globalPhotonCount > 0)
        optixManager->launchIndirectLighting(width, height, *camera, d_globalPhotonMap, globalPhotonCount, gatherRadius, indirectBrightness, globalKDTree.getDeviceTree(), d_indirectBuffer);
    else
        cudaMemset(d_indirectBuffer, 0, width * height * sizeof(float4));

    if (causticPhotonCount > 0)
        optixManager->launchCausticLighting(width, height, *camera, d_causticPhotonMap, causticPhotonCount, gatherRadius * Constants::Photon::CAUSTIC_RADIUS_MULTIPLIER, causticBrightness, causticKDTree.getDeviceTree(), d_causticBuffer);
    else
        cudaMemset(d_causticBuffer, 0, width * height * sizeof(float4));

    optixManager->launchSpecularLighting(width, height, *camera,
                                         d_globalPhotonMap, globalPhotonCount, globalKDTree.getDeviceTree(),
                                         d_causticPhotonMap, causticPhotonCount, causticKDTree.getDeviceTree(),
                                         specParams, d_specularBuffer);

    // Combine on CPU
    combineBuffers(width, height);

    // Upload to OpenGL
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, h_combinedBuffer.data());

    // Render
    glViewport(viewport.x, viewport.y, viewport.width, viewport.height);
    glDisable(GL_DEPTH_TEST);
    glUseProgram(shaderProgram);
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
    glEnable(GL_DEPTH_TEST);
}

void CombinedRenderer::exportToImage(const std::string& filename)
{
    if (!initialized || !camera || !optixManager)
        return;

    unsigned int width = viewport.width;
    unsigned int height = viewport.height;
    allocateBuffers(width, height);

    std::cout << "Rendering combined image..." << std::endl;

    // Run all pipelines
    optixManager->launchDirectLighting(width, height, *camera, directAmbient, directShadowAmbient, directIntensity, directAttenuation, d_directBuffer);
    
    if (globalPhotonCount > 0)
        optixManager->launchIndirectLighting(width, height, *camera, d_globalPhotonMap, globalPhotonCount, gatherRadius, indirectBrightness, globalKDTree.getDeviceTree(), d_indirectBuffer);
    else
        cudaMemset(d_indirectBuffer, 0, width * height * sizeof(float4));

    if (causticPhotonCount > 0)
        optixManager->launchCausticLighting(width, height, *camera, d_causticPhotonMap, causticPhotonCount, gatherRadius * Constants::Photon::CAUSTIC_RADIUS_MULTIPLIER, causticBrightness, causticKDTree.getDeviceTree(), d_causticBuffer);
    else
        cudaMemset(d_causticBuffer, 0, width * height * sizeof(float4));

    optixManager->launchSpecularLighting(width, height, *camera,
                                         d_globalPhotonMap, globalPhotonCount, globalKDTree.getDeviceTree(),
                                         d_causticPhotonMap, causticPhotonCount, causticKDTree.getDeviceTree(),
                                         specParams, d_specularBuffer);

    // Combine
    combineBuffers(width, height);

    // Write PPM file (simple format)
    std::string ppmFilename = filename + ".ppm";
    std::ofstream file(ppmFilename, std::ios::binary);
    file << "P6\n" << width << " " << height << "\n255\n";

    for (int y = height - 1; y >= 0; y--)  // Flip Y
    {
        for (unsigned int x = 0; x < width; x++)
        {
            float4 pixel = h_combinedBuffer[y * width + x];
            unsigned char r = static_cast<unsigned char>(std::min(255.0f, pixel.x * 255.0f));
            unsigned char g = static_cast<unsigned char>(std::min(255.0f, pixel.y * 255.0f));
            unsigned char b = static_cast<unsigned char>(std::min(255.0f, pixel.z * 255.0f));
            file.write(reinterpret_cast<char*>(&r), 1);
            file.write(reinterpret_cast<char*>(&g), 1);
            file.write(reinterpret_cast<char*>(&b), 1);
        }
    }
    file.close();

    std::cout << "Exported combined image to: " << ppmFilename << std::endl;
    std::cout << "  Direct weight: " << directWeight << std::endl;
    std::cout << "  Indirect weight: " << indirectWeight << std::endl;
    std::cout << "  Caustic weight: " << causticWeight << std::endl;
    std::cout << "  Specular weight: " << specularWeight << std::endl;
}

