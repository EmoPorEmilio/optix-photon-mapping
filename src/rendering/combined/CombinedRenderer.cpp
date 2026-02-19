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

// CPU implementation of fog for combined rendering (Jensen's algorithm)
float3 CombinedRenderer::applyFogToPixel(float3 color, float3 rayOrigin, float3 rayDir, float hitDistance)
{
    if (!fogEnabled || hitDistance <= 0.0f || hitDistance > 1e10f)
        return color;

    // Check if ray passes through volume bounds
    float3 invDir = make_float3(
        rayDir.x != 0.0f ? 1.0f / rayDir.x : 1e16f,
        rayDir.y != 0.0f ? 1.0f / rayDir.y : 1e16f,
        rayDir.z != 0.0f ? 1.0f / rayDir.z : 1e16f);

    float3 t0 = make_float3(
        (volumeProps.bounds_min.x - rayOrigin.x) * invDir.x,
        (volumeProps.bounds_min.y - rayOrigin.y) * invDir.y,
        (volumeProps.bounds_min.z - rayOrigin.z) * invDir.z);
    float3 t1 = make_float3(
        (volumeProps.bounds_max.x - rayOrigin.x) * invDir.x,
        (volumeProps.bounds_max.y - rayOrigin.y) * invDir.y,
        (volumeProps.bounds_max.z - rayOrigin.z) * invDir.z);

    float tmin_x = std::min(t0.x, t1.x);
    float tmax_x = std::max(t0.x, t1.x);
    float tmin_y = std::min(t0.y, t1.y);
    float tmax_y = std::max(t0.y, t1.y);
    float tmin_z = std::min(t0.z, t1.z);
    float tmax_z = std::max(t0.z, t1.z);

    float t_near = std::max({tmin_x, tmin_y, tmin_z, 0.0f});
    float t_far = std::min({tmax_x, tmax_y, tmax_z, hitDistance});

    if (t_far <= t_near)
        return color;

    // Ray march with density-weighted extinction
    const float step_size = 10.0f;
    float optical_depth = 0.0f;
    float3 in_scatter = make_float3(0.0f, 0.0f, 0.0f);
    float t = std::max(t_near, 0.001f);

    while (t < t_far)
    {
        float3 sample_pos = make_float3(
            rayOrigin.x + t * rayDir.x,
            rayOrigin.y + t * rayDir.y,
            rayOrigin.z + t * rayDir.z);

        // Get local density using height-based exponential falloff
        float density = volumeProps.densityAt(sample_pos);

        if (density > 0.0f)
        {
            float local_sigma_t = volumeProps.sigma_t * density;
            optical_depth += local_sigma_t * step_size;

            float transmittance_so_far = std::exp(-optical_depth);
            float scatter_contrib = density * (1.0f - std::exp(-local_sigma_t * step_size)) * transmittance_so_far;
            in_scatter.x += fogColor.x * scatter_contrib;
            in_scatter.y += fogColor.y * scatter_contrib;
            in_scatter.z += fogColor.z * scatter_contrib;
        }

        t += step_size;
    }

    float transmittance = std::exp(-optical_depth);
    return make_float3(
        color.x * transmittance + in_scatter.x,
        color.y * transmittance + in_scatter.y,
        color.z * transmittance + in_scatter.z);
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

    // Get camera parameters for fog ray computation
    float3 eye = camera->getPosition();
    float3 U = camera->getU();
    float3 V = camera->getV();
    float3 W = camera->getW();

    // Combine with weights - use MAX blending for specular (sphere areas)
    for (unsigned int y = 0; y < height; y++)
    {
        for (unsigned int x = 0; x < width; x++)
        {
            unsigned int i = y * width + x;

            // Check if this pixel has specular content (sphere area)
            float specLuminance = h_specular[i].x + h_specular[i].y + h_specular[i].z;

            float r, g, b;
            float hitDistance;

            if (specLuminance > 0.01f)
            {
                // Sphere pixel: use specular directly
                r = h_specular[i].x;
                g = h_specular[i].y;
                b = h_specular[i].z;
                hitDistance = h_specular[i].w;  // Hit distance stored in w
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
                hitDistance = h_direct[i].w;  // Hit distance from direct pass
            }

            // Apply fog (Jensen's algorithm - once to final combined result)
            if (fogEnabled)
            {
                // Compute ray direction for this pixel
                float u = (2.0f * (float(x) + 0.5f) / float(width)) - 1.0f;
                float v = (2.0f * (float(y) + 0.5f) / float(height)) - 1.0f;
                float3 rayDir = normalize(make_float3(
                    W.x + u * U.x + v * V.x,
                    W.y + u * U.y + v * V.y,
                    W.z + u * U.z + v * V.z));

                float3 color = make_float3(r, g, b);
                color = applyFogToPixel(color, eye, rayDir, hitDistance);
                r = color.x;
                g = color.y;
                b = color.z;
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
}

void CombinedRenderer::render()
{
    if (!initialized || !camera || !optixManager)
        return;

    unsigned int width = viewport.width;
    unsigned int height = viewport.height;
    allocateBuffers(width, height);

    // Run all 4 pipelines (skip fog in individual passes - will apply once at end if needed)
    optixManager->launchDirectLighting(width, height, *camera, directAmbient, directShadowAmbient, directIntensity, directAttenuation, d_directBuffer, true);
    
    if (globalPhotonCount > 0)
        optixManager->launchIndirectLighting(width, height, *camera, d_globalPhotonMap, globalPhotonCount, gatherRadius, indirectBrightness, globalKDTree.getDeviceTree(), d_indirectBuffer);
    else
        cudaMemset(d_indirectBuffer, 0, width * height * sizeof(float4));

    if (causticPhotonCount > 0)
        optixManager->launchCausticLighting(width, height, *camera, d_causticPhotonMap, causticPhotonCount, gatherRadius * Constants::Photon::CAUSTIC_RADIUS_MULTIPLIER, causticBrightness, causticKDTree.getDeviceTree(), d_causticBuffer);
    else
        cudaMemset(d_causticBuffer, 0, width * height * sizeof(float4));

    // Use specParams with skip_fog=true for combined rendering
    auto specParamsNoFog = specParams;
    specParamsNoFog.skip_fog = true;
    optixManager->launchSpecularLighting(width, height, *camera,
                                         d_globalPhotonMap, globalPhotonCount, globalKDTree.getDeviceTree(),
                                         d_causticPhotonMap, causticPhotonCount, causticKDTree.getDeviceTree(),
                                         specParamsNoFog, d_specularBuffer);

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

    // Run all pipelines (skip fog in individual passes)
    optixManager->launchDirectLighting(width, height, *camera, directAmbient, directShadowAmbient, directIntensity, directAttenuation, d_directBuffer, true);

    if (globalPhotonCount > 0)
        optixManager->launchIndirectLighting(width, height, *camera, d_globalPhotonMap, globalPhotonCount, gatherRadius, indirectBrightness, globalKDTree.getDeviceTree(), d_indirectBuffer);
    else
        cudaMemset(d_indirectBuffer, 0, width * height * sizeof(float4));

    if (causticPhotonCount > 0)
        optixManager->launchCausticLighting(width, height, *camera, d_causticPhotonMap, causticPhotonCount, gatherRadius * Constants::Photon::CAUSTIC_RADIUS_MULTIPLIER, causticBrightness, causticKDTree.getDeviceTree(), d_causticBuffer);
    else
        cudaMemset(d_causticBuffer, 0, width * height * sizeof(float4));

    // Use specParams with skip_fog=true for combined rendering
    auto specParamsNoFog = specParams;
    specParamsNoFog.skip_fog = true;
    optixManager->launchSpecularLighting(width, height, *camera,
                                         d_globalPhotonMap, globalPhotonCount, globalKDTree.getDeviceTree(),
                                         d_causticPhotonMap, causticPhotonCount, causticKDTree.getDeviceTree(),
                                         specParamsNoFog, d_specularBuffer);

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

