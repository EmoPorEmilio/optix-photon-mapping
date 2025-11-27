#include "ExporterManager.h"
#include "../optix/OptixManager.h"
#include "../scene/Scene.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../utils/stb_image_write.h"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <direct.h>  // For _mkdir on Windows

ExporterManager::~ExporterManager()
{
    freeBuffers();
}

void ExporterManager::initialize(OptixManager* optix, const Camera* cam, Scene* scn)
{
    optixManager = optix;
    camera = cam;
    scene = scn;
    initialized = true;
}

void ExporterManager::setPhotonData(const std::vector<Photon>& global, const std::vector<Photon>& caustic)
{
    globalPhotons = global;
    causticPhotons = caustic;
    photonsUploaded = false;  // Need to re-upload
}

void ExporterManager::setImageSize(unsigned int width, unsigned int height)
{
    if (width != imageWidth || height != imageHeight)
    {
        imageWidth = width;
        imageHeight = height;
        freeBuffers();  // Will reallocate on next use
    }
}

void ExporterManager::setBrightnessMultipliers(float indirect, float caustic)
{
    indirectBrightness = indirect;
    causticBrightness = caustic;
}

void ExporterManager::setDirectLightingParams(float ambient, float shadowAmbient, float intensity, float attenuation)
{
    directAmbient = ambient;
    directShadowAmbient = shadowAmbient;
    directIntensity = intensity;
    directAttenuation = attenuation;
}

void ExporterManager::createDirectory(const std::string& path)
{
    _mkdir(path.c_str());
}

void ExporterManager::allocateBuffers()
{
    if (buffersAllocated)
        return;

    size_t bufferSize = imageWidth * imageHeight * sizeof(float4);
    cudaMalloc(&d_outputBuffer, bufferSize);
    h_outputBuffer = new float4[imageWidth * imageHeight];
    buffersAllocated = true;
}

void ExporterManager::freeBuffers()
{
    if (d_outputBuffer)
    {
        cudaFree(d_outputBuffer);
        d_outputBuffer = nullptr;
    }
    if (h_outputBuffer)
    {
        delete[] h_outputBuffer;
        h_outputBuffer = nullptr;
    }
    if (d_globalPhotonMap)
    {
        cudaFree(d_globalPhotonMap);
        d_globalPhotonMap = nullptr;
    }
    if (d_causticPhotonMap)
    {
        cudaFree(d_causticPhotonMap);
        d_causticPhotonMap = nullptr;
    }
    buffersAllocated = false;
    photonsUploaded = false;
}

void ExporterManager::uploadPhotonsToGPU()
{
    if (photonsUploaded)
        return;

    // Upload global photons
    if (!globalPhotons.empty())
    {
        if (d_globalPhotonMap)
            cudaFree(d_globalPhotonMap);
        cudaMalloc(&d_globalPhotonMap, globalPhotons.size() * sizeof(Photon));
        cudaMemcpy(d_globalPhotonMap, globalPhotons.data(), 
                   globalPhotons.size() * sizeof(Photon), cudaMemcpyHostToDevice);
        globalKDTree.build(globalPhotons);
    }
    else
    {
        globalKDTree.clear();
    }

    // Upload caustic photons
    if (!causticPhotons.empty())
    {
        if (d_causticPhotonMap)
            cudaFree(d_causticPhotonMap);
        cudaMalloc(&d_causticPhotonMap, causticPhotons.size() * sizeof(Photon));
        cudaMemcpy(d_causticPhotonMap, causticPhotons.data(),
                   causticPhotons.size() * sizeof(Photon), cudaMemcpyHostToDevice);
        causticKDTree.build(causticPhotons);
    }
    else
    {
        causticKDTree.clear();
    }

    photonsUploaded = true;
}

void ExporterManager::copyBufferToHost()
{
    cudaMemcpy(h_outputBuffer, d_outputBuffer, 
               imageWidth * imageHeight * sizeof(float4), cudaMemcpyDeviceToHost);
}

void ExporterManager::applyGammaCorrection()
{
    for (unsigned int i = 0; i < imageWidth * imageHeight; ++i)
    {
        h_outputBuffer[i].x = powf(std::max(0.0f, std::min(1.0f, h_outputBuffer[i].x)), Constants::Render::INV_GAMMA);
        h_outputBuffer[i].y = powf(std::max(0.0f, std::min(1.0f, h_outputBuffer[i].y)), Constants::Render::INV_GAMMA);
        h_outputBuffer[i].z = powf(std::max(0.0f, std::min(1.0f, h_outputBuffer[i].z)), Constants::Render::INV_GAMMA);
    }
}

bool ExporterManager::saveBufferToPng(const std::string& filename)
{
    // Convert float4 buffer to RGB bytes (flipped vertically for image format)
    std::vector<unsigned char> rgbData(imageWidth * imageHeight * 3);

    for (unsigned int y = 0; y < imageHeight; ++y)
    {
        for (unsigned int x = 0; x < imageWidth; ++x)
        {
            // Flip Y for image output (OpenGL/CUDA has origin at bottom-left)
            unsigned int srcIdx = (imageHeight - 1 - y) * imageWidth + x;
            unsigned int dstIdx = (y * imageWidth + x) * 3;

            float4 pixel = h_outputBuffer[srcIdx];
            
            // Clamp and convert to bytes
            rgbData[dstIdx + 0] = static_cast<unsigned char>(std::min(255.0f, std::max(0.0f, pixel.x * 255.0f)));
            rgbData[dstIdx + 1] = static_cast<unsigned char>(std::min(255.0f, std::max(0.0f, pixel.y * 255.0f)));
            rgbData[dstIdx + 2] = static_cast<unsigned char>(std::min(255.0f, std::max(0.0f, pixel.z * 255.0f)));
        }
    }

    // stride_in_bytes = width * channels
    int result = stbi_write_png(filename.c_str(), imageWidth, imageHeight, 3, rgbData.data(), imageWidth * 3);
    return result != 0;
}

void ExporterManager::exportPhotonMap(const std::string& filename)
{
    PhotonMapIO::exportToFile(globalPhotons, causticPhotons, filename);
}

void ExporterManager::exportTrajectories(const std::vector<PhotonTrajectory>& trajectories, const std::string& filename)
{
    TrajectoryExporter::exportToFile(trajectories, filename);
}

void ExporterManager::exportAllRenderModes(const std::string& outputDir)
{
    if (!initialized || !optixManager || !camera)
    {
        std::cerr << "ExporterManager: Not initialized" << std::endl;
        return;
    }

    std::cout << "\n=== Exporting all render modes to " << outputDir << " ===" << std::endl;

    createDirectory(outputDir);
    allocateBuffers();
    uploadPhotonsToGPU();

    // 1. Direct lighting
    std::cout << "Rendering direct lighting..." << std::endl;
    optixManager->launchDirectLighting(imageWidth, imageHeight, *camera,
                                       directAmbient, directShadowAmbient,
                                       directIntensity, directAttenuation,
                                       d_outputBuffer);
    copyBufferToHost();
    applyGammaCorrection();
    std::string filename = outputDir + "/direct_lighting.png";
    if (saveBufferToPng(filename))
        std::cout << "  Saved: " << filename << std::endl;

    // 2. Indirect lighting
    if (!globalPhotons.empty())
    {
        std::cout << "Rendering indirect lighting..." << std::endl;
        optixManager->launchIndirectLighting(imageWidth, imageHeight, *camera,
                                             d_globalPhotonMap,
                                             static_cast<unsigned int>(globalPhotons.size()),
                                             gatherRadius, indirectBrightness,
                                             globalKDTree.getDeviceTree(),
                                             d_outputBuffer);
        copyBufferToHost();
        applyGammaCorrection();
        filename = outputDir + "/indirect_lighting.png";
        if (saveBufferToPng(filename))
            std::cout << "  Saved: " << filename << std::endl;
    }

    // 3. Caustic lighting
    if (!causticPhotons.empty())
    {
        std::cout << "Rendering caustic lighting..." << std::endl;
        optixManager->launchCausticLighting(imageWidth, imageHeight, *camera,
                                            d_causticPhotonMap,
                                            static_cast<unsigned int>(causticPhotons.size()),
                                            gatherRadius * Constants::Photon::CAUSTIC_RADIUS_MULTIPLIER,
                                            causticBrightness,
                                            causticKDTree.getDeviceTree(),
                                            d_outputBuffer);
        copyBufferToHost();
        applyGammaCorrection();
        filename = outputDir + "/caustic_lighting.png";
        if (saveBufferToPng(filename))
            std::cout << "  Saved: " << filename << std::endl;
    }

    // 4. Specular lighting
    std::cout << "Rendering specular lighting..." << std::endl;
    OptixManager::SpecularParams specParams;
    specParams.gather_radius = gatherRadius;
    specParams.indirect_brightness = indirectBrightness * 2.0f;
    specParams.caustic_brightness = causticBrightness * 2.0f;

    optixManager->launchSpecularLighting(imageWidth, imageHeight, *camera,
                                         d_globalPhotonMap,
                                         static_cast<unsigned int>(globalPhotons.size()),
                                         globalKDTree.getDeviceTree(),
                                         d_causticPhotonMap,
                                         static_cast<unsigned int>(causticPhotons.size()),
                                         causticKDTree.getDeviceTree(),
                                         specParams, d_outputBuffer);
    copyBufferToHost();
    applyGammaCorrection();
    filename = outputDir + "/specular_lighting.png";
    if (saveBufferToPng(filename))
        std::cout << "  Saved: " << filename << std::endl;

    // 5. Combined (render all and combine)
    std::cout << "Rendering combined image..." << std::endl;
    
    float4 *d_direct, *d_indirect, *d_caustic, *d_specular;
    size_t bufSize = imageWidth * imageHeight * sizeof(float4);
    cudaMalloc(&d_direct, bufSize);
    cudaMalloc(&d_indirect, bufSize);
    cudaMalloc(&d_caustic, bufSize);
    cudaMalloc(&d_specular, bufSize);

    optixManager->launchDirectLighting(imageWidth, imageHeight, *camera,
                                       directAmbient, directShadowAmbient,
                                       directIntensity, directAttenuation,
                                       d_direct);

    if (!globalPhotons.empty())
        optixManager->launchIndirectLighting(imageWidth, imageHeight, *camera,
                                             d_globalPhotonMap,
                                             static_cast<unsigned int>(globalPhotons.size()),
                                             gatherRadius, indirectBrightness,
                                             globalKDTree.getDeviceTree(),
                                             d_indirect);
    else
        cudaMemset(d_indirect, 0, bufSize);

    if (!causticPhotons.empty())
        optixManager->launchCausticLighting(imageWidth, imageHeight, *camera,
                                            d_causticPhotonMap,
                                            static_cast<unsigned int>(causticPhotons.size()),
                                            gatherRadius * Constants::Photon::CAUSTIC_RADIUS_MULTIPLIER,
                                            causticBrightness,
                                            causticKDTree.getDeviceTree(),
                                            d_caustic);
    else
        cudaMemset(d_caustic, 0, bufSize);

    optixManager->launchSpecularLighting(imageWidth, imageHeight, *camera,
                                         d_globalPhotonMap,
                                         static_cast<unsigned int>(globalPhotons.size()),
                                         globalKDTree.getDeviceTree(),
                                         d_causticPhotonMap,
                                         static_cast<unsigned int>(causticPhotons.size()),
                                         causticKDTree.getDeviceTree(),
                                         specParams, d_specular);

    // Copy all to host and combine
    std::vector<float4> h_direct(imageWidth * imageHeight);
    std::vector<float4> h_indirect(imageWidth * imageHeight);
    std::vector<float4> h_caustic(imageWidth * imageHeight);
    std::vector<float4> h_specular(imageWidth * imageHeight);

    cudaMemcpy(h_direct.data(), d_direct, bufSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_indirect.data(), d_indirect, bufSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_caustic.data(), d_caustic, bufSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_specular.data(), d_specular, bufSize, cudaMemcpyDeviceToHost);

    // Combine with weights
    for (unsigned int i = 0; i < imageWidth * imageHeight; ++i)
    {
        float r = h_direct[i].x + h_indirect[i].x + h_caustic[i].x + h_specular[i].x * 0.5f;
        float g = h_direct[i].y + h_indirect[i].y + h_caustic[i].y + h_specular[i].y * 0.5f;
        float b = h_direct[i].z + h_indirect[i].z + h_caustic[i].z + h_specular[i].z * 0.5f;

        // Apply gamma
        r = powf(std::max(0.0f, std::min(1.0f, r)), Constants::Render::INV_GAMMA);
        g = powf(std::max(0.0f, std::min(1.0f, g)), Constants::Render::INV_GAMMA);
        b = powf(std::max(0.0f, std::min(1.0f, b)), Constants::Render::INV_GAMMA);

        h_outputBuffer[i] = make_float4(r, g, b, 1.0f);
    }

    cudaFree(d_direct);
    cudaFree(d_indirect);
    cudaFree(d_caustic);
    cudaFree(d_specular);

    filename = outputDir + "/combined.png";
    if (saveBufferToPng(filename))
        std::cout << "  Saved: " << filename << std::endl;

    std::cout << "=== Export complete ===" << std::endl;
}

void ExporterManager::exportAll(const std::string& outputDir)
{
    std::cout << "\n=== Full Export to " << outputDir << " ===" << std::endl;

    createDirectory(outputDir);

    // Export photon map data
    exportPhotonMap(outputDir + "/photon_map.txt");

    // Export all rendered images
    exportAllRenderModes(outputDir);

    std::cout << "=== Full export complete ===" << std::endl;
}
