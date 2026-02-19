#include "ExporterManager.h"
#include "PerformanceManager.h"
#include "../optix/OptixManager.h"
#include "../scene/Scene.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../utils/stb_image_write.h"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <direct.h>  // For _mkdir on Windows
#include <sys/stat.h>

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
    photonsUploaded = false;
}

void ExporterManager::setImageSize(unsigned int width, unsigned int height)
{
    if (width != imageWidth || height != imageHeight)
    {
        imageWidth = width;
        imageHeight = height;
        freeBuffers();
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

void ExporterManager::createDirectoryRecursive(const std::string& path)
{
    // Create each directory in the path
    std::string currentPath;
    for (size_t i = 0; i < path.size(); ++i)
    {
        currentPath += path[i];
        if (path[i] == '/' || path[i] == '\\' || i == path.size() - 1)
        {
            _mkdir(currentPath.c_str());
        }
    }
}

std::string ExporterManager::getPerformanceMetricsPath(const std::string& baseDir)
{
    // Create performance_metrics subfolder
    std::string metricsBaseDir = baseDir + "/performance_metrics";
    createDirectory(metricsBaseDir);
    
    // Get current date
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::tm* localTime = std::localtime(&time_t_now);
    
    // Format: MM-DD-YYYY
    char dateStr[20];
    std::strftime(dateStr, sizeof(dateStr), "%m-%d-%Y", localTime);
    
    // Create date folder
    std::string dateDir = metricsBaseDir + "/" + dateStr;
    createDirectory(dateDir);
    
    // Find next available ID number
    int fileId = 1;
    while (true)
    {
        std::string testPath = dateDir + "/" + std::to_string(fileId) + ".txt";
        struct stat buffer;
        if (stat(testPath.c_str(), &buffer) != 0)
        {
            // File doesn't exist, use this ID
            return testPath;
        }
        fileId++;
    }
}

std::string ExporterManager::generateConfigSummary() const
{
    if (!hasConfig)
        return "";
    
    std::ostringstream oss;
    oss << "================================================================================\n";
    oss << "                            CONFIGURATION USED                                  \n";
    oss << "================================================================================\n";
    
    oss << "\n[Photon Mapping]\n";
    oss << "  Max Photons: " << currentConfig.max_photons << "\n";
    oss << "  Collision Radius: " << currentConfig.photon_collision_radius << "\n";
    
    oss << "\n[Animation]\n";
    oss << "  Enabled: " << (currentConfig.animation.enabled ? "true" : "false") << "\n";
    oss << "  Speed: " << currentConfig.animation.photonSpeed << "\n";
    oss << "  Emission Interval: " << currentConfig.animation.emissionInterval << "\n";
    
    oss << "\n[Debug]\n";
    oss << "  Record Trajectories: " << (currentConfig.debug.record_trajectories ? "true" : "false") << "\n";
    oss << "  Save Photon Map: " << (currentConfig.debug.save_photon_map ? "true" : "false") << "\n";
    oss << "  Load Photon Map: " << (currentConfig.debug.load_photon_map ? "true" : "false") << "\n";
    oss << "  Photon Map File: " << currentConfig.debug.photon_map_file << "\n";
    oss << "  Export Images: " << (currentConfig.debug.export_images ? "true" : "false") << "\n";
    oss << "  Export Metrics: " << (currentConfig.debug.export_metrics ? "true" : "false") << "\n";
    oss << "  Export Dir: " << currentConfig.debug.export_dir << "\n";
    
    oss << "\n[Gathering]\n";
    oss << "  Indirect Radius: " << currentConfig.gathering.indirect_radius << "\n";
    oss << "  Caustic Radius: " << currentConfig.gathering.caustic_radius << "\n";
    oss << "  Indirect Brightness: " << currentConfig.gathering.indirect_brightness << "\n";
    oss << "  Caustic Brightness: " << currentConfig.gathering.caustic_brightness << "\n";
    
    oss << "\n[Direct Lighting]\n";
    oss << "  Ambient: " << currentConfig.direct_lighting.ambient << "\n";
    oss << "  Shadow Ambient: " << currentConfig.direct_lighting.shadow_ambient << "\n";
    oss << "  Intensity: " << currentConfig.direct_lighting.intensity << "\n";
    oss << "  Attenuation: " << currentConfig.direct_lighting.attenuation_factor << "\n";
    
    oss << "\n[Specular]\n";
    oss << "  Max Depth: " << currentConfig.specular.max_recursion_depth << "\n";
    oss << "  Glass IOR: " << currentConfig.specular.glass_ior << "\n";
    oss << "  Mirror Reflectivity: " << currentConfig.specular.mirror_reflectivity << "\n";
    oss << "  Fresnel Min: " << currentConfig.specular.fresnel_min << "\n";
    oss << "  Glass Tint: (" << currentConfig.specular.glass_tint.x << ", " 
                            << currentConfig.specular.glass_tint.y << ", " 
                            << currentConfig.specular.glass_tint.z << ")\n";
    
    oss << "\n[Weights]\n";
    oss << "  Direct: " << currentConfig.weights.direct << "\n";
    oss << "  Indirect: " << currentConfig.weights.indirect << "\n";
    oss << "  Caustics: " << currentConfig.weights.caustics << "\n";
    oss << "  Specular: " << currentConfig.weights.specular << "\n";
    
    oss << "\n[Scene Objects]\n";
    oss << "  Spheres: " << currentConfig.spheres.size() << "\n";
    oss << "  Meshes: " << currentConfig.meshes.size() << "\n";
    oss << "  Quads: " << currentConfig.quads.size() << "\n";
    
    return oss.str();
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
    std::vector<unsigned char> rgbData(imageWidth * imageHeight * 3);

    for (unsigned int y = 0; y < imageHeight; ++y)
    {
        for (unsigned int x = 0; x < imageWidth; ++x)
        {
            unsigned int srcIdx = (imageHeight - 1 - y) * imageWidth + x;  // flip Y
            unsigned int dstIdx = (y * imageWidth + x) * 3;

            float4 pixel = h_outputBuffer[srcIdx];
            rgbData[dstIdx + 0] = static_cast<unsigned char>(std::min(255.0f, std::max(0.0f, pixel.x * 255.0f)));
            rgbData[dstIdx + 1] = static_cast<unsigned char>(std::min(255.0f, std::max(0.0f, pixel.y * 255.0f)));
            rgbData[dstIdx + 2] = static_cast<unsigned char>(std::min(255.0f, std::max(0.0f, pixel.z * 255.0f)));
        }
    }

    return stbi_write_png(filename.c_str(), imageWidth, imageHeight, 3, rgbData.data(), imageWidth * 3) != 0;
}

void ExporterManager::exportPhotonMap(const std::string& filename)
{
    PhotonMapIO::exportToFile(globalPhotons, causticPhotons, filename);
}

void ExporterManager::renderPhotonsToBuffer(const std::vector<Photon>& photons, std::vector<unsigned char>& rgbBuffer)
{
    std::fill(rgbBuffer.begin(), rgbBuffer.end(), static_cast<unsigned char>(5));

    if (!camera || photons.empty())
        return;

    float3 eye = camera->getPosition();
    float3 U_exact = camera->getU();
    float3 V_exact = camera->getV();
    float3 W_exact = camera->getW();

    float wlen = length(camera->getLookAt() - eye);
    float vlen = length(V_exact);
    float ulen = length(U_exact);

    float3 f = normalize(camera->getLookAt() - eye);

    for (size_t i = 0; i < photons.size(); ++i)
    {
        float3 p = photons[i].position;
        float3 d = p - eye;

        float dw = dot(d, f);
        if (dw <= 0.0f)
            continue;

        float a = dot(d, U_exact) / (ulen * ulen);
        float b = dot(d, V_exact) / (vlen * vlen);
        float c = dot(d, W_exact) / (wlen * wlen);
        if (c <= 0.0f)
            continue;

        float ndcX = a / c;
        float ndcY = b / c;
        if (ndcX < -1.0f || ndcX > 1.0f || ndcY < -1.0f || ndcY > 1.0f)
            continue;

        int px = static_cast<int>((ndcX * 0.5f + 0.5f) * (imageWidth - 1));
        int py = static_cast<int>((ndcY * 0.5f + 0.5f) * (imageHeight - 1));
        
        if (px < 0 || py < 0 || px >= (int)imageWidth || py >= (int)imageHeight)
            continue;

        int flippedY = imageHeight - 1 - py;
        size_t idx = (static_cast<size_t>(flippedY) * imageWidth + static_cast<size_t>(px)) * 3;

        float3 power = photons[i].power;
        float maxComp = fmaxf(fmaxf(power.x, power.y), power.z);
        float3 col = (maxComp > 0.0f) ? power / maxComp : make_float3(0.5f, 0.5f, 0.5f);

        rgbBuffer[idx + 0] = static_cast<unsigned char>(col.x * 255.0f);
        rgbBuffer[idx + 1] = static_cast<unsigned char>(col.y * 255.0f);
        rgbBuffer[idx + 2] = static_cast<unsigned char>(col.z * 255.0f);
    }
}

void ExporterManager::exportGlobalPhotonVisualization(const std::string& filename)
{
    if (globalPhotons.empty())
    {
        std::cerr << "ExporterManager: No global photons to visualize" << std::endl;
        return;
    }

    std::cout << "Rendering global photon visualization (" << globalPhotons.size() << " photons)..." << std::endl;

    std::vector<unsigned char> rgbBuffer(imageWidth * imageHeight * 3);
    renderPhotonsToBuffer(globalPhotons, rgbBuffer);

    int result = stbi_write_png(filename.c_str(), imageWidth, imageHeight, 3, rgbBuffer.data(), imageWidth * 3);
    if (result)
        std::cout << "  Saved: " << filename << std::endl;
    else
        std::cerr << "  Failed to save: " << filename << std::endl;
}

void ExporterManager::exportCausticPhotonVisualization(const std::string& filename)
{
    if (causticPhotons.empty())
    {
        std::cerr << "ExporterManager: No caustic photons to visualize" << std::endl;
        return;
    }

    std::cout << "Rendering caustic photon visualization (" << causticPhotons.size() << " photons)..." << std::endl;

    std::vector<unsigned char> rgbBuffer(imageWidth * imageHeight * 3);
    renderPhotonsToBuffer(causticPhotons, rgbBuffer);

    int result = stbi_write_png(filename.c_str(), imageWidth, imageHeight, 3, rgbBuffer.data(), imageWidth * 3);
    if (result)
        std::cout << "  Saved: " << filename << std::endl;
    else
        std::cerr << "  Failed to save: " << filename << std::endl;
}

void ExporterManager::exportTrajectories(const std::vector<PhotonTrajectory>& trajectories, const std::string& filename)
{
    TrajectoryExporter::exportToFile(trajectories, filename);
}

// CPU implementation of fog for combined export (Jensen's algorithm)
float3 ExporterManager::applyFogToPixel(float3 color, float3 rayOrigin, float3 rayDir, float hitDistance)
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

    if (!globalPhotons.empty())
        exportGlobalPhotonVisualization(outputDir + "/global_photons.png");

    if (!causticPhotons.empty())
        exportCausticPhotonVisualization(outputDir + "/caustic_photons.png");
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

    std::cout << "Rendering combined image..." << std::endl;

    float4 *d_direct, *d_indirect, *d_caustic, *d_specular;
    size_t bufSize = imageWidth * imageHeight * sizeof(float4);
    cudaMalloc(&d_direct, bufSize);
    cudaMalloc(&d_indirect, bufSize);
    cudaMalloc(&d_caustic, bufSize);
    cudaMalloc(&d_specular, bufSize);

    // Skip fog in individual passes for combined rendering (fog would double-apply)
    optixManager->launchDirectLighting(imageWidth, imageHeight, *camera,
                                       directAmbient, directShadowAmbient,
                                       directIntensity, directAttenuation,
                                       d_direct, true);  // skip_fog=true

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

    // Use specParams with skip_fog=true for combined rendering
    auto specParamsNoFog = specParams;
    specParamsNoFog.skip_fog = true;
    optixManager->launchSpecularLighting(imageWidth, imageHeight, *camera,
                                         d_globalPhotonMap,
                                         static_cast<unsigned int>(globalPhotons.size()),
                                         globalKDTree.getDeviceTree(),
                                         d_causticPhotonMap,
                                         static_cast<unsigned int>(causticPhotons.size()),
                                         causticKDTree.getDeviceTree(),
                                         specParamsNoFog, d_specular);

    std::vector<float4> h_direct(imageWidth * imageHeight);
    std::vector<float4> h_indirect(imageWidth * imageHeight);
    std::vector<float4> h_caustic(imageWidth * imageHeight);
    std::vector<float4> h_specular(imageWidth * imageHeight);

    cudaMemcpy(h_direct.data(), d_direct, bufSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_indirect.data(), d_indirect, bufSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_caustic.data(), d_caustic, bufSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_specular.data(), d_specular, bufSize, cudaMemcpyDeviceToHost);

    // Get camera parameters for fog ray computation
    float3 eye = camera->getPosition();
    float3 U = camera->getU();
    float3 V = camera->getV();
    float3 W = camera->getW();

    // Combine with fog applied (Jensen's algorithm)
    for (unsigned int y = 0; y < imageHeight; ++y)
    {
        for (unsigned int x = 0; x < imageWidth; ++x)
        {
            unsigned int i = y * imageWidth + x;

            // Check if this pixel has specular content
            float specLuminance = h_specular[i].x + h_specular[i].y + h_specular[i].z;

            float r, g, b;
            float hitDistance;

            if (specLuminance > 0.01f)
            {
                // Sphere pixel: use specular
                r = h_specular[i].x;
                g = h_specular[i].y;
                b = h_specular[i].z;
                hitDistance = h_specular[i].w;
            }
            else
            {
                // Non-sphere pixel: combine direct + indirect + caustic
                r = h_direct[i].x + h_indirect[i].x + h_caustic[i].x;
                g = h_direct[i].y + h_indirect[i].y + h_caustic[i].y;
                b = h_direct[i].z + h_indirect[i].z + h_caustic[i].z;
                hitDistance = h_direct[i].w;
            }

            // Apply fog (Jensen's algorithm - once to final combined result)
            if (fogEnabled)
            {
                float u = (2.0f * (float(x) + 0.5f) / float(imageWidth)) - 1.0f;
                float v = (2.0f * (float(y) + 0.5f) / float(imageHeight)) - 1.0f;
                float3 rayDir = normalize(make_float3(
                    W.x + u * U.x + v * V.x,
                    W.y + u * U.y + v * V.y,
                    W.z + u * U.z + v * V.z));

                float3 color = applyFogToPixel(make_float3(r, g, b), eye, rayDir, hitDistance);
                r = color.x;
                g = color.y;
                b = color.z;
            }

            r = powf(std::max(0.0f, std::min(1.0f, r)), Constants::Render::INV_GAMMA);
            g = powf(std::max(0.0f, std::min(1.0f, g)), Constants::Render::INV_GAMMA);
            b = powf(std::max(0.0f, std::min(1.0f, b)), Constants::Render::INV_GAMMA);

            h_outputBuffer[i] = make_float4(r, g, b, 1.0f);
        }
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
    std::cout << "\n=== Exporting to " << outputDir << " ===" << std::endl;
    createDirectory(outputDir);
    exportPhotonMap(outputDir + "/photon_map.txt");
    exportAllRenderModes(outputDir);
    
    // Export performance metrics to dated subfolder
    exportPerformanceMetrics(outputDir);
    
    std::cout << "=== Done ===" << std::endl;
}

void ExporterManager::exportPerformanceMetrics(const std::string& baseDir)
{
    if (!PerformanceManager::instance().hasMetrics())
    {
        std::cout << "ExporterManager: No performance metrics to export" << std::endl;
        return;
    }

    // Print summary to console
    PerformanceManager::instance().printSummary();

    // Get the dated path (creates folders automatically)
    std::string metricsPath = getPerformanceMetricsPath(baseDir);

    std::ofstream file(metricsPath);
    if (!file.is_open())
    {
        std::cerr << "ExporterManager: Failed to open " << metricsPath << " for writing" << std::endl;
        return;
    }

    // Generate config summary
    std::string configSummary = generateConfigSummary();
    
    // Generate and write the full report with config
    std::string report = PerformanceManager::instance().generateMetricsReport(configSummary);
    file << report;
    file.close();

    std::cout << "  Saved: " << metricsPath << std::endl;
}
