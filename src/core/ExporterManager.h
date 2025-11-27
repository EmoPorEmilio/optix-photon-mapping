#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <sutil/vec_math.h>

#include "../rendering/photon/Photon.h"
#include "../rendering/photon/PhotonMapIO.h"
#include "../rendering/photon/PhotonTrajectory.h"
#include "../rendering/photon/TrajectoryExporter.h"
#include "../rendering/photon/PhotonKDTree.h"
#include "../scene/Camera.h"
#include "Constants.h"

// Forward declarations
class OptixManager;
class Scene;

//=============================================================================
// ExporterManager
// Centralized manager for all export operations:
// - Rendered images (JPG) for each lighting mode
// - Photon map data (TXT)
// - Photon trajectories (TXT)
//=============================================================================
class ExporterManager
{
public:
    ExporterManager() = default;
    ~ExporterManager();

    // Initialize with rendering context
    void initialize(OptixManager* optix, const Camera* cam, Scene* scene);

    // Set photon data (required for rendering modes that use photon maps)
    void setPhotonData(const std::vector<Photon>& global, const std::vector<Photon>& caustic);

    // Export everything to a directory
    void exportAll(const std::string& outputDir);

    // Export individual components
    void exportPhotonMap(const std::string& filename);
    void exportTrajectories(const std::vector<PhotonTrajectory>& trajectories, const std::string& filename);
    void exportAllRenderModes(const std::string& outputDir);
    
    // Export photon visualizations (dot displays like window mode)
    void exportGlobalPhotonVisualization(const std::string& filename);
    void exportCausticPhotonVisualization(const std::string& filename);

    // Configuration
    void setImageSize(unsigned int width, unsigned int height);
    void setGatherRadius(float radius) { gatherRadius = radius; }
    void setBrightnessMultipliers(float indirect, float caustic);
    void setDirectLightingParams(float ambient, float shadowAmbient, float intensity, float attenuation);

private:
    // Rendering context
    OptixManager* optixManager = nullptr;
    const Camera* camera = nullptr;
    Scene* scene = nullptr;

    // Photon data
    std::vector<Photon> globalPhotons;
    std::vector<Photon> causticPhotons;

    // GPU photon buffers
    Photon* d_globalPhotonMap = nullptr;
    Photon* d_causticPhotonMap = nullptr;
    
    // GPU output buffer
    float4* d_outputBuffer = nullptr;
    float4* h_outputBuffer = nullptr;

    // KD-trees for photon queries
    PhotonKDTree globalKDTree;
    PhotonKDTree causticKDTree;

    // Configuration
    unsigned int imageWidth = 768;
    unsigned int imageHeight = 768;
    float gatherRadius = 200.0f;
    float indirectBrightness = 25000.0f;
    float causticBrightness = 50000.0f;
    float directAmbient = 0.03f;
    float directShadowAmbient = 0.02f;
    float directIntensity = 0.5f;
    float directAttenuation = 0.00001f;

    bool initialized = false;
    bool buffersAllocated = false;
    bool photonsUploaded = false;

    // Internal helpers
    void allocateBuffers();
    void freeBuffers();
    void uploadPhotonsToGPU();
    bool saveBufferToPng(const std::string& filename);
    void copyBufferToHost();
    void applyGammaCorrection();
    void createDirectory(const std::string& path);
    void renderPhotonsToBuffer(const std::vector<Photon>& photons, std::vector<unsigned char>& rgbBuffer);
};
