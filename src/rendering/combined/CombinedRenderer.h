#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <sutil/vec_math.h>
#include <vector>
#include <cuda_runtime.h>
#include <string>

#include "../../scene/Camera.h"
#include "../../scene/Scene.h"
#include "../../optix/OptixManager.h"
#include "../../ui/WindowManager.h"
#include "../photon/Photon.h"
#include "../photon/PhotonKDTree.h"
#include "../photon/VolumePhoton.h"  // For VolumeProperties

class CombinedRenderer
{
private:
    GLFWwindow* window;
    ViewportRect viewport;
    const Scene* scene;
    const Camera* camera;
    OptixManager* optixManager;

    GLuint textureID = 0;
    GLuint vao = 0;
    GLuint vbo = 0;
    GLuint shaderProgram = 0;

    unsigned int bufferWidth = 0;
    unsigned int bufferHeight = 0;
    
    // Individual buffers for each mode
    float4* d_directBuffer = nullptr;
    float4* d_indirectBuffer = nullptr;
    float4* d_causticBuffer = nullptr;
    float4* d_specularBuffer = nullptr;
    float4* d_combinedBuffer = nullptr;
    
    std::vector<float4> h_combinedBuffer;

    // Photon maps on GPU
    Photon* d_globalPhotonMap = nullptr;
    unsigned int globalPhotonCount = 0;
    Photon* d_causticPhotonMap = nullptr;
    unsigned int causticPhotonCount = 0;
    PhotonKDTree globalKDTree;
    PhotonKDTree causticKDTree;

    // Weights
    float directWeight = 1.0f;
    float indirectWeight = 1.0f;
    float causticWeight = 1.0f;
    float specularWeight = 1.0f;
    float gatherRadius = 100.0f;
    float indirectBrightness = 50000.0f;
    float causticBrightness = 100000.0f;
    
    // Direct lighting params
    float directAmbient = 0.03f;
    float directShadowAmbient = 0.02f;
    float directIntensity = 0.5f;
    float directAttenuation = 0.00001f;
    
    // Specular params
    OptixManager::SpecularParams specParams;

    // Fog parameters (Jensen's algorithm - applied once to final combined result)
    bool fogEnabled = false;
    VolumeProperties volumeProps;
    float3 fogColor = make_float3(0.15f, 0.15f, 0.18f);

    bool initialized = false;

    void setupOpenGL();
    void createFullscreenQuad();
    void createShader();
    void allocateBuffers(unsigned int width, unsigned int height);
    void combineBuffers(unsigned int width, unsigned int height);

    // Fog helpers (Jensen's algorithm - CPU implementation)
    float3 applyFogToPixel(float3 color, float3 rayOrigin, float3 rayDir, float hitDistance);

public:
    CombinedRenderer(GLFWwindow* win, const ViewportRect& vp);
    ~CombinedRenderer();

    bool initialize(OptixManager* om);
    void setCamera(const Camera* cam) { camera = cam; }
    void setScene(const Scene* s) { scene = s; }
    void setViewport(const ViewportRect& vp);

    void uploadGlobalPhotonMap(const std::vector<Photon>& photons);
    void uploadCausticPhotonMap(const std::vector<Photon>& caustics);
    
    void setWeights(float direct, float indirect, float caustic, float specular);
    void setGatherRadius(float radius) { gatherRadius = radius; }
    void setBrightnessMultipliers(float indirect, float caustic) { indirectBrightness = indirect; causticBrightness = caustic; }
    void setDirectLightingParams(float amb, float shadowAmb, float inten, float atten) {
        directAmbient = amb; directShadowAmbient = shadowAmb; directIntensity = inten; directAttenuation = atten;
    }
    void setSpecularParams(const OptixManager::SpecularParams& params) { specParams = params; }

    // Fog settings (Jensen's algorithm - fog applied once to final combined result)
    void setFogEnabled(bool enabled) { fogEnabled = enabled; }
    void setVolumeProperties(const VolumeProperties& props) { volumeProps = props; }
    void setFogColor(const float3& color) { fogColor = color; }

    void render();
    void exportToImage(const std::string& filename);
};

