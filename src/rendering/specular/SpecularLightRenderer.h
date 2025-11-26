#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <sutil/vec_math.h>
#include <vector>
#include <cuda_runtime.h>

#include "../../scene/Camera.h"
#include "../../scene/Scene.h"
#include "../../optix/OptixManager.h"
#include "../../ui/WindowManager.h"
#include "../photon/Photon.h"
#include "../photon/PhotonKDTree.h"

class SpecularLightRenderer
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
    float4* d_frameBuffer = nullptr;
    std::vector<float4> h_frameBuffer;

    // Photon maps on GPU for full lighting
    Photon* d_globalPhotonMap = nullptr;
    unsigned int globalPhotonCount = 0;
    Photon* d_causticPhotonMap = nullptr;
    unsigned int causticPhotonCount = 0;
    PhotonKDTree globalKDTree;
    PhotonKDTree causticKDTree;
    
    // Specular parameters
    OptixManager::SpecularParams specParams;

    bool initialized = false;

    void setupOpenGL();
    void createFullscreenQuad();
    void createShader();
    void allocateBuffers(unsigned int width, unsigned int height);

public:
    SpecularLightRenderer(GLFWwindow* win, const ViewportRect& vp);
    ~SpecularLightRenderer();

    bool initialize(OptixManager* om);
    void setCamera(const Camera* cam) { camera = cam; }
    void setScene(const Scene* s) { scene = s; }
    void setViewport(const ViewportRect& vp);
    
    void uploadGlobalPhotonMap(const std::vector<Photon>& photons);
    void uploadCausticPhotonMap(const std::vector<Photon>& caustics);
    void setGatherRadius(float radius) { specParams.gather_radius = radius; }
    void setSpecularParams(const OptixManager::SpecularParams& params) { specParams = params; }

    void render();
};

