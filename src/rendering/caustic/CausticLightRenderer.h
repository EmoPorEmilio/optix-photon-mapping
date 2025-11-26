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

class CausticLightRenderer
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

    // Caustic photon map on GPU
    Photon* d_causticMap = nullptr;
    unsigned int causticCount = 0;
    float gatherRadius = 50.0f;

    bool initialized = false;

    void setupOpenGL();
    void createFullscreenQuad();
    void createShader();
    void allocateBuffers(unsigned int width, unsigned int height);

public:
    CausticLightRenderer(GLFWwindow* win, const ViewportRect& vp);
    ~CausticLightRenderer();

    bool initialize(OptixManager* om);
    void setCamera(const Camera* cam) { camera = cam; }
    void setScene(const Scene* s) { scene = s; }
    void setViewport(const ViewportRect& vp);

    void uploadCausticMap(const std::vector<Photon>& caustics);
    void setGatherRadius(float radius) { gatherRadius = radius; }

    void render();
};

