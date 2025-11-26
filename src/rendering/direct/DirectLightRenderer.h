#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <sutil/vec_math.h>
#include <cuda_runtime.h>
#include "../../scene/Camera.h"
#include "../../scene/Scene.h"
#include "../../ui/WindowManager.h"

class OptixManager;

// Renderer for direct lighting using OptiX raytracing
// Traces camera rays and computes direct illumination with shadow rays
class DirectLightRenderer
{
private:
    ViewportRect viewport;
    const Camera* camera = nullptr;
    const Scene* scene = nullptr;
    OptixManager* optixManager = nullptr;

    // OpenGL resources for displaying the rendered image
    GLuint textureID = 0;
    GLuint vao = 0;
    GLuint vbo = 0;
    GLuint shaderProgram = 0;

    // CUDA/GPU resources
    float4* d_frameBuffer = nullptr;
    std::vector<float4> h_frameBuffer;
    unsigned int bufferWidth = 0;
    unsigned int bufferHeight = 0;

    // Configurable lighting parameters
    float ambient = 0.03f;
    float shadowAmbient = 0.02f;
    float intensity = 0.5f;
    float attenuationFactor = 0.00001f;

    bool initialized = false;

    void createShaders();
    void createQuad();
    void allocateBuffers(unsigned int width, unsigned int height);

public:
    DirectLightRenderer() = default;
    ~DirectLightRenderer();

    bool initialize(GLFWwindow* window);
    void shutdown();

    void setViewport(const ViewportRect& vp) { viewport = vp; }
    void setCamera(const Camera* cam) { camera = cam; }
    void setScene(const Scene* s) { scene = s; }
    void setOptixManager(OptixManager* mgr) { optixManager = mgr; }
    void setLightingParams(float amb, float shadowAmb, float inten, float atten) {
        ambient = amb; shadowAmbient = shadowAmb; intensity = inten; attenuationFactor = atten;
    }

    // Render one frame of direct lighting
    void render();

    bool isInitialized() const { return initialized; }
};

