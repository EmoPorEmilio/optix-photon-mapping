#pragma once

#include <sutil/vec_math.h>
#include "OpenGLManager.h"
#include "InputCommandManager.h"
#include "RasterRenderer.h"
#include "Camera.h"
#include "Scene.h"
#include "Sphere.h"
#include "Triangle.h"
#include "ConfigLoader.h"
#include "ObjLoader.h"
#include "PhotonMapper.h"
#include "PhotonMapRenderer.h"
#include "optix/OptixManager.h"
#include <iostream>
#include <memory>

class Application
{
private:
    OpenGLManager glManager;
    std::unique_ptr<InputCommandManager> inputCommandManager;
    std::unique_ptr<RasterRenderer> leftRenderer;
    std::unique_ptr<PhotonMapRenderer> rightRenderer;
    Camera camera;
    Scene scene;
    OptixManager optixManager;
    std::unique_ptr<PhotonMapper> photonMapper;
    bool isRunning = false;
    bool photonsEmitted = false;

    enum RenderMode
    {
        MODE_PHOTON_DOTS,
        MODE_GATHER
    };
    RenderMode currentMode = MODE_PHOTON_DOTS;

public:
    Application() = default;
    ~Application() = default;

    bool initialize();
    void run();
    void shutdown();
};



