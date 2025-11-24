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
#include <vector>
#include <chrono>

struct AnimatedPhoton
{
    float3 position;
    float3 direction;
    float3 velocity;
    bool isActive;
    
    AnimatedPhoton() : position(make_float3(0.0f, 0.0f, 0.0f)),
                       direction(make_float3(0.0f, 0.0f, 0.0f)),
                       velocity(make_float3(0.0f, 0.0f, 0.0f)),
                       isActive(true) {}
};

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

    // CPU Photon Animation
    std::vector<AnimatedPhoton> animatedPhotons;
    unsigned int maxPhotons = 10;
    std::chrono::steady_clock::time_point lastPhotonEmissionTime;
    float photonSpeed = 80.0f; // units per second - speed of photon movement
    float photonCollisionRadius = 10.0f; // collision detection radius

    void emitPhoton();
    void updatePhotons(float deltaTime);
    bool checkCollision(const float3& position);

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



