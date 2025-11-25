#pragma once

#include <sutil/vec_math.h>
#include "../ui/OpenGLManager.h"
#include "../ui/InputCommandManager.h"
#include "../rendering/raster/RasterRenderer.h"
#include "../scene/Camera.h"
#include "../scene/Scene.h"
#include "../scene/Sphere.h"
#include "../scene/Triangle.h"
#include "ConfigLoader.h"
#include "Constants.h"
#include "CollisionDetector.h"
#include "../scene/ObjLoader.h"
#include "../rendering/photon/PhotonMapper.h"
#include "../rendering/photon/PhotonMapRenderer.h"
#include "../rendering/photon/AnimatedPhoton.h"
#include "../rendering/photon/Photon.h"
#include "../optix/OptixManager.h"
#include <iostream>
#include <memory>
#include <vector>
#include <chrono>
#include <random>

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
    std::unique_ptr<CollisionDetector> collisionDetector;
    bool isRunning = false;
    bool photonsEmitted = false;

    // Animation mode settings
    bool animatedMode = true;      // If false, instant photon tracing
    float photonSpeed = 200.0f;    // units per second (animated mode)
    float emissionInterval = 0.5f; // seconds between emissions (animated mode)

    // CPU Photon Animation
    std::vector<AnimatedPhoton> animatedPhotons;
    unsigned int maxPhotons = 10;
    std::chrono::steady_clock::time_point lastPhotonEmissionTime;
    float photonCollisionRadius = 1.0f;

    // Photon Map Storage (stores photons after first diffuse bounce)
    std::vector<Photon> photonMap;

    // Random number generator for Russian Roulette and bounce directions
    std::mt19937 rng;
    std::uniform_real_distribution<float> uniformDist;

    // Animated mode: emit one photon
    void emitPhoton();
    // Animated mode: update photon positions with physics
    void updatePhotons(float deltaTime);

    // Instant mode: emit all photons and trace them immediately
    void emitAllPhotonsInstant();
    // Trace a single photon to completion (all bounces)
    void tracePhotonToCompletion(AnimatedPhoton &photon);

    // Generate a cosine-weighted random direction in hemisphere around normal
    float3 sampleCosineHemisphere(const float3 &normal);

    enum RenderMode
    {
        MODE_PHOTON_DOTS,
        MODE_GATHER
    };
    RenderMode currentMode = MODE_PHOTON_DOTS;

public:
    Application();
    ~Application() = default;

    bool initialize();
    void run();
    void shutdown();

    // Access to photon map for debugging/visualization
    const std::vector<Photon> &getPhotonMap() const { return photonMap; }
    size_t getPhotonMapSize() const { return photonMap.size(); }
};
