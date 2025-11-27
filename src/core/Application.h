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
#include "../rendering/direct/DirectLightRenderer.h"
#include "../rendering/indirect/IndirectLightRenderer.h"
#include "../rendering/caustic/CausticLightRenderer.h"
#include "../rendering/specular/SpecularLightRenderer.h"
#include "../rendering/combined/CombinedRenderer.h"
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
    std::unique_ptr<PhotonMapRenderer> photonMapRenderer;
    std::unique_ptr<DirectLightRenderer> directLightRenderer;
    std::unique_ptr<IndirectLightRenderer> indirectLightRenderer;
    std::unique_ptr<CausticLightRenderer> causticLightRenderer;
    std::unique_ptr<SpecularLightRenderer> specularLightRenderer;
    std::unique_ptr<CombinedRenderer> combinedRenderer;
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
    
    // Caustic Photon Map (stores photons that hit S/T then diffuse)
    std::vector<Photon> causticPhotonMap;

    // Sphere materials from config (for OptiX)
    struct SphereMaterialInfo
    {
        int type;
        float3 color;
        float ior;
    };
    std::vector<SphereMaterialInfo> sphereMaterials;

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

    // Right viewport display mode
    enum RightViewportMode
    {
        MODE_GLOBAL_PHOTONS,    // Global photon map (dot display)
        MODE_CAUSTIC_PHOTONS,   // Caustic photon map (dot display)
        MODE_DIRECT_LIGHTING,   // Direct lighting raytracing
        MODE_INDIRECT_LIGHTING, // Indirect lighting (color bleeding from photon map)
        MODE_CAUSTIC_LIGHTING,  // Caustic highlights on walls
        MODE_SPECULAR_LIGHTING, // Reflection/refraction on spheres
        MODE_COMBINED           // All modes combined with weights
    };
    RightViewportMode rightViewportMode = MODE_GLOBAL_PHOTONS;

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
