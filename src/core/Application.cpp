#include "Application.h"
#include "../scene/Sphere.h"
#include "../scene/Triangle.h"
#include "../lighting/QuadLight.h"
#include "../rendering/photon/PhotonMapper.h"
#include "../rendering/photon/PhotonMapRenderer.h"
#include "../scene/ObjLoader.h"
#include "ConfigLoader.h"
#include "../rendering/raster/RasterRenderer.h"
#include <optix_types.h>
#include <cmath>

static unsigned int NUM_PHOTONS = 100000;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

Application::Application()
    : rng(std::random_device{}()), uniformDist(0.0f, 1.0f)
{
}

bool Application::initialize()
{
    // Load photon mapping configuration from JSON in the Debug folder.
    PhotonMappingConfig pmc = ConfigLoader::load("bin/Debug/configuration.json");
    NUM_PHOTONS = pmc.max_photons;
    maxPhotons = pmc.max_photons;
    photonCollisionRadius = pmc.photon_collision_radius;

    // Load animation settings
    animatedMode = pmc.animation.enabled;
    photonSpeed = pmc.animation.photonSpeed;
    emissionInterval = pmc.animation.emissionInterval;

    // Initialize photon emission timer
    lastPhotonEmissionTime = std::chrono::steady_clock::now();

    inputCommandManager = std::make_unique<InputCommandManager>();
    photonMapper = std::make_unique<PhotonMapper>(NUM_PHOTONS);

    if (!glManager.initialize())
    {
        std::cerr << "Failed to initialize graphics" << std::endl;
        return false;
    }

    rightRenderer = std::make_unique<PhotonMapRenderer>();
    if (!rightRenderer->initialize(glManager.getWindow()))
    {
        std::cerr << "Failed to initialize photon map renderer" << std::endl;
        return false;
    }

    inputCommandManager->initialize(glManager.getWindow()->get());
    inputCommandManager->bindKey(GLFW_KEY_ESCAPE, [this]()
                                 { glManager.getWindow()->requestClose(); });

    inputCommandManager->bindKey(GLFW_KEY_M, [this]()
                                 {
                                     if (currentMode == MODE_PHOTON_DOTS)
                                     {
                                         currentMode = MODE_GATHER;
                                         std::cout << "Switched to MODE_GATHER" << std::endl;
                                     }
                                     else
                                     {
                                         currentMode = MODE_PHOTON_DOTS;
                                         std::cout << "Switched to MODE_PHOTON_DOTS" << std::endl;
                                     } });

    inputCommandManager->setMouseDragLeftHandler([this](int deltaX, int deltaY)
                                                 { camera.orbit(static_cast<float>(deltaX), static_cast<float>(deltaY)); });

    inputCommandManager->setMouseDragRightHandler([this](int deltaX, int deltaY)
                                                  { camera.pan(static_cast<float>(deltaX), static_cast<float>(deltaY)); });

    inputCommandManager->setMouseWheelHandler([this](double yoffset)
                                              { camera.dolly(static_cast<float>(yoffset)); });

    inputCommandManager->setMouseDragMiddleHandler([this](int deltaX, int deltaY)
                                                   { camera.pan(static_cast<float>(deltaX), static_cast<float>(deltaY)); });

    float aspectRatio = static_cast<float>(glManager.getLeftViewport().width) / static_cast<float>(glManager.getLeftViewport().height);

    float3 cornellCenter = make_float3(278.0f, 273.0f, 279.0f);
    float3 cameraPos = make_float3(278.0f, 273.0f, -800.0f);

    if (pmc.camera.hasCamera)
    {
        cameraPos = pmc.camera.eye;
        cornellCenter = pmc.camera.lookAt;
        camera = Camera(cameraPos, cornellCenter, pmc.camera.up, pmc.camera.fov, aspectRatio);
    }
    else
    {
        camera = Camera(cameraPos, cornellCenter, make_float3(0, 1, 0), 45.0f, aspectRatio);
    }
    camera.setTarget(cornellCenter);
    camera.setMoveSpeed(5.0f);

    std::cout << "=== CAMERA INITIALIZATION ===" << std::endl;
    std::cout << "Camera position: (" << camera.getPosition().x << ", " << camera.getPosition().y << ", " << camera.getPosition().z << ")" << std::endl;
    std::cout << "Camera lookAt: (" << camera.getLookAt().x << ", " << camera.getLookAt().y << ", " << camera.getLookAt().z << ")" << std::endl;
    std::cout << "Camera aspect ratio: " << camera.getAspectRatio() << std::endl;
    std::cout << "Camera FOV: 45.0 degrees" << std::endl;
    std::cout << "Left viewport aspect: " << aspectRatio << std::endl;

    const float cornellWidth = Constants::Cornell::WIDTH;
    const float cornellHeight = Constants::Cornell::HEIGHT;
    const float cornellDepth = Constants::Cornell::DEPTH;

    float3 white = make_float3(Constants::Cornell::WHITE_R, Constants::Cornell::WHITE_G, Constants::Cornell::WHITE_B);
    float3 red = make_float3(Constants::Cornell::RED_R, Constants::Cornell::RED_G, Constants::Cornell::RED_B);
    float3 blue = make_float3(Constants::Cornell::BLUE_R, Constants::Cornell::BLUE_G, Constants::Cornell::BLUE_B);

    scene.addObject(std::make_unique<Triangle>(
        make_float3(0.0f, 0.0f, 0.0f),
        make_float3(0.0f, 0.0f, cornellDepth),
        make_float3(cornellWidth, 0.0f, cornellDepth),
        white));
    scene.addObject(std::make_unique<Triangle>(
        make_float3(0.0f, 0.0f, 0.0f),
        make_float3(cornellWidth, 0.0f, cornellDepth),
        make_float3(cornellWidth, 0.0f, 0.0f),
        white));

    scene.addObject(std::make_unique<Triangle>(
        make_float3(0.0f, cornellHeight, 0.0f),
        make_float3(cornellWidth, cornellHeight, 0.0f),
        make_float3(cornellWidth, cornellHeight, cornellDepth),
        white));
    scene.addObject(std::make_unique<Triangle>(
        make_float3(0.0f, cornellHeight, 0.0f),
        make_float3(cornellWidth, cornellHeight, cornellDepth),
        make_float3(0.0f, cornellHeight, cornellDepth),
        white));

    scene.addObject(std::make_unique<Triangle>(
        make_float3(0.0f, 0.0f, cornellDepth),
        make_float3(0.0f, cornellHeight, cornellDepth),
        make_float3(cornellWidth, cornellHeight, cornellDepth),
        white));
    scene.addObject(std::make_unique<Triangle>(
        make_float3(0.0f, 0.0f, cornellDepth),
        make_float3(cornellWidth, cornellHeight, cornellDepth),
        make_float3(cornellWidth, 0.0f, cornellDepth),
        white));

    scene.addObject(std::make_unique<Triangle>(
        make_float3(0.0f, 0.0f, 0.0f),
        make_float3(0.0f, 0.0f, cornellDepth),
        make_float3(0.0f, cornellHeight, cornellDepth),
        red));
    scene.addObject(std::make_unique<Triangle>(
        make_float3(0.0f, 0.0f, 0.0f),
        make_float3(0.0f, cornellHeight, cornellDepth),
        make_float3(0.0f, cornellHeight, 0.0f),
        red));

    scene.addObject(std::make_unique<Triangle>(
        make_float3(cornellWidth, 0.0f, 0.0f),
        make_float3(cornellWidth, 0.0f, cornellDepth),
        make_float3(cornellWidth, cornellHeight, 0.0f),
        blue));
    scene.addObject(std::make_unique<Triangle>(
        make_float3(cornellWidth, 0.0f, cornellDepth),
        make_float3(cornellWidth, cornellHeight, cornellDepth),
        make_float3(cornellWidth, cornellHeight, 0.0f),
        blue));

    // Place the quad light flush with the ceiling, pointing downwards.
    float3 lightCenter = make_float3(278.0f, cornellHeight - 1.0f, 279.6f);
    float3 lightNormal = make_float3(0.0f, -1.0f, 0.0f);
    float3 lightU = make_float3(1.0f, 0.0f, 0.0f);
    float lightWidth = 200.0f;
    float lightHeight = 200.0f;
    float3 lightIntensity = make_float3(50.0f, 50.0f, 50.0f);

    scene.addLight(std::make_unique<QuadLight>(
        lightCenter, lightNormal, lightU, lightWidth, lightHeight, lightIntensity));

    // Initialize collision detector with scene and Cornell box dimensions
    collisionDetector = std::make_unique<CollisionDetector>(
        &scene, photonCollisionRadius, cornellWidth, cornellHeight, cornellDepth);

    if (!optixManager.initialize())
    {
        std::cerr << "Failed to initialize OptiX" << std::endl;
        return false;
    }

    if (!optixManager.createPipeline())
    {
        std::cerr << "Failed to create OptiX pipeline" << std::endl;
        return false;
    }

    if (!optixManager.createPhotonPipeline())
    {
        std::cerr << "Failed to create photon pipeline" << std::endl;
        return false;
    }

    std::vector<OptixVertex> vertices = scene.exportTriangleVertices();
    std::vector<float3> colors = scene.exportTriangleColors();

    if (!optixManager.buildTriangleGAS(vertices, colors))
    {
        std::cerr << "Failed to build triangle GAS" << std::endl;
        return false;
    }

    if (!optixManager.buildIAS())
    {
        std::cerr << "Failed to build IAS" << std::endl;
        return false;
    }

    leftRenderer = std::make_unique<RasterRenderer>(
        glManager.getWindow(),
        glManager.getLeftViewport());
    leftRenderer->setCamera(&camera);
    leftRenderer->setScene(&scene);

    float leftAspect = static_cast<float>(glManager.getLeftViewport().width) / static_cast<float>(glManager.getLeftViewport().height);
    std::cout << "Left Renderer (Raster) - Viewport: " << glManager.getLeftViewport().width << "x" << glManager.getLeftViewport().height
              << ", Aspect: " << leftAspect << std::endl;
    std::cout << "Left Renderer camera U: (" << camera.getU().x << ", " << camera.getU().y << ", " << camera.getU().z << ")" << std::endl;
    std::cout << "Left Renderer camera V: (" << camera.getV().x << ", " << camera.getV().y << ", " << camera.getV().z << ")" << std::endl;
    std::cout << "Left Renderer camera W: (" << camera.getW().x << ", " << camera.getW().y << ", " << camera.getW().z << ")" << std::endl;

    rightRenderer->setViewport(glManager.getRightViewport());
    rightRenderer->setCamera(&camera);
    rightRenderer->setScene(&scene);

    float rightAspect = static_cast<float>(glManager.getRightViewport().width) / static_cast<float>(glManager.getRightViewport().height);
    std::cout << "Right Renderer (PhotonMap) - Viewport: " << glManager.getRightViewport().width << "x" << glManager.getRightViewport().height
              << ", Aspect: " << rightAspect << std::endl;

    std::cout << "Application::initialize() completed successfully" << std::endl;
    std::cout.flush();
    return true;
}

void Application::run()
{
    std::cout << "Application::run() called" << std::endl;
    std::cout.flush();

    if (!glManager.isInitialized())
    {
        std::cerr << "Application not initialized" << std::endl;
        return;
    }

    std::cout << "GL Manager is initialized, proceeding..." << std::endl;
    std::cout.flush();

    // INSTANT MODE: Emit and trace all photons immediately
    if (!animatedMode)
    {
        std::cout << "=== INSTANT MODE: Tracing " << maxPhotons << " photons ===" << std::endl;
        auto startTime = std::chrono::steady_clock::now();

        emitAllPhotonsInstant();

        auto endTime = std::chrono::steady_clock::now();
        float elapsedMs = std::chrono::duration<float, std::milli>(endTime - startTime).count();

        std::cout << "=== Photon tracing complete ===" << std::endl;
        std::cout << "  Photons traced: " << maxPhotons << std::endl;
        std::cout << "  Photons stored in map: " << photonMap.size() << std::endl;
        std::cout << "  Time elapsed: " << elapsedMs << " ms" << std::endl;
        std::cout << "  Rate: " << (maxPhotons / (elapsedMs / 1000.0f)) << " photons/sec" << std::endl;

        // Upload photon map to the right renderer for visualization
        if (!photonMap.empty())
        {
            rightRenderer->uploadFromHost(photonMap.data(), photonMap.size());
            std::cout << "  Uploaded " << photonMap.size() << " photons to renderer" << std::endl;
        }
    }

    std::cout << "Entering main loop. Press ESC to exit." << std::endl;
    std::cout << "Mode: " << (animatedMode ? "ANIMATED" : "INSTANT (photons already traced)") << std::endl;
    std::cout << "isRunning: " << isRunning << ", shouldClose: " << glManager.shouldClose() << std::endl;
    std::cout.flush();
    isRunning = true;

    auto lastFrameTime = std::chrono::steady_clock::now();

    while (isRunning && !glManager.shouldClose())
    {
        // Calculate delta time
        auto currentTime = std::chrono::steady_clock::now();
        float deltaTime = std::chrono::duration<float>(currentTime - lastFrameTime).count();
        lastFrameTime = currentTime;

        inputCommandManager->pollEvents();

        // ANIMATED MODE: Emit and update photons over time
        if (animatedMode)
        {
            // Emit photon at configured interval
            auto timeSinceLastEmission = std::chrono::duration<float>(currentTime - lastPhotonEmissionTime).count();
            if (timeSinceLastEmission >= emissionInterval && animatedPhotons.size() < maxPhotons)
            {
                emitPhoton();
                lastPhotonEmissionTime = currentTime;
            }

            // Update photon positions
            size_t prevMapSize = photonMap.size();
            updatePhotons(deltaTime);

            // If new photons were added to the map, upload to right renderer
            if (photonMap.size() > prevMapSize && !photonMap.empty())
            {
                rightRenderer->uploadFromHost(photonMap.data(), photonMap.size());
            }

            // Pass photons to renderer for visualization
            leftRenderer->setAnimatedPhotons(animatedPhotons);
        }

        try
        {
            leftRenderer->renderFrame();
        }
        catch (const std::exception &e)
        {
            std::cerr << "Exception in left renderer: " << e.what() << std::endl;
        }

        if (currentMode == MODE_PHOTON_DOTS)
        {
            rightRenderer->render();
        }
        else
        {
            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);
        }

        glManager.getWindow()->swapBuffers();
    }

    std::cout << "EXITED MAIN LOOP - isRunning: " << isRunning << ", shouldClose: " << glManager.shouldClose() << std::endl;
    std::cout << "Final photon map size: " << photonMap.size() << " photons stored" << std::endl;
    std::cout << "Cleaning up..." << std::endl;
}

void Application::shutdown()
{
    isRunning = false;
}

// ============================================================================
// INSTANT MODE: Emit all photons and trace them to completion immediately
// ============================================================================

void Application::emitAllPhotonsInstant()
{
    const auto &lights = scene.getLights();
    if (lights.empty() || !lights[0]->isAreaLight())
    {
        std::cerr << "No area light found for photon emission!" << std::endl;
        return;
    }

    const QuadLight *quadLight = static_cast<const QuadLight *>(lights[0].get());
    float3 lightIntensity = quadLight->getIntensity();
    float maxIntensity = fmaxf(fmaxf(lightIntensity.x, lightIntensity.y), lightIntensity.z);

    // Reserve space for efficiency
    photonMap.reserve(maxPhotons * 3); // Estimate ~3 stored photons per emitted

    for (unsigned int i = 0; i < maxPhotons; i++)
    {
        AnimatedPhoton photon;

        // Generate random samples for position and direction
        float u_pos = uniformDist(rng);
        float v_pos = uniformDist(rng);
        float u_dir_1 = uniformDist(rng);
        float u_dir_2 = uniformDist(rng);

        // Sample photon emission from quad light
        quadLight->samplePhotonEmission(u_pos, v_pos, u_dir_1, u_dir_2,
                                        photon.position, photon.direction);

        photon.emissionPosition = photon.position;

        // Set initial power
        if (maxIntensity > 0.0f)
        {
            photon.power = lightIntensity / maxIntensity;
        }
        else
        {
            photon.power = make_float3(1.0f, 1.0f, 1.0f);
        }

        photon.isActive = true;
        photon.bounceCount = 0;
        photon.maxBounces = Constants::Photon::MAX_BOUNCES;
        photon.wasAbsorbed = false;

        // Trace this photon to completion
        tracePhotonToCompletion(photon);

        // Progress output every 10%
        if ((i + 1) % (maxPhotons / 10) == 0)
        {
            std::cout << "  Progress: " << ((i + 1) * 100 / maxPhotons) << "% ("
                      << (i + 1) << "/" << maxPhotons << " photons, "
                      << photonMap.size() << " stored)" << std::endl;
        }
    }
}

void Application::tracePhotonToCompletion(AnimatedPhoton &photon)
{
    while (photon.isActive)
    {
        // Raycast to find next intersection
        CollisionResult collision = collisionDetector->raycast(
            photon.position, photon.direction, 10000.0f);

        if (!collision.hit)
        {
            // Photon escaped the scene
            photon.isActive = false;
            break;
        }

        // Move to hit point
        photon.position = collision.hitPoint;
        photon.lastHitNormal = collision.normal;
        photon.lastSurfaceColor = collision.surfaceColor;

        // Store photon in map AFTER first bounce (bounceCount > 0)
        if (photon.bounceCount > 0)
        {
            Photon storedPhoton(
                collision.hitPoint,
                photon.power,
                photon.direction);
            photonMap.push_back(storedPhoton);
        }

        // Russian Roulette: decide if photon survives
        float survivalProbability = collision.diffuseProb;
        float randomValue = uniformDist(rng);

        if (randomValue > survivalProbability)
        {
            // Absorbed
            photon.isActive = false;
            photon.wasAbsorbed = true;
            break;
        }

        // Check max bounces
        photon.bounceCount++;
        if (photon.bounceCount >= photon.maxBounces)
        {
            photon.isActive = false;
            break;
        }

        // Modulate power by surface color
        photon.power = photon.power * collision.surfaceColor / survivalProbability;

        // Clamp power
        float maxPower = fmaxf(fmaxf(photon.power.x, photon.power.y), photon.power.z);
        if (maxPower > 3.0f)
        {
            photon.power = photon.power / maxPower * 3.0f;
        }

        // Generate new diffuse direction
        photon.direction = sampleCosineHemisphere(collision.normal);

        // Offset position to prevent self-intersection
        photon.position = collision.hitPoint + collision.normal * 0.01f;
    }
}

// ============================================================================
// ANIMATED MODE: Emit photons one at a time with visual movement
// ============================================================================

void Application::emitPhoton()
{
    if (animatedPhotons.size() >= maxPhotons)
        return;

    const auto &lights = scene.getLights();
    if (lights.empty() || !lights[0]->isAreaLight())
        return;

    const QuadLight *quadLight = static_cast<const QuadLight *>(lights[0].get());

    AnimatedPhoton photon;

    // Generate random samples for position and direction
    float u_pos = uniformDist(rng);
    float v_pos = uniformDist(rng);
    float u_dir_1 = uniformDist(rng);
    float u_dir_2 = uniformDist(rng);

    // Sample photon emission from quad light
    quadLight->samplePhotonEmission(u_pos, v_pos, u_dir_1, u_dir_2,
                                    photon.position, photon.direction);

    // Store emission position for path visualization
    photon.emissionPosition = photon.position;

    // Set initial power from light intensity (normalized for visualization)
    float3 lightIntensity = quadLight->getIntensity();
    float maxIntensity = fmaxf(fmaxf(lightIntensity.x, lightIntensity.y), lightIntensity.z);
    if (maxIntensity > 0.0f)
    {
        photon.power = lightIntensity / maxIntensity; // Normalize to [0,1] for display
    }
    else
    {
        photon.power = make_float3(1.0f, 1.0f, 1.0f); // Default white
    }

    // Set velocity based on direction and speed
    photon.velocity = photon.direction * photonSpeed;
    photon.isActive = true;
    photon.bounceCount = 0;
    photon.maxBounces = Constants::Photon::MAX_BOUNCES;
    photon.wasAbsorbed = false;

    // Record initial position in path history
    photon.recordPathPoint();

    animatedPhotons.push_back(photon);

    std::cout << "Emitted photon #" << animatedPhotons.size()
              << " at (" << photon.position.x << ", " << photon.position.y << ", " << photon.position.z << ")"
              << " direction (" << photon.direction.x << ", " << photon.direction.y << ", " << photon.direction.z << ")"
              << " power (" << photon.power.x << ", " << photon.power.y << ", " << photon.power.z << ")" << std::endl;
}

float3 Application::sampleCosineHemisphere(const float3 &normal)
{
    // Generate two uniform random numbers
    float u1 = uniformDist(rng);
    float u2 = uniformDist(rng);

    // Cosine-weighted hemisphere sampling
    float phi = 2.0f * M_PI * u1;
    float cosTheta = sqrtf(u2);
    float sinTheta = sqrtf(1.0f - u2);

    // Local direction in tangent space
    float3 localDir = make_float3(
        cosf(phi) * sinTheta,
        cosTheta,
        sinf(phi) * sinTheta);

    // Build orthonormal basis around normal
    float3 up = fabsf(normal.y) < 0.999f ? make_float3(0.0f, 1.0f, 0.0f) : make_float3(1.0f, 0.0f, 0.0f);
    float3 tangent = normalize(cross(up, normal));
    float3 bitangent = cross(normal, tangent);

    // Transform to world space
    float3 worldDir = tangent * localDir.x + normal * localDir.y + bitangent * localDir.z;
    return normalize(worldDir);
}

void Application::updatePhotons(float deltaTime)
{
    static int frameCount = 0;
    bool shouldPrint = (frameCount++ % 120 == 0); // Print every 120 frames

    for (size_t i = 0; i < animatedPhotons.size(); i++)
    {
        auto &photon = animatedPhotons[i];
        if (!photon.isActive)
            continue;

        // Update position based on velocity
        float3 oldPos = photon.position;
        photon.position = photon.position + photon.velocity * deltaTime;

        if (shouldPrint && i == 0) // Print first active photon
        {
            std::cout << "Photon #1: pos(" << photon.position.x << ", " << photon.position.y
                      << ", " << photon.position.z << ") bounces=" << photon.bounceCount
                      << " power=(" << photon.power.x << ", " << photon.power.y << ", " << photon.power.z << ")" << std::endl;
        }

        // Check for collision using raycast from old position
        CollisionResult collision = collisionDetector->checkPositionCollision(photon.position);

        if (collision.hit)
        {
            // Store hit information
            photon.position = collision.hitPoint;
            photon.lastHitNormal = collision.normal;
            photon.lastSurfaceColor = collision.surfaceColor;

            // Record this position in path history
            photon.recordPathPoint();

            std::cout << "Photon #" << (i + 1) << " hit surface at ("
                      << collision.hitPoint.x << ", " << collision.hitPoint.y << ", " << collision.hitPoint.z << ")"
                      << " surface color (" << collision.surfaceColor.x << ", " << collision.surfaceColor.y << ", " << collision.surfaceColor.z << ")"
                      << " bounce #" << (photon.bounceCount + 1) << std::endl;

            // Store photon in map AFTER first bounce (bounceCount > 0)
            // This follows Jensen's algorithm - we don't store direct illumination
            if (photon.bounceCount > 0)
            {
                Photon storedPhoton(
                    collision.hitPoint,
                    photon.power,
                    photon.direction // Incident direction before bounce
                );
                photonMap.push_back(storedPhoton);

                std::cout << "  -> Stored in photon map (total: " << photonMap.size() << ")" << std::endl;
            }

            // Russian Roulette: decide if photon survives
            float survivalProbability = collision.diffuseProb;
            float randomValue = uniformDist(rng);

            if (randomValue > survivalProbability)
            {
                // Absorbed! Photon terminates
                photon.isActive = false;
                photon.wasAbsorbed = true;
                photon.velocity = make_float3(0.0f, 0.0f, 0.0f);

                std::cout << "  -> ABSORBED by Russian Roulette (rand=" << randomValue
                          << " > prob=" << survivalProbability << ")" << std::endl;
                continue;
            }

            // Photon survives - check max bounces
            photon.bounceCount++;
            if (photon.bounceCount >= photon.maxBounces)
            {
                photon.isActive = false;
                photon.velocity = make_float3(0.0f, 0.0f, 0.0f);

                std::cout << "  -> Terminated (max bounces reached)" << std::endl;
                continue;
            }

            // Modulate power by surface color (color bleeding effect)
            // Divide by survival probability to maintain unbiased estimate
            photon.power = photon.power * collision.surfaceColor / survivalProbability;

            // Clamp power to prevent explosion (for visualization)
            float maxPower = fmaxf(fmaxf(photon.power.x, photon.power.y), photon.power.z);
            if (maxPower > 3.0f)
            {
                photon.power = photon.power / maxPower * 3.0f;
            }

            // Generate new diffuse direction (cosine-weighted hemisphere)
            photon.direction = sampleCosineHemisphere(collision.normal);
            photon.velocity = photon.direction * photonSpeed;

            // Offset position slightly along normal to prevent self-intersection
            photon.position = collision.hitPoint + collision.normal * 0.5f;

            std::cout << "  -> BOUNCED! New direction (" << photon.direction.x << ", " << photon.direction.y << ", " << photon.direction.z << ")"
                      << " new power (" << photon.power.x << ", " << photon.power.y << ", " << photon.power.z << ")" << std::endl;
        }
    }
}
