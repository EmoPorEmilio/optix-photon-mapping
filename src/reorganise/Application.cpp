#include "Application.h"
#include "Sphere.h"
#include "Triangle.h"
#include "QuadLight.h"
#include "PhotonMapper.h"
#include "PhotonMapRenderer.h"
#include "ObjLoader.h"
#include "ConfigLoader.h"
#include "RasterRenderer.h"
#include <optix_types.h>

static unsigned int NUM_PHOTONS = 100000;

bool Application::initialize()
{
    // Load photon mapping configuration from JSON in the Debug folder.
    // The compile script runs the executable from the project root, so we
    // reference the config relative to that (bin/Debug/...).
    PhotonMappingConfig pmc = ConfigLoader::load("bin/Debug/configuration.json");
    NUM_PHOTONS = pmc.max_photons;

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
                                     }
                                 });

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

    const float cornellWidth = 556.0f;
    const float cornellHeight = 548.8f;
    const float cornellDepth = 559.2f;

    float3 white = make_float3(0.8f, 0.8f, 0.8f);
    float3 red = make_float3(0.8f, 0.0f, 0.0f);
    float3 blue = make_float3(0.0f, 0.0f, 0.8f);

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

    // Bare bones: Removed Spheres and Bunny model.
    // scene.addObject(std::make_unique<Sphere>(make_float3(185.0f, 82.5f, 169.0f), 82.5f, white));
    // scene.addObject(std::make_unique<Sphere>(make_float3(368.0f, 103.5f, 351.0f), 103.5f, make_float3(0.6f, 0.6f, 0.6f)));
    // ObjLoader::addModelToScene(scene, bunnyPath, bunnyPosition, bunnyScale, bunnyColor);

    // Place the quad light flush with the ceiling, pointing downwards.
    float3 lightCenter = make_float3(278.0f, cornellHeight - 1.0f, 279.6f);
    float3 lightNormal = make_float3(0.0f, -1.0f, 0.0f);
    float3 lightU = make_float3(1.0f, 0.0f, 0.0f);
    float lightWidth = 200.0f;
    float lightHeight = 200.0f;
    float3 lightIntensity = make_float3(50.0f, 50.0f, 50.0f);

    scene.addLight(std::make_unique<QuadLight>(
        lightCenter, lightNormal, lightU, lightWidth, lightHeight, lightIntensity));

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

    // Bare bones: No spheres
    /*
    const float3 sphere1_center = make_float3(185.0f, 82.5f, 169.0f);
    const float sphere1_radius = 82.5f;
    const float3 sphere2_center = make_float3(368.0f, 103.5f, 351.0f);
    const float sphere2_radius = 103.5f;

    if (!optixManager.buildSphereGAS(sphere1_center, sphere1_radius, sphere2_center, sphere2_radius))
    {
        std::cerr << "Failed to build sphere GAS" << std::endl;
        return false;
    }
    */

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

    std::cout << "Entering main loop. Press ESC to exit." << std::endl;
    std::cout << "isRunning: " << isRunning << ", shouldClose: " << glManager.shouldClose() << std::endl;
    std::cout.flush();
    isRunning = true;

    std::cout << "About to enter while loop" << std::endl;
    std::cout.flush();

    while (isRunning && !glManager.shouldClose())
    {
        inputCommandManager->pollEvents();

        try
        {
            leftRenderer->renderFrame();
        }
        catch (const std::exception &e)
        {
            std::cerr << "Exception in left renderer: " << e.what() << std::endl;
        }

        if (!photonsEmitted)
        {
            // Bare bones: No photon emission
            photonsEmitted = true;
            /*
            const auto &lights = scene.getLights();
            if (!lights.empty() && lights[0]->isAreaLight())
            {
                const QuadLight *quadLight = static_cast<const QuadLight *>(lights[0].get());

                std::cout << "PHOTON EMISSION: About to launch photon pass" << std::endl;
                std::cout.flush();

                CUdeviceptr photonBuffer;
                unsigned int storedCount;
                optixManager.launchPhotonPass(NUM_PHOTONS, *quadLight, scene.getQuadLightStartIndex(), photonBuffer, storedCount);

                std::cout << "PHOTON EMISSION: Photon pass completed" << std::endl;
                std::cout.flush();

                std::vector<Photon> hostPhotons(storedCount);
                if (storedCount > 0)
                {
                    CUDA_CHECK(cudaMemcpy(hostPhotons.data(), reinterpret_cast<void *>(photonBuffer),
                                          storedCount * sizeof(Photon), cudaMemcpyDeviceToHost));

                    // Build CPU-side kd-tree for photon mapping queries (Jensen-style).
                    photonMapper->buildFromArray(hostPhotons.data(), storedCount);

                    // Still upload flat array to the debug photon map renderer.
                    rightRenderer->uploadFromHost(hostPhotons.data(), storedCount);
                }

                photonsEmitted = true;
            }
            */
        }

        if (currentMode == MODE_PHOTON_DOTS)
        {
            rightRenderer->render();
        }
        else
        {
            // Placeholder for Gather Renderer
            // rightRenderer->renderGathered(photonMapper.get(), camera); 
            // For now just clear screen or show dots to indicate it's not implemented yet
             glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
             glClear(GL_COLOR_BUFFER_BIT);
        }

        glManager.getWindow()->swapBuffers();
    }

    std::cout << "EXITED MAIN LOOP - isRunning: " << isRunning << ", shouldClose: " << glManager.shouldClose() << std::endl;
    std::cout << "Cleaning up..." << std::endl;
}

void Application::shutdown()
{
    isRunning = false;
}
