#include "Application.h"
#include "PerformanceManager.h"
#include "../lighting/QuadLight.h"
#include "../rendering/photon/PhotonMapRenderer.h"
#include "../rendering/raster/RasterRenderer.h"
#include "../scene/Material.h"
#include "../scene/ObjLoader.h"
#include "../scene/Sphere.h"
#include "../scene/Triangle.h"
#include "ConfigLoader.h"
#include <cmath>
#include <cuda_runtime.h>
#include <optix_types.h>

#ifdef _WIN32
#include <windows.h>
#endif

// Get the directory where the executable is located
static std::string getExecutableDir()
{
#ifdef _WIN32
  char path[MAX_PATH];
  GetModuleFileNameA(NULL, path, MAX_PATH);
  std::string fullPath(path);
  size_t lastSlash = fullPath.find_last_of("\\/");
  if (lastSlash != std::string::npos)
  {
    return fullPath.substr(0, lastSlash + 1);
  }
#endif
  return "";
}

static std::string exeDir;

static unsigned int NUM_PHOTONS = 100000;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

Application::Application()
    : rng(std::random_device{}()), uniformDist(0.0f, 1.0f) {}

bool Application::initialize()
{
  PERF_START("Application::initialize (total)");

  // Get the executable's directory for loading files
  exeDir = getExecutableDir();
  std::cout << "Executable directory: " << exeDir << std::endl;

  // Load photon mapping configuration from XML
  PERF_START("ConfigLoader::load");
  PhotonMappingConfig pmc = ConfigLoader::load(exeDir + "configuration.xml");
  PERF_STOP("ConfigLoader::load");
  NUM_PHOTONS = pmc.max_photons;
  maxPhotons = pmc.max_photons;
  photonCollisionRadius = pmc.photon_collision_radius;

  // Load animation settings
  animatedMode = pmc.animation.enabled;
  photonSpeed = pmc.animation.photonSpeed;
  emissionInterval = pmc.animation.emissionInterval;

  // Load debug/trajectory settings
  recordTrajectories = pmc.debug.record_trajectories;
  trajectoryOutputFile = pmc.debug.trajectory_file;

  // Load photon map I/O settings
  savePhotonMap = pmc.debug.save_photon_map;
  loadPhotonMap = pmc.debug.load_photon_map;
  photonMapFile = pmc.debug.photon_map_file;

  // Load export settings
  exportImages = pmc.debug.export_images;
  exportMetrics = pmc.debug.export_metrics;
  exportDir = pmc.debug.export_dir;

  // Initialize photon emission timer
  lastPhotonEmissionTime = std::chrono::steady_clock::now();

  inputCommandManager = std::make_unique<InputCommandManager>();

  if (!glManager.initialize())
  {
    std::cerr << "Failed to initialize graphics" << std::endl;
    return false;
  }

  photonMapRenderer = std::make_unique<PhotonMapRenderer>();
  if (!photonMapRenderer->initialize(glManager.getWindow()))
  {
    std::cerr << "Failed to initialize photon map renderer" << std::endl;
    return false;
  }

  directLightRenderer = std::make_unique<DirectLightRenderer>();
  if (!directLightRenderer->initialize(glManager.getWindow()->get()))
  {
    std::cerr << "Failed to initialize direct light renderer" << std::endl;
    return false;
  }

  indirectLightRenderer = std::make_unique<IndirectLightRenderer>(
      glManager.getWindow()->get(), glManager.getRightViewport());
  if (!indirectLightRenderer->initialize(&optixManager))
  {
    std::cerr << "Failed to initialize indirect light renderer" << std::endl;
    return false;
  }

  causticLightRenderer = std::make_unique<CausticLightRenderer>(
      glManager.getWindow()->get(), glManager.getRightViewport());
  if (!causticLightRenderer->initialize(&optixManager))
  {
    std::cerr << "Failed to initialize caustic light renderer" << std::endl;
    return false;
  }

  specularLightRenderer = std::make_unique<SpecularLightRenderer>(
      glManager.getWindow()->get(), glManager.getRightViewport());
  if (!specularLightRenderer->initialize(&optixManager))
  {
    std::cerr << "Failed to initialize specular light renderer" << std::endl;
    return false;
  }

  combinedRenderer = std::make_unique<CombinedRenderer>(
      glManager.getWindow()->get(), glManager.getRightViewport());
  if (!combinedRenderer->initialize(&optixManager))
  {
    std::cerr << "Failed to initialize combined renderer" << std::endl;
    return false;
  }

  // Initialize ExporterManager for image/data exports
  exporterManager = std::make_unique<ExporterManager>();
  exporterManager->initialize(&optixManager, &camera, &scene);
  exporterManager->setConfig(pmc); // Store config for metrics export

  inputCommandManager->initialize(glManager.getWindow()->get());
  inputCommandManager->bindKey(
      GLFW_KEY_ESCAPE, [this]()
      { glManager.getWindow()->requestClose(); });

  // E key exports combined image
  inputCommandManager->bindKey(GLFW_KEY_E, [this]()
                               { combinedRenderer->exportToImage("combined_render"); });

  // M key cycles through right viewport modes: Global -> Caustic dots -> Direct
  // -> Indirect -> Caustic light -> Global
  inputCommandManager->bindKey(GLFW_KEY_M, [this]()
                               {
    if (rightViewportMode == MODE_GLOBAL_PHOTONS) {
      rightViewportMode = MODE_CAUSTIC_PHOTONS;
      std::cout << "Switched to CAUSTIC photon map display ("
                << causticPhotonMap.size() << " photons)" << std::endl;
      if (!causticPhotonMap.empty()) {
        photonMapRenderer->uploadFromHost(causticPhotonMap.data(),
                                          causticPhotonMap.size());
      }
    } else if (rightViewportMode == MODE_CAUSTIC_PHOTONS) {
      rightViewportMode = MODE_DIRECT_LIGHTING;
      std::cout << "Switched to DIRECT LIGHTING raytracing" << std::endl;
    } else if (rightViewportMode == MODE_DIRECT_LIGHTING) {
      rightViewportMode = MODE_INDIRECT_LIGHTING;
      std::cout << "Switched to INDIRECT LIGHTING (color bleeding) - "
                << photonMap.size() << " photons" << std::endl;
      if (!photonMap.empty()) {
        indirectLightRenderer->uploadPhotonMap(photonMap);
      }
    } else if (rightViewportMode == MODE_INDIRECT_LIGHTING) {
      rightViewportMode = MODE_CAUSTIC_LIGHTING;
      std::cout << "Switched to CAUSTIC LIGHTING (highlights on walls) - "
                << causticPhotonMap.size() << " caustics" << std::endl;
      if (!causticPhotonMap.empty()) {
        causticLightRenderer->uploadCausticMap(causticPhotonMap);
      }
    } else if (rightViewportMode == MODE_CAUSTIC_LIGHTING) {
      rightViewportMode = MODE_SPECULAR_LIGHTING;
      std::cout << "Switched to SPECULAR LIGHTING (reflection/refraction with "
                   "full scene)"
                << std::endl;
      specularLightRenderer->uploadGlobalPhotonMap(photonMap);
      specularLightRenderer->uploadCausticPhotonMap(causticPhotonMap);
    } else if (rightViewportMode == MODE_SPECULAR_LIGHTING) {
      rightViewportMode = MODE_COMBINED;
      std::cout << "Switched to COMBINED mode (all weighted)" << std::endl;
      combinedRenderer->uploadGlobalPhotonMap(photonMap);
      combinedRenderer->uploadCausticPhotonMap(causticPhotonMap);
    } else {
      rightViewportMode = MODE_GLOBAL_PHOTONS;
      std::cout << "Switched to GLOBAL photon map display (" << photonMap.size()
                << " photons)" << std::endl;
      if (!photonMap.empty()) {
        photonMapRenderer->uploadFromHost(photonMap.data(), photonMap.size());
      }
    } });

  inputCommandManager->setMouseDragLeftHandler([this](int deltaX, int deltaY)
                                               { camera.orbit(static_cast<float>(deltaX), static_cast<float>(deltaY)); });

  inputCommandManager->setMouseDragRightHandler([this](int deltaX, int deltaY)
                                                { camera.pan(static_cast<float>(deltaX), static_cast<float>(deltaY)); });

  inputCommandManager->setMouseWheelHandler(
      [this](double yoffset)
      { camera.dolly(static_cast<float>(yoffset)); });

  inputCommandManager->setMouseDragMiddleHandler(
      [this](int deltaX, int deltaY)
      {
        camera.pan(static_cast<float>(deltaX), static_cast<float>(deltaY));
      });

  float aspectRatio = static_cast<float>(glManager.getLeftViewport().width) /
                      static_cast<float>(glManager.getLeftViewport().height);

  float3 cornellCenter = make_float3(278.0f, 273.0f, 279.0f);
  float3 cameraPos = make_float3(278.0f, 273.0f, -800.0f);

  if (pmc.camera.hasCamera)
  {
    cameraPos = pmc.camera.eye;
    cornellCenter = pmc.camera.lookAt;
    camera = Camera(cameraPos, cornellCenter, pmc.camera.up, pmc.camera.fov,
                    aspectRatio);
  }
  else
  {
    camera = Camera(cameraPos, cornellCenter, make_float3(0, 1, 0), 45.0f,
                    aspectRatio);
  }
  camera.setTarget(cornellCenter);
  camera.setMoveSpeed(5.0f);

  std::cout << "=== CAMERA INITIALIZATION ===" << std::endl;
  std::cout << "Camera position: (" << camera.getPosition().x << ", "
            << camera.getPosition().y << ", " << camera.getPosition().z << ")"
            << std::endl;
  std::cout << "Camera lookAt: (" << camera.getLookAt().x << ", "
            << camera.getLookAt().y << ", " << camera.getLookAt().z << ")"
            << std::endl;
  std::cout << "Camera aspect ratio: " << camera.getAspectRatio() << std::endl;
  std::cout << "Camera FOV: 45.0 degrees" << std::endl;
  std::cout << "Left viewport aspect: " << aspectRatio << std::endl;

  const float cornellWidth = Constants::Cornell::WIDTH;
  const float cornellHeight = Constants::Cornell::HEIGHT;
  const float cornellDepth = Constants::Cornell::DEPTH;

  // Wall colors from config
  float3 floorColor = pmc.walls.floor;
  float3 ceilingColor = pmc.walls.ceiling;
  float3 backColor = pmc.walls.back;
  float3 leftColor = pmc.walls.left;   // Was red
  float3 rightColor = pmc.walls.right; // Was blue

  // Floor (2 triangles)
  scene.addObject(std::make_unique<Triangle>(
      make_float3(0.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, cornellDepth),
      make_float3(cornellWidth, 0.0f, cornellDepth), floorColor));
  scene.addObject(std::make_unique<Triangle>(
      make_float3(0.0f, 0.0f, 0.0f),
      make_float3(cornellWidth, 0.0f, cornellDepth),
      make_float3(cornellWidth, 0.0f, 0.0f), floorColor));

  // Ceiling (2 triangles)
  scene.addObject(std::make_unique<Triangle>(
      make_float3(0.0f, cornellHeight, 0.0f),
      make_float3(cornellWidth, cornellHeight, 0.0f),
      make_float3(cornellWidth, cornellHeight, cornellDepth), ceilingColor));
  scene.addObject(std::make_unique<Triangle>(
      make_float3(0.0f, cornellHeight, 0.0f),
      make_float3(cornellWidth, cornellHeight, cornellDepth),
      make_float3(0.0f, cornellHeight, cornellDepth), ceilingColor));

  // Back wall (2 triangles)
  scene.addObject(std::make_unique<Triangle>(
      make_float3(0.0f, 0.0f, cornellDepth),
      make_float3(0.0f, cornellHeight, cornellDepth),
      make_float3(cornellWidth, cornellHeight, cornellDepth), backColor));
  scene.addObject(std::make_unique<Triangle>(
      make_float3(0.0f, 0.0f, cornellDepth),
      make_float3(cornellWidth, cornellHeight, cornellDepth),
      make_float3(cornellWidth, 0.0f, cornellDepth), backColor));

  // Add quad objects from configuration (mirrors, etc.)
  for (const auto &quadConfig : pmc.quads)
  {
    int matType = MATERIAL_DIFFUSE;
    if (quadConfig.materialType == "specular")
      matType = MATERIAL_SPECULAR;
    else if (quadConfig.materialType == "transmissive")
      matType = MATERIAL_TRANSMISSIVE;

    // Quad is defined by corner + edge1 + edge2
    // Create two triangles
    float3 v0 = quadConfig.corner;
    float3 v1 = quadConfig.corner + quadConfig.edge1;
    float3 v2 = quadConfig.corner + quadConfig.edge1 + quadConfig.edge2;
    float3 v3 = quadConfig.corner + quadConfig.edge2;

    scene.addObject(
        std::make_unique<Triangle>(v0, v3, v2, quadConfig.color, matType));
    scene.addObject(
        std::make_unique<Triangle>(v0, v2, v1, quadConfig.color, matType));
  }

  // Left wall (2 triangles) - configurable color (was red)
  scene.addObject(std::make_unique<Triangle>(
      make_float3(0.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, cornellDepth),
      make_float3(0.0f, cornellHeight, cornellDepth), leftColor));
  scene.addObject(std::make_unique<Triangle>(
      make_float3(0.0f, 0.0f, 0.0f),
      make_float3(0.0f, cornellHeight, cornellDepth),
      make_float3(0.0f, cornellHeight, 0.0f), leftColor));

  // Right wall (2 triangles) - configurable color (was blue)
  scene.addObject(std::make_unique<Triangle>(
      make_float3(cornellWidth, 0.0f, 0.0f),
      make_float3(cornellWidth, 0.0f, cornellDepth),
      make_float3(cornellWidth, cornellHeight, 0.0f), rightColor));
  scene.addObject(std::make_unique<Triangle>(
      make_float3(cornellWidth, 0.0f, cornellDepth),
      make_float3(cornellWidth, cornellHeight, cornellDepth),
      make_float3(cornellWidth, cornellHeight, 0.0f), rightColor));

  // Place the quad light flush with the ceiling, pointing downwards.
  float3 lightCenter = make_float3(278.0f, cornellHeight - 1.0f, 279.6f);
  float3 lightNormal = make_float3(0.0f, -1.0f, 0.0f);
  float3 lightU = make_float3(1.0f, 0.0f, 0.0f);
  float lightWidth = 200.0f;
  float lightHeight = 200.0f;
  float3 lightIntensity = make_float3(50.0f, 50.0f, 50.0f);

  scene.addLight(std::make_unique<QuadLight>(lightCenter, lightNormal, lightU,
                                             lightWidth, lightHeight,
                                             lightIntensity));

  // Load mesh objects from configuration
  for (const auto &meshConfig : pmc.meshes)
  {
    auto triangles = ObjLoader::load(meshConfig.path, meshConfig.position,
                                     meshConfig.scale, meshConfig.color);
    for (auto &tri : triangles)
    {
      scene.addObject(std::move(tri));
    }
  }

  // Initialize collision detector with scene and Cornell box dimensions
  collisionDetector = std::make_unique<CollisionDetector>(
      &scene, photonCollisionRadius, cornellWidth, cornellHeight, cornellDepth);
  collisionDetector->setWallColors(floorColor, ceilingColor, backColor,
                                   leftColor, rightColor);

  PERF_START("OptiX::initialize");
  if (!optixManager.initialize())
  {
    std::cerr << "Failed to initialize OptiX" << std::endl;
    return false;
  }
  PERF_STOP("OptiX::initialize");

  PERF_START("OptiX::createPipelines");
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

  if (!optixManager.createDirectLightingPipeline())
  {
    std::cerr << "Failed to create direct lighting pipeline" << std::endl;
    return false;
  }

  if (!optixManager.createIndirectLightingPipeline())
  {
    std::cerr << "Failed to create indirect lighting pipeline" << std::endl;
    return false;
  }

  if (!optixManager.createCausticLightingPipeline())
  {
    std::cerr << "Failed to create caustic lighting pipeline" << std::endl;
    return false;
  }

  if (!optixManager.createSpecularLightingPipeline())
  {
    std::cerr << "Failed to create specular lighting pipeline" << std::endl;
    return false;
  }
  PERF_STOP("OptiX::createPipelines");

  PERF_START("Scene::exportGeometry");
  std::vector<OptixVertex> vertices = scene.exportTriangleVertices();
  std::vector<float3> colors = scene.exportTriangleColors();
  std::vector<int> materialTypes = scene.exportTriangleMaterialTypes();
  PERF_STOP("Scene::exportGeometry");

  // Set the light start index (must be done after exportTriangleVertices which
  // computes it)
  optixManager.setQuadLightStartIndex(scene.getQuadLightStartIndex());
  std::cout << "quadLightStartIndex = " << scene.getQuadLightStartIndex()
            << " (total triangles before light)" << std::endl;

  PERF_START("OptiX::buildTriangleGAS");
  if (!optixManager.buildTriangleGAS(vertices, colors, materialTypes))
  {
    std::cerr << "Failed to build triangle GAS" << std::endl;
    return false;
  }
  PERF_STOP("OptiX::buildTriangleGAS");

  // Add spheres from configuration (up to 2 supported by current GAS builder)
  // Default: invisible spheres far outside scene
  float3 sphere1Center = make_float3(-10000.0f, -10000.0f, -10000.0f);
  float sphere1Radius = 0.1f;
  float3 sphere2Center = make_float3(-10000.0f, -10000.0f, -10000.0f);
  float sphere2Radius = 0.1f;

  // Override with spheres from config
  if (pmc.spheres.size() >= 1)
  {
    sphere1Center = pmc.spheres[0].center;
    sphere1Radius = pmc.spheres[0].radius;
    scene.addObject(std::make_unique<Sphere>(sphere1Center, sphere1Radius,
                                             pmc.spheres[0].color));
  }
  if (pmc.spheres.size() >= 2)
  {
    sphere2Center = pmc.spheres[1].center;
    sphere2Radius = pmc.spheres[1].radius;
    scene.addObject(std::make_unique<Sphere>(sphere2Center, sphere2Radius,
                                             pmc.spheres[1].color));
  }

  // Store sphere material types for OptiX (used in launch params)
  // Material type: 0=transmissive, 1=specular based on config
  sphereMaterials.clear();
  for (size_t i = 0; i < 2; i++)
  {
    if (i < pmc.spheres.size())
    {
      if (pmc.spheres[i].materialType == "transmissive")
        sphereMaterials.push_back(
            {MATERIAL_TRANSMISSIVE, pmc.spheres[i].color, pmc.spheres[i].ior});
      else if (pmc.spheres[i].materialType == "specular")
        sphereMaterials.push_back(
            {MATERIAL_SPECULAR, pmc.spheres[i].color, 1.0f});
      else
        sphereMaterials.push_back(
            {MATERIAL_DIFFUSE, pmc.spheres[i].color, 1.0f});
    }
    else
    {
      // Dummy sphere - make it diffuse
      sphereMaterials.push_back({MATERIAL_DIFFUSE, make_float3(0.0f), 1.0f});
    }
  }

  PERF_START("OptiX::buildSphereGAS");
  if (!optixManager.buildSphereGAS(sphere1Center, sphere1Radius, sphere2Center,
                                   sphere2Radius))
  {
    std::cerr << "Failed to build sphere GAS" << std::endl;
    return false;
  }
  PERF_STOP("OptiX::buildSphereGAS");

  PERF_START("OptiX::buildIAS");
  if (!optixManager.buildIAS())
  {
    std::cerr << "Failed to build IAS" << std::endl;
    return false;
  }
  PERF_STOP("OptiX::buildIAS");

  leftRenderer = std::make_unique<RasterRenderer>(glManager.getWindow(),
                                                  glManager.getLeftViewport());
  leftRenderer->setCamera(&camera);
  leftRenderer->setScene(&scene);

  float leftAspect = static_cast<float>(glManager.getLeftViewport().width) /
                     static_cast<float>(glManager.getLeftViewport().height);
  std::cout << "Left Renderer (Raster) - Viewport: "
            << glManager.getLeftViewport().width << "x"
            << glManager.getLeftViewport().height << ", Aspect: " << leftAspect
            << std::endl;
  std::cout << "Left Renderer camera U: (" << camera.getU().x << ", "
            << camera.getU().y << ", " << camera.getU().z << ")" << std::endl;
  std::cout << "Left Renderer camera V: (" << camera.getV().x << ", "
            << camera.getV().y << ", " << camera.getV().z << ")" << std::endl;
  std::cout << "Left Renderer camera W: (" << camera.getW().x << ", "
            << camera.getW().y << ", " << camera.getW().z << ")" << std::endl;

  photonMapRenderer->setViewport(glManager.getRightViewport());
  photonMapRenderer->setCamera(&camera);
  photonMapRenderer->setScene(&scene);

  directLightRenderer->setViewport(glManager.getRightViewport());
  directLightRenderer->setCamera(&camera);
  directLightRenderer->setScene(&scene);
  directLightRenderer->setOptixManager(&optixManager);
  directLightRenderer->setLightingParams(
      pmc.direct_lighting.ambient, pmc.direct_lighting.shadow_ambient,
      pmc.direct_lighting.intensity, pmc.direct_lighting.attenuation_factor);

  indirectLightRenderer->setViewport(glManager.getRightViewport());
  indirectLightRenderer->setCamera(&camera);
  indirectLightRenderer->setScene(&scene);
  indirectLightRenderer->setGatherRadius(pmc.gathering.indirect_radius);
  indirectLightRenderer->setBrightnessMultiplier(
      pmc.gathering.indirect_brightness);

  causticLightRenderer->setViewport(glManager.getRightViewport());
  causticLightRenderer->setCamera(&camera);
  causticLightRenderer->setScene(&scene);
  causticLightRenderer->setGatherRadius(pmc.gathering.caustic_radius);
  causticLightRenderer->setBrightnessMultiplier(
      pmc.gathering.caustic_brightness);

  specularLightRenderer->setViewport(glManager.getRightViewport());
  specularLightRenderer->setCamera(&camera);
  specularLightRenderer->setScene(&scene);
  {
    OptixManager::SpecularParams sp;
    sp.gather_radius = pmc.gathering.indirect_radius;
    sp.max_recursion_depth = pmc.specular.max_recursion_depth;
    sp.glass_ior = pmc.specular.glass_ior;
    sp.glass_tint = pmc.specular.glass_tint;
    sp.mirror_reflectivity = pmc.specular.mirror_reflectivity;
    sp.fresnel_min = pmc.specular.fresnel_min;
    sp.specular_ambient = pmc.specular.ambient;
    sp.indirect_brightness = pmc.specular.indirect_brightness;
    sp.caustic_brightness = pmc.specular.caustic_brightness;
    specularLightRenderer->setSpecularParams(sp);
  }

  combinedRenderer->setViewport(glManager.getRightViewport());
  combinedRenderer->setCamera(&camera);
  combinedRenderer->setScene(&scene);
  combinedRenderer->setWeights(pmc.weights.direct, pmc.weights.indirect,
                               pmc.weights.caustics, pmc.weights.specular);
  combinedRenderer->setGatherRadius(pmc.gathering.indirect_radius);
  combinedRenderer->setBrightnessMultipliers(pmc.gathering.indirect_brightness,
                                             pmc.gathering.caustic_brightness);
  combinedRenderer->setDirectLightingParams(
      pmc.direct_lighting.ambient, pmc.direct_lighting.shadow_ambient,
      pmc.direct_lighting.intensity, pmc.direct_lighting.attenuation_factor);

  // Configure ExporterManager with same parameters
  exporterManager->setGatherRadius(pmc.gathering.indirect_radius);
  exporterManager->setBrightnessMultipliers(pmc.gathering.indirect_brightness,
                                            pmc.gathering.caustic_brightness);
  exporterManager->setDirectLightingParams(
      pmc.direct_lighting.ambient, pmc.direct_lighting.shadow_ambient,
      pmc.direct_lighting.intensity, pmc.direct_lighting.attenuation_factor);
  {
    OptixManager::SpecularParams sp;
    sp.gather_radius = pmc.gathering.indirect_radius;
    sp.max_recursion_depth = pmc.specular.max_recursion_depth;
    sp.glass_ior = pmc.specular.glass_ior;
    sp.glass_tint = pmc.specular.glass_tint;
    sp.mirror_reflectivity = pmc.specular.mirror_reflectivity;
    sp.fresnel_min = pmc.specular.fresnel_min;
    sp.specular_ambient = pmc.specular.ambient;
    sp.indirect_brightness = pmc.specular.indirect_brightness;
    sp.caustic_brightness = pmc.specular.caustic_brightness;
    combinedRenderer->setSpecularParams(sp);
  }

  float rightAspect = static_cast<float>(glManager.getRightViewport().width) /
                      static_cast<float>(glManager.getRightViewport().height);
  std::cout << "Right Renderer - Viewport: "
            << glManager.getRightViewport().width << "x"
            << glManager.getRightViewport().height
            << ", Aspect: " << rightAspect << std::endl;
  std::cout
      << "  Press M to cycle: Global Photons -> Caustics -> Direct Lighting"
      << std::endl;

  PERF_STOP("Application::initialize (total)");
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

  // ANIMATED MODE: Load trajectories for playback
  if (animatedMode)
  {
    std::cout << "=== ANIMATED MODE: Loading trajectories ===" << std::endl;

    // Try to load from existing trajectory file first
    PERF_START("AnimatedMode::loadTrajectories");
    if (trajectoryAnimator.loadFromFile(trajectoryOutputFile))
    {
      std::cout << "  Loaded " << trajectoryAnimator.getTrajectoryCount()
                << " trajectories from " << trajectoryOutputFile << std::endl;
    }
    else
    {
      // No file exists - trace photons and record trajectories
      std::cout << "  No trajectory file found, tracing " << maxPhotons << " photons..." << std::endl;

      const auto &lights = scene.getLights();
      if (!lights.empty() && lights[0]->isAreaLight())
      {
        const QuadLight *quadLight = static_cast<const QuadLight *>(lights[0].get());

        CUdeviceptr d_photonBuffer;
        unsigned int storedCount = 0;
        CUdeviceptr d_causticBuffer;
        unsigned int causticCount = 0;
        std::vector<PhotonTrajectory> trajectories;

        optixManager.launchPhotonPassWithTrajectories(
            maxPhotons, *quadLight, scene.getQuadLightStartIndex(),
            d_photonBuffer, storedCount, d_causticBuffer, causticCount,
            trajectories);

        // Save for future use
        TrajectoryExporter::exportToFile(trajectories, trajectoryOutputFile);
        std::cout << "  Traced and saved " << trajectories.size() << " trajectories" << std::endl;

        // Load into animator
        trajectoryAnimator.loadTrajectories(trajectories);
      }
    }
    PERF_STOP("AnimatedMode::loadTrajectories");
  }

  // INSTANT MODE: Emit and trace all photons using OptiX (GPU) OR load from file
  if (!animatedMode)
  {
    auto startTime = std::chrono::steady_clock::now();

    // Option 1: Load photon map from file (skip tracing)
    if (loadPhotonMap)
    {
      std::cout << "=== INSTANT MODE: Loading photon map from " << photonMapFile << " ===" << std::endl;

      PERF_START("PhotonMapIO::importFromFile");
      if (PhotonMapIO::importFromFile(photonMapFile, photonMap, causticPhotonMap))
      {
        std::cout << "  Loaded " << photonMap.size() << " global + "
                  << causticPhotonMap.size() << " caustic photons" << std::endl;
      }
      else
      {
        std::cerr << "Failed to load photon map! Falling back to tracing..." << std::endl;
        loadPhotonMap = false; // Fall through to tracing
      }
      PERF_STOP("PhotonMapIO::importFromFile");
    }

    // Option 2: Trace photons using OptiX GPU
    if (!loadPhotonMap)
    {
      std::cout << "=== INSTANT MODE (OptiX GPU): Tracing " << maxPhotons
                << " photons ===" << std::endl;

      // Get the quad light for photon emission
      const auto &lights = scene.getLights();
      if (!lights.empty() && lights[0]->isAreaLight())
      {
        const QuadLight *quadLight =
            static_cast<const QuadLight *>(lights[0].get());

        CUdeviceptr d_photonBuffer;
        unsigned int storedCount = 0;
        CUdeviceptr d_causticBuffer;
        unsigned int causticCount = 0;

        // Launch OptiX photon pass on GPU (returns both global and caustic photons)
        // Optionally record full trajectories for debugging/visualization
        std::vector<PhotonTrajectory> trajectories;

        PERF_START("PhotonTracing::launchPhotonPass");
        if (recordTrajectories)
        {
          std::cout << "Trajectory recording enabled - will export to: "
                    << trajectoryOutputFile << std::endl;
          optixManager.launchPhotonPassWithTrajectories(
              maxPhotons, *quadLight, scene.getQuadLightStartIndex(),
              d_photonBuffer, storedCount, d_causticBuffer, causticCount,
              trajectories);

          // Export trajectories to file
          if (TrajectoryExporter::exportToFile(trajectories, trajectoryOutputFile))
          {
            std::cout << "Exported " << trajectories.size() << " trajectories to "
                      << trajectoryOutputFile << std::endl;
            std::cout << TrajectoryExporter::getSummary(trajectories) << std::endl;
          }
          else
          {
            std::cerr << "Failed to export trajectories!" << std::endl;
          }
        }
        else
        {
          optixManager.launchPhotonPass(
              maxPhotons, *quadLight, scene.getQuadLightStartIndex(),
              d_photonBuffer, storedCount, d_causticBuffer, causticCount);
        }
        PERF_STOP("PhotonTracing::launchPhotonPass");

        std::cout << "=== OptiX Photon tracing complete ===" << std::endl;
        std::cout << "  Photons launched: " << maxPhotons << std::endl;
        std::cout << "  Global photons stored: " << storedCount << std::endl;
        std::cout << "  Caustic photons stored: " << causticCount << std::endl;

        // Copy GLOBAL photons from GPU to CPU
        PERF_START("PhotonTracing::copyToHost");
        photonMap.clear();
        if (storedCount > 0)
        {
          photonMap.resize(storedCount);
          cudaMemcpy(photonMap.data(), reinterpret_cast<void *>(d_photonBuffer),
                     storedCount * sizeof(Photon), cudaMemcpyDeviceToHost);
        }

        // Copy CAUSTIC photons from GPU to CPU
        causticPhotonMap.clear();
        if (causticCount > 0)
        {
          causticPhotonMap.resize(causticCount);
          cudaMemcpy(causticPhotonMap.data(),
                     reinterpret_cast<void *>(d_causticBuffer),
                     causticCount * sizeof(Photon), cudaMemcpyDeviceToHost);
        }
        PERF_STOP("PhotonTracing::copyToHost");

        // Save photon map to file if requested
        if (savePhotonMap && (!photonMap.empty() || !causticPhotonMap.empty()))
        {
          PERF_START("PhotonMapIO::exportToFile");
          PhotonMapIO::exportToFile(photonMap, causticPhotonMap, photonMapFile);
          PERF_STOP("PhotonMapIO::exportToFile");
        }
      }
      else
      {
        std::cerr << "No area light found for OptiX photon emission!" << std::endl;
      }
    }

    auto endTime = std::chrono::steady_clock::now();
    float elapsedMs =
        std::chrono::duration<float, std::milli>(endTime - startTime).count();
    std::cout << "  Time elapsed: " << elapsedMs << " ms" << std::endl;

    // Upload photons to all renderers that need them
    // 1. Upload to dot renderer (for visualization modes)
    PERF_START("PhotonUpload::dotRenderer");
    if (!photonMap.empty())
    {
      photonMapRenderer->uploadFromHost(photonMap.data(), photonMap.size());
      std::cout << "  Uploaded " << photonMap.size() << " GLOBAL photons to dot renderer" << std::endl;
    }
    PERF_STOP("PhotonUpload::dotRenderer");

    // 2. Upload to combined renderer (for actual rendering with KD-tree)
    PERF_START("PhotonUpload::combinedRenderer (KD-tree)");
    combinedRenderer->uploadGlobalPhotonMap(photonMap);
    combinedRenderer->uploadCausticPhotonMap(causticPhotonMap);
    PERF_STOP("PhotonUpload::combinedRenderer (KD-tree)");
    std::cout << "  Uploaded photon maps to combined renderer (KD-tree built)" << std::endl;

    // 3. Upload to specular renderer
    PERF_START("PhotonUpload::specularRenderer (KD-tree)");
    specularLightRenderer->uploadGlobalPhotonMap(photonMap);
    specularLightRenderer->uploadCausticPhotonMap(causticPhotonMap);
    PERF_STOP("PhotonUpload::specularRenderer (KD-tree)");

    // 4. Export rendered images if requested
    if (exportImages && (!photonMap.empty() || !causticPhotonMap.empty()))
    {
      PERF_START("ExporterManager::exportAll");
      exporterManager->setPhotonData(photonMap, causticPhotonMap);
      exporterManager->exportAll(exportDir);
      PERF_STOP("ExporterManager::exportAll");
    }
    // 5. Export performance metrics separately if images not exported but metrics requested
    else if (exportMetrics)
    {
      exporterManager->createDirectory(exportDir);
      exporterManager->exportPerformanceMetrics(exportDir);
    }

    std::cout << "  Press M to cycle: Global Photons -> Caustics -> Direct Lighting"
              << std::endl;
  }

  std::cout << "Entering main loop. Press ESC to exit." << std::endl;
  std::cout << "Mode: "
            << (animatedMode ? "ANIMATED" : "INSTANT (photons already traced)")
            << std::endl;
  std::cout << "isRunning: " << isRunning
            << ", shouldClose: " << glManager.shouldClose() << std::endl;
  std::cout.flush();
  isRunning = true;

  auto lastFrameTime = std::chrono::steady_clock::now();
  auto lastFpsDisplayTime = std::chrono::steady_clock::now();
  int frameCount = 0;

  while (isRunning && !glManager.shouldClose())
  {
    // Calculate delta time
    auto currentTime = std::chrono::steady_clock::now();
    float deltaTime =
        std::chrono::duration<float>(currentTime - lastFrameTime).count();
    lastFrameTime = currentTime;

    inputCommandManager->pollEvents();

    // ANIMATED MODE: Animate through recorded trajectories
    if (animatedMode)
    {
      trajectoryAnimator.update(deltaTime, photonSpeed, emissionInterval);

      // Get current photon positions and pass to renderer
      std::vector<Photon> activePhotons = trajectoryAnimator.getActivePhotons();
      leftRenderer->setPhotons(activePhotons);
    }

    // Render left viewport (raster scene)
    PERF_START("Render::LeftViewport");
    try
    {
      leftRenderer->renderFrame();
    }
    catch (const std::exception &e)
    {
      std::cerr << "Exception in left renderer: " << e.what() << std::endl;
    }
    PERF_STOP("Render::LeftViewport");

    // Render right viewport (skip in animated mode - only show photon animation)
    if (!animatedMode)
    {
      if (rightViewportMode == MODE_DIRECT_LIGHTING)
      {
        PERF_START("Render::DirectLighting");
        directLightRenderer->render();
        PERF_STOP("Render::DirectLighting");
      }
      else if (rightViewportMode == MODE_INDIRECT_LIGHTING)
      {
        PERF_START("Render::IndirectLighting");
        indirectLightRenderer->render();
        PERF_STOP("Render::IndirectLighting");
      }
      else if (rightViewportMode == MODE_CAUSTIC_LIGHTING)
      {
        PERF_START("Render::CausticLighting");
        causticLightRenderer->render();
        PERF_STOP("Render::CausticLighting");
      }
      else if (rightViewportMode == MODE_SPECULAR_LIGHTING)
      {
        PERF_START("Render::SpecularLighting");
        specularLightRenderer->render();
        PERF_STOP("Render::SpecularLighting");
      }
      else if (rightViewportMode == MODE_COMBINED)
      {
        PERF_START("Render::Combined");
        combinedRenderer->render();
        PERF_STOP("Render::Combined");
      }
      else
      {
        // MODE_GLOBAL_PHOTONS or MODE_CAUSTIC_PHOTONS
        PERF_START("Render::PhotonDots");
        photonMapRenderer->render();
        PERF_STOP("Render::PhotonDots");
      }
    }

    glManager.getWindow()->swapBuffers();
    frameCount++;

    // Display FPS every 2 seconds
    auto fpsElapsed = std::chrono::duration<float>(currentTime - lastFpsDisplayTime).count();
    if (fpsElapsed >= 2.0f)
    {
      float fps = frameCount / fpsElapsed;
      std::cout << "[FPS] " << std::fixed << std::setprecision(1) << fps << " fps";

      // Show current render mode timing
      if (!animatedMode)
      {
        const char *modeName = nullptr;
        double avgMs = 0.0;
        unsigned int calls = 0;

        if (rightViewportMode == MODE_DIRECT_LIGHTING)
        {
          modeName = "DirectLighting";
          avgMs = PerformanceManager::instance().getTotalTime("Render::DirectLighting");
          calls = PerformanceManager::instance().getCallCount("Render::DirectLighting");
        }
        else if (rightViewportMode == MODE_INDIRECT_LIGHTING)
        {
          modeName = "IndirectLighting";
          avgMs = PerformanceManager::instance().getTotalTime("Render::IndirectLighting");
          calls = PerformanceManager::instance().getCallCount("Render::IndirectLighting");
        }
        else if (rightViewportMode == MODE_CAUSTIC_LIGHTING)
        {
          modeName = "CausticLighting";
          avgMs = PerformanceManager::instance().getTotalTime("Render::CausticLighting");
          calls = PerformanceManager::instance().getCallCount("Render::CausticLighting");
        }
        else if (rightViewportMode == MODE_SPECULAR_LIGHTING)
        {
          modeName = "SpecularLighting";
          avgMs = PerformanceManager::instance().getTotalTime("Render::SpecularLighting");
          calls = PerformanceManager::instance().getCallCount("Render::SpecularLighting");
        }
        else if (rightViewportMode == MODE_COMBINED)
        {
          modeName = "Combined";
          avgMs = PerformanceManager::instance().getTotalTime("Render::Combined");
          calls = PerformanceManager::instance().getCallCount("Render::Combined");
        }
        else
        {
          modeName = "PhotonDots";
          avgMs = PerformanceManager::instance().getTotalTime("Render::PhotonDots");
          calls = PerformanceManager::instance().getCallCount("Render::PhotonDots");
        }

        if (calls > 0)
        {
          double avgPerFrame = avgMs / calls;
          std::cout << " | " << modeName << ": " << std::setprecision(2) << avgPerFrame << " ms/frame";
        }
      }
      std::cout << std::endl;

      frameCount = 0;
      lastFpsDisplayTime = currentTime;
    }
  }

  std::cout << "EXITED MAIN LOOP - isRunning: " << isRunning
            << ", shouldClose: " << glManager.shouldClose() << std::endl;
  std::cout << "Final photon map size: " << photonMap.size()
            << " photons stored" << std::endl;
  std::cout << "Cleaning up..." << std::endl;
}

void Application::shutdown() { isRunning = false; }

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
  float maxIntensity =
      fmaxf(fmaxf(lightIntensity.x, lightIntensity.y), lightIntensity.z);

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
    CollisionResult collision =
        collisionDetector->raycast(photon.position, photon.direction, 10000.0f);

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

    // Russian Roulette: decide if photon survives
    float survivalProbability = collision.diffuseProb;
    float randomValue = uniformDist(rng);

    if (randomValue > survivalProbability)
    {
      // Absorbed - but still store the photon if bounceCount > 0
      // This captures the incoming flux before absorption
      if (photon.bounceCount > 0)
      {
        Photon storedPhoton(
            collision.hitPoint,
            photon.power, // Store INCOMING power before absorption
            photon.direction);
        photonMap.push_back(storedPhoton);
      }
      photon.isActive = false;
      photon.wasAbsorbed = true;
      break;
    }

    // Store photon in map AFTER first bounce (bounceCount > 0)
    // Store BEFORE power modulation - this is the incoming flux
    if (photon.bounceCount > 0)
    {
      Photon storedPhoton(
          collision.hitPoint,
          photon.power, // Store INCOMING power (before surface modulation)
          photon.direction);
      photonMap.push_back(storedPhoton);
    }

    // NOW modulate power by surface color for the NEXT bounce
    photon.power = photon.power * collision.surfaceColor / survivalProbability;

    // Clamp power
    float maxPower =
        fmaxf(fmaxf(photon.power.x, photon.power.y), photon.power.z);
    if (maxPower > 3.0f)
    {
      photon.power = photon.power / maxPower * 3.0f;
    }

    // Check max bounces
    photon.bounceCount++;
    if (photon.bounceCount >= photon.maxBounces)
    {
      photon.isActive = false;
      break;
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
  float maxIntensity =
      fmaxf(fmaxf(lightIntensity.x, lightIntensity.y), lightIntensity.z);
  if (maxIntensity > 0.0f)
  {
    photon.power =
        lightIntensity / maxIntensity; // Normalize to [0,1] for display
  }
  else
  {
    photon.power = make_float3(1.0f, 1.0f, 1.0f); // Default white
  }

  // arrivalPower starts the same as power (the initial emission power)
  photon.arrivalPower = photon.power;

  // Set velocity based on direction and speed
  photon.velocity = photon.direction * photonSpeed;
  photon.isActive = true;
  photon.bounceCount = 0;
  photon.maxBounces = Constants::Photon::MAX_BOUNCES;
  photon.wasAbsorbed = false;

  // Record initial position in path history
  photon.recordPathPoint();

  animatedPhotons.push_back(photon);

  std::cout << "Emitted photon #" << animatedPhotons.size() << " at ("
            << photon.position.x << ", " << photon.position.y << ", "
            << photon.position.z << ")"
            << " direction (" << photon.direction.x << ", "
            << photon.direction.y << ", " << photon.direction.z << ")"
            << " power (" << photon.power.x << ", " << photon.power.y << ", "
            << photon.power.z << ")" << std::endl;
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
  float3 localDir =
      make_float3(cosf(phi) * sinTheta, cosTheta, sinf(phi) * sinTheta);

  // Build orthonormal basis around normal
  float3 up = fabsf(normal.y) < 0.999f ? make_float3(0.0f, 1.0f, 0.0f)
                                       : make_float3(1.0f, 0.0f, 0.0f);
  float3 tangent = normalize(cross(up, normal));
  float3 bitangent = cross(normal, tangent);

  // Transform to world space
  float3 worldDir =
      tangent * localDir.x + normal * localDir.y + bitangent * localDir.z;
  return normalize(worldDir);
}

void Application::updatePhotons(float deltaTime)
{
  // ANIMATED MODE: Simple visualization only - photons travel until first hit,
  // then stop. For actual photon mapping with bounces, use INSTANT MODE
  // (OptiX-based).

  for (size_t i = 0; i < animatedPhotons.size(); i++)
  {
    auto &photon = animatedPhotons[i];
    if (!photon.isActive)
      continue;

    // Update position based on velocity
    photon.position = photon.position + photon.velocity * deltaTime;

    // Simple boundary check - stop photon when it hits a wall
    // This is for visualization only, not actual photon tracing
    bool hitWall = false;
    float3 hitColor = make_float3(0.8f, 0.8f, 0.8f);

    // Check Cornell box boundaries
    const float margin = 1.0f;

    // Floor (y = 0)
    if (photon.position.y <= margin)
    {
      photon.position.y = margin;
      hitWall = true;
      hitColor = make_float3(0.8f, 0.8f, 0.8f); // white floor
    }
    // Ceiling
    else if (photon.position.y >= Constants::Cornell::HEIGHT - margin)
    {
      photon.position.y = Constants::Cornell::HEIGHT - margin;
      hitWall = true;
      hitColor = make_float3(0.8f, 0.8f, 0.8f); // white ceiling
    }

    // Left wall (x = 0) - RED
    if (photon.position.x <= margin)
    {
      photon.position.x = margin;
      hitWall = true;
      hitColor = make_float3(0.8f, 0.0f, 0.0f); // red
    }
    // Right wall - BLUE
    else if (photon.position.x >= Constants::Cornell::WIDTH - margin)
    {
      photon.position.x = Constants::Cornell::WIDTH - margin;
      hitWall = true;
      hitColor = make_float3(0.0f, 0.0f, 0.8f); // blue
    }

    // Back wall
    if (photon.position.z >= Constants::Cornell::DEPTH - margin)
    {
      photon.position.z = Constants::Cornell::DEPTH - margin;
      hitWall = true;
      hitColor = make_float3(0.8f, 0.8f, 0.8f); // white back
    }
    // Front (camera side) - photon escapes
    else if (photon.position.z <= margin)
    {
      photon.isActive = false;
      photon.velocity = make_float3(0.0f, 0.0f, 0.0f);
      continue;
    }

    if (hitWall)
    {
      // Stop the photon at first hit - no CPU bouncing
      // This is just for visualizing emission direction
      photon.isActive = false;
      photon.velocity = make_float3(0.0f, 0.0f, 0.0f);
      photon.lastSurfaceColor = hitColor;

      // First hit = direct illumination, photon disappears (not stored)
      // For actual photon mapping with indirect bounces, use instant mode
      // (OptiX)
    }
  }
}
