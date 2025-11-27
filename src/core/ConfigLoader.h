#pragma once

#include <string>
#include <vector>
#include <sutil/vec_math.h>

// Simple camera configuration read from XML.
struct CameraConfig
{
    bool hasCamera = false;
    float3 eye = make_float3(278.0f, 273.0f, -800.0f);
    float3 lookAt = make_float3(278.0f, 273.0f, 279.0f);
    float3 up = make_float3(0.0f, 1.0f, 0.0f);
    float fov = 45.0f;
};

// Animation settings for photon visualization
struct AnimationConfig
{
    bool enabled = true;           // If false, photons are traced instantly
    float photonSpeed = 200.0f;    // Units per second for animated movement
    float emissionInterval = 0.5f; // Seconds between photon emissions
};

// Debug settings for trajectory recording and photon map I/O
struct DebugConfig
{
    bool record_trajectories = false;  // Record full photon paths
    std::string trajectory_file = "photon_trajectories.txt";  // Trajectory output file
    
    bool save_photon_map = false;      // Save photon map after tracing
    bool load_photon_map = false;      // Load photon map from file instead of tracing
    std::string photon_map_file = "photon_map.txt";  // Photon map file path
    
    bool export_images = false;        // Export rendered images for all modes
    std::string export_dir = "export"; // Directory for exported files
};

// Photon gathering parameters
struct GatheringConfig
{
    float indirect_radius = 100.0f;       // Search radius for indirect lighting
    float caustic_radius = 50.0f;         // Search radius for caustics (tighter = sharper)
    float indirect_brightness = 50000.0f; // Multiplier for indirect lighting visibility
    float caustic_brightness = 100000.0f; // Multiplier for caustic visibility
};

// Direct lighting parameters
struct DirectLightingConfig
{
    float ambient = 0.03f;               // Base ambient light
    float shadow_ambient = 0.02f;        // Ambient in shadowed areas
    float intensity = 0.5f;              // Direct lighting intensity multiplier
    float attenuation_factor = 0.00001f; // Light falloff factor
};

// Specular/glass material parameters
struct SpecularConfig
{
    unsigned int max_recursion_depth = 10;               // Max ray bounces for reflections/refractions
    float glass_ior = 1.5f;                              // Index of refraction for glass
    float3 glass_tint = make_float3(0.98f, 0.99f, 1.0f); // Glass color tint
    float mirror_reflectivity = 0.95f;                   // Mirror reflection intensity
    float fresnel_min = 0.1f;                            // Fresnel effect minimum
    float indirect_brightness = 100000.0f;               // Indirect brightness in specular reflections
    float caustic_brightness = 200000.0f;                // Caustic brightness in specular reflections
    float ambient = 0.15f;                               // Ambient in specular mode
};

// Render mode weights
struct WeightsConfig
{
    float direct = 1.0f;
    float indirect = 1.0f;
    float caustics = 1.0f;
    float specular = 0.5f;
};

// Cornell box wall colors
struct WallColorsConfig
{
    float3 floor = make_float3(0.8f, 0.8f, 0.8f);
    float3 ceiling = make_float3(0.8f, 0.8f, 0.8f);
    float3 back = make_float3(0.8f, 0.8f, 0.8f);
    float3 left = make_float3(0.8f, 0.0f, 0.0f);  // Red wall
    float3 right = make_float3(0.0f, 0.0f, 0.8f); // Blue wall
};

// Mesh object configuration
struct MeshObjectConfig
{
    std::string path;
    float3 position = make_float3(0.0f, 0.0f, 0.0f);
    float scale = 1.0f;
    std::string materialType = "diffuse";
    float3 color = make_float3(0.8f, 0.8f, 0.8f);
    float ior = 1.5f;
};

// Sphere object configuration
struct SphereObjectConfig
{
    float3 center = make_float3(0.0f, 0.0f, 0.0f);
    float radius = 50.0f;
    std::string materialType = "diffuse";  // "diffuse", "specular", "transmissive"
    float3 color = make_float3(0.8f, 0.8f, 0.8f);
    float ior = 1.5f;  // For transmissive materials
};

// Quad object configuration (for mirrors, etc.)
struct QuadObjectConfig
{
    float3 corner = make_float3(0.0f, 0.0f, 0.0f);  // Bottom-left corner
    float3 edge1 = make_float3(1.0f, 0.0f, 0.0f);   // First edge direction (width)
    float3 edge2 = make_float3(0.0f, 1.0f, 0.0f);   // Second edge direction (height)
    std::string materialType = "diffuse";  // "diffuse", "specular"
    float3 color = make_float3(0.8f, 0.8f, 0.8f);
};

// Simple configuration loader for photon mapping parameters from an XML file.
struct PhotonMappingConfig
{
    unsigned int max_photons = 100000;
    float photon_collision_radius = 5.0f;

    AnimationConfig animation;
    DebugConfig debug;
    GatheringConfig gathering;
    DirectLightingConfig direct_lighting;
    SpecularConfig specular;
    WeightsConfig weights;
    WallColorsConfig walls;
    CameraConfig camera;

    std::vector<MeshObjectConfig> meshes;
    std::vector<SphereObjectConfig> spheres;
    std::vector<QuadObjectConfig> quads;
};

class ConfigLoader
{
public:
    // Loads configuration from the given XML file.
    // On failure, returns a config filled with default values.
    static PhotonMappingConfig load(const std::string &path);
};
