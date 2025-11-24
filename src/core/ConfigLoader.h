#pragma once

#include <string>
#include <sutil/vec_math.h>

// Simple camera configuration read from JSON.
struct CameraConfig
{
    bool hasCamera = false;
    float3 eye = make_float3(278.0f, 273.0f, -800.0f);
    float3 lookAt = make_float3(278.0f, 273.0f, 279.0f);
    float3 up = make_float3(0.0f, 1.0f, 0.0f);
    float fov = 45.0f;
};

// Simple configuration loader for photon mapping parameters from a JSON file.
// This is intentionally minimal and only extracts the fields we currently use.
struct PhotonMappingConfig
{
    unsigned int max_photons = 100000;
    float direct_weight = 1.0f;
    float indirect_weight = 1.0f;
    float caustics_weight = 1.0f;
    float participating_media_weight = 1.0f;

    CameraConfig camera;
};

class ConfigLoader
{
public:
    // Loads configuration from the given JSON file.
    // On failure, returns a config filled with default values.
    static PhotonMappingConfig load(const std::string &path);
};


