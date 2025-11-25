#include "ConfigLoader.h"

#include <fstream>
#include <sstream>
#include <iostream>

// Very small helper to extract a numeric value from a JSON-like string by key.
// This is not a full JSON parser, but sufficient for our controlled config file.
template <typename T>
static bool extractNumber(const std::string &text, const std::string &key, T &out)
{
    const std::string pattern = "\"" + key + "\"";
    size_t pos = text.find(pattern);
    if (pos == std::string::npos)
        return false;

    pos = text.find(':', pos);
    if (pos == std::string::npos)
        return false;

    ++pos; // move past ':'
    while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t'))
        ++pos;

    std::stringstream ss(text.substr(pos));
    T value;
    ss >> value;
    if (ss.fail())
        return false;

    out = value;
    return true;
}

// Helper to extract a boolean value like: "enabled": true
static bool extractBool(const std::string &text, const std::string &key, bool &out)
{
    const std::string pattern = "\"" + key + "\"";
    size_t pos = text.find(pattern);
    if (pos == std::string::npos)
        return false;

    pos = text.find(':', pos);
    if (pos == std::string::npos)
        return false;

    ++pos; // move past ':'
    while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t'))
        ++pos;

    // Check for true/false
    if (text.substr(pos, 4) == "true")
    {
        out = true;
        return true;
    }
    else if (text.substr(pos, 5) == "false")
    {
        out = false;
        return true;
    }
    return false;
}

// Helper to extract a float3 array like: "eye": [x, y, z]
static bool extractFloat3(const std::string &text, const std::string &key, float3 &out)
{
    const std::string pattern = "\"" + key + "\"";
    size_t pos = text.find(pattern);
    if (pos == std::string::npos)
        return false;

    pos = text.find('[', pos);
    if (pos == std::string::npos)
        return false;

    ++pos; // move past '['
    std::stringstream ss(text.substr(pos));
    float x, y, z;
    char comma;
    ss >> x >> comma >> y >> comma >> z;
    if (ss.fail())
        return false;

    out = make_float3(x, y, z);
    return true;
}

PhotonMappingConfig ConfigLoader::load(const std::string &path)
{
    PhotonMappingConfig cfg;

    std::ifstream file(path);
    if (!file)
    {
        std::cerr << "ConfigLoader: could not open configuration file: " << path
                  << " – using default photon mapping parameters." << std::endl;
        return cfg;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    const std::string text = buffer.str();

    unsigned int uval = 0;
    float fval = 0.0f;
    bool bval = false;

    if (extractNumber<unsigned int>(text, "max_photons", uval))
        cfg.max_photons = uval;
    if (extractNumber<float>(text, "photon_collision_radius", fval))
        cfg.photon_collision_radius = fval;
    if (extractNumber<float>(text, "direct_weight", fval))
        cfg.direct_weight = fval;
    if (extractNumber<float>(text, "indirect_weight", fval))
        cfg.indirect_weight = fval;
    if (extractNumber<float>(text, "caustics_weight", fval))
        cfg.caustics_weight = fval;
    if (extractNumber<float>(text, "participating_media_weight", fval))
        cfg.participating_media_weight = fval;

    // Animation configuration
    if (extractBool(text, "enabled", bval))
        cfg.animation.enabled = bval;
    if (extractNumber<float>(text, "photon_speed", fval))
        cfg.animation.photonSpeed = fval;
    if (extractNumber<float>(text, "emission_interval", fval))
        cfg.animation.emissionInterval = fval;

    // Optional camera configuration.
    float3 eye, lookAt, up;
    float fovVal = 0.0f;
    bool haveEye = extractFloat3(text, "eye", eye);
    bool haveLookAt = extractFloat3(text, "lookAt", lookAt);
    bool haveUp = extractFloat3(text, "up", up);
    bool haveFov = extractNumber<float>(text, "fov", fovVal);
    if (haveEye && haveLookAt && haveUp)
    {
        cfg.camera.hasCamera = true;
        cfg.camera.eye = eye;
        cfg.camera.lookAt = lookAt;
        cfg.camera.up = up;
        if (haveFov)
            cfg.camera.fov = fovVal;
    }

    std::cout << "ConfigLoader: loaded photon mapping config from " << path << std::endl;
    std::cout << "  max_photons=" << cfg.max_photons << std::endl;
    std::cout << "  animated=" << (cfg.animation.enabled ? "true" : "false");
    if (cfg.animation.enabled)
    {
        std::cout << " (speed=" << cfg.animation.photonSpeed
                  << ", interval=" << cfg.animation.emissionInterval << "s)";
    }
    std::cout << std::endl;

    return cfg;
}
