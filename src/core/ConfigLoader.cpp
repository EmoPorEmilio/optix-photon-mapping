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

    // Animation configuration
    if (extractBool(text, "enabled", bval))
        cfg.animation.enabled = bval;
    if (extractNumber<float>(text, "photon_speed", fval))
        cfg.animation.photonSpeed = fval;
    if (extractNumber<float>(text, "emission_interval", fval))
        cfg.animation.emissionInterval = fval;

    // Gathering configuration
    if (extractNumber<float>(text, "indirect_radius", fval))
        cfg.gathering.indirect_radius = fval;
    if (extractNumber<float>(text, "caustic_radius", fval))
        cfg.gathering.caustic_radius = fval;
    if (extractNumber<float>(text, "indirect_brightness", fval))
        cfg.gathering.indirect_brightness = fval;
    if (extractNumber<float>(text, "caustic_brightness", fval))
        cfg.gathering.caustic_brightness = fval;

    // Direct lighting configuration
    if (extractNumber<float>(text, "ambient", fval))
        cfg.direct_lighting.ambient = fval;
    if (extractNumber<float>(text, "shadow_ambient", fval))
        cfg.direct_lighting.shadow_ambient = fval;
    if (extractNumber<float>(text, "intensity", fval))
        cfg.direct_lighting.intensity = fval;
    if (extractNumber<float>(text, "attenuation_factor", fval))
        cfg.direct_lighting.attenuation_factor = fval;

    // Specular configuration
    if (extractNumber<unsigned int>(text, "max_recursion_depth", uval))
        cfg.specular.max_recursion_depth = uval;
    if (extractNumber<float>(text, "glass_ior", fval))
        cfg.specular.glass_ior = fval;
    float3 glass_tint;
    if (extractFloat3(text, "glass_tint", glass_tint))
        cfg.specular.glass_tint = glass_tint;
    if (extractNumber<float>(text, "mirror_reflectivity", fval))
        cfg.specular.mirror_reflectivity = fval;
    if (extractNumber<float>(text, "fresnel_min", fval))
        cfg.specular.fresnel_min = fval;
    // Specular-specific brightness (unique keys to avoid conflicts)
    if (extractNumber<float>(text, "specular_indirect_brightness", fval))
        cfg.specular.indirect_brightness = fval;
    if (extractNumber<float>(text, "specular_caustic_brightness", fval))
        cfg.specular.caustic_brightness = fval;
    if (extractNumber<float>(text, "specular_ambient", fval))
        cfg.specular.ambient = fval;

    // Weights configuration (in "weights" section)
    // These use simple key names under the weights object
    float direct_w, indirect_w, caustics_w, specular_w;
    if (extractNumber<float>(text, "\"direct\"", direct_w))
        cfg.weights.direct = direct_w;
    if (extractNumber<float>(text, "\"indirect\"", indirect_w))
        cfg.weights.indirect = indirect_w;
    if (extractNumber<float>(text, "\"caustics\"", caustics_w))
        cfg.weights.caustics = caustics_w;
    if (extractNumber<float>(text, "\"specular\"", specular_w))
        cfg.weights.specular = specular_w;

    // Wall colors configuration
    float3 wallColor;
    if (extractFloat3(text, "floor", wallColor))
        cfg.walls.floor = wallColor;
    if (extractFloat3(text, "ceiling", wallColor))
        cfg.walls.ceiling = wallColor;
    if (extractFloat3(text, "back", wallColor))
        cfg.walls.back = wallColor;
    if (extractFloat3(text, "left", wallColor))
        cfg.walls.left = wallColor;
    if (extractFloat3(text, "right", wallColor))
        cfg.walls.right = wallColor;

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

    // Parse mesh objects - find all "type": "mesh" entries
    size_t meshPos = 0;
    while ((meshPos = text.find("\"type\"", meshPos)) != std::string::npos)
    {
        // Check if this is a mesh type
        size_t colonPos = text.find(':', meshPos);
        if (colonPos == std::string::npos)
            break;

        size_t quoteStart = text.find('"', colonPos + 1);
        size_t quoteEnd = text.find('"', quoteStart + 1);
        if (quoteStart == std::string::npos || quoteEnd == std::string::npos)
            break;

        std::string typeValue = text.substr(quoteStart + 1, quoteEnd - quoteStart - 1);

        if (typeValue == "mesh")
        {
            MeshObjectConfig mesh;

            // Find the enclosing object block
            size_t blockStart = text.rfind('{', meshPos);
            size_t blockEnd = text.find('}', meshPos);
            if (blockStart != std::string::npos && blockEnd != std::string::npos)
            {
                std::string block = text.substr(blockStart, blockEnd - blockStart + 1);

                // Extract path
                size_t pathPos = block.find("\"path\"");
                if (pathPos != std::string::npos)
                {
                    size_t pathColonPos = block.find(':', pathPos);
                    size_t pathQuoteStart = block.find('"', pathColonPos + 1);
                    size_t pathQuoteEnd = block.find('"', pathQuoteStart + 1);
                    if (pathQuoteStart != std::string::npos && pathQuoteEnd != std::string::npos)
                    {
                        mesh.path = block.substr(pathQuoteStart + 1, pathQuoteEnd - pathQuoteStart - 1);
                    }
                }

                // Extract position
                float3 pos;
                if (extractFloat3(block, "position", pos))
                    mesh.position = pos;

                // Extract scale
                float scaleVal;
                if (extractNumber<float>(block, "scale", scaleVal))
                    mesh.scale = scaleVal;

                // Extract color from material
                float3 col;
                if (extractFloat3(block, "color", col))
                    mesh.color = col;

                if (!mesh.path.empty())
                {
                    cfg.meshes.push_back(mesh);
                }
            }
        }

        meshPos = quoteEnd + 1;
    }

    std::cout << "ConfigLoader: loaded photon mapping config from " << path << std::endl;
    std::cout << "  max_photons=" << cfg.max_photons << std::endl;
    std::cout << "  animated=" << (cfg.animation.enabled ? "true" : "false") << std::endl;
    std::cout << "  gathering: indirect_radius=" << cfg.gathering.indirect_radius
              << ", caustic_radius=" << cfg.gathering.caustic_radius << std::endl;
    std::cout << "  gathering: indirect_brightness=" << cfg.gathering.indirect_brightness
              << ", caustic_brightness=" << cfg.gathering.caustic_brightness << std::endl;
    std::cout << "  direct: ambient=" << cfg.direct_lighting.ambient
              << ", intensity=" << cfg.direct_lighting.intensity << std::endl;
    std::cout << "  specular: glass_ior=" << cfg.specular.glass_ior
              << ", mirror_reflectivity=" << cfg.specular.mirror_reflectivity << std::endl;
    std::cout << "  weights: direct=" << cfg.weights.direct
              << ", indirect=" << cfg.weights.indirect
              << ", caustics=" << cfg.weights.caustics
              << ", specular=" << cfg.weights.specular << std::endl;
    std::cout << "  walls: left=(" << cfg.walls.left.x << "," << cfg.walls.left.y << "," << cfg.walls.left.z << ")"
              << ", right=(" << cfg.walls.right.x << "," << cfg.walls.right.y << "," << cfg.walls.right.z << ")" << std::endl;
    std::cout << "  meshes: " << cfg.meshes.size() << " mesh object(s)" << std::endl;
    for (const auto &m : cfg.meshes)
    {
        std::cout << "    - " << m.path << " at (" << m.position.x << "," << m.position.y << "," << m.position.z
                  << ") scale=" << m.scale << std::endl;
    }

    return cfg;
}
