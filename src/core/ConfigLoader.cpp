#include "ConfigLoader.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cctype>

//=============================================================================
// Simple XML parsing helpers (not a full parser, but sufficient for config)
//=============================================================================

static std::string trim(const std::string &s)
{
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start])))
        ++start;
    size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1])))
        --end;
    return s.substr(start, end - start);
}

// Extract content between <tag>...</tag>
static bool extractTagContent(const std::string &text, const std::string &tag, std::string &out)
{
    std::string openTagStart = "<" + tag;
    std::string closeTag = "</" + tag + ">";

    size_t tagStart = text.find(openTagStart);
    if (tagStart == std::string::npos)
        return false;

    size_t tagEnd = text.find(">", tagStart);
    if (tagEnd == std::string::npos)
        return false;

    if (tagEnd > 0 && text[tagEnd - 1] == '/')
        return false; // Self-closing

    size_t contentStart = tagEnd + 1;
    size_t end = text.find(closeTag, contentStart);
    if (end == std::string::npos)
        return false;

    out = trim(text.substr(contentStart, end - contentStart));
    return true;
}

// Find the entire element string for a tag (including attributes)
static bool findElement(const std::string &text, const std::string &tag, std::string &element, size_t startFrom = 0)
{
    std::string pattern = "<" + tag;
    size_t pos = text.find(pattern, startFrom);
    if (pos == std::string::npos)
        return false;

    // Make sure it's the actual tag (not a prefix of another tag)
    size_t afterTag = pos + pattern.length();
    if (afterTag < text.length() && !std::isspace(text[afterTag]) && 
        text[afterTag] != '>' && text[afterTag] != '/')
        return false;

    size_t endPos = text.find(">", pos);
    if (endPos == std::string::npos)
        return false;

    element = text.substr(pos, endPos - pos + 1);
    return true;
}

// Extract a single attribute value from an element string
static bool extractAttr(const std::string &element, const std::string &attr, std::string &out)
{
    std::string pattern = attr + "=\"";
    size_t pos = element.find(pattern);
    if (pos == std::string::npos)
        return false;

    pos += pattern.length();
    size_t endPos = element.find('"', pos);
    if (endPos == std::string::npos)
        return false;

    out = element.substr(pos, endPos - pos);
    return true;
}

// Extract float attribute
static bool extractFloatAttr(const std::string &element, const std::string &attr, float &out)
{
    std::string val;
    if (!extractAttr(element, attr, val))
        return false;
    std::stringstream ss(val);
    ss >> out;
    return !ss.fail();
}

// Extract unsigned int attribute
static bool extractUintAttr(const std::string &element, const std::string &attr, unsigned int &out)
{
    std::string val;
    if (!extractAttr(element, attr, val))
        return false;
    std::stringstream ss(val);
    ss >> out;
    return !ss.fail();
}

// Extract bool attribute (true/false or 1/0)
static bool extractBoolAttr(const std::string &element, const std::string &attr, bool &out)
{
    std::string val;
    if (!extractAttr(element, attr, val))
        return false;
    
    std::string lower = val;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    
    if (lower == "true" || lower == "1") { out = true; return true; }
    if (lower == "false" || lower == "0") { out = false; return true; }
    return false;
}

// Extract float3 from r/g/b or x/y/z attributes
static bool extractFloat3Attr(const std::string &element, float3 &out, bool rgb = false)
{
    const char *a1 = rgb ? "r" : "x";
    const char *a2 = rgb ? "g" : "y";
    const char *a3 = rgb ? "b" : "z";
    
    float x, y, z;
    if (!extractFloatAttr(element, a1, x) || 
        !extractFloatAttr(element, a2, y) || 
        !extractFloatAttr(element, a3, z))
        return false;
    
    out = make_float3(x, y, z);
    return true;
}

// Extract float3 from a named sub-element with r/g/b or x/y/z attributes
static bool extractFloat3Element(const std::string &text, const std::string &tag, float3 &out, bool rgb = false)
{
    std::string element;
    if (!findElement(text, tag, element))
        return false;
    return extractFloat3Attr(element, out, rgb);
}

//=============================================================================
// ConfigLoader implementation
//=============================================================================

PhotonMappingConfig ConfigLoader::load(const std::string &path)
{
    PhotonMappingConfig cfg;

    std::ifstream file(path);
    if (!file)
    {
        std::cerr << "ConfigLoader: could not open " << path << " - using defaults." << std::endl;
        return cfg;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    const std::string text = buffer.str();

    std::string element;

    //-------------------------------------------------------------------------
    // Cornell Box
    //-------------------------------------------------------------------------
    if (findElement(text, "cornell_box", element))
    {
        // Dimensions are attributes (ignored for now, using Constants.h)
    }

    std::string wallsBlock;
    if (extractTagContent(text, "walls", wallsBlock))
    {
        extractFloat3Element(wallsBlock, "floor", cfg.walls.floor, true);
        extractFloat3Element(wallsBlock, "ceiling", cfg.walls.ceiling, true);
        extractFloat3Element(wallsBlock, "back", cfg.walls.back, true);
        extractFloat3Element(wallsBlock, "left", cfg.walls.left, true);
        extractFloat3Element(wallsBlock, "right", cfg.walls.right, true);
    }

    //-------------------------------------------------------------------------
    // Photon Mapping Parameters
    //-------------------------------------------------------------------------
    if (findElement(text, "photon_mapping", element))
    {
        extractUintAttr(element, "max_photons", cfg.max_photons);
        extractFloatAttr(element, "collision_radius", cfg.photon_collision_radius);
    }

    // Animation
    if (findElement(text, "animated", element))
    {
        extractBoolAttr(element, "enabled", cfg.animation.enabled);
        extractFloatAttr(element, "speed", cfg.animation.photonSpeed);
        extractFloatAttr(element, "emission_interval", cfg.animation.emissionInterval);
    }

    // Debug / Trajectory Recording / Photon Map I/O
    if (findElement(text, "debug", element))
    {
        extractBoolAttr(element, "record_trajectories", cfg.debug.record_trajectories);
        extractAttr(element, "trajectory_file", cfg.debug.trajectory_file);
        extractBoolAttr(element, "save_photon_map", cfg.debug.save_photon_map);
        extractBoolAttr(element, "load_photon_map", cfg.debug.load_photon_map);
        extractAttr(element, "photon_map_file", cfg.debug.photon_map_file);
        extractBoolAttr(element, "export_images", cfg.debug.export_images);
        extractAttr(element, "export_dir", cfg.debug.export_dir);
    }

    // Gathering
    if (findElement(text, "gathering", element))
    {
        extractFloatAttr(element, "indirect_radius", cfg.gathering.indirect_radius);
        extractFloatAttr(element, "caustic_radius", cfg.gathering.caustic_radius);
        extractFloatAttr(element, "indirect_brightness", cfg.gathering.indirect_brightness);
        extractFloatAttr(element, "caustic_brightness", cfg.gathering.caustic_brightness);
    }

    // Direct Lighting
    if (findElement(text, "direct_lighting", element))
    {
        extractFloatAttr(element, "ambient", cfg.direct_lighting.ambient);
        extractFloatAttr(element, "shadow_ambient", cfg.direct_lighting.shadow_ambient);
        extractFloatAttr(element, "intensity", cfg.direct_lighting.intensity);
        extractFloatAttr(element, "attenuation", cfg.direct_lighting.attenuation_factor);
    }

    // Specular
    std::string specularBlock;
    if (extractTagContent(text, "specular", specularBlock))
    {
        if (findElement(text, "specular", element))
        {
            extractUintAttr(element, "max_depth", cfg.specular.max_recursion_depth);
            extractFloatAttr(element, "glass_ior", cfg.specular.glass_ior);
            extractFloatAttr(element, "mirror_reflectivity", cfg.specular.mirror_reflectivity);
            extractFloatAttr(element, "fresnel_min", cfg.specular.fresnel_min);
            extractFloatAttr(element, "ambient", cfg.specular.ambient);
            extractFloatAttr(element, "indirect_brightness", cfg.specular.indirect_brightness);
            extractFloatAttr(element, "caustic_brightness", cfg.specular.caustic_brightness);
        }
        extractFloat3Element(specularBlock, "glass_tint", cfg.specular.glass_tint, true);
    }

    // Weights
    if (findElement(text, "weights", element))
    {
        extractFloatAttr(element, "direct", cfg.weights.direct);
        extractFloatAttr(element, "indirect", cfg.weights.indirect);
        extractFloatAttr(element, "caustics", cfg.weights.caustics);
        extractFloatAttr(element, "specular", cfg.weights.specular);
    }

    //-------------------------------------------------------------------------
    // Camera
    //-------------------------------------------------------------------------
    std::string cameraBlock;
    if (extractTagContent(text, "camera", cameraBlock))
    {
        if (findElement(text, "camera", element))
            extractFloatAttr(element, "fov", cfg.camera.fov);
        
        if (extractFloat3Element(cameraBlock, "eye", cfg.camera.eye, false) &&
            extractFloat3Element(cameraBlock, "lookAt", cfg.camera.lookAt, false) &&
            extractFloat3Element(cameraBlock, "up", cfg.camera.up, false))
        {
            cfg.camera.hasCamera = true;
        }
    }

    //-------------------------------------------------------------------------
    // Scene Objects
    //-------------------------------------------------------------------------
    std::string objectsBlock;
    if (extractTagContent(text, "objects", objectsBlock))
    {
        // Parse spheres: <sphere x="..." y="..." z="..." radius="...">
        size_t searchPos = 0;
        while (true)
        {
            std::string sphereElement;
            size_t spherePos = objectsBlock.find("<sphere", searchPos);
            if (spherePos == std::string::npos)
                break;

            size_t sphereEnd = objectsBlock.find("</sphere>", spherePos);
            if (sphereEnd == std::string::npos)
                break;

            std::string sphereBlock = objectsBlock.substr(spherePos, sphereEnd - spherePos + 9);
            
            if (findElement(sphereBlock, "sphere", sphereElement))
            {
                SphereObjectConfig sphere;
                
                float x, y, z;
                if (extractFloatAttr(sphereElement, "x", x) &&
                    extractFloatAttr(sphereElement, "y", y) &&
                    extractFloatAttr(sphereElement, "z", z))
                {
                    sphere.center = make_float3(x, y, z);
                }
                extractFloatAttr(sphereElement, "radius", sphere.radius);
                
                // Material
                std::string matElement;
                if (findElement(sphereBlock, "material", matElement))
                {
                    extractAttr(matElement, "type", sphere.materialType);
                    extractFloatAttr(matElement, "ior", sphere.ior);
                    
                    float r, g, b;
                    if (extractFloatAttr(matElement, "r", r) &&
                        extractFloatAttr(matElement, "g", g) &&
                        extractFloatAttr(matElement, "b", b))
                    {
                        sphere.color = make_float3(r, g, b);
                    }
                }
                
                cfg.spheres.push_back(sphere);
            }
            
            searchPos = sphereEnd + 9;
        }

        // Parse meshes: <mesh path="..." x="..." y="..." z="..." scale="...">
        searchPos = 0;
        while (true)
        {
            size_t meshPos = objectsBlock.find("<mesh", searchPos);
            if (meshPos == std::string::npos)
                break;

            size_t meshEnd = objectsBlock.find("</mesh>", meshPos);
            if (meshEnd == std::string::npos)
                break;

            std::string meshBlock = objectsBlock.substr(meshPos, meshEnd - meshPos + 7);
            std::string meshElement;
            
            if (findElement(meshBlock, "mesh", meshElement))
            {
                MeshObjectConfig mesh;
                
                extractAttr(meshElement, "path", mesh.path);
                
                float x, y, z;
                if (extractFloatAttr(meshElement, "x", x) &&
                    extractFloatAttr(meshElement, "y", y) &&
                    extractFloatAttr(meshElement, "z", z))
                {
                    mesh.position = make_float3(x, y, z);
                }
                extractFloatAttr(meshElement, "scale", mesh.scale);
                
                // Material
                std::string matElement;
                if (findElement(meshBlock, "material", matElement))
                {
                    extractAttr(matElement, "type", mesh.materialType);
                    extractFloatAttr(matElement, "ior", mesh.ior);
                    
                    float r, g, b;
                    if (extractFloatAttr(matElement, "r", r) &&
                        extractFloatAttr(matElement, "g", g) &&
                        extractFloatAttr(matElement, "b", b))
                    {
                        mesh.color = make_float3(r, g, b);
                    }
                }
                
                if (!mesh.path.empty())
                    cfg.meshes.push_back(mesh);
            }
            
            searchPos = meshEnd + 7;
        }
    }

    //-------------------------------------------------------------------------
    // Print loaded configuration
    //-------------------------------------------------------------------------
    std::cout << "ConfigLoader: loaded from " << path << std::endl;
    std::cout << "  max_photons=" << cfg.max_photons << std::endl;
    std::cout << "  animated=" << (cfg.animation.enabled ? "true" : "false")
              << " speed=" << cfg.animation.photonSpeed << std::endl;
    std::cout << "  gathering: indirect_r=" << cfg.gathering.indirect_radius
              << " caustic_r=" << cfg.gathering.caustic_radius << std::endl;
    std::cout << "  direct: ambient=" << cfg.direct_lighting.ambient
              << " intensity=" << cfg.direct_lighting.intensity << std::endl;
    std::cout << "  specular: ior=" << cfg.specular.glass_ior
              << " mirror=" << cfg.specular.mirror_reflectivity << std::endl;
    std::cout << "  weights: d=" << cfg.weights.direct
              << " i=" << cfg.weights.indirect
              << " c=" << cfg.weights.caustics
              << " s=" << cfg.weights.specular << std::endl;
    std::cout << "  spheres: " << cfg.spheres.size() << std::endl;
    std::cout << "  meshes: " << cfg.meshes.size() << std::endl;

    return cfg;
}
