#include "ConfigLoader.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cctype>

// Simple XML parsing helpers - not a full XML parser, but sufficient for our config file

// Trim whitespace from both ends of a string
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

// Extract content between <tag ...>...</tag> (handles tags with attributes)
static bool extractTagContent(const std::string &text, const std::string &tag, std::string &out)
{
    std::string openTagStart = "<" + tag;
    std::string closeTag = "</" + tag + ">";

    // Find opening tag (may have attributes)
    size_t tagStart = text.find(openTagStart);
    if (tagStart == std::string::npos)
        return false;

    // Find the end of the opening tag (the '>')
    size_t tagEnd = text.find(">", tagStart);
    if (tagEnd == std::string::npos)
        return false;

    // Check for self-closing tag
    if (tagEnd > 0 && text[tagEnd - 1] == '/')
        return false;  // Self-closing tags have no content

    size_t contentStart = tagEnd + 1;
    size_t end = text.find(closeTag, contentStart);
    if (end == std::string::npos)
        return false;

    out = trim(text.substr(contentStart, end - contentStart));
    return true;
}

// Extract a numeric value from <tag>value</tag>
template <typename T>
static bool extractNumber(const std::string &text, const std::string &tag, T &out)
{
    std::string content;
    if (!extractTagContent(text, tag, content))
        return false;

    std::stringstream ss(content);
    T value;
    ss >> value;
    if (ss.fail())
        return false;

    out = value;
    return true;
}

// Extract a boolean value from <tag>true/false</tag>
static bool extractBool(const std::string &text, const std::string &tag, bool &out)
{
    std::string content;
    if (!extractTagContent(text, tag, content))
        return false;

    // Convert to lowercase for comparison
    std::string lower = content;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c)
                   { return std::tolower(c); });

    if (lower == "true" || lower == "1")
    {
        out = true;
        return true;
    }
    else if (lower == "false" || lower == "0")
    {
        out = false;
        return true;
    }
    return false;
}

// Extract a float3 from an element with x, y, z or r, g, b attributes
// e.g., <eye x="1.0" y="2.0" z="3.0"/> or <color r="0.5" g="0.5" b="0.5"/>
static bool extractFloat3Attr(const std::string &text, const std::string &tag, float3 &out, bool rgb = false)
{
    // Find the tag opening
    std::string pattern = "<" + tag;
    size_t pos = text.find(pattern);
    if (pos == std::string::npos)
        return false;

    // Find the end of this element (either /> or >)
    size_t endPos = text.find(">", pos);
    if (endPos == std::string::npos)
        return false;

    std::string element = text.substr(pos, endPos - pos + 1);

    const char *attr1 = rgb ? "r" : "x";
    const char *attr2 = rgb ? "g" : "y";
    const char *attr3 = rgb ? "b" : "z";

    auto extractAttr = [&element](const char *attr, float &val) -> bool
    {
        std::string attrPattern = std::string(attr) + "=\"";
        size_t attrPos = element.find(attrPattern);
        if (attrPos == std::string::npos)
            return false;

        attrPos += attrPattern.length();
        size_t attrEnd = element.find('"', attrPos);
        if (attrEnd == std::string::npos)
            return false;

        std::string valStr = element.substr(attrPos, attrEnd - attrPos);
        std::stringstream ss(valStr);
        ss >> val;
        return !ss.fail();
    };

    float x, y, z;
    if (!extractAttr(attr1, x) || !extractAttr(attr2, y) || !extractAttr(attr3, z))
        return false;

    out = make_float3(x, y, z);
    return true;
}

// Extract string content from <tag>text</tag>
static bool extractString(const std::string &text, const std::string &tag, std::string &out)
{
    return extractTagContent(text, tag, out);
}

// Extract attribute value from an element
static bool extractAttribute(const std::string &text, const std::string &tag, const std::string &attr, std::string &out)
{
    std::string pattern = "<" + tag;
    size_t pos = text.find(pattern);
    if (pos == std::string::npos)
        return false;

    size_t endPos = text.find(">", pos);
    if (endPos == std::string::npos)
        return false;

    std::string element = text.substr(pos, endPos - pos + 1);

    std::string attrPattern = attr + "=\"";
    size_t attrPos = element.find(attrPattern);
    if (attrPos == std::string::npos)
        return false;

    attrPos += attrPattern.length();
    size_t attrEnd = element.find('"', attrPos);
    if (attrEnd == std::string::npos)
        return false;

    out = element.substr(attrPos, attrEnd - attrPos);
    return true;
}

PhotonMappingConfig ConfigLoader::load(const std::string &path)
{
    PhotonMappingConfig cfg;

    std::ifstream file(path);
    if (!file)
    {
        std::cerr << "ConfigLoader: could not open configuration file: " << path
                  << " - using default photon mapping parameters." << std::endl;
        return cfg;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    const std::string text = buffer.str();

    unsigned int uval = 0;
    float fval = 0.0f;
    bool bval = false;

    // Basic photon mapping params
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
    if (extractFloat3Attr(text, "glass_tint", glass_tint, true))
        cfg.specular.glass_tint = glass_tint;
    if (extractNumber<float>(text, "mirror_reflectivity", fval))
        cfg.specular.mirror_reflectivity = fval;
    if (extractNumber<float>(text, "fresnel_min", fval))
        cfg.specular.fresnel_min = fval;
    if (extractNumber<float>(text, "specular_indirect_brightness", fval))
        cfg.specular.indirect_brightness = fval;
    if (extractNumber<float>(text, "specular_caustic_brightness", fval))
        cfg.specular.caustic_brightness = fval;
    if (extractNumber<float>(text, "specular_ambient", fval))
        cfg.specular.ambient = fval;

    // Weights configuration (inside <weights> block)
    std::string weightsBlock;
    if (extractTagContent(text, "weights", weightsBlock))
    {
        float w;
        if (extractNumber<float>(weightsBlock, "direct", w))
            cfg.weights.direct = w;
        if (extractNumber<float>(weightsBlock, "indirect", w))
            cfg.weights.indirect = w;
        if (extractNumber<float>(weightsBlock, "caustics", w))
            cfg.weights.caustics = w;
        if (extractNumber<float>(weightsBlock, "specular", w))
            cfg.weights.specular = w;
    }

    // Wall colors configuration
    std::string wallsBlock;
    if (extractTagContent(text, "walls", wallsBlock))
    {
        float3 wallColor;
        if (extractFloat3Attr(wallsBlock, "floor", wallColor, true))
            cfg.walls.floor = wallColor;
        if (extractFloat3Attr(wallsBlock, "ceiling", wallColor, true))
            cfg.walls.ceiling = wallColor;
        if (extractFloat3Attr(wallsBlock, "back", wallColor, true))
            cfg.walls.back = wallColor;
        if (extractFloat3Attr(wallsBlock, "left", wallColor, true))
            cfg.walls.left = wallColor;
        if (extractFloat3Attr(wallsBlock, "right", wallColor, true))
            cfg.walls.right = wallColor;
    }

    // Camera configuration
    std::string cameraBlock;
    if (extractTagContent(text, "camera", cameraBlock))
    {
        float3 eye, lookAt, up;
        float fovVal = 0.0f;
        bool haveEye = extractFloat3Attr(cameraBlock, "eye", eye, false);
        bool haveLookAt = extractFloat3Attr(cameraBlock, "lookAt", lookAt, false);
        bool haveUp = extractFloat3Attr(cameraBlock, "up", up, false);
        bool haveFov = extractNumber<float>(cameraBlock, "fov", fovVal);

        if (haveEye && haveLookAt && haveUp)
        {
            cfg.camera.hasCamera = true;
            cfg.camera.eye = eye;
            cfg.camera.lookAt = lookAt;
            cfg.camera.up = up;
            if (haveFov)
                cfg.camera.fov = fovVal;
        }
    }

    // Parse all objects from <objects> block
    std::string objectsBlock;
    if (extractTagContent(text, "objects", objectsBlock))
    {
        size_t searchPos = 0;
        while (true)
        {
            // Find next <object tag
            size_t objStart = objectsBlock.find("<object", searchPos);
            if (objStart == std::string::npos)
                break;

            // Find the end of the opening tag
            size_t tagEnd = objectsBlock.find(">", objStart);
            if (tagEnd == std::string::npos)
                break;

            size_t objEnd;
            // Check if it's self-closing (ends with />)
            if (tagEnd > 0 && objectsBlock[tagEnd - 1] == '/')
            {
                objEnd = tagEnd + 1;
            }
            else
            {
                // Look for matching </object>
                size_t objEndTag = objectsBlock.find("</object>", tagEnd);
                if (objEndTag == std::string::npos)
                    break;
                objEnd = objEndTag + 9;
            }

            std::string objBlock = objectsBlock.substr(objStart, objEnd - objStart);
            std::string typeAttr;
            extractAttribute(objBlock, "object", "type", typeAttr);

            // Parse MESH objects
            if (typeAttr == "mesh")
            {
                MeshObjectConfig mesh;
                extractString(objBlock, "path", mesh.path);

                float3 pos;
                if (extractFloat3Attr(objBlock, "position", pos, false))
                    mesh.position = pos;

                float scaleVal;
                if (extractNumber<float>(objBlock, "scale", scaleVal))
                    mesh.scale = scaleVal;

                std::string materialBlock;
                if (extractTagContent(objBlock, "material", materialBlock))
                {
                    std::string matType;
                    if (extractAttribute(objBlock, "material", "type", matType))
                        mesh.materialType = matType;

                    float3 col;
                    if (extractFloat3Attr(materialBlock, "color", col, true))
                        mesh.color = col;

                    float iorVal;
                    if (extractNumber<float>(materialBlock, "ior", iorVal))
                        mesh.ior = iorVal;
                }

                if (!mesh.path.empty())
                    cfg.meshes.push_back(mesh);
            }
            // Parse SPHERE objects
            else if (typeAttr == "sphere")
            {
                SphereObjectConfig sphere;

                float3 center;
                if (extractFloat3Attr(objBlock, "center", center, false))
                    sphere.center = center;

                float radiusVal;
                if (extractNumber<float>(objBlock, "radius", radiusVal))
                    sphere.radius = radiusVal;

                std::string materialBlock;
                if (extractTagContent(objBlock, "material", materialBlock))
                {
                    std::string matType;
                    if (extractAttribute(objBlock, "material", "type", matType))
                        sphere.materialType = matType;

                    float3 col;
                    if (extractFloat3Attr(materialBlock, "color", col, true))
                        sphere.color = col;

                    float iorVal;
                    if (extractNumber<float>(materialBlock, "ior", iorVal))
                        sphere.ior = iorVal;
                }

                cfg.spheres.push_back(sphere);
            }
            // Parse QUAD objects (mirrors, etc.)
            else if (typeAttr == "quad")
            {
                QuadObjectConfig quad;

                float3 corner;
                if (extractFloat3Attr(objBlock, "corner", corner, false))
                    quad.corner = corner;

                float3 edge1;
                if (extractFloat3Attr(objBlock, "edge1", edge1, false))
                    quad.edge1 = edge1;

                float3 edge2;
                if (extractFloat3Attr(objBlock, "edge2", edge2, false))
                    quad.edge2 = edge2;

                std::string materialBlock;
                if (extractTagContent(objBlock, "material", materialBlock))
                {
                    std::string matType;
                    if (extractAttribute(objBlock, "material", "type", matType))
                        quad.materialType = matType;

                    float3 col;
                    if (extractFloat3Attr(materialBlock, "color", col, true))
                        quad.color = col;
                }

                cfg.quads.push_back(quad);
            }

            searchPos = objEnd;
        }
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
              << ", right=(" << cfg.walls.right.x << "," << cfg.walls.right.y << "," << cfg.walls.right.z << ")"
              << std::endl;
    std::cout << "  spheres: " << cfg.spheres.size() << " sphere(s)" << std::endl;
    for (const auto &s : cfg.spheres)
    {
        std::cout << "    - " << s.materialType << " at (" << s.center.x << "," << s.center.y << "," << s.center.z
                  << ") r=" << s.radius << std::endl;
    }
    std::cout << "  quads: " << cfg.quads.size() << " quad(s)" << std::endl;
    for (const auto &q : cfg.quads)
    {
        std::cout << "    - " << q.materialType << " at (" << q.corner.x << "," << q.corner.y << "," << q.corner.z
                  << ")" << std::endl;
    }
    std::cout << "  meshes: " << cfg.meshes.size() << " mesh(es)" << std::endl;
    for (const auto &m : cfg.meshes)
    {
        std::cout << "    - " << m.path << " at (" << m.position.x << "," << m.position.y << "," << m.position.z
                  << ") scale=" << m.scale << std::endl;
    }

    return cfg;
}
