#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "WindowManager.h"
#include "Camera.h"
#include "Scene.h"
#include <vector>
#include <sutil/vec_math.h>

struct AnimatedPhoton; // Forward declaration

class RasterRenderer
{
private:
    Window *window;
    const ViewportRect &viewport;
    Camera *camera = nullptr;
    Scene *scene = nullptr;

    unsigned int shaderProgram = 0;
    unsigned int VAO = 0;
    unsigned int VBO = 0;
    unsigned int vertexCount = 0;

    // Photon rendering
    unsigned int photonVAO = 0;
    unsigned int photonVBO = 0;
    std::vector<AnimatedPhoton> photons;

    void createShaderProgram();
    void buildSceneGeometry();
    void renderPhotons();

public:
    RasterRenderer(Window *w, const ViewportRect &vp);
    ~RasterRenderer();

    void setCamera(Camera *cam);
    void setScene(Scene *s);
    void setAnimatedPhotons(const std::vector<AnimatedPhoton>& p);
    
    void renderFrame();
};
