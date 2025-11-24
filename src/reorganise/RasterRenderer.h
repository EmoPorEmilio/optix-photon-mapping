#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "WindowManager.h"
#include "Camera.h"
#include "Scene.h"
#include <vector>
#include <sutil/vec_math.h>

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

    void createShaderProgram();
    void buildSceneGeometry();

public:
    RasterRenderer(Window *w, const ViewportRect &vp);
    ~RasterRenderer();

    void setCamera(Camera *cam);
    void setScene(Scene *s);
    
    void renderFrame();
};
