

#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "WindowManager.h"
#include <iostream>
#include <memory>

class OpenGLManager
{
private:
    WindowManager windowManager;
    bool initialized = false;

public:
    OpenGLManager() = default;
    
    bool initialize()
    {
        if (initialized)
            return true;

        
        if (!windowManager.initializeGLFW())
        {
            std::cerr << "Failed to initialize GLFW" << std::endl;
            return false;
        }

        
        windowManager.setHintsForOpenGL();

        
        if (!windowManager.create(1536, 768, "Optix PhotonMapping RayTracer"))
        {
            std::cerr << "Failed to create window" << std::endl;
            return false;
        }

        Window *window = windowManager.getWindow();

        
        window->makeCurrent();
        if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
        {
            std::cerr << "Failed to initialize GLAD" << std::endl;
            return false;
        }

        std::cout << "GLAD initialized successfully" << std::endl;

        initialized = true;
        return true;
    }

    bool isInitialized() const
    {
        return initialized;
    }

    bool shouldClose() const
    {
        return windowManager.shouldClose();
    }

    Window *getWindow()
    {
        return windowManager.getWindow();
    }

    const ViewportRect &getLeftViewport() const
    {
        return windowManager.getLeftViewport();
    }

    const ViewportRect &getRightViewport() const
    {
        return windowManager.getRightViewport();
    }
};



