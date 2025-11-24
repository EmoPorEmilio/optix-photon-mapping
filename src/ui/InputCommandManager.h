

#pragma once

#include <GLFW/glfw3.h>
#include <functional>
#include <unordered_map>

class InputCommandManager
{
private:
    std::unordered_map<int, std::function<void()>> keyBindings;
    std::function<void(int, int)> onMouseDragLeft;
    std::function<void(int, int)> onMouseDragRight;
    std::function<void(int, int)> onMouseDragMiddle;
    std::function<void(double)> onMouseWheel;
    std::function<void(int, int, int)> onMouseClick;

    
    bool isDraggingLeft = false;
    bool isDraggingRight = false;
    bool isDraggingMiddle = false;
    double lastMouseX = 0.0;
    double lastMouseY = 0.0;

    
    static void glfwKeyCallback(GLFWwindow *glfwWindow, int key, int scancode, int action, int mods);
    static void glfwMouseButtonCallback(GLFWwindow *glfwWindow, int button, int action, int mods);
    static void glfwCursorPosCallback(GLFWwindow *glfwWindow, double x, double y);
    static void glfwScrollCallback(GLFWwindow *glfwWindow, double xoffset, double yoffset);

    void handleKey(int key, int action);
    void handleMouseButton(int button, int action, int x, int y);
    void handleMouseMove(double x, double y);
    void handleMouseWheel(double yoffset);

public:
    InputCommandManager() = default;
    ~InputCommandManager() = default;

    void initialize(GLFWwindow *glfwWindow);
    void pollEvents();

    void bindKey(int key, std::function<void()> action)
    {
        keyBindings[key] = action;
    }

    
    void setMouseDragHandler(std::function<void(int, int)> handler)
    {
        onMouseDragLeft = handler;
    }

    void setMouseDragLeftHandler(std::function<void(int, int)> handler)
    {
        onMouseDragLeft = handler;
    }

    void setMouseDragRightHandler(std::function<void(int, int)> handler)
    {
        onMouseDragRight = handler;
    }

    void setMouseDragMiddleHandler(std::function<void(int, int)> handler)
    {
        onMouseDragMiddle = handler;
    }

    void setMouseWheelHandler(std::function<void(double)> handler)
    {
        onMouseWheel = handler;
    }

    void setMouseClickHandler(std::function<void(int, int, int)> handler)
    {
        onMouseClick = handler;
    }
};



