

#include "InputCommandManager.h"


void InputCommandManager::glfwKeyCallback(GLFWwindow *glfwWindow, int key, int scancode, int action, int mods)
{
    auto *manager = reinterpret_cast<InputCommandManager *>(glfwGetWindowUserPointer(glfwWindow));
    if (manager)
    {
        manager->handleKey(key, action);
    }
}

void InputCommandManager::glfwMouseButtonCallback(GLFWwindow *glfwWindow, int button, int action, int mods)
{
    auto *manager = reinterpret_cast<InputCommandManager *>(glfwGetWindowUserPointer(glfwWindow));
    if (manager)
    {
        double x, y;
        glfwGetCursorPos(glfwWindow, &x, &y);
        manager->handleMouseButton(button, action, static_cast<int>(x), static_cast<int>(y));
    }
}

void InputCommandManager::glfwCursorPosCallback(GLFWwindow *glfwWindow, double x, double y)
{
    auto *manager = reinterpret_cast<InputCommandManager *>(glfwGetWindowUserPointer(glfwWindow));
    if (manager)
    {
        manager->handleMouseMove(x, y);
    }
}

void InputCommandManager::glfwScrollCallback(GLFWwindow *glfwWindow, double xoffset, double yoffset)
{
    auto *manager = reinterpret_cast<InputCommandManager *>(glfwGetWindowUserPointer(glfwWindow));
    if (manager)
    {
        manager->handleMouseWheel(yoffset);
    }
}

void InputCommandManager::handleKey(int key, int action)
{
    if (action == GLFW_PRESS && keyBindings.count(key))
    {
        keyBindings[key]();
    }
}

void InputCommandManager::handleMouseButton(int button, int action, int x, int y)
{
    if (action == GLFW_PRESS)
    {
        if (button == GLFW_MOUSE_BUTTON_LEFT)
            isDraggingLeft = true;
        else if (button == GLFW_MOUSE_BUTTON_RIGHT)
            isDraggingRight = true;
        else if (button == GLFW_MOUSE_BUTTON_MIDDLE)
            isDraggingMiddle = true;

        lastMouseX = x;
        lastMouseY = y;

        if (onMouseClick)
        {
            onMouseClick(button, x, y);
        }
    }
    else if (action == GLFW_RELEASE)
    {
        if (button == GLFW_MOUSE_BUTTON_LEFT)
            isDraggingLeft = false;
        else if (button == GLFW_MOUSE_BUTTON_RIGHT)
            isDraggingRight = false;
        else if (button == GLFW_MOUSE_BUTTON_MIDDLE)
            isDraggingMiddle = false;
    }
}

void InputCommandManager::handleMouseMove(double x, double y)
{
    double deltaX = x - lastMouseX;
    double deltaY = y - lastMouseY;

    lastMouseX = x;
    lastMouseY = y;

    if (isDraggingLeft && onMouseDragLeft)
    {
        onMouseDragLeft(static_cast<int>(deltaX), static_cast<int>(deltaY));
    }
    else if (isDraggingRight && onMouseDragRight)
    {
        onMouseDragRight(static_cast<int>(deltaX), static_cast<int>(deltaY));
    }
    else if (isDraggingMiddle && onMouseDragMiddle)
    {
        onMouseDragMiddle(static_cast<int>(deltaX), static_cast<int>(deltaY));
    }
}

void InputCommandManager::handleMouseWheel(double yoffset)
{
    if (onMouseWheel)
    {
        onMouseWheel(yoffset);
    }
}

void InputCommandManager::initialize(GLFWwindow *glfwWindow)
{
    if (glfwWindow)
    {
        glfwSetWindowUserPointer(glfwWindow, this);
        glfwSetKeyCallback(glfwWindow, glfwKeyCallback);
        glfwSetMouseButtonCallback(glfwWindow, glfwMouseButtonCallback);
        glfwSetCursorPosCallback(glfwWindow, glfwCursorPosCallback);
        glfwSetScrollCallback(glfwWindow, glfwScrollCallback);
    }
}

void InputCommandManager::pollEvents()
{
    glfwPollEvents();
}



