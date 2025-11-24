

#pragma once

#include <GLFW/glfw3.h>
#include <iostream>
#include <string>
#include <memory>
#include <functional>


struct ViewportRect
{
    int x, y, width, height;

    ViewportRect(int x = 0, int y = 0, int w = 0, int h = 0)
        : x(x), y(y), width(w), height(h) {}
};


class Window
{
private:
    GLFWwindow *handle = nullptr;
    std::string title;
    int width;
    int height;
    std::function<void(int, int)> viewportCallback = nullptr;

    static void windowSizeCallback(GLFWwindow *glfwWindow, int32_t res_x, int32_t res_y)
    {
        auto *window = reinterpret_cast<Window *>(glfwGetWindowUserPointer(glfwWindow));
        if (window)
        {
            window->width = res_x;
            window->height = res_y;
            std::cout << window->title << " resized to " << res_x << "x" << res_y << std::endl;

            
            if (window->viewportCallback)
            {
                window->viewportCallback(res_x, res_y);
            }
        }
    }

public:
    Window() = default;
    Window(const Window &) = delete;
    Window &operator=(const Window &) = delete;

    ~Window()
    {
        destroy();
    }

    bool create(int w, int h, const std::string &t, std::function<void(int, int)> callback = nullptr)
    {
        title = t;
        width = w;
        height = h;
        viewportCallback = callback;

        handle = glfwCreateWindow(w, h, t.c_str(), nullptr, nullptr);
        if (!handle)
        {
            std::cerr << "Failed to create window: " << t << std::endl;
            return false;
        }

        
        glfwSetWindowUserPointer(handle, this);
        glfwSetWindowSizeCallback(handle, windowSizeCallback);

        std::cout << "Window created: " << t << std::endl;
        return true;
    }

    void destroy()
    {
        if (handle)
        {
            glfwDestroyWindow(handle);
            handle = nullptr;
        }
    }

    void makeCurrent()
    {
        if (handle)
        {
            glfwMakeContextCurrent(handle);
        }
    }

    void swapBuffers()
    {
        if (handle)
        {
            glfwSwapBuffers(handle);
        }
    }

    bool shouldClose() const
    {
        return handle ? glfwWindowShouldClose(handle) : true;
    }

    void requestClose()
    {
        if (handle)
        {
            glfwSetWindowShouldClose(handle, true);
        }
    }

    GLFWwindow *get() const
    {
        return handle;
    }

    int getWidth() const { return width; }
    int getHeight() const { return height; }
    const std::string &getTitle() const { return title; }
};

class WindowManager
{
private:
    std::unique_ptr<Window> window;

    
    ViewportRect leftViewport;
    ViewportRect rightViewport;

public:
    WindowManager() = default;
    WindowManager(const WindowManager &) = delete;
    WindowManager &operator=(const WindowManager &) = delete;

    ~WindowManager()
    {
        shutdown();
    }

    bool initializeGLFW()
    {
        return glfwInit() != 0;
    }

    void setHintsForOpenGL()
    {
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); 
    }

    bool create(int width, int height, const std::string &title)
    {
        window = std::make_unique<Window>();

        
        auto resizeCallback = [this](int w, int h)
        {
            this->updateViewports(w, h);
        };

        if (!window->create(width, height, title, resizeCallback))
        {
            return false;
        }

        
        leftViewport = ViewportRect(0, 0, width / 2, height);
        rightViewport = ViewportRect(width / 2, 0, width / 2, height);

        return true;
    }

    
    void updateViewports(int width, int height)
    {
        leftViewport = ViewportRect(0, 0, width / 2, height);
        rightViewport = ViewportRect(width / 2, 0, width / 2, height);
    }

    bool shouldClose() const
    {
        return window && window->shouldClose();
    }

    Window *getWindow()
    {
        return window.get();
    }

    const ViewportRect &getLeftViewport() const
    {
        return leftViewport;
    }

    const ViewportRect &getRightViewport() const
    {
        return rightViewport;
    }

    void shutdown()
    {
        window.reset();
        if (glfwGetCurrentContext())
        {
            glfwTerminate();
        }
    }
};



