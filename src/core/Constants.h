#pragma once

namespace Constants
{
    namespace Math
    {
        constexpr float PI = 3.14159265358979323846f;
        constexpr float INV_PI = 0.31830988618379067154f;
        constexpr float TWO_PI = 6.28318530717958647692f;
    }

    namespace Photon
    {
        constexpr float CONE_FILTER_K = 3.0f;  // Jensen cone filter k for w(d) = 1 - d/r
        constexpr float DEFAULT_DIFFUSE_PROB = 0.7f;
        constexpr unsigned int MAX_BOUNCES = 10;
        constexpr float CAUSTIC_RADIUS_MULTIPLIER = 0.5f;
    }

    namespace Render
    {
        constexpr float GAMMA = 2.2f;
        constexpr float INV_GAMMA = 1.0f / 2.2f;
    }

    namespace Cornell
    {
        constexpr float WHITE_R = 0.8f;
        constexpr float WHITE_G = 0.8f;
        constexpr float WHITE_B = 0.8f;
        constexpr float RED_R = 0.8f;
        constexpr float RED_G = 0.0f;
        constexpr float RED_B = 0.0f;
        constexpr float BLUE_R = 0.0f;
        constexpr float BLUE_G = 0.0f;
        constexpr float BLUE_B = 0.8f;
        constexpr float WIDTH = 556.0f;
        constexpr float HEIGHT = 548.8f;
        constexpr float DEPTH = 559.2f;
    }
}
