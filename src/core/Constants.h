#pragma once

namespace Constants
{
    namespace Cornell
    {
        // White surfaces (floor, ceiling, back wall)
        constexpr float WHITE_R = 0.8f;
        constexpr float WHITE_G = 0.8f;
        constexpr float WHITE_B = 0.8f;

        // Red wall (left)
        constexpr float RED_R = 0.8f;
        constexpr float RED_G = 0.0f;
        constexpr float RED_B = 0.0f;

        // Blue wall (right)
        constexpr float BLUE_R = 0.0f;
        constexpr float BLUE_G = 0.0f;
        constexpr float BLUE_B = 0.8f;

        // Box dimensions
        constexpr float WIDTH = 556.0f;
        constexpr float HEIGHT = 548.8f;
        constexpr float DEPTH = 559.2f;
    }

    namespace Photon
    {
        // Default diffuse reflection probability for Russian Roulette
        constexpr float DEFAULT_DIFFUSE_PROB = 0.7f;

        // Maximum bounces before forced termination
        constexpr unsigned int MAX_BOUNCES = 10;
    }
}
