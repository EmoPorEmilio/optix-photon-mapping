#pragma once

namespace Constants
{
    //=========================================================================
    // Mathematical constants
    //=========================================================================
    namespace Math
    {
        constexpr float PI = 3.14159265358979323846f;
        constexpr float INV_PI = 0.31830988618379067154f;  // 1/π
        constexpr float TWO_PI = 6.28318530717958647692f;
    }

    //=========================================================================
    // Photon mapping constants (Jensen's algorithm)
    //=========================================================================
    namespace Photon
    {
        // Cone filter normalization factor k for w(d) = 1 - d/r
        // Jensen's formula: L = (ρ/π) * (k/(π*r²)) * Σ(Φ_p * w_p)
        constexpr float CONE_FILTER_K = 3.0f;

        // Default diffuse reflection probability for Russian Roulette
        constexpr float DEFAULT_DIFFUSE_PROB = 0.7f;

        // Maximum bounces before forced termination
        constexpr unsigned int MAX_BOUNCES = 10;

        // Caustic gather radius multiplier (relative to global radius)
        // Caustics need tighter radius for sharper patterns
        constexpr float CAUSTIC_RADIUS_MULTIPLIER = 0.5f;
    }

    //=========================================================================
    // Rendering constants
    //=========================================================================
    namespace Render
    {
        // Gamma correction exponent for sRGB display
        constexpr float GAMMA = 2.2f;
        constexpr float INV_GAMMA = 1.0f / 2.2f;  // ~0.4545
    }

    //=========================================================================
    // Cornell box geometry
    //=========================================================================
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
}
