#pragma once

#include <sutil/vec_math.h>
#include <vector>

// Maximum number of bounces to track for path visualization
constexpr unsigned int MAX_PHOTON_BOUNCES = 10;

struct AnimatedPhoton
{
    float3 position;         // Current position (may be at collision)
    float3 emissionPosition; // Original emission position on light surface
    float3 direction;
    float3 velocity;
    bool isActive;

    // === Phase 1: Bouncing & Color Tracking ===
    float3 power;             // Current photon power (RGB) - starts as light emission, modulated by surfaces
    unsigned int bounceCount; // How many times this photon has bounced
    unsigned int maxBounces;  // Maximum bounces before termination

    // Path history for visualization (stores positions at each bounce)
    std::vector<float3> pathHistory;
    std::vector<float3> pathColors; // Color at each path point (shows color transfer)

    // Surface hit information (for debugging)
    float3 lastHitNormal;
    float3 lastSurfaceColor;
    bool wasAbsorbed; // True if terminated by Russian Roulette

    AnimatedPhoton() : position(make_float3(0.0f, 0.0f, 0.0f)),
                       emissionPosition(make_float3(0.0f, 0.0f, 0.0f)),
                       direction(make_float3(0.0f, 0.0f, 0.0f)),
                       velocity(make_float3(0.0f, 0.0f, 0.0f)),
                       isActive(true),
                       power(make_float3(1.0f, 1.0f, 1.0f)),
                       bounceCount(0),
                       maxBounces(MAX_PHOTON_BOUNCES),
                       lastHitNormal(make_float3(0.0f, 1.0f, 0.0f)),
                       lastSurfaceColor(make_float3(1.0f, 1.0f, 1.0f)),
                       wasAbsorbed(false)
    {
        pathHistory.reserve(MAX_PHOTON_BOUNCES + 1);
        pathColors.reserve(MAX_PHOTON_BOUNCES + 1);
    }

    // Record current position in path history
    void recordPathPoint()
    {
        pathHistory.push_back(position);
        pathColors.push_back(power);
    }

    // Get visual color (normalized power for display)
    float3 getDisplayColor() const
    {
        // Normalize power to [0,1] range for display
        float maxComp = fmaxf(fmaxf(power.x, power.y), power.z);
        if (maxComp > 0.0f)
        {
            return power / maxComp;
        }
        // Power is zero (e.g., blue photon hit red wall) - show last surface color
        // This represents a photon that was fully absorbed by an incompatible surface
        return lastSurfaceColor;
    }
};
