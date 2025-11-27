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
    float3 power;             // Current photon power (RGB) - modulated for NEXT bounce
    float3 arrivalPower;      // Power when photon ARRIVED at current position (for display)
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
                       arrivalPower(make_float3(1.0f, 1.0f, 1.0f)),
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

    // Get visual color for display based on photon power
    float3 getDisplayColor() const
    {
        // Normalize power to visible range
        float maxPow = fmaxf(fmaxf(power.x, power.y), power.z);
        if (maxPow > 0.0f)
        {
            // Boost visibility while preserving color ratio
            float scale = 1.0f / maxPow;
            return make_float3(
                fminf(power.x * scale, 1.0f),
                fminf(power.y * scale, 1.0f),
                fminf(power.z * scale, 1.0f)
            );
        }
        return make_float3(1.0f, 1.0f, 1.0f);
    }
};
