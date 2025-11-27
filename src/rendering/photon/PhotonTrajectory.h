#pragma once

#include <sutil/vec_math.h>

//=============================================================================
// Photon Trajectory Recording System
// Records the complete journey of photons for debugging and visualization
//=============================================================================

// CUDA host/device compatibility
#ifdef __CUDACC__
#define TRAJECTORY_HOSTDEVICE __host__ __device__
#else
#define TRAJECTORY_HOSTDEVICE
#endif

//-----------------------------------------------------------------------------
// Event types that can occur during photon transport
//-----------------------------------------------------------------------------
enum PhotonEventType
{
    EVENT_NONE = 0,          // Unused slot
    EVENT_EMITTED,           // Initial emission from light source
    EVENT_DIFFUSE_BOUNCE,    // Hit diffuse surface, bounced (survived Russian roulette)
    EVENT_DIFFUSE_STORED,    // Hit diffuse surface, stored to photon map
    EVENT_DIFFUSE_ABSORBED,  // Hit diffuse surface, absorbed (Russian roulette termination)
    EVENT_SPECULAR_REFLECT,  // Hit mirror surface, reflected
    EVENT_GLASS_REFLECT,     // Hit glass, reflected (Fresnel or TIR)
    EVENT_GLASS_REFRACT,     // Hit glass, refracted through
    EVENT_MISS,              // Ray escaped scene (hit nothing)
    EVENT_MAX_DEPTH,         // Terminated due to max bounce limit
    EVENT_HIT_LIGHT          // Hit light source geometry
};

//-----------------------------------------------------------------------------
// Material types (mirrors Material.h for trajectory recording)
//-----------------------------------------------------------------------------
enum TrajectoryMaterialType
{
    TRAJ_MAT_NONE = -1,
    TRAJ_MAT_DIFFUSE = 0,
    TRAJ_MAT_SPECULAR = 1,
    TRAJ_MAT_TRANSMISSIVE = 2
};

//-----------------------------------------------------------------------------
// Single event in a photon's trajectory
//-----------------------------------------------------------------------------
struct PhotonEvent
{
    float3 position;         // World position of this event
    float3 direction;        // Direction vector (outgoing for bounces, incoming for terminal)
    float3 power;            // Photon power/throughput at this point
    int event_type;          // PhotonEventType
    int material_type;       // TrajectoryMaterialType (-1 if N/A)
    float extra_data;        // Extra info (e.g., Fresnel reflectance for glass)
    int _pad;                // Padding for alignment

    TRAJECTORY_HOSTDEVICE PhotonEvent()
        : position(make_float3(0.0f, 0.0f, 0.0f)),
          direction(make_float3(0.0f, 0.0f, 0.0f)),
          power(make_float3(0.0f, 0.0f, 0.0f)),
          event_type(EVENT_NONE),
          material_type(TRAJ_MAT_NONE),
          extra_data(0.0f),
          _pad(0) {}

    TRAJECTORY_HOSTDEVICE PhotonEvent(int type, const float3 &pos, const float3 &dir, 
                                       const float3 &pow, int mat = TRAJ_MAT_NONE)
        : position(pos),
          direction(dir),
          power(pow),
          event_type(type),
          material_type(mat),
          extra_data(0.0f),
          _pad(0) {}
};

//-----------------------------------------------------------------------------
// Complete trajectory of a single photon
//-----------------------------------------------------------------------------
#define MAX_TRAJECTORY_EVENTS 14  // max_depth(10) + emission + possible extra events

struct PhotonTrajectory
{
    unsigned int photon_id;                      // Which photon this is
    unsigned int event_count;                    // Number of valid events recorded
    PhotonEvent events[MAX_TRAJECTORY_EVENTS];   // Event history

    TRAJECTORY_HOSTDEVICE PhotonTrajectory()
        : photon_id(0), event_count(0) {}

    // Add an event to this trajectory (GPU-safe)
    TRAJECTORY_HOSTDEVICE bool addEvent(const PhotonEvent &evt)
    {
        if (event_count < MAX_TRAJECTORY_EVENTS)
        {
            events[event_count++] = evt;
            return true;
        }
        return false;
    }

    // Add event with individual parameters (convenience)
    TRAJECTORY_HOSTDEVICE bool addEvent(int type, const float3 &pos, const float3 &dir,
                                         const float3 &pow, int mat = TRAJ_MAT_NONE)
    {
        return addEvent(PhotonEvent(type, pos, dir, pow, mat));
    }
};

//-----------------------------------------------------------------------------
// Helper to get string name for event type (CPU only)
//-----------------------------------------------------------------------------
#ifndef __CUDACC__
inline const char* getEventTypeName(int type)
{
    switch (type)
    {
        case EVENT_NONE:             return "NONE";
        case EVENT_EMITTED:          return "EMITTED";
        case EVENT_DIFFUSE_BOUNCE:   return "DIFFUSE_BOUNCE";
        case EVENT_DIFFUSE_STORED:   return "DIFFUSE_STORED";
        case EVENT_DIFFUSE_ABSORBED: return "DIFFUSE_ABSORBED";
        case EVENT_SPECULAR_REFLECT: return "SPECULAR_REFLECT";
        case EVENT_GLASS_REFLECT:    return "GLASS_REFLECT";
        case EVENT_GLASS_REFRACT:    return "GLASS_REFRACT";
        case EVENT_MISS:             return "MISS";
        case EVENT_MAX_DEPTH:        return "MAX_DEPTH";
        case EVENT_HIT_LIGHT:        return "HIT_LIGHT";
        default:                     return "UNKNOWN";
    }
}

inline const char* getMaterialTypeName(int type)
{
    switch (type)
    {
        case TRAJ_MAT_NONE:         return "NONE";
        case TRAJ_MAT_DIFFUSE:      return "DIFFUSE";
        case TRAJ_MAT_SPECULAR:     return "SPECULAR";
        case TRAJ_MAT_TRANSMISSIVE: return "TRANSMISSIVE";
        default:                    return "UNKNOWN";
    }
}
#endif

