#pragma once

#include "PhotonTrajectory.h"
#include "Photon.h"
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>

// Animates photons through their recorded trajectories
class TrajectoryAnimator
{
public:
    struct AnimatedPhotonState
    {
        float3 position;
        float3 power;
        int currentEvent;
        float progress;      // 0-1 progress between current and next event
        bool active;
        bool terminated;
    };

    TrajectoryAnimator() = default;

    void loadTrajectories(const std::vector<PhotonTrajectory>& trajectories)
    {
        m_trajectories = trajectories;
        reset();
    }

    bool loadFromFile(const std::string& filename)
    {
        m_trajectories.clear();
        std::ifstream file(filename);
        if (!file.is_open())
            return false;

        std::string line;
        PhotonTrajectory currentTraj;
        bool inPhoton = false;

        while (std::getline(file, line))
        {
            if (line.empty() || line[0] == '#')
                continue;

            if (line.rfind("PHOTON ", 0) == 0)
            {
                if (inPhoton && currentTraj.event_count > 0)
                    m_trajectories.push_back(currentTraj);

                currentTraj = PhotonTrajectory();
                sscanf(line.c_str(), "PHOTON %u", &currentTraj.photon_id);
                inPhoton = true;
                continue;
            }

            if (inPhoton && line.find('[') != std::string::npos)
            {
                PhotonEvent evt;
                float px, py, pz, dx, dy, dz, powr, powg, powb;
                char eventType[64] = {0};

                // Parse: [idx] EVENT_TYPE pos=(x,y,z) dir=(x,y,z) power=(r,g,b)
                if (sscanf(line.c_str(), " [%*d] %63s pos=(%f,%f,%f) dir=(%f,%f,%f) power=(%f,%f,%f)",
                           eventType, &px, &py, &pz, &dx, &dy, &dz, &powr, &powg, &powb) >= 10)
                {
                    evt.position = make_float3(px, py, pz);
                    evt.direction = make_float3(dx, dy, dz);
                    evt.power = make_float3(powr, powg, powb);
                    evt.event_type = parseEventType(eventType);
                    currentTraj.addEvent(evt);
                }
            }
        }

        if (inPhoton && currentTraj.event_count > 0)
            m_trajectories.push_back(currentTraj);

        file.close();
        reset();
        return !m_trajectories.empty();
    }

    void reset()
    {
        m_time = 0.0f;
        m_photonStates.clear();
        m_photonStates.resize(m_trajectories.size());

        for (size_t i = 0; i < m_trajectories.size(); ++i)
        {
            auto& state = m_photonStates[i];
            state.currentEvent = 0;
            state.progress = 0.0f;
            state.active = false;
            state.terminated = false;

            if (m_trajectories[i].event_count > 0)
            {
                state.position = m_trajectories[i].events[0].position;
                state.power = m_trajectories[i].events[0].power;
            }
        }
        m_nextPhotonToActivate = 0;
    }

    void update(float deltaTime, float photonSpeed, float emissionInterval)
    {
        m_time += deltaTime;

        // Activate new photons based on emission interval
        float photonsToActivate = m_time / emissionInterval;
        while (m_nextPhotonToActivate < m_photonStates.size() &&
               m_nextPhotonToActivate < static_cast<size_t>(photonsToActivate))
        {
            m_photonStates[m_nextPhotonToActivate].active = true;
            m_nextPhotonToActivate++;
        }

        // Update each active photon
        for (size_t i = 0; i < m_photonStates.size(); ++i)
        {
            auto& state = m_photonStates[i];
            if (!state.active || state.terminated)
                continue;

            const auto& traj = m_trajectories[i];
            if (traj.event_count < 2)
            {
                state.terminated = true;
                continue;
            }

            int currEvt = state.currentEvent;
            int nextEvt = currEvt + 1;

            if (nextEvt >= static_cast<int>(traj.event_count))
            {
                state.terminated = true;
                continue;
            }

            // Calculate distance between events
            float3 from = traj.events[currEvt].position;
            float3 to = traj.events[nextEvt].position;
            float dist = length(to - from);

            if (dist < 0.001f)
            {
                // Events at same position, skip to next
                state.currentEvent++;
                state.progress = 0.0f;
                continue;
            }

            // Move progress based on speed
            float travelTime = dist / photonSpeed;
            state.progress += deltaTime / travelTime;

            if (state.progress >= 1.0f)
            {
                // Move to next segment
                state.currentEvent++;
                state.progress = 0.0f;

                if (state.currentEvent + 1 >= static_cast<int>(traj.event_count))
                {
                    state.terminated = true;
                    state.position = traj.events[traj.event_count - 1].position;
                    state.power = traj.events[traj.event_count - 1].power;
                }
            }
            else
            {
                // Interpolate position
                state.position = from + (to - from) * state.progress;
                // Interpolate power
                float3 powFrom = traj.events[currEvt].power;
                float3 powTo = traj.events[nextEvt].power;
                state.power = powFrom + (powTo - powFrom) * state.progress;
            }
        }
    }

    // Get active photon positions for rendering
    std::vector<Photon> getActivePhotons() const
    {
        std::vector<Photon> result;
        for (size_t i = 0; i < m_photonStates.size(); ++i)
        {
            const auto& state = m_photonStates[i];
            if (state.active && !state.terminated)
            {
                Photon p;
                p.position = state.position;
                p.power = state.power;
                p.incidentDir = make_float3(0, -1, 0);
                p.flag = 0;
                result.push_back(p);
            }
        }
        return result;
    }

    // Get all photon positions (including terminated ones at their final position)
    std::vector<Photon> getAllPhotonPositions() const
    {
        std::vector<Photon> result;
        for (size_t i = 0; i < m_photonStates.size(); ++i)
        {
            const auto& state = m_photonStates[i];
            if (state.active)
            {
                Photon p;
                p.position = state.position;
                p.power = state.power;
                p.incidentDir = make_float3(0, -1, 0);
                p.flag = state.terminated ? 1 : 0;
                result.push_back(p);
            }
        }
        return result;
    }

    size_t getTrajectoryCount() const { return m_trajectories.size(); }
    size_t getActiveCount() const
    {
        size_t count = 0;
        for (const auto& s : m_photonStates)
            if (s.active && !s.terminated) count++;
        return count;
    }
    float getTime() const { return m_time; }
    bool isComplete() const
    {
        for (const auto& s : m_photonStates)
            if (!s.terminated) return false;
        return m_photonStates.size() > 0;
    }

private:
    int parseEventType(const char* name)
    {
        if (strcmp(name, "EMITTED") == 0) return EVENT_EMITTED;
        if (strcmp(name, "DIFFUSE_BOUNCE") == 0) return EVENT_DIFFUSE_BOUNCE;
        if (strcmp(name, "DIFFUSE_STORED") == 0) return EVENT_DIFFUSE_STORED;
        if (strcmp(name, "DIFFUSE_ABSORBED") == 0) return EVENT_DIFFUSE_ABSORBED;
        if (strcmp(name, "SPECULAR_REFLECT") == 0) return EVENT_SPECULAR_REFLECT;
        if (strcmp(name, "GLASS_REFLECT") == 0) return EVENT_GLASS_REFLECT;
        if (strcmp(name, "GLASS_REFRACT") == 0) return EVENT_GLASS_REFRACT;
        if (strcmp(name, "MISS") == 0) return EVENT_MISS;
        if (strcmp(name, "MAX_DEPTH") == 0) return EVENT_MAX_DEPTH;
        if (strcmp(name, "HIT_LIGHT") == 0) return EVENT_HIT_LIGHT;
        return EVENT_NONE;
    }

    std::vector<PhotonTrajectory> m_trajectories;
    std::vector<AnimatedPhotonState> m_photonStates;
    float m_time = 0.0f;
    size_t m_nextPhotonToActivate = 0;
};

