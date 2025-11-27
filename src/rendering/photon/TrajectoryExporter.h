#pragma once

#include "PhotonTrajectory.h"
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>

class TrajectoryExporter
{
public:
    // Export trajectories to a text file
    static bool exportToFile(const std::vector<PhotonTrajectory> &trajectories,
                             const std::string &filename)
    {
        std::ofstream file(filename);
        if (!file.is_open())
            return false;

        file << std::fixed << std::setprecision(6);
        file << "# Photon Trajectory Export\n";
        file << "# Total photons: " << trajectories.size() << "\n";
        file << "# Format: [event_idx] EVENT_TYPE pos=(x,y,z) dir=(x,y,z) power=(r,g,b) mat=MATERIAL\n";
        file << "#\n\n";

        unsigned int totalEvents = 0;
        for (const auto &traj : trajectories)
        {
            if (traj.event_count == 0)
                continue;

            file << "PHOTON " << traj.photon_id << " (" << traj.event_count << " events)\n";

            for (unsigned int i = 0; i < traj.event_count; ++i)
            {
                const PhotonEvent &evt = traj.events[i];
                file << "  [" << i << "] " << getEventTypeName(evt.event_type);
                file << " pos=(" << evt.position.x << "," << evt.position.y << "," << evt.position.z << ")";
                file << " dir=(" << evt.direction.x << "," << evt.direction.y << "," << evt.direction.z << ")";
                file << " power=(" << evt.power.x << "," << evt.power.y << "," << evt.power.z << ")";
                
                if (evt.material_type != TRAJ_MAT_NONE)
                    file << " mat=" << getMaterialTypeName(evt.material_type);
                
                file << "\n";
                totalEvents++;
            }
            file << "\n";
        }

        file << "# End of export. Total events: " << totalEvents << "\n";
        file.close();
        return true;
    }

    // Export to CSV format for analysis in spreadsheet software
    static bool exportToCSV(const std::vector<PhotonTrajectory> &trajectories,
                            const std::string &filename)
    {
        std::ofstream file(filename);
        if (!file.is_open())
            return false;

        file << std::fixed << std::setprecision(6);
        
        // Header
        file << "photon_id,event_idx,event_type,pos_x,pos_y,pos_z,dir_x,dir_y,dir_z,power_r,power_g,power_b,material\n";

        for (const auto &traj : trajectories)
        {
            for (unsigned int i = 0; i < traj.event_count; ++i)
            {
                const PhotonEvent &evt = traj.events[i];
                file << traj.photon_id << ","
                     << i << ","
                     << getEventTypeName(evt.event_type) << ","
                     << evt.position.x << "," << evt.position.y << "," << evt.position.z << ","
                     << evt.direction.x << "," << evt.direction.y << "," << evt.direction.z << ","
                     << evt.power.x << "," << evt.power.y << "," << evt.power.z << ","
                     << getMaterialTypeName(evt.material_type) << "\n";
            }
        }

        file.close();
        return true;
    }

    // Get summary statistics
    static std::string getSummary(const std::vector<PhotonTrajectory> &trajectories)
    {
        std::stringstream ss;

        unsigned int totalPhotons = 0;
        unsigned int totalEvents = 0;
        unsigned int eventCounts[12] = {0};  // Count by event type
        unsigned int maxBounces = 0;

        for (const auto &traj : trajectories)
        {
            if (traj.event_count > 0)
            {
                totalPhotons++;
                totalEvents += traj.event_count;
                if (traj.event_count > maxBounces)
                    maxBounces = traj.event_count;

                for (unsigned int i = 0; i < traj.event_count; ++i)
                {
                    int type = traj.events[i].event_type;
                    if (type >= 0 && type < 12)
                        eventCounts[type]++;
                }
            }
        }

        ss << "=== Trajectory Summary ===\n";
        ss << "Total photons with events: " << totalPhotons << "\n";
        ss << "Total events: " << totalEvents << "\n";
        ss << "Max events per photon: " << maxBounces << "\n";
        ss << "Average events per photon: " << (totalPhotons > 0 ? (float)totalEvents / totalPhotons : 0) << "\n";
        ss << "\nEvent breakdown:\n";
        
        const char* eventNames[] = {
            "NONE", "EMITTED", "DIFFUSE_BOUNCE", "DIFFUSE_STORED", 
            "DIFFUSE_ABSORBED", "SPECULAR_REFLECT", "GLASS_REFLECT", 
            "GLASS_REFRACT", "MISS", "MAX_DEPTH", "HIT_LIGHT"
        };
        
        for (int i = 1; i <= 10; ++i)
        {
            if (eventCounts[i] > 0)
                ss << "  " << eventNames[i] << ": " << eventCounts[i] << "\n";
        }

        return ss.str();
    }
};

