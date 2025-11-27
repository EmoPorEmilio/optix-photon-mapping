#pragma once

#include "Photon.h"
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>

//=============================================================================
// PhotonMapIO
// Import/Export photon maps to/from parseable text files
// Allows saving traced photons and reloading without re-tracing
//=============================================================================
class PhotonMapIO
{
public:
    static constexpr int VERSION = 1;

    //-------------------------------------------------------------------------
    // Export photon maps to file
    //-------------------------------------------------------------------------
    static bool exportToFile(const std::vector<Photon>& globalPhotons,
                             const std::vector<Photon>& causticPhotons,
                             const std::string& filename)
    {
        std::ofstream file(filename);
        if (!file.is_open())
        {
            std::cerr << "PhotonMapIO: Failed to open " << filename << " for writing" << std::endl;
            return false;
        }

        file << std::fixed << std::setprecision(6);

        // Header
        file << "# Photon Map Export\n";
        file << "VERSION " << VERSION << "\n";
        file << "GLOBAL_COUNT " << globalPhotons.size() << "\n";
        file << "CAUSTIC_COUNT " << causticPhotons.size() << "\n";
        file << "\n";

        // Global photons
        file << "GLOBAL_PHOTONS\n";
        file << "# pos_x,pos_y,pos_z,power_r,power_g,power_b,dir_x,dir_y,dir_z\n";
        for (const auto& p : globalPhotons)
        {
            file << p.position.x << "," << p.position.y << "," << p.position.z << ","
                 << p.power.x << "," << p.power.y << "," << p.power.z << ","
                 << p.incidentDir.x << "," << p.incidentDir.y << "," << p.incidentDir.z << "\n";
        }
        file << "\n";

        // Caustic photons
        file << "CAUSTIC_PHOTONS\n";
        file << "# pos_x,pos_y,pos_z,power_r,power_g,power_b,dir_x,dir_y,dir_z\n";
        for (const auto& p : causticPhotons)
        {
            file << p.position.x << "," << p.position.y << "," << p.position.z << ","
                 << p.power.x << "," << p.power.y << "," << p.power.z << ","
                 << p.incidentDir.x << "," << p.incidentDir.y << "," << p.incidentDir.z << "\n";
        }

        file << "\n# End of photon map\n";
        file.close();

        std::cout << "PhotonMapIO: Exported " << globalPhotons.size() << " global + "
                  << causticPhotons.size() << " caustic photons to " << filename << std::endl;
        return true;
    }

    //-------------------------------------------------------------------------
    // Import photon maps from file
    //-------------------------------------------------------------------------
    static bool importFromFile(const std::string& filename,
                               std::vector<Photon>& globalPhotons,
                               std::vector<Photon>& causticPhotons)
    {
        std::ifstream file(filename);
        if (!file.is_open())
        {
            std::cerr << "PhotonMapIO: Failed to open " << filename << " for reading" << std::endl;
            return false;
        }

        globalPhotons.clear();
        causticPhotons.clear();

        std::string line;
        int version = 0;
        size_t expectedGlobal = 0;
        size_t expectedCaustic = 0;
        
        enum Section { NONE, GLOBAL, CAUSTIC };
        Section currentSection = NONE;

        while (std::getline(file, line))
        {
            // Skip empty lines and comments
            if (line.empty() || line[0] == '#')
                continue;

            // Parse header
            if (line.rfind("VERSION ", 0) == 0)
            {
                version = std::stoi(line.substr(8));
                if (version != VERSION)
                {
                    std::cerr << "PhotonMapIO: Version mismatch (file: " << version 
                              << ", expected: " << VERSION << ")" << std::endl;
                    return false;
                }
                continue;
            }

            if (line.rfind("GLOBAL_COUNT ", 0) == 0)
            {
                expectedGlobal = std::stoull(line.substr(13));
                globalPhotons.reserve(expectedGlobal);
                continue;
            }

            if (line.rfind("CAUSTIC_COUNT ", 0) == 0)
            {
                expectedCaustic = std::stoull(line.substr(14));
                causticPhotons.reserve(expectedCaustic);
                continue;
            }

            if (line == "GLOBAL_PHOTONS")
            {
                currentSection = GLOBAL;
                continue;
            }

            if (line == "CAUSTIC_PHOTONS")
            {
                currentSection = CAUSTIC;
                continue;
            }

            // Parse photon data
            if (currentSection != NONE)
            {
                Photon p;
                if (parsePhotonLine(line, p))
                {
                    p.flag = (currentSection == CAUSTIC) ? 1 : 0;
                    if (currentSection == GLOBAL)
                        globalPhotons.push_back(p);
                    else
                        causticPhotons.push_back(p);
                }
            }
        }

        file.close();

        std::cout << "PhotonMapIO: Imported " << globalPhotons.size() << " global + "
                  << causticPhotons.size() << " caustic photons from " << filename << std::endl;

        if (globalPhotons.size() != expectedGlobal || causticPhotons.size() != expectedCaustic)
        {
            std::cerr << "PhotonMapIO: Warning - count mismatch (expected " 
                      << expectedGlobal << "+" << expectedCaustic << ")" << std::endl;
        }

        return true;
    }

private:
    //-------------------------------------------------------------------------
    // Parse a single photon line (CSV format)
    //-------------------------------------------------------------------------
    static bool parsePhotonLine(const std::string& line, Photon& p)
    {
        std::stringstream ss(line);
        std::string token;
        float values[9];
        int idx = 0;

        while (std::getline(ss, token, ',') && idx < 9)
        {
            try
            {
                values[idx++] = std::stof(token);
            }
            catch (...)
            {
                return false;
            }
        }

        if (idx != 9)
            return false;

        p.position = make_float3(values[0], values[1], values[2]);
        p.power = make_float3(values[3], values[4], values[5]);
        p.incidentDir = make_float3(values[6], values[7], values[8]);
        p.flag = 0;

        return true;
    }
};

