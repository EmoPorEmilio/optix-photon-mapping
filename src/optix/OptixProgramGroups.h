

#pragma once

#include <optix.h>

void createDefaultProgramGroups(
    OptixDeviceContext context,
    OptixModule module,
    OptixProgramGroup &raygen,
    OptixProgramGroup &miss,
    OptixProgramGroup &triangleHit,
    OptixProgramGroup &sphereHit);





