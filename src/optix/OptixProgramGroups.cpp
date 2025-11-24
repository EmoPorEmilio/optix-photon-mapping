#include "OptixProgramGroups.h"
#include <sutil/Exception.h>

void createDefaultProgramGroups(
    OptixDeviceContext context,
    OptixModule module,
    OptixProgramGroup &raygen,
    OptixProgramGroup &miss,
    OptixProgramGroup &triangleHit,
    OptixProgramGroup &sphereHit)
{
    char log[2048];
    size_t logSize = sizeof(log);
    OptixProgramGroupOptions pg_options = {};
    OptixProgramGroupDesc pg_desc = {};

    
    pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pg_desc.raygen.module = module;
    pg_desc.raygen.entryFunctionName = "__raygen__rg";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &pg_desc, 1, &pg_options, log, &logSize, &raygen), log, logSize);

    
    pg_desc = {};
    pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pg_desc.miss.module = module;
    pg_desc.miss.entryFunctionName = "__miss__ms";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &pg_desc, 1, &pg_options, log, &logSize, &miss), log, logSize);

    
    pg_desc = {};
    pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pg_desc.hitgroup.moduleCH = module;
    pg_desc.hitgroup.entryFunctionNameCH = "__closesthit__triangle_ch";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &pg_desc, 1, &pg_options, log, &logSize, &triangleHit), log, logSize);

    
    pg_desc = {};
    pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pg_desc.hitgroup.moduleCH = module;
    pg_desc.hitgroup.entryFunctionNameCH = "__closesthit__sphere_ch";
    pg_desc.hitgroup.moduleIS = module;
    pg_desc.hitgroup.entryFunctionNameIS = "__intersection__sphere";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &pg_desc, 1, &pg_options, log, &logSize, &sphereHit), log, logSize);
}

