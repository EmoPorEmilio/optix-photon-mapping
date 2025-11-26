#include "OptixManager.h"
#include <optix_function_table_definition.h>
#include <sutil/Exception.h>
#include "../scene/Material.h"
#include "../cuda/photon_emission/photon_launch_params.h"

bool OptixManager::initialize()
{
    if (initialized)
        return true;
    contextOwner.create(&contextLogCallback, 4);
    context = contextOwner.get();
    stream = contextOwner.stream();
    initialized = true;
    std::cout << "OptixManager initialized successfully" << std::endl;
    return true;
}

bool OptixManager::loadModule()
{
    OptixModuleCompileOptions module_options = {};
    pipeline_compile_options.usesMotionBlur = false;
    pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    pipeline_compile_options.numPayloadValues = 3;
    pipeline_compile_options.numAttributeValues = 3;
    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
    pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE | OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;
    module = createOptixModule(context, module_options, &pipeline_compile_options, "ptx/raytrace.optixir");
    return true;
}

bool OptixManager::loadPhotonModule()
{
    OptixModuleCompileOptions module_options = {};
    OptixPipelineCompileOptions photon_pipeline_options = {};
    photon_pipeline_options.usesMotionBlur = false;
    photon_pipeline_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    // Payload layout (12 values, 32-bit each):
    // 0-2: throughput (float3)
    // 3-5: origin (float3)
    // 6-8: direction (float3)
    // 9:   depth (uint)
    // 10:  photon index (uint)
    // 11:  continue flag (uint, 0/1)
    photon_pipeline_options.numPayloadValues = 12;
    photon_pipeline_options.numAttributeValues = 3;
    photon_pipeline_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    photon_pipeline_options.pipelineLaunchParamsVariableName = "params";
    photon_pipeline_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE | OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;

    photon_module = createOptixModule(context, module_options, &photon_pipeline_options, "ptx/photon_emission.optixir");
    return true;
}

bool OptixManager::createProgramGroups()
{
    createDefaultProgramGroups(context, module, raygen_group, miss_group, triangle_hit_group, sphere_hit_group);
    return true;
}

bool OptixManager::createPhotonProgramGroups()
{
    char log[2048];
    size_t logSize = sizeof(log);
    OptixProgramGroupOptions pg_options = {};
    OptixProgramGroupDesc pg_desc = {};

    pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pg_desc.raygen.module = photon_module;
    pg_desc.raygen.entryFunctionName = "__raygen__photon_emitter";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &pg_desc, 1, &pg_options, log, &logSize, &photon_raygen_group), log, logSize);

    pg_desc = {};
    pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pg_desc.miss.module = photon_module;
    pg_desc.miss.entryFunctionName = "__miss__photon_miss";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &pg_desc, 1, &pg_options, log, &logSize, &photon_miss_group), log, logSize);

    pg_desc = {};
    pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pg_desc.hitgroup.moduleCH = photon_module;
    pg_desc.hitgroup.entryFunctionNameCH = "__closesthit__photon_hit";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &pg_desc, 1, &pg_options, log, &logSize, &photon_triangle_hit_group), log, logSize);

    pg_desc = {};
    pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pg_desc.hitgroup.moduleCH = photon_module;
    pg_desc.hitgroup.entryFunctionNameCH = "__closesthit__photon_sphere_hit";
    pg_desc.hitgroup.moduleIS = photon_module;
    pg_desc.hitgroup.entryFunctionNameIS = "__intersection__photon_sphere";

    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &pg_desc, 1, &pg_options, log, &logSize, &photon_sphere_hit_group), log, logSize);

    return true;
}

bool OptixManager::linkPipeline()
{
    OptixProgramGroup groups[] = {raygen_group, miss_group, triangle_hit_group, sphere_hit_group};
    OptixPipelineBuilder::createAndLink(context, &pipeline_compile_options, groups, 4, pipeline);
    return true;
}

bool OptixManager::linkPhotonPipeline()
{
    OptixPipelineCompileOptions photon_pipeline_options = {};
    photon_pipeline_options.usesMotionBlur = false;
    photon_pipeline_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    photon_pipeline_options.numPayloadValues = 12;
    photon_pipeline_options.numAttributeValues = 3;
    photon_pipeline_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    photon_pipeline_options.pipelineLaunchParamsVariableName = "params";
    photon_pipeline_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE | OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;

    OptixProgramGroup photon_groups[] = {photon_raygen_group, photon_miss_group, photon_triangle_hit_group, photon_sphere_hit_group};
    OptixPipelineBuilder::createAndLink(context, &photon_pipeline_options, photon_groups, 4, photon_pipeline);
    return true;
}

void OptixManager::buildSBT()
{
    OptixSBTBuilder::build(raygen_group, miss_group, triangle_hit_group, sphere_hit_group, sbt);
}

void OptixManager::buildPhotonSBT()
{

    SbtRecord<RaygenData> rg = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(photon_raygen_group, &rg));
    CUdeviceptr d_rg;
    CUDA_CHECK(cudaMalloc((void **)&d_rg, sizeof(rg)));
    CUDA_CHECK(cudaMemcpy((void *)d_rg, &rg, sizeof(rg), cudaMemcpyHostToDevice));
    photon_sbt.raygenRecord = d_rg;

    SbtRecord<MissData> ms = {};
    ms.data.bg = make_float4(0.f, 0.f, 0.f, 1.f);
    OPTIX_CHECK(optixSbtRecordPackHeader(photon_miss_group, &ms));
    CUdeviceptr d_ms;
    CUDA_CHECK(cudaMalloc((void **)&d_ms, sizeof(ms)));
    CUDA_CHECK(cudaMemcpy((void *)d_ms, &ms, sizeof(ms), cudaMemcpyHostToDevice));
    photon_sbt.missRecordBase = d_ms;
    photon_sbt.missRecordStrideInBytes = sizeof(SbtRecord<MissData>);
    photon_sbt.missRecordCount = 1;

    SbtRecord<HitData> hg[2] = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(photon_triangle_hit_group, &hg[0]));
    OPTIX_CHECK(optixSbtRecordPackHeader(photon_sphere_hit_group, &hg[1]));
    CUdeviceptr d_hg;
    CUDA_CHECK(cudaMalloc((void **)&d_hg, sizeof(hg)));
    CUDA_CHECK(cudaMemcpy((void *)d_hg, hg, sizeof(hg), cudaMemcpyHostToDevice));
    photon_sbt.hitgroupRecordBase = d_hg;
    photon_sbt.hitgroupRecordStrideInBytes = sizeof(SbtRecord<HitData>);
    photon_sbt.hitgroupRecordCount = 2;
}

void OptixManager::launchPhotonPass(unsigned int num_photons, const QuadLight &light,
                                    unsigned int quadLightStartIndex,
                                    CUdeviceptr &out_photons, unsigned int &out_count,
                                    CUdeviceptr &out_caustic_photons, unsigned int &out_caustic_count)
{
    if (!photon_pipeline)
    {
        std::cerr << "Photon pipeline not created!" << std::endl;
        return;
    }

    // Allocate global photon buffer
    if (!d_photon_buffer)
    {
        CUDA_CHECK(cudaMalloc((void **)&d_photon_buffer, num_photons * sizeof(Photon)));
    }

    if (!d_photon_counter)
    {
        CUDA_CHECK(cudaMalloc((void **)&d_photon_counter, sizeof(unsigned int)));
    }

    // Allocate caustic photon buffer
    if (!d_caustic_photon_buffer)
    {
        CUDA_CHECK(cudaMalloc((void **)&d_caustic_photon_buffer, num_photons * sizeof(Photon)));
    }

    if (!d_caustic_photon_counter)
    {
        CUDA_CHECK(cudaMalloc((void **)&d_caustic_photon_counter, sizeof(unsigned int)));
    }

    // Reset counters
    unsigned int zero = 0;
    CUDA_CHECK(cudaMemcpy((void *)d_photon_counter, &zero, sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy((void *)d_caustic_photon_counter, &zero, sizeof(unsigned int), cudaMemcpyHostToDevice));

    PhotonLaunchParams params = {};

    params.handle = ias_handle;
    params.light = light;
    params.num_photons = num_photons;
    
    // Photon power: distribute light intensity across all photons
    // Using original formula for compatibility with existing brightness tuning
    params.photon_power = light.getIntensity() / static_cast<float>(num_photons);
    
    params.quadLightStartIndex = quadLightStartIndex;
    params.triangle_colors = reinterpret_cast<float3 *>(d_triangle_colors);

    // Bind triangle materials (all walls: diffuse with 50% survive prob).
    params.triangle_materials = reinterpret_cast<Material *>(d_triangle_materials);

    // Sphere 0: fully transmissive glass (no color modulation)
    params.sphere_materials[0].type = MATERIAL_TRANSMISSIVE;
    params.sphere_materials[0].albedo = make_float3(1.0f, 1.0f, 1.0f);
    params.sphere_materials[0].diffuseProb = 0.0f;
    params.sphere_materials[0].transmissiveCoeff = 1.0f;

    // Sphere 1: fully specular mirror (no color modulation)
    params.sphere_materials[1].type = MATERIAL_SPECULAR;
    params.sphere_materials[1].albedo = make_float3(1.0f, 1.0f, 1.0f);
    params.sphere_materials[1].diffuseProb = 0.0f;
    params.sphere_materials[1].transmissiveCoeff = 0.0f;

    // Match the analytic sphere geometry used in the render pass.
    params.sphere1.center = make_float3(185.0f, 82.5f, 169.0f);
    params.sphere1.radius = 82.5f;
    params.sphere2.center = make_float3(368.0f, 103.5f, 351.0f);
    params.sphere2.radius = 103.5f;
    params.max_depth = 8;

    // Global photon map output
    params.photons_out = reinterpret_cast<Photon *>(d_photon_buffer);
    params.photon_counter = d_photon_counter;

    // Caustic photon map output
    params.caustic_photons_out = reinterpret_cast<Photon *>(d_caustic_photon_buffer);
    params.caustic_photon_counter = d_caustic_photon_counter;

    CUdeviceptr d_params;
    CUDA_CHECK(cudaMalloc((void **)&d_params, sizeof(PhotonLaunchParams)));
    CUDA_CHECK(cudaMemcpy((void *)d_params, &params, sizeof(PhotonLaunchParams), cudaMemcpyHostToDevice));

    // Launch one photon per x-dimension index.
    OPTIX_CHECK(optixLaunch(photon_pipeline, stream, d_params, sizeof(PhotonLaunchParams), &photon_sbt, num_photons, 1, 1));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Retrieve global photon count
    CUDA_CHECK(cudaMemcpy(&out_count, (void *)d_photon_counter, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    if (out_count > num_photons)
        out_count = num_photons;
    out_photons = d_photon_buffer;

    // Retrieve caustic photon count
    CUDA_CHECK(cudaMemcpy(&out_caustic_count, (void *)d_caustic_photon_counter, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    if (out_caustic_count > num_photons)
        out_caustic_count = num_photons;
    out_caustic_photons = d_caustic_photon_buffer;

    CUDA_CHECK(cudaFree((void *)d_params));

    std::cout << "Launched " << num_photons << " photons, stored " << out_count << " global + "
              << out_caustic_count << " caustic hits" << std::endl;
}

void OptixManager::cleanup()
{
    if (d_triangle_vertices)
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_triangle_vertices)));
    if (d_triangle_gas_buffer)
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_triangle_gas_buffer)));
    if (d_triangle_colors)
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_triangle_colors)));
    if (d_triangle_materials)
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_triangle_materials)));
    if (d_sphere_aabb_buffer)
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_sphere_aabb_buffer)));
    if (d_sphere_gas_buffer)
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_sphere_gas_buffer)));
    if (d_ias_buffer)
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_ias_buffer)));
    if (d_ias_instances)
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_ias_instances)));
    if (module)
        OPTIX_CHECK(optixModuleDestroy(module));
    if (pipeline)
        OPTIX_CHECK(optixPipelineDestroy(pipeline));

    if (d_photon_buffer)
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_photon_buffer)));
    if (d_photon_counter)
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_photon_counter)));
    if (photon_module)
        OPTIX_CHECK(optixModuleDestroy(photon_module));
    if (photon_pipeline)
        OPTIX_CHECK(optixPipelineDestroy(photon_pipeline));

    if (direct_module)
        OPTIX_CHECK(optixModuleDestroy(direct_module));
    if (direct_pipeline)
        OPTIX_CHECK(optixPipelineDestroy(direct_pipeline));

    contextOwner.destroy();
}

// ============== Direct Lighting Pipeline ==============

bool OptixManager::loadDirectModule()
{
    OptixModuleCompileOptions module_options = {};
    OptixPipelineCompileOptions direct_pipeline_options = {};
    direct_pipeline_options.usesMotionBlur = false;
    direct_pipeline_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    direct_pipeline_options.numPayloadValues = 4;   // RGBA
    direct_pipeline_options.numAttributeValues = 3; // normal
    direct_pipeline_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    direct_pipeline_options.pipelineLaunchParamsVariableName = "params";
    direct_pipeline_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE | OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;

    direct_module = createOptixModule(context, module_options, &direct_pipeline_options, "ptx/direct_lighting.optixir");
    return direct_module != nullptr;
}

bool OptixManager::createDirectProgramGroups()
{
    char log[2048];
    size_t logSize = sizeof(log);
    OptixProgramGroupOptions pg_options = {};
    OptixProgramGroupDesc pg_desc = {};

    // Raygen
    pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pg_desc.raygen.module = direct_module;
    pg_desc.raygen.entryFunctionName = "__raygen__direct";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &pg_desc, 1, &pg_options, log, &logSize, &direct_raygen_group), log, logSize);

    // Primary miss
    pg_desc = {};
    pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pg_desc.miss.module = direct_module;
    pg_desc.miss.entryFunctionName = "__miss__direct";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &pg_desc, 1, &pg_options, log, &logSize, &direct_miss_group), log, logSize);

    // Shadow miss
    pg_desc = {};
    pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pg_desc.miss.module = direct_module;
    pg_desc.miss.entryFunctionName = "__miss__shadow";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &pg_desc, 1, &pg_options, log, &logSize, &direct_shadow_miss_group), log, logSize);

    // Triangle hit (primary rays)
    pg_desc = {};
    pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pg_desc.hitgroup.moduleCH = direct_module;
    pg_desc.hitgroup.entryFunctionNameCH = "__closesthit__direct_triangle";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &pg_desc, 1, &pg_options, log, &logSize, &direct_triangle_hit_group), log, logSize);

    // Sphere hit (primary rays) with custom intersection
    pg_desc = {};
    pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pg_desc.hitgroup.moduleCH = direct_module;
    pg_desc.hitgroup.entryFunctionNameCH = "__closesthit__direct_sphere";
    pg_desc.hitgroup.moduleIS = direct_module;
    pg_desc.hitgroup.entryFunctionNameIS = "__intersection__direct_sphere";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &pg_desc, 1, &pg_options, log, &logSize, &direct_sphere_hit_group), log, logSize);

    // Shadow hit for triangles (built-in intersection)
    pg_desc = {};
    pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pg_desc.hitgroup.moduleCH = direct_module;
    pg_desc.hitgroup.entryFunctionNameCH = "__closesthit__shadow";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &pg_desc, 1, &pg_options, log, &logSize, &direct_shadow_triangle_hit_group), log, logSize);

    // Shadow hit for spheres (needs custom intersection)
    pg_desc = {};
    pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pg_desc.hitgroup.moduleCH = direct_module;
    pg_desc.hitgroup.entryFunctionNameCH = "__closesthit__shadow";
    pg_desc.hitgroup.moduleIS = direct_module;
    pg_desc.hitgroup.entryFunctionNameIS = "__intersection__direct_sphere";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &pg_desc, 1, &pg_options, log, &logSize, &direct_shadow_sphere_hit_group), log, logSize);

    return true;
}

bool OptixManager::linkDirectPipeline()
{
    OptixProgramGroup groups[] = {
        direct_raygen_group,
        direct_miss_group,
        direct_shadow_miss_group,
        direct_triangle_hit_group,
        direct_sphere_hit_group,
        direct_shadow_triangle_hit_group,
        direct_shadow_sphere_hit_group};

    OptixPipelineLinkOptions link_options = {};
    link_options.maxTraceDepth = 2; // Primary + shadow

    char log[2048];
    size_t logSize = sizeof(log);

    OptixPipelineCompileOptions direct_pipeline_options = {};
    direct_pipeline_options.usesMotionBlur = false;
    direct_pipeline_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    direct_pipeline_options.numPayloadValues = 4;
    direct_pipeline_options.numAttributeValues = 3;
    direct_pipeline_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    direct_pipeline_options.pipelineLaunchParamsVariableName = "params";
    direct_pipeline_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE | OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;

    OPTIX_CHECK_LOG(optixPipelineCreate(context, &direct_pipeline_options, &link_options, groups, 7, log, &logSize, &direct_pipeline), log, logSize);

    return true;
}

void OptixManager::buildDirectSBT()
{
    // Raygen record
    SbtRecord<RaygenData> rg = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(direct_raygen_group, &rg));
    CUdeviceptr d_rg;
    CUDA_CHECK(cudaMalloc((void **)&d_rg, sizeof(rg)));
    CUDA_CHECK(cudaMemcpy((void *)d_rg, &rg, sizeof(rg), cudaMemcpyHostToDevice));
    direct_sbt.raygenRecord = d_rg;

    // Miss records (2: primary + shadow)
    SbtRecord<MissData> ms[2] = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(direct_miss_group, &ms[0]));
    OPTIX_CHECK(optixSbtRecordPackHeader(direct_shadow_miss_group, &ms[1]));
    CUdeviceptr d_ms;
    CUDA_CHECK(cudaMalloc((void **)&d_ms, sizeof(ms)));
    CUDA_CHECK(cudaMemcpy((void *)d_ms, ms, sizeof(ms), cudaMemcpyHostToDevice));
    direct_sbt.missRecordBase = d_ms;
    direct_sbt.missRecordStrideInBytes = sizeof(SbtRecord<MissData>);
    direct_sbt.missRecordCount = 2;

    // Hit group records layout for 2 ray types x 2 geometry types:
    // Index = rayType * numGeomTypes + geomType
    // [0] = primary (0) * 2 + triangle (0) = primary triangle
    // [1] = primary (0) * 2 + sphere (1) = primary sphere
    // [2] = shadow (1) * 2 + triangle (0) = shadow triangle
    // [3] = shadow (1) * 2 + sphere (1) = shadow sphere
    SbtRecord<HitData> hg[4] = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(direct_triangle_hit_group, &hg[0]));        // Primary triangle
    OPTIX_CHECK(optixSbtRecordPackHeader(direct_sphere_hit_group, &hg[1]));          // Primary sphere
    OPTIX_CHECK(optixSbtRecordPackHeader(direct_shadow_triangle_hit_group, &hg[2])); // Shadow triangle
    OPTIX_CHECK(optixSbtRecordPackHeader(direct_shadow_sphere_hit_group, &hg[3]));   // Shadow sphere
    CUdeviceptr d_hg;
    CUDA_CHECK(cudaMalloc((void **)&d_hg, sizeof(hg)));
    CUDA_CHECK(cudaMemcpy((void *)d_hg, hg, sizeof(hg), cudaMemcpyHostToDevice));
    direct_sbt.hitgroupRecordBase = d_hg;
    direct_sbt.hitgroupRecordStrideInBytes = sizeof(SbtRecord<HitData>);
    direct_sbt.hitgroupRecordCount = 4;
}

bool OptixManager::createDirectLightingPipeline()
{
    if (!initialized)
    {
        std::cerr << "Must initialize OptixManager before creating direct lighting pipeline" << std::endl;
        return false;
    }

    if (!loadDirectModule())
    {
        std::cerr << "Failed to load direct lighting module" << std::endl;
        return false;
    }

    if (!createDirectProgramGroups())
    {
        std::cerr << "Failed to create direct lighting program groups" << std::endl;
        return false;
    }

    if (!linkDirectPipeline())
    {
        std::cerr << "Failed to link direct lighting pipeline" << std::endl;
        return false;
    }

    buildDirectSBT();

    std::cout << "Direct lighting pipeline created successfully" << std::endl;
    return true;
}

void OptixManager::launchDirectLighting(unsigned int width, unsigned int height, const Camera &camera,
                                        float ambient, float shadow_ambient, float intensity, float attenuation,
                                        float4 *d_output)
{
    if (!direct_pipeline)
    {
        std::cerr << "Direct lighting pipeline not created!" << std::endl;
        return;
    }

// Include the launch params header
#include "../cuda/direct_lighting/direct_launch_params.h"

    DirectLaunchParams params = {};
    params.frame_buffer = d_output;
    params.width = width;
    params.height = height;

    // Camera setup
    params.eye = camera.getPosition();
    params.U = camera.getU();
    params.V = camera.getV();
    params.W = camera.getW();

    // Debug camera
    static int debugCount = 0;
    if (debugCount++ % 60 == 0)
    {
        std::cout << "Direct lighting camera: eye=(" << params.eye.x << "," << params.eye.y << "," << params.eye.z << ")"
                  << " U=(" << params.U.x << "," << params.U.y << "," << params.U.z << ")"
                  << " V=(" << params.V.x << "," << params.V.y << "," << params.V.z << ")"
                  << " W=(" << params.W.x << "," << params.W.y << "," << params.W.z << ")" << std::endl;
    }

    params.handle = ias_handle;

    // Materials
    params.triangle_materials = reinterpret_cast<Material *>(d_triangle_materials);
    params.sphere_materials[0].type = MATERIAL_TRANSMISSIVE;
    params.sphere_materials[0].albedo = make_float3(0.95f, 0.95f, 1.0f);
    params.sphere_materials[1].type = MATERIAL_SPECULAR;
    params.sphere_materials[1].albedo = make_float3(0.95f, 0.95f, 0.95f);

    // Light (area light at ceiling)
    params.light.position = make_float3(278.0f, 548.8f - 1.0f, 279.6f);
    params.light.normal = make_float3(0.0f, -1.0f, 0.0f);
    params.light.u = make_float3(1.0f, 0.0f, 0.0f);
    params.light.v = make_float3(0.0f, 0.0f, 1.0f);
    params.light.halfWidth = 100.0f;
    params.light.halfHeight = 100.0f;
    params.light.intensity = make_float3(50.0f, 50.0f, 50.0f);

    // Use stored quadLightStartIndex (set from scene)
    params.quadLightStartIndex = quadLightStartIndex;

    // Sphere geometry
    params.sphere1_center = sphere1_center;
    params.sphere1_radius = sphere1_radius;
    params.sphere2_center = sphere2_center;
    params.sphere2_radius = sphere2_radius;

    // Configurable lighting parameters
    params.ambient = ambient;
    params.shadow_ambient = shadow_ambient;
    params.intensity_multiplier = intensity;
    params.attenuation_factor = attenuation;

    CUdeviceptr d_params;
    CUDA_CHECK(cudaMalloc((void **)&d_params, sizeof(DirectLaunchParams)));
    CUDA_CHECK(cudaMemcpy((void *)d_params, &params, sizeof(DirectLaunchParams), cudaMemcpyHostToDevice));

    OPTIX_CHECK(optixLaunch(direct_pipeline, stream, d_params, sizeof(DirectLaunchParams), &direct_sbt, width, height, 1));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree((void *)d_params));
}

// ============= Indirect Lighting Pipeline (Color Bleeding) =============

bool OptixManager::loadIndirectModule()
{
    OptixModuleCompileOptions module_options = {};
    OptixPipelineCompileOptions indirect_pipeline_options = {};
    indirect_pipeline_options.usesMotionBlur = false;
    indirect_pipeline_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    indirect_pipeline_options.numPayloadValues = 4; // color RGB + t_hit
    indirect_pipeline_options.numAttributeValues = 3;
    indirect_pipeline_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    indirect_pipeline_options.pipelineLaunchParamsVariableName = "params";
    indirect_pipeline_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE | OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;

    indirect_module = createOptixModule(context, module_options, &indirect_pipeline_options, "ptx/indirect_lighting.optixir");
    return indirect_module != nullptr;
}

bool OptixManager::createIndirectProgramGroups()
{
    char log[2048];
    size_t log_size;

    // Raygen
    OptixProgramGroupOptions pg_options = {};
    OptixProgramGroupDesc raygen_desc = {};
    raygen_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_desc.raygen.module = indirect_module;
    raygen_desc.raygen.entryFunctionName = "__raygen__indirect";
    log_size = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(context, &raygen_desc, 1, &pg_options, log, &log_size, &indirect_raygen_group));

    // Miss
    OptixProgramGroupDesc miss_desc = {};
    miss_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_desc.miss.module = indirect_module;
    miss_desc.miss.entryFunctionName = "__miss__indirect";
    log_size = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(context, &miss_desc, 1, &pg_options, log, &log_size, &indirect_miss_group));

    // Triangle hit
    OptixProgramGroupDesc tri_hit_desc = {};
    tri_hit_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    tri_hit_desc.hitgroup.moduleCH = indirect_module;
    tri_hit_desc.hitgroup.entryFunctionNameCH = "__closesthit__indirect_triangle";
    log_size = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(context, &tri_hit_desc, 1, &pg_options, log, &log_size, &indirect_triangle_hit_group));

    // Sphere hit (with intersection)
    OptixProgramGroupDesc sphere_hit_desc = {};
    sphere_hit_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    sphere_hit_desc.hitgroup.moduleCH = indirect_module;
    sphere_hit_desc.hitgroup.entryFunctionNameCH = "__closesthit__indirect_sphere";
    sphere_hit_desc.hitgroup.moduleIS = indirect_module;
    sphere_hit_desc.hitgroup.entryFunctionNameIS = "__intersection__indirect_sphere";
    log_size = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(context, &sphere_hit_desc, 1, &pg_options, log, &log_size, &indirect_sphere_hit_group));

    return true;
}

bool OptixManager::linkIndirectPipeline()
{
    char log[2048];
    size_t log_size = sizeof(log);

    OptixProgramGroup groups[] = {
        indirect_raygen_group,
        indirect_miss_group,
        indirect_triangle_hit_group,
        indirect_sphere_hit_group};

    OptixPipelineLinkOptions link_options = {};
    link_options.maxTraceDepth = 1;

    OptixPipelineCompileOptions pipeline_options = {};
    pipeline_options.usesMotionBlur = false;
    pipeline_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    pipeline_options.numPayloadValues = 4;
    pipeline_options.numAttributeValues = 3;
    pipeline_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_options.pipelineLaunchParamsVariableName = "params";
    pipeline_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE | OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;

    OPTIX_CHECK(optixPipelineCreate(context, &pipeline_options, &link_options, groups, 4, log, &log_size, &indirect_pipeline));
    return true;
}

void OptixManager::buildIndirectSBT()
{
    // Raygen record
    SbtRecord<RaygenData> rg = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(indirect_raygen_group, &rg));
    CUdeviceptr d_rg;
    CUDA_CHECK(cudaMalloc((void **)&d_rg, sizeof(rg)));
    CUDA_CHECK(cudaMemcpy((void *)d_rg, &rg, sizeof(rg), cudaMemcpyHostToDevice));

    // Miss record
    SbtRecord<MissData> ms = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(indirect_miss_group, &ms));
    CUdeviceptr d_ms;
    CUDA_CHECK(cudaMalloc((void **)&d_ms, sizeof(ms)));
    CUDA_CHECK(cudaMemcpy((void *)d_ms, &ms, sizeof(ms), cudaMemcpyHostToDevice));

    // Hit group records - ALL must be same size for uniform stride
    struct SphereHitData
    {
        float3 center;
        float radius;
    };

    // Use the larger size for all records
    SbtRecord<SphereHitData> tri_hit = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(indirect_triangle_hit_group, &tri_hit));
    // Triangle doesn't need the data, but we use same struct size

    SbtRecord<SphereHitData> sphere1_hit = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(indirect_sphere_hit_group, &sphere1_hit));
    sphere1_hit.data.center = sphere1_center;
    sphere1_hit.data.radius = sphere1_radius;

    SbtRecord<SphereHitData> sphere2_hit = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(indirect_sphere_hit_group, &sphere2_hit));
    sphere2_hit.data.center = sphere2_center;
    sphere2_hit.data.radius = sphere2_radius;

    // All records now same size
    const size_t record_size = sizeof(SbtRecord<SphereHitData>);
    const size_t hg_size = 3 * record_size;
    CUdeviceptr d_hg;
    CUDA_CHECK(cudaMalloc((void **)&d_hg, hg_size));

    char *hg_ptr = new char[hg_size];
    memcpy(hg_ptr, &tri_hit, record_size);
    memcpy(hg_ptr + record_size, &sphere1_hit, record_size);
    memcpy(hg_ptr + 2 * record_size, &sphere2_hit, record_size);
    CUDA_CHECK(cudaMemcpy((void *)d_hg, hg_ptr, hg_size, cudaMemcpyHostToDevice));
    delete[] hg_ptr;

    // Build SBT
    indirect_sbt.raygenRecord = d_rg;
    indirect_sbt.missRecordBase = d_ms;
    indirect_sbt.missRecordStrideInBytes = sizeof(ms);
    indirect_sbt.missRecordCount = 1;
    indirect_sbt.hitgroupRecordBase = d_hg;
    indirect_sbt.hitgroupRecordStrideInBytes = record_size;
    indirect_sbt.hitgroupRecordCount = 3; // triangle + 2 spheres
}

bool OptixManager::createIndirectLightingPipeline()
{
    if (!initialized)
    {
        std::cerr << "Must initialize OptixManager before creating indirect lighting pipeline" << std::endl;
        return false;
    }

    if (!loadIndirectModule())
    {
        std::cerr << "Failed to load indirect lighting module" << std::endl;
        return false;
    }

    if (!createIndirectProgramGroups())
    {
        std::cerr << "Failed to create indirect lighting program groups" << std::endl;
        return false;
    }

    if (!linkIndirectPipeline())
    {
        std::cerr << "Failed to link indirect lighting pipeline" << std::endl;
        return false;
    }

    buildIndirectSBT();

    std::cout << "Indirect lighting pipeline created successfully" << std::endl;
    return true;
}

void OptixManager::launchIndirectLighting(unsigned int width, unsigned int height, const Camera &camera,
                                          const Photon *d_photon_map, unsigned int photon_count,
                                          float gather_radius, float brightness_multiplier,
                                          const PhotonKDTreeDevice &kdtree,
                                          float4 *d_output)
{
    if (!indirect_pipeline)
    {
        std::cerr << "Indirect lighting pipeline not created!" << std::endl;
        return;
    }

#include "../cuda/indirect_lighting/indirect_launch_params.h"

    IndirectLaunchParams params = {};
    params.frame_buffer = d_output;
    params.width = width;
    params.height = height;

    // Camera setup
    params.eye = camera.getPosition();
    params.U = camera.getU();
    params.V = camera.getV();
    params.W = camera.getW();

    params.handle = ias_handle;

    // Materials
    params.triangle_materials = reinterpret_cast<Material *>(d_triangle_materials);
    params.sphere_materials[0].type = MATERIAL_TRANSMISSIVE;
    params.sphere_materials[0].albedo = make_float3(0.95f, 0.95f, 1.0f);
    params.sphere_materials[1].type = MATERIAL_SPECULAR;
    params.sphere_materials[1].albedo = make_float3(0.95f, 0.95f, 0.95f);

    // Photon map for gathering
    params.photon_map = const_cast<Photon *>(d_photon_map);
    params.photon_count = photon_count;
    params.gather_radius = gather_radius;
    params.brightness_multiplier = brightness_multiplier;
    params.kdtree = kdtree;

    // Initialize kd-tree as invalid (use linear fallback)
    params.kdtree.nodes = nullptr;
    params.kdtree.num_nodes = 0;
    params.kdtree.max_depth = 0;
    params.kdtree.valid = false;

    params.quadLightStartIndex = quadLightStartIndex;

    CUdeviceptr d_params;
    CUDA_CHECK(cudaMalloc((void **)&d_params, sizeof(IndirectLaunchParams)));
    CUDA_CHECK(cudaMemcpy((void *)d_params, &params, sizeof(IndirectLaunchParams), cudaMemcpyHostToDevice));

    OPTIX_CHECK(optixLaunch(indirect_pipeline, stream, d_params, sizeof(IndirectLaunchParams), &indirect_sbt, width, height, 1));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree((void *)d_params));
}

// ============= Caustic Lighting Pipeline (Glossy/Specular Spheres) =============

bool OptixManager::loadCausticModule()
{
    OptixModuleCompileOptions module_options = {};
    OptixPipelineCompileOptions caustic_pipeline_options = {};
    caustic_pipeline_options.usesMotionBlur = false;
    caustic_pipeline_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    caustic_pipeline_options.numPayloadValues = 4;
    caustic_pipeline_options.numAttributeValues = 3;
    caustic_pipeline_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    caustic_pipeline_options.pipelineLaunchParamsVariableName = "params";
    caustic_pipeline_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE | OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;

    caustic_module = createOptixModule(context, module_options, &caustic_pipeline_options, "ptx/caustic_lighting.optixir");
    return caustic_module != nullptr;
}

bool OptixManager::createCausticProgramGroups()
{
    char log[2048];
    size_t log_size;

    OptixProgramGroupOptions pg_options = {};

    // Raygen
    OptixProgramGroupDesc raygen_desc = {};
    raygen_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_desc.raygen.module = caustic_module;
    raygen_desc.raygen.entryFunctionName = "__raygen__caustic";
    log_size = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(context, &raygen_desc, 1, &pg_options, log, &log_size, &caustic_raygen_group));

    // Miss
    OptixProgramGroupDesc miss_desc = {};
    miss_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_desc.miss.module = caustic_module;
    miss_desc.miss.entryFunctionName = "__miss__caustic";
    log_size = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(context, &miss_desc, 1, &pg_options, log, &log_size, &caustic_miss_group));

    // Triangle hit
    OptixProgramGroupDesc tri_hit_desc = {};
    tri_hit_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    tri_hit_desc.hitgroup.moduleCH = caustic_module;
    tri_hit_desc.hitgroup.entryFunctionNameCH = "__closesthit__caustic_triangle";
    log_size = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(context, &tri_hit_desc, 1, &pg_options, log, &log_size, &caustic_triangle_hit_group));

    // Sphere hit
    OptixProgramGroupDesc sphere_hit_desc = {};
    sphere_hit_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    sphere_hit_desc.hitgroup.moduleCH = caustic_module;
    sphere_hit_desc.hitgroup.entryFunctionNameCH = "__closesthit__caustic_sphere";
    sphere_hit_desc.hitgroup.moduleIS = caustic_module;
    sphere_hit_desc.hitgroup.entryFunctionNameIS = "__intersection__caustic_sphere";
    log_size = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(context, &sphere_hit_desc, 1, &pg_options, log, &log_size, &caustic_sphere_hit_group));

    return true;
}

bool OptixManager::linkCausticPipeline()
{
    char log[2048];
    size_t log_size = sizeof(log);

    OptixProgramGroup groups[] = {
        caustic_raygen_group,
        caustic_miss_group,
        caustic_triangle_hit_group,
        caustic_sphere_hit_group};

    OptixPipelineLinkOptions link_options = {};
    link_options.maxTraceDepth = 1;

    OptixPipelineCompileOptions pipeline_options = {};
    pipeline_options.usesMotionBlur = false;
    pipeline_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    pipeline_options.numPayloadValues = 4;
    pipeline_options.numAttributeValues = 3;
    pipeline_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_options.pipelineLaunchParamsVariableName = "params";
    pipeline_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE | OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;

    OPTIX_CHECK(optixPipelineCreate(context, &pipeline_options, &link_options, groups, 4, log, &log_size, &caustic_pipeline));
    return true;
}

void OptixManager::buildCausticSBT()
{
    // Raygen record
    SbtRecord<RaygenData> rg = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(caustic_raygen_group, &rg));
    CUdeviceptr d_rg;
    CUDA_CHECK(cudaMalloc((void **)&d_rg, sizeof(rg)));
    CUDA_CHECK(cudaMemcpy((void *)d_rg, &rg, sizeof(rg), cudaMemcpyHostToDevice));

    // Miss record
    SbtRecord<MissData> ms = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(caustic_miss_group, &ms));
    CUdeviceptr d_ms;
    CUDA_CHECK(cudaMalloc((void **)&d_ms, sizeof(ms)));
    CUDA_CHECK(cudaMemcpy((void *)d_ms, &ms, sizeof(ms), cudaMemcpyHostToDevice));

    // Hit group records - uniform size
    struct SphereHitData
    {
        float3 center;
        float radius;
    };
    const size_t record_size = sizeof(SbtRecord<SphereHitData>);

    SbtRecord<SphereHitData> tri_hit = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(caustic_triangle_hit_group, &tri_hit));

    SbtRecord<SphereHitData> sphere1_hit = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(caustic_sphere_hit_group, &sphere1_hit));
    sphere1_hit.data.center = sphere1_center;
    sphere1_hit.data.radius = sphere1_radius;

    SbtRecord<SphereHitData> sphere2_hit = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(caustic_sphere_hit_group, &sphere2_hit));
    sphere2_hit.data.center = sphere2_center;
    sphere2_hit.data.radius = sphere2_radius;

    const size_t hg_size = 3 * record_size;
    CUdeviceptr d_hg;
    CUDA_CHECK(cudaMalloc((void **)&d_hg, hg_size));

    char *hg_ptr = new char[hg_size];
    memcpy(hg_ptr, &tri_hit, record_size);
    memcpy(hg_ptr + record_size, &sphere1_hit, record_size);
    memcpy(hg_ptr + 2 * record_size, &sphere2_hit, record_size);
    CUDA_CHECK(cudaMemcpy((void *)d_hg, hg_ptr, hg_size, cudaMemcpyHostToDevice));
    delete[] hg_ptr;

    caustic_sbt.raygenRecord = d_rg;
    caustic_sbt.missRecordBase = d_ms;
    caustic_sbt.missRecordStrideInBytes = sizeof(ms);
    caustic_sbt.missRecordCount = 1;
    caustic_sbt.hitgroupRecordBase = d_hg;
    caustic_sbt.hitgroupRecordStrideInBytes = record_size;
    caustic_sbt.hitgroupRecordCount = 3;
}

bool OptixManager::createCausticLightingPipeline()
{
    if (!initialized)
    {
        std::cerr << "Must initialize OptixManager before creating caustic lighting pipeline" << std::endl;
        return false;
    }

    if (!loadCausticModule())
    {
        std::cerr << "Failed to load caustic lighting module" << std::endl;
        return false;
    }

    if (!createCausticProgramGroups())
    {
        std::cerr << "Failed to create caustic lighting program groups" << std::endl;
        return false;
    }

    if (!linkCausticPipeline())
    {
        std::cerr << "Failed to link caustic lighting pipeline" << std::endl;
        return false;
    }

    buildCausticSBT();

    std::cout << "Caustic lighting pipeline created successfully" << std::endl;
    return true;
}

void OptixManager::launchCausticLighting(unsigned int width, unsigned int height, const Camera &camera,
                                         const Photon *d_caustic_map, unsigned int caustic_count,
                                         float gather_radius, float brightness_multiplier,
                                         const PhotonKDTreeDevice &kdtree,
                                         float4 *d_output)
{
    if (!caustic_pipeline)
    {
        std::cerr << "Caustic lighting pipeline not created!" << std::endl;
        return;
    }

#include "../cuda/caustic_lighting/caustic_launch_params.h"

    CausticLaunchParams params = {};
    params.frame_buffer = d_output;
    params.width = width;
    params.height = height;

    params.eye = camera.getPosition();
    params.U = camera.getU();
    params.V = camera.getV();
    params.W = camera.getW();

    params.handle = ias_handle;

    params.sphere_materials[0].type = MATERIAL_TRANSMISSIVE;
    params.sphere_materials[0].albedo = make_float3(0.95f, 0.95f, 1.0f);
    params.sphere_materials[1].type = MATERIAL_SPECULAR;
    params.sphere_materials[1].albedo = make_float3(0.95f, 0.95f, 0.95f);

    params.caustic_photon_map = const_cast<Photon *>(d_caustic_map);
    params.caustic_photon_count = caustic_count;
    params.gather_radius = gather_radius;
    params.brightness_multiplier = brightness_multiplier;
    params.caustic_kdtree = kdtree;

    // Initialize kd-tree as invalid (use linear fallback)
    params.caustic_kdtree.nodes = nullptr;
    params.caustic_kdtree.num_nodes = 0;
    params.caustic_kdtree.max_depth = 0;
    params.caustic_kdtree.valid = false;

    params.quadLightStartIndex = quadLightStartIndex;

    CUdeviceptr d_params;
    CUDA_CHECK(cudaMalloc((void **)&d_params, sizeof(CausticLaunchParams)));
    CUDA_CHECK(cudaMemcpy((void *)d_params, &params, sizeof(CausticLaunchParams), cudaMemcpyHostToDevice));

    OPTIX_CHECK(optixLaunch(caustic_pipeline, stream, d_params, sizeof(CausticLaunchParams), &caustic_sbt, width, height, 1));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree((void *)d_params));
}

// ============= Specular Lighting Pipeline (Reflection/Refraction) =============

bool OptixManager::loadSpecularModule()
{
    OptixModuleCompileOptions module_options = {};
    OptixPipelineCompileOptions specular_pipeline_options = {};
    specular_pipeline_options.usesMotionBlur = false;
    specular_pipeline_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    specular_pipeline_options.numPayloadValues = 4;
    specular_pipeline_options.numAttributeValues = 3;
    specular_pipeline_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    specular_pipeline_options.pipelineLaunchParamsVariableName = "params";
    specular_pipeline_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE | OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;

    specular_module = createOptixModule(context, module_options, &specular_pipeline_options, "ptx/specular_lighting.optixir");
    return specular_module != nullptr;
}

bool OptixManager::createSpecularProgramGroups()
{
    char log[2048];
    size_t log_size;

    OptixProgramGroupOptions pg_options = {};

    // Raygen
    OptixProgramGroupDesc raygen_desc = {};
    raygen_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_desc.raygen.module = specular_module;
    raygen_desc.raygen.entryFunctionName = "__raygen__specular";
    log_size = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(context, &raygen_desc, 1, &pg_options, log, &log_size, &specular_raygen_group));

    // Miss
    OptixProgramGroupDesc miss_desc = {};
    miss_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_desc.miss.module = specular_module;
    miss_desc.miss.entryFunctionName = "__miss__specular";
    log_size = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(context, &miss_desc, 1, &pg_options, log, &log_size, &specular_miss_group));

    // Triangle hit
    OptixProgramGroupDesc tri_hit_desc = {};
    tri_hit_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    tri_hit_desc.hitgroup.moduleCH = specular_module;
    tri_hit_desc.hitgroup.entryFunctionNameCH = "__closesthit__specular_triangle";
    log_size = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(context, &tri_hit_desc, 1, &pg_options, log, &log_size, &specular_triangle_hit_group));

    // Sphere hit
    OptixProgramGroupDesc sphere_hit_desc = {};
    sphere_hit_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    sphere_hit_desc.hitgroup.moduleCH = specular_module;
    sphere_hit_desc.hitgroup.entryFunctionNameCH = "__closesthit__specular_sphere";
    sphere_hit_desc.hitgroup.moduleIS = specular_module;
    sphere_hit_desc.hitgroup.entryFunctionNameIS = "__intersection__specular_sphere";
    log_size = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(context, &sphere_hit_desc, 1, &pg_options, log, &log_size, &specular_sphere_hit_group));

    return true;
}

bool OptixManager::linkSpecularPipeline()
{
    char log[2048];
    size_t log_size = sizeof(log);

    OptixProgramGroup groups[] = {
        specular_raygen_group,
        specular_miss_group,
        specular_triangle_hit_group,
        specular_sphere_hit_group};

    OptixPipelineLinkOptions link_options = {};
    link_options.maxTraceDepth = 12; // Support deep reflections/refractions

    OptixPipelineCompileOptions pipeline_options = {};
    pipeline_options.usesMotionBlur = false;
    pipeline_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    pipeline_options.numPayloadValues = 4;
    pipeline_options.numAttributeValues = 3;
    pipeline_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_options.pipelineLaunchParamsVariableName = "params";
    pipeline_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE | OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;

    OPTIX_CHECK(optixPipelineCreate(context, &pipeline_options, &link_options, groups, 4, log, &log_size, &specular_pipeline));
    return true;
}

void OptixManager::buildSpecularSBT()
{
    // Raygen record
    SbtRecord<RaygenData> rg = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(specular_raygen_group, &rg));
    CUdeviceptr d_rg;
    CUDA_CHECK(cudaMalloc((void **)&d_rg, sizeof(rg)));
    CUDA_CHECK(cudaMemcpy((void *)d_rg, &rg, sizeof(rg), cudaMemcpyHostToDevice));

    // Miss record
    SbtRecord<MissData> ms = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(specular_miss_group, &ms));
    CUdeviceptr d_ms;
    CUDA_CHECK(cudaMalloc((void **)&d_ms, sizeof(ms)));
    CUDA_CHECK(cudaMemcpy((void *)d_ms, &ms, sizeof(ms), cudaMemcpyHostToDevice));

    // Hit group records - uniform size
    struct SphereHitData
    {
        float3 center;
        float radius;
    };
    const size_t record_size = sizeof(SbtRecord<SphereHitData>);

    SbtRecord<SphereHitData> tri_hit = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(specular_triangle_hit_group, &tri_hit));

    SbtRecord<SphereHitData> sphere1_hit = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(specular_sphere_hit_group, &sphere1_hit));
    sphere1_hit.data.center = sphere1_center;
    sphere1_hit.data.radius = sphere1_radius;

    SbtRecord<SphereHitData> sphere2_hit = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(specular_sphere_hit_group, &sphere2_hit));
    sphere2_hit.data.center = sphere2_center;
    sphere2_hit.data.radius = sphere2_radius;

    const size_t hg_size = 3 * record_size;
    CUdeviceptr d_hg;
    CUDA_CHECK(cudaMalloc((void **)&d_hg, hg_size));

    char *hg_ptr = new char[hg_size];
    memcpy(hg_ptr, &tri_hit, record_size);
    memcpy(hg_ptr + record_size, &sphere1_hit, record_size);
    memcpy(hg_ptr + 2 * record_size, &sphere2_hit, record_size);
    CUDA_CHECK(cudaMemcpy((void *)d_hg, hg_ptr, hg_size, cudaMemcpyHostToDevice));
    delete[] hg_ptr;

    specular_sbt.raygenRecord = d_rg;
    specular_sbt.missRecordBase = d_ms;
    specular_sbt.missRecordStrideInBytes = sizeof(ms);
    specular_sbt.missRecordCount = 1;
    specular_sbt.hitgroupRecordBase = d_hg;
    specular_sbt.hitgroupRecordStrideInBytes = record_size;
    specular_sbt.hitgroupRecordCount = 3;
}

bool OptixManager::createSpecularLightingPipeline()
{
    if (!initialized)
    {
        std::cerr << "Must initialize OptixManager before creating specular lighting pipeline" << std::endl;
        return false;
    }

    if (!loadSpecularModule())
    {
        std::cerr << "Failed to load specular lighting module" << std::endl;
        return false;
    }

    if (!createSpecularProgramGroups())
    {
        std::cerr << "Failed to create specular lighting program groups" << std::endl;
        return false;
    }

    if (!linkSpecularPipeline())
    {
        std::cerr << "Failed to link specular lighting pipeline" << std::endl;
        return false;
    }

    buildSpecularSBT();

    std::cout << "Specular lighting pipeline created successfully" << std::endl;
    return true;
}

void OptixManager::launchSpecularLighting(unsigned int width, unsigned int height, const Camera &camera,
                                          const Photon *d_global_map, unsigned int global_count,
                                          const PhotonKDTreeDevice &global_kdtree,
                                          const Photon *d_caustic_map, unsigned int caustic_count,
                                          const PhotonKDTreeDevice &caustic_kdtree,
                                          const SpecularParams &spec_params, float4 *d_output)
{
    if (!specular_pipeline)
    {
        std::cerr << "Specular lighting pipeline not created!" << std::endl;
        return;
    }

#include "../cuda/specular_lighting/specular_launch_params.h"

    SpecularLaunchParams params = {};
    params.frame_buffer = d_output;
    params.width = width;
    params.height = height;

    params.eye = camera.getPosition();
    params.U = camera.getU();
    params.V = camera.getV();
    params.W = camera.getW();

    params.handle = ias_handle;

    params.triangle_materials = reinterpret_cast<Material *>(d_triangle_materials);
    params.sphere_materials[0].type = MATERIAL_TRANSMISSIVE;
    params.sphere_materials[0].albedo = make_float3(0.95f, 0.95f, 1.0f);
    params.sphere_materials[1].type = MATERIAL_SPECULAR;
    params.sphere_materials[1].albedo = make_float3(0.95f, 0.95f, 0.95f);

    // Photon maps for full lighting in reflections/refractions
    params.global_photon_map = const_cast<Photon *>(d_global_map);
    params.global_photon_count = global_count;
    params.caustic_photon_map = const_cast<Photon *>(d_caustic_map);
    params.caustic_photon_count = caustic_count;
    params.gather_radius = spec_params.gather_radius;
    params.global_kdtree = global_kdtree;
    params.caustic_kdtree = caustic_kdtree;

    // Light info
    params.light_position = make_float3(278.0f, 548.8f - 1.0f, 279.6f);
    params.light_intensity = make_float3(50.0f, 50.0f, 50.0f);

    params.quadLightStartIndex = quadLightStartIndex;

    // Initialize kd-trees as invalid (use linear fallback)
    params.global_kdtree.nodes = nullptr;
    params.global_kdtree.num_nodes = 0;
    params.global_kdtree.max_depth = 0;
    params.global_kdtree.valid = false;
    
    params.caustic_kdtree.nodes = nullptr;
    params.caustic_kdtree.num_nodes = 0;
    params.caustic_kdtree.max_depth = 0;
    params.caustic_kdtree.valid = false;

    // Configurable specular parameters
    params.max_recursion_depth = spec_params.max_recursion_depth;
    params.glass_ior = spec_params.glass_ior;
    params.glass_tint = spec_params.glass_tint;
    params.mirror_reflectivity = spec_params.mirror_reflectivity;
    params.fresnel_min = spec_params.fresnel_min;
    params.specular_ambient = spec_params.specular_ambient;
    params.indirect_brightness = spec_params.indirect_brightness;
    params.caustic_brightness = spec_params.caustic_brightness;

    CUdeviceptr d_params;
    CUDA_CHECK(cudaMalloc((void **)&d_params, sizeof(SpecularLaunchParams)));
    CUDA_CHECK(cudaMemcpy((void *)d_params, &params, sizeof(SpecularLaunchParams), cudaMemcpyHostToDevice));

    OPTIX_CHECK(optixLaunch(specular_pipeline, stream, d_params, sizeof(SpecularLaunchParams), &specular_sbt, width, height, 1));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree((void *)d_params));
}

bool OptixManager::buildTriangleGAS(const std::vector<OptixVertex> &vertices, const std::vector<float3> &colors)
{
    if (!initialized)
    {
        std::cerr << "Must initialize OptixManager before building geometry" << std::endl;
        return false;
    }
    if (vertices.empty() || vertices.size() % 3 != 0)
    {
        std::cerr << "Invalid vertex count: must be multiple of 3" << std::endl;
        return false;
    }

    if (colors.size() != vertices.size() / 3)
    {
        std::cerr << "Color count mismatch: expected " << (vertices.size() / 3) << " colors, got " << colors.size() << std::endl;
        return false;
    }

    OptixAccelerationStructuresBuilder::buildTriangleGAS(context, stream, vertices, d_triangle_vertices, d_triangle_gas_buffer, triangle_gas_handle);

    const size_t color_size = colors.size() * sizeof(float3);
    CUDA_CHECK(cudaMalloc((void **)&d_triangle_colors, color_size));
    CUDA_CHECK(cudaMemcpy((void *)d_triangle_colors, colors.data(), color_size, cudaMemcpyHostToDevice));

    // Allocate and upload diffuse materials for all triangle walls.
    // Use the per-triangle color as the diffuse albedo so photons inherit
    // the Cornell box colors (red/blue/white), while diffuseProb controls
    // the 50% reflect / 50% absorb behavior.
    const size_t material_count = colors.size();
    std::vector<Material> host_materials(material_count);
    for (size_t i = 0; i < material_count; ++i)
    {
        host_materials[i].type = MATERIAL_DIFFUSE;
        host_materials[i].albedo = colors[i]; // carry wall color into photons
        host_materials[i].diffuseProb = 0.5f; // 50% chance to continue
        host_materials[i].transmissiveCoeff = 0.0f;
    }

    const size_t material_size = material_count * sizeof(Material);
    CUDA_CHECK(cudaMalloc((void **)&d_triangle_materials, material_size));
    CUDA_CHECK(cudaMemcpy((void *)d_triangle_materials, host_materials.data(), material_size, cudaMemcpyHostToDevice));

    std::cout << "Built Triangle GAS with " << (vertices.size() / 3) << " triangles, " << colors.size()
              << " colors and basic diffuse materials" << std::endl;
    return true;
}

bool OptixManager::buildSphereGAS(float3 center1, float radius1, float3 center2, float radius2)
{
    if (!initialized)
    {
        std::cerr << "Must initialize OptixManager before building geometry" << std::endl;
        return false;
    }
    OptixAabb aabbs[2];
    aabbs[0].minX = center1.x - radius1;
    aabbs[0].minY = center1.y - radius1;
    aabbs[0].minZ = center1.z - radius1;
    aabbs[0].maxX = center1.x + radius1;
    aabbs[0].maxY = center1.y + radius1;
    aabbs[0].maxZ = center1.z + radius1;
    aabbs[1].minX = center2.x - radius2;
    aabbs[1].minY = center2.y - radius2;
    aabbs[1].minZ = center2.z - radius2;
    aabbs[1].maxX = center2.x + radius2;
    aabbs[1].maxY = center2.y + radius2;
    aabbs[1].maxZ = center2.z + radius2;
    OptixAccelerationStructuresBuilder::buildSphereGAS(context, stream, aabbs, 2, d_sphere_aabb_buffer, d_sphere_gas_buffer, sphere_gas_handle);
    std::cout << "Built Sphere GAS with 2 analytical spheres" << std::endl;
    return true;
}

bool OptixManager::buildIAS()
{
    if (!initialized)
    {
        std::cerr << "Must initialize OptixManager before building IAS" << std::endl;
        return false;
    }
    if (!triangle_gas_handle)
    {
        std::cerr << "Must build triangle GAS before building IAS" << std::endl;
        return false;
    }

    std::vector<OptixInstance> hostInstances;
    float identity[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};

    // Triangle Instance
    OptixInstance triInst = {};
    memcpy(triInst.transform, identity, sizeof(float) * 12);
    triInst.instanceId = 0;
    triInst.visibilityMask = 1;
    triInst.sbtOffset = 0;
    triInst.flags = OPTIX_INSTANCE_FLAG_NONE;
    triInst.traversableHandle = triangle_gas_handle;
    hostInstances.push_back(triInst);

    // Sphere Instance (only if built)
    if (sphere_gas_handle)
    {
        OptixInstance sphereInst = {};
        memcpy(sphereInst.transform, identity, sizeof(float) * 12);
        sphereInst.instanceId = 1;
        sphereInst.visibilityMask = 1;
        sphereInst.sbtOffset = 1;
        sphereInst.flags = OPTIX_INSTANCE_FLAG_NONE;
        sphereInst.traversableHandle = sphere_gas_handle;
        hostInstances.push_back(sphereInst);
    }

    OptixAccelerationStructuresBuilder::buildIAS(context, stream, hostInstances, d_ias_instances, d_ias_buffer, ias_handle);
    std::cout << "Built IAS with " << hostInstances.size() << " instances" << std::endl;
    return true;
}

void OptixManager::render(unsigned int width, unsigned int height, const Camera &camera, unsigned char *output_buffer)
{
    if (!initialized || !pipeline || !ias_handle)
    {
        std::cerr << "OptixManager not properly initialized (need IAS)" << std::endl;
        return;
    }
    launcher.allocateBuffers(width, height);

    float3 sphere1_color = make_float3(0.8f, 0.8f, 0.8f);
    float3 sphere2_color = make_float3(0.6f, 0.6f, 0.6f);

    launcher.launch(pipeline, stream, sbt, width, height, camera, ias_handle, output_buffer,
                    d_triangle_colors, sphere1_color, sphere2_color);
}

bool OptixManager::createPipeline()
{
    if (!initialized)
    {
        std::cerr << "Must initialize OptixManager before creating pipeline" << std::endl;
        return false;
    }
    if (!loadModule())
        return false;
    if (!createProgramGroups())
        return false;
    if (!linkPipeline())
        return false;
    buildSBT();
    std::cout << "OptixManager pipeline created successfully" << std::endl;
    return true;
}

bool OptixManager::createPhotonPipeline()
{
    if (!initialized)
    {
        std::cerr << "Must initialize OptixManager before creating photon pipeline" << std::endl;
        return false;
    }
    if (!loadPhotonModule())
        return false;
    if (!createPhotonProgramGroups())
        return false;
    if (!linkPhotonPipeline())
        return false;
    buildPhotonSBT();
    std::cout << "Photon emission pipeline created successfully" << std::endl;
    return true;
}
