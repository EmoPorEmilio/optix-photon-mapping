#include "OptixManager.h"
#include <optix_function_table_definition.h>
#include <sutil/Exception.h>
#include "../Material.h"
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
                                    unsigned int quadLightStartIndex, CUdeviceptr &out_photons, unsigned int &out_count)
{
    if (!photon_pipeline)
    {
        std::cerr << "Photon pipeline not created!" << std::endl;
        return;
    }

    if (!d_photon_buffer)
    {
        CUDA_CHECK(cudaMalloc((void **)&d_photon_buffer, num_photons * sizeof(Photon)));
    }

    if (!d_photon_counter)
    {
        CUDA_CHECK(cudaFree((void *)d_photon_counter));
        CUDA_CHECK(cudaMalloc((void **)&d_photon_counter, sizeof(unsigned int)));
    }

    unsigned int zero = 0;
    CUDA_CHECK(cudaMemcpy((void *)d_photon_counter, &zero, sizeof(unsigned int), cudaMemcpyHostToDevice));

    PhotonLaunchParams params = {};

    // esto por ahora solo maneja una luz, podría refactorear para agregar de manera dinámica
    params.handle = ias_handle;
    params.light = light;
    params.num_photons = num_photons;
    params.photon_power = light.getIntensity() / static_cast<float>(num_photons);
    params.quadLightStartIndex = quadLightStartIndex;
    params.triangle_colors = reinterpret_cast<float3 *>(d_triangle_colors);

    // Bind triangle materials (all walls: 0.5 gray diffuse with 50% survive prob).
    params.triangle_materials = reinterpret_cast<Material *>(d_triangle_materials);

    // Simple material + geometry setup for the two analytic spheres:
    // sphere 0: fully transmissive "glass" (no absorption, straight-through in photon pass)
    // sphere 1: fully reflective mirror.
    params.sphere_materials[0].type = MATERIAL_TRANSMISSIVE;
    params.sphere_materials[0].albedo = make_float3(1.0f, 1.0f, 1.0f);
    params.sphere_materials[0].diffuseProb = 0.0f;
    params.sphere_materials[0].transmissiveCoeff = 1.0f;

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
    params.photons_out = reinterpret_cast<Photon *>(d_photon_buffer);
    params.photon_counter = d_photon_counter;

    CUdeviceptr d_params;
    CUDA_CHECK(cudaMalloc((void **)&d_params, sizeof(PhotonLaunchParams)));
    CUDA_CHECK(cudaMemcpy((void *)d_params, &params, sizeof(PhotonLaunchParams), cudaMemcpyHostToDevice));

    // Launch one photon per x-dimension index (no CUDA-style blocks/threads).
    OPTIX_CHECK(optixLaunch(photon_pipeline, stream, d_params, sizeof(PhotonLaunchParams), &photon_sbt, num_photons, 1, 1));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaMemcpy(&out_count, (void *)d_photon_counter, sizeof(unsigned int), cudaMemcpyDeviceToHost));

    // With multiple bounces we may attempt to store more hits than there is
    // space in the buffer; clamp the reported count to the buffer capacity.
    if (out_count > num_photons)
        out_count = num_photons;

    out_photons = d_photon_buffer;

    CUDA_CHECK(cudaFree((void *)d_params));

    std::cout << "Launched " << num_photons << " photons, stored " << out_count << " hits (clamped)" << std::endl;
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

    contextOwner.destroy();
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
        host_materials[i].albedo = colors[i];          // carry wall color into photons
        host_materials[i].diffuseProb = 0.5f;          // 50% chance to continue
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
