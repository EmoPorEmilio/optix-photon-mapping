#pragma once

#include <optix.h>
#include <optix_stack_size.h>
#include <sutil/Exception.h>

class OptixPipelineBuilder
{
public:
    
    
    static void createAndLink(
        OptixDeviceContext context,
        const OptixPipelineCompileOptions *compileOptions,
        const OptixProgramGroup *groups,
        unsigned int groupCount,
        OptixPipeline &outPipeline)
    {
        OptixPipelineLinkOptions link_options = {};
        link_options.maxTraceDepth = 1;

        char log[2048];
        size_t logSize = sizeof(log);
        OPTIX_CHECK_LOG(optixPipelineCreate(
                            context,
                            compileOptions,
                            &link_options,
                            groups,
                            groupCount,
                            log,
                            &logSize,
                            &outPipeline),
                        log, logSize);

        
        OptixStackSizes stack_sizes = {};
        for (unsigned int i = 0; i < groupCount; ++i)
        {
            OPTIX_CHECK(optixUtilAccumulateStackSizes(groups[i], &stack_sizes, outPipeline));
        }

        uint32_t dcss_trav, dcss_state, css;
        OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, 1, 0, 0, &dcss_trav, &dcss_state, &css));
        OPTIX_CHECK(optixPipelineSetStackSize(outPipeline, dcss_trav, dcss_state, css, 1));
    }
};



