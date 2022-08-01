from .base cimport OptixObject
from .common cimport OptixResult, CUdeviceptr, CUstream
from .context cimport OptixDeviceContext, OptixContextObject
from .program_group cimport ProgramGroup, OptixStackSizes
from .shader_binding_table cimport OptixShaderBindingTable

cdef extern from "optix_includes.h" nogil:
    cdef size_t OPTIX_COMPILE_DEFAULT_MAX_PAYLOAD_VALUE_COUNT

    # pipeline functions and structs
    ctypedef struct OptixPipeline:
        pass

    cdef enum OptixExceptionFlags:
        OPTIX_EXCEPTION_FLAG_NONE,
        OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW,
        OPTIX_EXCEPTION_FLAG_TRACE_DEPTH,
        OPTIX_EXCEPTION_FLAG_USER,
        OPTIX_EXCEPTION_FLAG_DEBUG

    cdef enum OptixTraversableGraphFlags:
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY,
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS,
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING

    IF _OPTIX_VERSION > 70400:  # switch to new primitive type flags
        cdef enum OptixPrimitiveTypeFlags:
            OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM,
            OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_QUADRATIC_BSPLINE,
            OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BSPLINE,
            OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_LINEAR,
            OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CATMULLROM,
            OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE,
            OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE,
    ELIF _OPTIX_VERSION > 70300:
        cdef enum OptixPrimitiveTypeFlags:
            OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM,
            OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_QUADRATIC_BSPLINE,
            OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BSPLINE,
            OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_LINEAR,
            OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CATMULLROM,
            OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE,
    ELSE:
        cdef enum OptixPrimitiveTypeFlags:
            OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM,
            OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_QUADRATIC_BSPLINE,
            OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BSPLINE,
            OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_LINEAR,
            OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE


    IF _OPTIX_VERSION > 70300:  # switch to new compile debug level
        cdef enum OptixCompileDebugLevel:
            OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT,
            OPTIX_COMPILE_DEBUG_LEVEL_NONE,
            OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL,
            OPTIX_COMPILE_DEBUG_LEVEL_MODERATE,
            OPTIX_COMPILE_DEBUG_LEVEL_FULL
    ELSE:
        cdef enum OptixCompileDebugLevel:
            OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT,
            OPTIX_COMPILE_DEBUG_LEVEL_NONE,
            OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO,
            OPTIX_COMPILE_DEBUG_LEVEL_FULL




    cdef struct OptixPipelineCompileOptions:
        int usesMotionBlur
        unsigned int traversableGraphFlags
        int numPayloadValues
        int numAttributeValues
        unsigned int exceptionFlags
        const char * pipelineLaunchParamsVariableName
        unsigned int usesPrimitiveTypeFlags


    cdef struct OptixPipelineLinkOptions:
        unsigned int maxTraceDepth
        OptixCompileDebugLevel debugLevel # OptixCompileDebugLevel


    ctypedef struct OptixProgramGroup:
        pass


    cdef OptixResult optixPipelineCreate(OptixDeviceContext context,
                                         const OptixPipelineCompileOptions * pipelineCompileOptions,
                                         const OptixPipelineLinkOptions * pipelineLinkOptions,
                                         const OptixProgramGroup *    programGroups,
                                         unsigned int numProgramGroups,
                                         char * logString,
                                         size_t * logStringSize,
                                         OptixPipeline * pipeline)


    cdef OptixResult optixPipelineDestroy(OptixPipeline pipeline)


    cdef OptixResult optixPipelineSetStackSize(OptixPipeline pipeline,
                                               unsigned int directCallableStackSizeFromTraversal,
                                               unsigned int directCallableStackSizeFromState,
                                               unsigned int continuationStackSize,
                                               unsigned int maxTraversableGraphDepth)



    OptixResult optixLaunch(OptixPipeline pipeline,
                            CUstream stream,
                            CUdeviceptr pipelineParams,
                            size_t pipelineParamsSize,
                            const OptixShaderBindingTable *sbt,
                            unsigned int width,
                            unsigned int height,
                            unsigned int depth
                            )

cdef extern from "optix_stack_size.h" nogil:
    OptixResult optixUtilAccumulateStackSizes(OptixProgramGroup programGroup,
                                                  OptixStackSizes* stackSizes)


    OptixResult optixUtilComputeStackSizes(const OptixStackSizes* stackSizes,
                                           unsigned int maxTraceDepth,
                                           unsigned int maxCCDepth,
                                           unsigned int maxDCDepth,
                                           unsigned int* directCallableStackSizeFromTraversal,
                                           unsigned int* directCallableStackSizeFromState,
                                           unsigned int* continuationStackSize
                                           )

    OptixResult optixUtilComputeStackSizesCssCCTree(const OptixStackSizes* stackSizes,
                                                    unsigned int cssCCTree,
                                                    unsigned int maxTraceDepth,
                                                    unsigned int maxDCDepth,
                                                    unsigned int* directCallableStackSizeFromTraversal,
                                                    unsigned int* directCallableStackSizeFromState,
                                                    unsigned int* continuationStackSize
                                                    )

    OptixResult optixUtilComputeStackSizesDCSplit(const OptixStackSizes* stackSizes,
                                                  unsigned int dssDCFromTraversal,
                                                  unsigned int dssDCFromState,
                                                  unsigned int maxTraceDepth,
                                                  unsigned int maxCCDepth,
                                                  unsigned int maxDCDepthFromTraversal,
                                                  unsigned int maxDCDepthFromState,
                                                  unsigned int* directCallableStackSizeFromTraversal,
                                                  unsigned int* directCallableStackSizeFromState,
                                                  unsigned int* continuationStackSize
                                                  )

    OptixResult optixUtilComputeStackSizesSimplePathTracer(OptixProgramGroup programGroupRG,
                                                           OptixProgramGroup programGroupMS1,
                                                           const OptixProgramGroup* programGroupCH1,
                                                           unsigned int programGroupCH1Count,
                                                           OptixProgramGroup programGroupMS2,
                                                           const OptixProgramGroup* programGroupCH2,
                                                           unsigned int programGroupCH2Count,
                                                           unsigned int* directCallableStackSizeFromTraversal,
                                                           unsigned int* directCallableStackSizeFromState,
                                                           unsigned int* continuationStackSize
                                                           )

cdef class PipelineCompileOptions(OptixObject):
    cdef OptixPipelineCompileOptions compile_options
    cdef bytes _pipeline_launch_params_variable_name

cdef class PipelineLinkOptions(OptixObject):
    cdef OptixPipelineLinkOptions link_options

cdef class Pipeline(OptixContextObject):
    cdef OptixPipeline pipeline
    cdef OptixStackSizes _stack_sizes
    cdef unsigned int _max_traversable_graph_depth
    cpdef set_stack_sizes(self,
                          unsigned int direct_callable_stack_size_from_traversal,
                          unsigned int direct_callable_stack_size_from_state,
                          unsigned int continuation_stack_size)
    cpdef compute_stack_sizes(self,
                              unsigned int max_trace_depth,
                              unsigned int max_cc_depth,
                              unsigned int max_dc_depth)
    cpdef compute_stack_sizes_css_cc_tree(self,
                                          unsigned int css_cc_tree,
                                          unsigned int max_trace_depth,
                                          unsigned int max_dc_depth)
    cpdef compute_stack_sizes_dc_split(self,
                                       unsigned int dss_dc_from_traversal,
                                       unsigned int dss_dc_from_state,
                                       unsigned int max_trace_depth,
                                       unsigned int max_cc_depth,
                                       unsigned int max_dc_depth_from_traversal,
                                       unsigned int max_dc_depth_from_state)
    cpdef compute_stack_sizes_simple_path_tracer(self,
                                                 ProgramGroup program_group_raygen,
                                                 ProgramGroup program_group_miss_1,
                                                 object program_groups_closesthit_1,
                                                 ProgramGroup program_group_miss_2,
                                                 object program_groups_closesthit_2)
