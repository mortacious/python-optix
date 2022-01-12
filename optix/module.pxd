from .base cimport OptixObject
from .common cimport OptixResult, OptixModule
from .context cimport OptixDeviceContext, OptixContextObject
from .build cimport OptixPrimitiveType
from .pipeline cimport OptixPipelineCompileOptions, OptixCompileDebugLevel
from libcpp.vector cimport vector

cdef extern from "optix_includes.h" nogil:

    cdef OptixResult optixInit()

    cdef size_t OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT
    cdef size_t OPTIX_COMPILE_DEFAULT_MAX_PAYLOAD_VALUE_COUNT


    cdef enum OptixCompileOptimizationLevel:
        OPTIX_COMPILE_OPTIMIZATION_DEFAULT,
        OPTIX_COMPILE_OPTIMIZATION_LEVEL_0,
        OPTIX_COMPILE_OPTIMIZATION_LEVEL_1,
        OPTIX_COMPILE_OPTIMIZATION_LEVEL_2,
        OPTIX_COMPILE_OPTIMIZATION_LEVEL_3


    cdef struct OptixModuleCompileBoundValueEntry:
        size_t pipelineParamOffsetInBytes
        size_t sizeInBytes
        const void* boundValuePtr
        const char* annotation


    IF _OPTIX_VERSION > 70300:  # switch to new version
        cdef size_t OPTIX_COMPILE_DEFAULT_MAX_PAYLOAD_TYPE_COUNT

        cdef enum OptixPayloadSemantics:
            OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_NONE,
            OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ,
            OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE,
            OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE,
            OPTIX_PAYLOAD_SEMANTICS_CH_NONE,
            OPTIX_PAYLOAD_SEMANTICS_CH_READ,
            OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
            OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE,
            OPTIX_PAYLOAD_SEMANTICS_MS_NONE,
            OPTIX_PAYLOAD_SEMANTICS_MS_READ,
            OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
            OPTIX_PAYLOAD_SEMANTICS_MS_READ_WRITE,
            OPTIX_PAYLOAD_SEMANTICS_AH_NONE,
            OPTIX_PAYLOAD_SEMANTICS_AH_READ,
            OPTIX_PAYLOAD_SEMANTICS_AH_WRITE,
            OPTIX_PAYLOAD_SEMANTICS_AH_READ_WRITE,
            OPTIX_PAYLOAD_SEMANTICS_IS_NONE,
            OPTIX_PAYLOAD_SEMANTICS_IS_READ,
            OPTIX_PAYLOAD_SEMANTICS_IS_WRITE,
            OPTIX_PAYLOAD_SEMANTICS_IS_READ_WRITE,

        cdef struct OptixPayloadType:
            unsigned int numPayloadValues
            const unsigned int* payloadSemantics

        cdef struct OptixModuleCompileOptions:
            int maxRegisterCount
            OptixCompileOptimizationLevel optLevel
            OptixCompileDebugLevel 	debugLevel
            const OptixModuleCompileBoundValueEntry* boundValues
            unsigned int numBoundValues
            unsigned int numPayloadTypes
            OptixPayloadType* payloadTypes

        cdef struct OptixBuiltinISOptions:
            OptixPrimitiveType builtinISModuleType
            int usesMotionBlur
            unsigned int buildFlags
            unsigned int curveEndcapFlags
    ELSE:
        cdef struct OptixModuleCompileOptions:
            int maxRegisterCount
            OptixCompileOptimizationLevel optLevel
            OptixCompileDebugLevel 	debugLevel
            const OptixModuleCompileBoundValueEntry* boundValues
            unsigned int numBoundValues

        cdef struct OptixBuiltinISOptions:
            OptixPrimitiveType builtinISModuleType
            int usesMotionBlur


    OptixResult optixModuleCreateFromPTX(OptixDeviceContext context,
                                         const OptixModuleCompileOptions *moduleCompileOptions,
                                         const OptixPipelineCompileOptions *pipelineCompileOptions,
                                         const char *PTX,
                                         size_t PTXsize,
                                         char *logString,
                                         size_t *logStringSize,
                                         OptixModule *module)


    cdef OptixResult optixModuleDestroy(OptixModule module)


    cdef OptixResult optixBuiltinISModuleGet(OptixDeviceContext context,
                                        const OptixModuleCompileOptions *moduleCompileOptions,
                                        const OptixPipelineCompileOptions *pipelineCompileOptions,
                                        const OptixBuiltinISOptions *builtinISOptions,
                                        OptixModule *builtinModule)


    IF _OPTIX_VERSION > 70300:  # switch to new version
        ctypedef struct OptixTask:
            pass

        cdef enum OptixModuleCompileState:
            OPTIX_MODULE_COMPILE_STATE_NOT_STARTED
            OPTIX_MODULE_COMPILE_STATE_STARTED
            OPTIX_MODULE_COMPILE_STATE_IMPENDING_FAILURE
            OPTIX_MODULE_COMPILE_STATE_FAILED
            OPTIX_MODULE_COMPILE_STATE_COMPLETED

        cdef OptixResult optixModuleGetCompilationState(OptixModule module,
                                                        OptixModuleCompileState * state)

        cdef OptixResult optixModuleCreateFromPTXWithTasks(OptixDeviceContext context,
                                                           const OptixModuleCompileOptions * moduleCompileOptions,
                                                           const OptixPipelineCompileOptions * pipelineCompileOptions,
                                                           const char * PTX,
                                                           size_t PTXsize,
                                                           char * logString,
                                                           size_t * logStringSize,
                                                           OptixModule * module,
                                                           OptixTask * firstTask)

        cdef OptixResult optixTaskExecute(OptixTask task,
                                     OptixTask * additionalTasks,
                                     unsigned int maxNumAdditionalTasks,
                                     unsigned int *numAdditionalTasksCreated)


cdef class BuiltinISOptions(OptixObject):
    cdef OptixBuiltinISOptions options

cdef class Module(OptixContextObject):
    cdef OptixModule module
    cdef list _compile_flags

IF _OPTIX_VERSION > 70300:  # switch to new version
    cdef class ModuleCompileOptions(OptixObject):
        cdef OptixModuleCompileOptions compile_options
        cdef vector[OptixPayloadType] payload_types
        cdef vector[vector[unsigned int]] payload_values # WTF!

    cdef class Task(OptixObject):
        cdef OptixTask task
        cdef Module module
ELSE:
    cdef class ModuleCompileOptions(OptixObject):
        cdef OptixModuleCompileOptions compile_options