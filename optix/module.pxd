from .base cimport OptixObject
from .common cimport OptixResult, OptixModule
from .context cimport OptixDeviceContext, OptixContextObject
from .build cimport OptixPrimitiveType
from .pipeline cimport OptixPipelineCompileOptions

cdef extern from "optix_includes.h" nogil:
    cdef OptixResult optixInit()

    cdef size_t OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT

    cdef enum OptixCompileOptimizationLevel:
        OPTIX_COMPILE_OPTIMIZATION_DEFAULT,
        OPTIX_COMPILE_OPTIMIZATION_LEVEL_0,
        OPTIX_COMPILE_OPTIMIZATION_LEVEL_1,
        OPTIX_COMPILE_OPTIMIZATION_LEVEL_2,
        OPTIX_COMPILE_OPTIMIZATION_LEVEL_3


    cdef enum OptixCompileDebugLevel:
        OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT,
        OPTIX_COMPILE_DEBUG_LEVEL_NONE,
        OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO,
        OPTIX_COMPILE_DEBUG_LEVEL_FULL


    cdef struct OptixModuleCompileBoundValueEntry:
        size_t pipelineParamOffsetInBytes
        size_t sizeInBytes
        const void* boundValuePtr
        const char* annotation


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


cdef class ModuleCompileOptions(OptixObject):
    cdef OptixModuleCompileOptions compile_options


cdef class Module(OptixContextObject):
    cdef OptixModule module
    cdef list _compile_flags

    #cpdef size_t c_obj(self)