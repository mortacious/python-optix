from .common cimport OptixResult, OptixModule
from .context cimport OptixDeviceContext, OptixContextObject

cdef extern from "optix_includes.h" nogil:
    ctypedef struct OptixProgramGroup:
        pass

    cdef struct OptixStackSizes:
        unsigned int cssRG
        unsigned int cssMS
        unsigned int cssCH
        unsigned int cssAH
        unsigned int cssIS
        unsigned int cssCC
        unsigned int dssDC


    cdef enum OptixProgramGroupKind:
        OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
        OPTIX_PROGRAM_GROUP_KIND_MISS,
        OPTIX_PROGRAM_GROUP_KIND_EXCEPTION,
        OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
        OPTIX_PROGRAM_GROUP_KIND_CALLABLES


    cdef enum OptixProgramGroupFlags:
        OPTIX_PROGRAM_GROUP_FLAGS_NONE


    cdef struct OptixProgramGroupSingleModule:
        OptixModule module
        const char * entryFunctionName


    cdef struct OptixProgramGroupCallables:
        OptixModule moduleDC
        const char * entryFunctionNameDC
        OptixModule moduleCC
        const char * entryFunctionNameCC


    cdef struct OptixProgramGroupHitgroup:
        OptixModule moduleCH
        const char * entryFunctionNameCH
        OptixModule moduleAH
        const char * entryFunctionNameAH
        OptixModule moduleIS
        const char * entryFunctionNameIS


    cdef struct OptixProgramGroupDesc:
        OptixProgramGroupKind kind
        unsigned int flags
        OptixProgramGroupSingleModule raygen
        OptixProgramGroupSingleModule miss
        OptixProgramGroupSingleModule exception
        OptixProgramGroupCallables callables
        OptixProgramGroupHitgroup hitgroup


    cdef struct OptixProgramGroupOptions:
        pass


    OptixResult optixProgramGroupGetStackSize(OptixProgramGroup programGroup, OptixStackSizes* stackSizes)

    OptixResult optixProgramGroupCreate(OptixDeviceContext context,
                                        const OptixProgramGroupDesc* programDescriptions,
                                        unsigned int numProgramGroups,
                                        const OptixProgramGroupOptions *options,
                                        char* logString,
                                        size_t* logStringSize,
                                        OptixProgramGroup* programGroups)

    OptixResult optixProgramGroupDestroy(OptixProgramGroup programGroup)




cdef class ProgramGroup(OptixContextObject):
    cdef OptixProgramGroup program_group
    cdef OptixProgramGroupKind _kind
