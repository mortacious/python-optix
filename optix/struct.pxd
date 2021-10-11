from .common cimport OptixResult
from .base cimport OptixObject
from .program_group cimport OptixProgramGroup, ProgramGroup


cdef extern from "optix_includes.h" nogil:
    cdef size_t OPTIX_SBT_RECORD_HEADER_SIZE
    cdef size_t OPTIX_SBT_RECORD_ALIGNMENT

    OptixResult optixSbtRecordPackHeader(OptixProgramGroup 	programGroup,
                                         void * sbtRecordHeaderHostPointer)


cdef class _StructHelper(OptixObject):
    cdef object _dtype
    cdef dict array_values
    cdef object _array


cdef class SbtRecord(_StructHelper):
    cdef list program_groups
    cdef str header_format


cdef class LaunchParamsRecord(_StructHelper):
    pass
