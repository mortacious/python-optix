from libc.stdint cimport uintptr_t

cdef extern from *:
    ctypedef void * CUcontext

cdef extern from "optix_includes.h" nogil:
    cdef enum OptixResult:
        OPTIX_SUCCESS
        # do not need the rest

    cdef OptixResult optixInit()

    ctypedef struct OptixModule:
        pass

    # declare this so the symbol is not removed
    ctypedef struct OptixFunctionTable:
        pass

    cdef const char* optixGetErrorName(OptixResult result)
    cdef const char* optixGetErrorString(OptixResult result)

    ctypedef uintptr_t CUstream

    ctypedef uintptr_t CUdeviceptr

    cdef void optix_check_return(OptixResult result) except+

cdef inline void optix_init():
    optix_check_return(optixInit())  # init the function table