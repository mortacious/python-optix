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
    cdef struct OptixFunctionTable:
        pass

    cdef const char* optixGetErrorName(OptixResult result)
    cdef const char* optixGetErrorString(OptixResult result)

    ctypedef CUstream

    ctypedef unsigned long long CUdeviceptr

cdef inline int optix_check_return(OptixResult result, const char * log_buffer = NULL) except -1 with gil:
    cdef bytes py_string

    if result != OPTIX_SUCCESS:
        py_name = optixGetErrorName(result)
        py_string = optixGetErrorString(result)
        py_error_string = f"{py_name.decode()}: {py_string.decode()}{f': {log_buffer.decode()}' if log_buffer != NULL else ''}"
        raise RuntimeError(py_error_string)


cdef inline void optix_init():
    optix_check_return(optixInit())  # init the function table
