from .context cimport DeviceContext

cdef class OptixObject:
    cdef DeviceContext context