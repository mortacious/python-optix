from libc.stdint cimport uint32_t

cdef extern from "optix_micromap.h" nogil:
    cdef struct float2:
        float x
        float y

    void optixMicromapIndexToBaseBarycentrics(uint32_t microTriangleIndex,
                                              uint32_t subdivisionLevel,
                                              float2&  baseBarycentrics0,
                                              float2&  baseBarycentrics1,
                                              float2&  baseBarycentrics2)