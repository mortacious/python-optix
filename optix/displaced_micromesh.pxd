from libc.stdint cimport uint32_t
from libcpp.vector cimport vector
from .base cimport OptixObject
from .common cimport OptixResult, CUstream, CUdeviceptr
from .context cimport OptixDeviceContext, OptixContextObject
from .opacity_micromap cimport optixMicromapIndexToBaseBarycentrics



cdef extern from "optix.h" nogil:
    cdef enum OptixDisplacementMicromapArrayIndexingMode:
        OPTIX_DISPLACEMENT_MICROMAP_ARRAY_INDEXING_MODE_NONE
        OPTIX_DISPLACEMENT_MICROMAP_ARRAY_INDEXING_MODE_LINEAR
        OPTIX_DISPLACEMENT_MICROMAP_ARRAY_INDEXING_MODE_INDEXED

    cdef enum OptixDisplacementMicromapBiasAndScaleFormat:
        OPTIX_DISPLACEMENT_MICROMAP_BIAS_AND_SCALE_FORMAT_NONE
        OPTIX_DISPLACEMENT_MICROMAP_BIAS_AND_SCALE_FORMAT_FLOAT2
        OPTIX_DISPLACEMENT_MICROMAP_BIAS_AND_SCALE_FORMAT_HALF2

    cdef enum OptixDisplacementMicromapDirectionFormat:
        OPTIX_DISPLACEMENT_MICROMAP_DIRECTION_FORMAT_NONE
        OPTIX_DISPLACEMENT_MICROMAP_DIRECTION_FORMAT_FLOAT3
        OPTIX_DISPLACEMENT_MICROMAP_DIRECTION_FORMAT_HALF3

    cdef enum OptixDisplacementMicromapFlags:
        OPTIX_DISPLACEMENT_MICROMAP_FLAG_NONE
        OPTIX_DISPLACEMENT_MICROMAP_FLAG_PREFER_FAST_TRACE
        OPTIX_DISPLACEMENT_MICROMAP_FLAG_PREFER_FAST_BUILD

    cdef enum OptixDisplacementMicromapFormat:
        OPTIX_DISPLACEMENT_MICROMAP_FORMAT_NONE
        OPTIX_DISPLACEMENT_MICROMAP_FORMAT_64_MICRO_TRIS_64_BYTES
        OPTIX_DISPLACEMENT_MICROMAP_FORMAT_256_MICRO_TRIS_128_BYTES
        OPTIX_DISPLACEMENT_MICROMAP_FORMAT_1024_MICRO_TRIS_128_BYTES

    cdef enum OptixDisplacementMicromapTriangleFlags:
        OPTIX_DISPLACEMENT_MICROMAP_TRIANGLE_FLAG_NONE
        OPTIX_DISPLACEMENT_MICROMAP_TRIANGLE_FLAG_DECIMATE_EDGE_01
        OPTIX_DISPLACEMENT_MICROMAP_TRIANGLE_FLAG_DECIMATE_EDGE_12
        OPTIX_DISPLACEMENT_MICROMAP_TRIANGLE_FLAG_DECIMATE_EDGE_20

    cdef struct OptixDisplacementMicromapHistogramEntry:
        unsigned int count
        unsigned int subdivisionLevel
        OptixDisplacementMicromapFormat format

    cdef struct OptixDisplacementMicromapArrayBuildInput:
        OptixDisplacementMicromapFlags flags
        CUdeviceptr displacementValuesBuffer
        CUdeviceptr perDisplacementMicromapDescBuffer
        unsigned int perDisplacementMicromapDescStrideInBytes
        unsigned int numDisplacementMicromapHistogramEntries
        const OptixDisplacementMicromapHistogramEntry* displacementMicromapHistogramEntries

cdef class DisplacedMicromapInput(OptixObject):
    cdef object buffer
    cdef OptixDisplacementMicromapFormat c_format
    cdef unsigned int c_subdivision_level


cdef class OpacityMicromapArray(OptixContextObject):
    cdef object d_micromap_array_buffer
    cdef OptixOpacityMicromapFlags _build_flags
    cdef size_t _buffer_size
    cdef unsigned int c_num_micromaps
    cdef tuple _micromap_types

    cdef void build(self, inputs, stream=*)


cdef class BuildInputOpacityMicromap(OptixObject):
    cdef OptixBuildInputOpacityMicromap build_input
    cdef OpacityMicromapArray c_micromap_array
    cdef object _d_index_buffer
    cdef object _usage_counts
    cdef vector[OptixOpacityMicromapUsageCount] c_usage_counts

