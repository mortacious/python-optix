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

    cdef struct OptixDisplacementMicromapDesc:
        unsigned int byteOffset
        unsigned short subdivisionLevel
        unsigned short format

    cdef struct OptixDisplacementMicromapArrayBuildInput:
        OptixDisplacementMicromapFlags flags
        CUdeviceptr displacementValuesBuffer
        CUdeviceptr perDisplacementMicromapDescBuffer
        unsigned int perDisplacementMicromapDescStrideInBytes
        unsigned int numDisplacementMicromapHistogramEntries
        const OptixDisplacementMicromapHistogramEntry* displacementMicromapHistogramEntries

    cdef struct OptixMicromapBufferSizes:
        size_t outputSizeInBytes
        size_t tempSizeInBytes


    cdef struct OptixMicromapBuffers:
        CUdeviceptr output
        size_t outputSizeInBytes
        CUdeviceptr temp
        size_t tempSizeInBytes

    
    cdef struct OptixDisplacementMicromapUsageCount:
        unsigned int count
        unsigned int subdivisionLevel
        OptixDisplacementMicromapFormat format

    cdef struct OptixBuildInputDisplacementMicromap:
        OptixDisplacementMicromapArrayIndexingMode indexingMode
        CUdeviceptr displacementMicromapArray
        CUdeviceptr displacementMicromapIndexBuffer
        CUdeviceptr vertexDirectionsBuffer
        CUdeviceptr vertexBiasAndScaleBuffer
        CUdeviceptr triangleFlagsBuffer
        unsigned int displacementMicromapIndexOffset
        unsigned int displacementMicromapIndexStrideInBytes
        unsigned int displacementMicromapIndexSizeInBytes
        OptixDisplacementMicromapDirectionFormat vertexDirectionFormat
        unsigned int vertexDirectionStrideInBytes
        OptixDisplacementMicromapBiasAndScaleFormat vertexBiasAndScaleFormat
        unsigned int vertexBiasAndScaleStrideInBytes
        unsigned int triangleFlagsStrideInBytes
        unsigned int numDisplacementMicromapUsageCounts
        const OptixDisplacementMicromapUsageCount* displacementMicromapUsageCounts


    OptixResult optixDisplacementMicromapArrayComputeMemoryUsage(OptixDeviceContext context,
                                                                 const OptixDisplacementMicromapArrayBuildInput* buildInput,
                                                                 OptixMicromapBufferSizes* bufferSizes)

    OptixResult optixDisplacementMicromapArrayBuild(OptixDeviceContext context,
                                                    CUstream stream,
                                                    const OptixDisplacementMicromapArrayBuildInput* buildInput,
                                                    const OptixMicromapBuffers* buffers) 		

cdef class DisplacementMicromapInput(OptixObject):
    cdef object buffer
    cdef OptixDisplacementMicromapFormat c_format
    cdef unsigned int c_subdivision_level


cdef class DisplacementMicromapArray(OptixContextObject):
    cdef object d_micromap_array_buffer
    cdef OptixDisplacementMicromapFlags _build_flags
    cdef size_t _buffer_size
    cdef unsigned int c_num_micromaps
    cdef tuple _micromap_types
    cdef void build(self, inputs, stream=*)


cdef class BuildInputDisplacementMicromap(OptixObject):
    cdef OptixBuildInputDisplacementMicromap build_input
    cdef DisplacementMicromapArray c_micromap_array
    cdef object _d_displacement_directions
    cdef object _d_index_buffer
    cdef object _d_bias_and_scale
    cdef object _usage_counts
    cdef vector[OptixDisplacementMicromapUsageCount] c_usage_counts

