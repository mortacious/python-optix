from libc.stdint cimport uint32_t
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from .base cimport OptixObject
from .common cimport OptixResult, CUstream, CUdeviceptr
from .context cimport OptixDeviceContext, OptixContextObject


cdef extern from "optix_micromap.h" nogil:
    cdef packed struct float2:
        float x
        float y

    void optixMicromapIndexToBaseBarycentrics(uint32_t microTriangleIndex,
                                              uint32_t subdivisionLevel,
                                              float2&  baseBarycentrics0,
                                              float2&  baseBarycentrics1,
                                              float2&  baseBarycentrics2)


cdef extern from "optix.h" nogil:
    cdef enum OptixOpacityMicromapArrayIndexingMode:
        OPTIX_OPACITY_MICROMAP_ARRAY_INDEXING_MODE_NONE,
        OPTIX_OPACITY_MICROMAP_ARRAY_INDEXING_MODE_LINEAR,
        OPTIX_OPACITY_MICROMAP_ARRAY_INDEXING_MODE_INDEXED


    cdef enum OptixOpacityMicromapFlags:
        OPTIX_OPACITY_MICROMAP_FLAG_NONE,
        OPTIX_OPACITY_MICROMAP_FLAG_PREFER_FAST_TRACE,
        OPTIX_OPACITY_MICROMAP_FLAG_PREFER_FAST_BUILD


    cdef enum OptixOpacityMicromapFormat:
        OPTIX_OPACITY_MICROMAP_FORMAT_NONE,
        OPTIX_OPACITY_MICROMAP_FORMAT_2_STATE  # 0: Transparent, 1: Opaque
        OPTIX_OPACITY_MICROMAP_FORMAT_4_STATE  # 0: Transparent, 1: Opaque, 2: Unknown-Transparent, 3: Unknown-Opaque


    cdef struct OptixOpacityMicromapHistogramEntry:
        unsigned int count
        unsigned int subdivisionLevel
        OptixOpacityMicromapFormat format


    cdef struct OptixOpacityMicromapUsageCount:
        unsigned int count
        unsigned int subdivisionLevel
        OptixOpacityMicromapFormat format


    cdef struct OptixOpacityMicromapDesc:
        unsigned int byteOffset
        unsigned short subdivisionLevel
        unsigned short format


    # get the defines for the micromap state as constant variables to access them from cython
    cdef const unsigned char OPTIX_OPACITY_MICROMAP_STATE_TRANSPARENT_DEFINE "OPTIX_OPACITY_MICROMAP_STATE_TRANSPARENT"  # = 0
    cdef const unsigned char OPTIX_OPACITY_MICROMAP_STATE_OPAQUE_DEFINE "OPTIX_OPACITY_MICROMAP_STATE_OPAQUE"  # = 1
    cdef const unsigned char OPTIX_OPACITY_MICROMAP_STATE_UNKNOWN_TRANSPARENT_DEFINE "OPTIX_OPACITY_MICROMAP_STATE_UNKNOWN_TRANSPARENT"  # = 2
    cdef const unsigned char OPTIX_OPACITY_MICROMAP_STATE_UNKNOWN_OPAQUE_DEFINE "OPTIX_OPACITY_MICROMAP_STATE_UNKNOWN_OPAQUE"  # = 3

    cdef const unsigned long long OPTIX_OPACITY_MICROMAP_ARRAY_BUFFER_BYTE_ALIGNMENT "OPTIX_OPACITY_MICROMAP_ARRAY_BUFFER_BYTE_ALIGNMENT"

    cdef const int OPTIX_OPACITY_MICROMAP_PREDEFINED_INDEX_FULLY_TRANSPARENT_DEFINE "OPTIX_OPACITY_MICROMAP_PREDEFINED_INDEX_FULLY_TRANSPARENT"  # = 3
    cdef const int OPTIX_OPACITY_MICROMAP_PREDEFINED_INDEX_FULLY_OPAQUE_DEFINE "OPTIX_OPACITY_MICROMAP_PREDEFINED_INDEX_FULLY_OPAQUE"  # = 3
    cdef const int OPTIX_OPACITY_MICROMAP_PREDEFINED_INDEX_FULLY_UNKNOWN_TRANSPARENT_DEFINE "OPTIX_OPACITY_MICROMAP_PREDEFINED_INDEX_FULLY_UNKNOWN_TRANSPARENT"  # = 3
    cdef const int OPTIX_OPACITY_MICROMAP_PREDEFINED_INDEX_FULLY_UNKNOWN_OPAQUE_DEFINE "OPTIX_OPACITY_MICROMAP_PREDEFINED_INDEX_FULLY_UNKNOWN_OPAQUE"  # = 3



    cdef struct OptixOpacityMicromapArrayBuildInput:
        OptixOpacityMicromapFlags flags
        CUdeviceptr inputBuffer
        CUdeviceptr perMicromapDescBuffer
        unsigned int perMicromapDescStrideInBytes
        unsigned int numMicromapHistogramEntries
        const OptixOpacityMicromapHistogramEntry * micromapHistogramEntries


    cdef struct OptixBuildInputOpacityMicromap:
        OptixOpacityMicromapArrayIndexingMode indexingMode
        CUdeviceptr opacityMicromapArray
        CUdeviceptr indexBuffer
        unsigned int indexSizeInBytes
        unsigned int indexStrideInBytes
        unsigned int indexOffset
        unsigned int numMicromapUsageCounts
        const OptixOpacityMicromapUsageCount * micromapUsageCounts


    cdef struct OptixMicromapBufferSizes:
        size_t outputSizeInBytes
        size_t tempSizeInBytes


    cdef struct OptixMicromapBuffers:
        CUdeviceptr output
        size_t outputSizeInBytes
        CUdeviceptr temp
        size_t tempSizeInBytes


    cdef struct OptixRelocateInputOpacityMicromap:
        CUdeviceptr opacityMicromapArray


    cdef struct OptixRelocationInfo:
        unsigned long long info[4]


    OptixResult optixOpacityMicromapArrayBuild(OptixDeviceContext context,
                                               CUstream stream,
                                               const OptixOpacityMicromapArrayBuildInput * buildInput,
                                               const OptixMicromapBuffers * buffers)


    OptixResult optixOpacityMicromapArrayComputeMemoryUsage(OptixDeviceContext context,
                                                            const OptixOpacityMicromapArrayBuildInput * buildInput,
                                                            OptixMicromapBufferSizes * bufferSizes)


    OptixResult optixOpacityMicromapArrayGetRelocationInfo(OptixDeviceContext context,
                                                           CUdeviceptr opacityMicromapArray,
                                                           OptixRelocationInfo * info)


    OptixResult optixOpacityMicromapArrayRelocate(OptixDeviceContext context,
                                                  CUstream stream,
                                                  const OptixRelocationInfo * info,
                                                  CUdeviceptr targetOpacityMicromapArray,
                                                  size_t targetOpacityMicromapArraySizeInBytes)


    OptixResult optixCheckRelocationCompatibility(OptixDeviceContext context,
                                           const OptixRelocationInfo * info,
                                           int * compatible
                                           )
# cdef extern from "<functional>" namespace "std" nogil:
#     cdef cppclass std_hash "hash"[T]:
#         function() except +
#         bint operator()(const T&) const



cdef class OpacityMicromapInput(OptixObject):
    cdef object buffer
    cdef OptixOpacityMicromapFormat c_format
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

