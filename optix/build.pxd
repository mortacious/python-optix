from .common cimport OptixResult
from .context cimport OptixDeviceContext, DeviceContext
from libcpp.vector cimport vector


cdef extern from "optix.h" nogil:
    # build functions and structs

    cdef enum OptixBuildFlags:
        OPTIX_BUILD_FLAG_NONE,
        OPTIX_BUILD_FLAG_ALLOW_UPDATE,
        OPTIX_BUILD_FLAG_ALLOW_COMPACTION,
        OPTIX_BUILD_FLAG_PREFER_FAST_TRACE,
        OPTIX_BUILD_FLAG_PREFER_FAST_BUILD,
        OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS,
        OPTIX_BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS,


    cdef enum OptixBuildOperation:
        OPTIX_BUILD_OPERATION_BUILD,
        OPTIX_BUILD_OPERATION_UPDATE


    cdef enum OptixMotionFlags:
        OPTIX_MOTION_FLAG_NONE,
        OPTIX_MOTION_FLAG_START_VANISH,
        OPTIX_MOTION_FLAG_END_VANISH


    cdef struct OptixMotionOptions:
        unsigned short numKeys
        unsigned short flags
        float timeBegin
        float timeEnd

    cdef struct OptixAccelBuildOptions:
        unsigned int buildFlags
        OptixBuildOperation operation
        OptixMotionOptions motionOptions

    cdef enum OptixBuildInputType:
        OPTIX_BUILD_INPUT_TYPE_TRIANGLES,
        OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES,
        OPTIX_BUILD_INPUT_TYPE_INSTANCES,
        OPTIX_BUILD_INPUT_TYPE_INSTANCE_POINTERS,
        OPTIX_BUILD_INPUT_TYPE_CURVES

    ctypedef unsigned long long CUdeviceptr

    cdef struct OptixBuildInputInstanceArray:
        CUdeviceptr instances
        unsigned int numInstances

    cdef struct OptixAabb:
        float minX
        float minY
        float minZ
        float maxX
        float maxY
        float maxZ

    cdef struct OptixBuildInputCustomPrimitiveArray:
        const CUdeviceptr * aabbBuffers
        unsigned int numPrimitives
        unsigned int strideInBytes
        const unsigned int * flags
        unsigned int numSbtRecords
        CUdeviceptr sbtIndexOffsetBuffer
        unsigned int sbtIndexOffsetSizeInBytes
        unsigned int sbtIndexOffsetStrideInBytes
        unsigned int primitiveIndexOffset

    cdef enum OptixPrimitiveType:
        OPTIX_PRIMITIVE_TYPE_CUSTOM,
        OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE,
        OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE,
        OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR,
        OPTIX_PRIMITIVE_TYPE_TRIANGLE,


    cdef struct OptixBuildInputCurveArray:
        OptixPrimitiveType curveType
        unsigned int numPrimitives
        const CUdeviceptr * vertexBuffers
        unsigned int numVertices
        unsigned int vertexStrideInBytes
        const CUdeviceptr * widthBuffers
        unsigned int widthStrideInBytes
        const CUdeviceptr * normalBuffers
        unsigned int normalStrideInBytes
        CUdeviceptr indexBuffer
        unsigned int indexStrideInBytes
        unsigned int flag
        unsigned int primitiveIndexOffset

    cdef enum OptixIndicesFormat:
        OPTIX_INDICES_FORMAT_NONE,
        OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3,
        OPTIX_INDICES_FORMAT_UNSIGNED_INT3

    cdef enum OptixVertexFormat:
        OPTIX_VERTEX_FORMAT_NONE,
        OPTIX_VERTEX_FORMAT_FLOAT3,
        OPTIX_VERTEX_FORMAT_FLOAT2,
        OPTIX_VERTEX_FORMAT_HALF3,
        OPTIX_VERTEX_FORMAT_HALF2,
        OPTIX_VERTEX_FORMAT_SNORM16_3,
        OPTIX_VERTEX_FORMAT_SNORM16_2

    cdef enum OptixTransformFormat:
        OPTIX_TRANSFORM_FORMAT_NONE,
        OPTIX_TRANSFORM_FORMAT_MATRIX_FLOAT12,

    cdef enum OptixGeometryFlags:
        OPTIX_GEOMETRY_FLAG_NONE,
        OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
        OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL

    cdef struct OptixBuildInputTriangleArray:
        const CUdeviceptr * vertexBuffers
        unsigned int numVertices
        OptixVertexFormat vertexFormat
        unsigned int vertexStrideInBytes
        CUdeviceptr indexBuffer
        unsigned int numIndexTriplets
        OptixIndicesFormat indexFormat
        unsigned int indexStrideInBytes
        CUdeviceptr preTransform
        const unsigned int * flags
        unsigned int numSbtRecords
        CUdeviceptr sbtIndexOffsetBuffer
        unsigned int sbtIndexOffsetSizeInBytes
        unsigned int sbtIndexOffsetStrideInBytes
        unsigned int primitiveIndexOffset
        OptixTransformFormat transformFormat

    cdef struct OptixBuildInput:
        OptixBuildInputType type
        # union
        OptixBuildInputTriangleArray triangleArray
        OptixBuildInputCurveArray curveArray
        OptixBuildInputCustomPrimitiveArray customPrimitiveArray
        OptixBuildInputInstanceArray instanceArray

    cdef struct OptixAccelBufferSizes:
        size_t outputSizeInBytes
        size_t tempSizeInBytes
        size_t tempUpdateSizeInBytes

    cdef enum OptixAccelPropertyType:
        OPTIX_PROPERTY_TYPE_COMPACTED_SIZE,
        OPTIX_PROPERTY_TYPE_AABBS,

    cdef struct OptixAccelEmitDesc:
        CUdeviceptr result
        OptixAccelPropertyType type

    ctypedef size_t CUstream

    ctypedef unsigned long long OptixTraversableHandle

    cdef struct OptixAccelRelocationInfo:
        unsigned long long info[4]

    cdef enum OptixTraversableType:
        OPTIX_TRAVERSABLE_TYPE_STATIC_TRANSFORM,
        OPTIX_TRAVERSABLE_TYPE_MATRIX_MOTION_TRANSFORM,
        OPTIX_TRAVERSABLE_TYPE_SRT_MOTION_TRANSFORM,

    cdef enum OptixInstanceFlags:
        OPTIX_INSTANCE_FLAG_NONE
        OPTIX_INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_CULLING
        OPTIX_INSTANCE_FLAG_FLIP_TRIANGLE_FACING
        OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT
        OPTIX_INSTANCE_FLAG_ENFORCE_ANYHIT
        OPTIX_INSTANCE_FLAG_DISABLE_TRANSFORM

    cdef struct OptixInstance:
        float transform [12]
        unsigned int instanceId
        unsigned int sbtOffset
        unsigned int visibilityMask
        unsigned int flags
        OptixTraversableHandle traversableHandle


    OptixResult optixAccelComputeMemoryUsage(OptixDeviceContext context,
                                 const OptixAccelBuildOptions * accelOptions,
                                 const OptixBuildInput * buildInputs,
                                 unsigned int numBuildInputs,
                                 OptixAccelBufferSizes * bufferSizes
                                 )


    OptixResult optixAccelBuild(OptixDeviceContext context,
                    CUstream stream,
                    const OptixAccelBuildOptions * accelOptions,
                    const OptixBuildInput * buildInputs,
                    unsigned int numBuildInputs,
                    CUdeviceptr tempBuffer,
                    size_t tempBufferSizeInBytes,
                    CUdeviceptr outputBuffer,
                    size_t outputBufferSizeInBytes,
                    OptixTraversableHandle * outputHandle,
                    const OptixAccelEmitDesc * emittedProperties,
                    unsigned int numEmittedProperties
                    )

    OptixResult optixAccelCompact(OptixDeviceContext context,
                                  CUstream stream,
                                  OptixTraversableHandle inputHandle,
                                  CUdeviceptr outputBuffer,
                                  size_t outputBufferSizeInBytes,
                                  OptixTraversableHandle * outputHandle
                                  )

    OptixResult optixAccelRelocate(OptixDeviceContext context,
                       CUstream stream,
                       const OptixAccelRelocationInfo * info,
                       CUdeviceptr instanceTraversableHandles,
                       size_t numInstanceTraversableHandles,
                       CUdeviceptr targetAccel,
                       size_t targetAccelSizeInBytes,
                       OptixTraversableHandle * targetHandle
                       )

    OptixResult optixAccelCheckRelocationCompatibility(OptixDeviceContext context,
                                           const OptixAccelRelocationInfo * info,
                                           int * compatible
                                           )

    OptixResult optixAccelGetRelocationInfo(OptixDeviceContext context,
                                OptixTraversableHandle handle,
                                OptixAccelRelocationInfo * info
                                )

    OptixResult optixConvertPointerToTraversableHandle(OptixDeviceContext onDevice,
                                           CUdeviceptr pointer,
                                           OptixTraversableType traversableType,
                                           OptixTraversableHandle * traversableHandle
                                           )


# cdef extern from "cuda_runtime.h":
#     ctypedef enum cudaError_t:
#         pass
#
#     cdef enum cudaMemcpyKind:
#         cudaMemcpyHostToHost,
#         cudaMemcpyHostToDevice,
#         cudaMemcpyDeviceToHost,
#         cudaMemcpyDeviceToDevice,
#         cudaMemcpyDefault
#
#     cudaError_t cudaMemcpy( void* dst, const void* src, size_t count, cudaMemcpyKind kind )


cdef class BuildInputArray:
    cdef void prepare_build_input(self, OptixBuildInput* build_input) except *


cdef class BuildInputTriangleArray(BuildInputArray):
    cdef OptixBuildInputTriangleArray _build_input
    cdef list _d_vertex_buffers
    cdef vector[CUdeviceptr] _d_vertex_buffer_ptrs
    cdef object _d_index_buffer
    cdef object _d_sbt_offset_buffer
    cdef object _d_pre_transform
    cdef vector[unsigned int] _flags

    cdef void prepare_build_input(self, OptixBuildInput * build_input) except *

cdef class BuildInputCustomPrimitiveArray(BuildInputArray):
    cdef OptixBuildInputCustomPrimitiveArray _build_input
    cdef list _d_aabb_buffers
    cdef vector[CUdeviceptr] _d_aabb_buffer_ptrs
    cdef object _d_sbt_offset_buffer
    cdef object _d_pre_transform
    cdef vector[unsigned int] _flags


cdef class BuildInputCurveArray(BuildInputArray):
    cdef OptixBuildInputCurveArray _build_input
    cdef list _d_vertex_buffers
    cdef vector[CUdeviceptr] _d_vertex_buffer_ptrs
    cdef list _d_width_buffers
    cdef vector[CUdeviceptr] _d_width_buffer_ptrs
    cdef list _d_normal_buffers
    cdef vector[CUdeviceptr] _d_normal_buffer_ptrs
    cdef object _d_index_buffer


cdef class Instance:
    cdef OptixInstance _instance
    cdef object _traversable


cdef class BuildInputInstanceArray(BuildInputArray):
    cdef OptixBuildInputInstanceArray _build_input
    cdef object _d_instances


cdef class AccelerationStructure:
    cdef DeviceContext context
    cdef unsigned int _build_flags
    cdef object _gas_buffer
    cdef OptixTraversableHandle _handle
    cpdef size_t c_obj(self)