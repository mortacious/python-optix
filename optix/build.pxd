from .common cimport OptixResult, CUstream, CUdeviceptr
from .context cimport OptixDeviceContext, OptixContextObject
from libcpp.vector cimport vector
from .base cimport OptixObject
from libc.stdint cimport uintptr_t, uint32_t
from .opacity_micromap cimport OptixBuildInputOpacityMicromap, BuildInputOpacityMicromap

cdef extern from "optix.h" nogil:
    cdef enum OptixBuildFlags:
        OPTIX_BUILD_FLAG_NONE,
        OPTIX_BUILD_FLAG_ALLOW_UPDATE,
        OPTIX_BUILD_FLAG_ALLOW_COMPACTION,
        OPTIX_BUILD_FLAG_PREFER_FAST_TRACE,
        OPTIX_BUILD_FLAG_PREFER_FAST_BUILD,
        OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS,
        OPTIX_BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS,
        OPTIX_BUILD_FLAG_ALLOW_OPACITY_MICROMAP_UPDATE,
        OPTIX_BUILD_FLAG_ALLOW_DISABLE_OPACITY_MICROMAPS


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
        OPTIX_BUILD_INPUT_TYPE_CURVES,
        OPTIX_BUILD_INPUT_TYPE_SPHERES


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
        OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM,
        OPTIX_PRIMITIVE_TYPE_FLAT_QUADRATIC_BSPLINE,
        OPTIX_PRIMITIVE_TYPE_SPHERE,
        OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BEZIER,
        OPTIX_PRIMITIVE_TYPE_TRIANGLE,


    cdef enum OptixCurveEndcapFlags:
        OPTIX_CURVE_ENDCAP_DEFAULT,
        OPTIX_CURVE_ENDCAP_ON


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
        unsigned int endcapFlags


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
        OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING


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
        OptixBuildInputOpacityMicromap opacityMicromap


    cdef struct OptixBuildInputSphereArray:
        const CUdeviceptr* vertexBuffers
        unsigned int vertexStrideInBytes
        unsigned int numVertices
        const CUdeviceptr *radiusBuffers
        unsigned int radiusStrideInBytes
        int singleRadius
        const unsigned int *flags
        unsigned int numSbtRecords
        CUdeviceptr sbtIndexOffsetBuffer
        unsigned int sbtIndexOffsetSizeInBytes
        unsigned int sbtIndexOffsetStrideInBytes
        unsigned int primitiveIndexOffset


    cdef struct OptixBuildInput:
        OptixBuildInputType type
        # union
        OptixBuildInputTriangleArray triangleArray
        OptixBuildInputCurveArray curveArray
        OptixBuildInputSphereArray   sphereArray
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


    ctypedef uintptr_t OptixTraversableHandle


    cdef struct OptixRelocationInfo:
        unsigned long long[4] info


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
        OPTIX_INSTANCE_FLAG_FORCE_OPACITY_MICROMAP_2_STATE
        OPTIX_INSTANCE_FLAG_DISABLE_OPACITY_MICROMAPS


    cdef struct OptixInstance:
        float[12] transform
        unsigned int instanceId
        unsigned int sbtOffset
        unsigned int visibilityMask
        unsigned int flags
        OptixTraversableHandle traversableHandle


    cdef struct OptixRelocateInputInstanceArray:
        unsigned int numInstances
        CUdeviceptr traversableHandles


    cdef struct OptixRelocateInputOpacityMicromap:
        CUdeviceptr opacityMicromapArray


    cdef struct OptixRelocateInputTriangleArray:
        unsigned int numSbtRecords
        OptixRelocateInputOpacityMicromap opacityMicromap


    cdef struct OptixRelocateInput:
        OptixBuildInputType type
        OptixRelocateInputInstanceArray instanceArray
        OptixRelocateInputTriangleArray triangleArray


    # Cluster API
    cdef enum OptixClusterAccelBuildMode:
        OPTIX_CLUSTER_ACCEL_BUILD_MODE_IMPLICIT_DESTINATIONS
        OPTIX_CLUSTER_ACCEL_BUILD_MODE_EXPLICIT_DESTINATIONS
        OPTIX_CLUSTER_ACCEL_BUILD_MODE_GET_SIZES


    cdef struct OptixClusterAccelBuildModeDescImplicitDest:
        CUdeviceptr outputBuffer
        size_t outputBufferSizeInBytes
        CUdeviceptr tempBuffer
        size_t tempBufferSizeInBytes
        CUdeviceptr outputHandlesBuffer
        unsigned int outputHandlesStrideInBytes
        CUdeviceptr outputSizesBuffer
        unsigned int outputSizesStrideInBytes


    cdef struct OptixClusterAccelBuildModeDescExplicitDest:
        CUdeviceptr tempBuffer
        size_t 	tempBufferSizeInBytes
        CUdeviceptr destAddressesBuffer
        unsigned int destAddressesStrideInBytes
        CUdeviceptr outputHandlesBuffer
        unsigned int outputHandlesStrideInBytes
        CUdeviceptr outputSizesBuffer
        unsigned int outputSizesStrideInBytes


    cdef struct OptixClusterAccelBuildModeDescGetSize:
        CUdeviceptr outputSizesBuffer
        unsigned int outputSizesStrideInBytes
        CUdeviceptr tempBuffer
        size_t tempBufferSizeInBytes


    cdef struct OptixClusterAccelBuildModeDesc:
        OptixClusterAccelBuildMode mode
        # union
        OptixClusterAccelBuildModeDescImplicitDest implicitDest
        OptixClusterAccelBuildModeDescExplicitDest explicitDest
        OptixClusterAccelBuildModeDescGetSize getSize

    
    cdef enum OptixClusterAccelBuildFlags:
        OPTIX_CLUSTER_ACCEL_BUILD_FLAG_NONE
        OPTIX_CLUSTER_ACCEL_BUILD_FLAG_PREFER_FAST_TRACE
        OPTIX_CLUSTER_ACCEL_BUILD_FLAG_PREFER_FAST_BUILD
        OPTIX_CLUSTER_ACCEL_BUILD_FLAG_ALLOW_OPACITY_MICROMAPS

    cdef struct OptixClusterAccelBuildInputTriangles:
        OptixClusterAccelBuildFlags flags
        unsigned int maxArgCount
        OptixVertexFormat vertexFormat
        unsigned int maxSbtIndexValue
        unsigned int maxUniqueSbtIndexCountPerArg
        unsigned int maxTriangleCountPerArg
        unsigned int maxVertexCountPerArg
        unsigned int maxTotalTriangleCount
        unsigned int maxTotalVertexCount
        unsigned int minPositionTruncateBitCount


    cdef struct OptixClusterAccelBuildInputGrids:
        OptixClusterAccelBuildFlags flags
        unsigned int maxArgCount
        OptixVertexFormat vertexFormat
        unsigned int maxSbtIndexValue
        unsigned int maxWidth
        unsigned int maxHeight

    
    cdef struct OptixClusterAccelBuildInputClusters:
        OptixClusterAccelBuildFlags flags
        unsigned int maxArgCount
        unsigned int maxTotalClusterCount
        unsigned int maxClusterCountPerArg


    cdef struct OptixClusterAccelPrimitiveInfo:
        unsigned int sbtIndex # 24 bits
        unsigned int reserved # 5 bits
        unsigned int primitiveFlags # 3 bits

    
    cdef struct OptixClusterAccelBuildInputTrianglesArgs:
        unsigned int clusterId
        unsigned int clusterFlags
        unsigned int triangleCount # 9 bits
        unsigned int vertexCount # 9 bits
        unsigned int positionTruncateBitCount # 6 bits
        unsigned int indexFormat # 4 bits
        unsigned int opacityMicromapIndexFormat # 4 bits
        OptixClusterAccelPrimitiveInfo basePrimitiveInfo
        unsigned short indexBufferStrideInBytes
        unsigned short vertexBufferStrideInBytes
        unsigned short primitiveInfoBufferStrideInBytes
        unsigned short opacityMicromapIndexBufferStrideInBytes
        CUdeviceptr indexBuffer
        CUdeviceptr vertexBuffer
        CUdeviceptr primitiveInfoBuffer
        CUdeviceptr opacityMicromapArray
        CUdeviceptr opacityMicromapIndexBuffer
        CUdeviceptr instantiationBoundingBoxLimit

    cdef struct OptixClusterAccelBuildInputGridsArgs:
        unsigned int baseClusterId
        unsigned int clusterFlags
        OptixClusterAccelPrimitiveInfo basePrimitiveInfo
        unsigned int positionTruncateBitCount # 6 bits
        unsigned int reserved # 26 bits
        unsigned char[2] dimensions
        unsigned short reserved2


    cdef struct OptixClusterAccelBuildInputTemplatesArgs:
        unsigned int clusterIdOffset
        unsigned int sbtIndexOffset
        CUdeviceptr clusterTemplate
        CUdeviceptr vertexBuffer
        unsigned int vertexStrideInBytes
        unsigned int reserved


    cdef struct OptixClusterAccelBuildInputClustersArgs:
        unsigned int clusterHandlesCount
        unsigned int clusterHandlesBufferStrideInBytes
        CUdeviceptr clusterHandlesBuffer


    cdef enum OptixClusterAccelBuildType:
        OPTIX_CLUSTER_ACCEL_BUILD_TYPE_GASES_FROM_CLUSTERS 	
        OPTIX_CLUSTER_ACCEL_BUILD_TYPE_CLUSTERS_FROM_TRIANGLES 	
        OPTIX_CLUSTER_ACCEL_BUILD_TYPE_TEMPLATES_FROM_TRIANGLES 	
        OPTIX_CLUSTER_ACCEL_BUILD_TYPE_CLUSTERS_FROM_TEMPLATES 	
        OPTIX_CLUSTER_ACCEL_BUILD_TYPE_TEMPLATES_FROM_GRIDS 


    cdef struct OptixClusterAccelBuildInput:
        OptixClusterAccelBuildType type
        # union
        OptixClusterAccelBuildInputClusters clusters
        OptixClusterAccelBuildInputTriangles triangles
        OptixClusterAccelBuildInputGrids grids


    cdef enum OptixDevicePropertyClusterAccelFlags:
        OPTIX_DEVICE_PROPERTY_CLUSTER_ACCEL_FLAG_NONE
        OPTIX_DEVICE_PROPERTY_CLUSTER_ACCEL_FLAG_STANDARD


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
                       const OptixRelocationInfo * info,
                       const OptixRelocateInput * relocateInputs,
                       size_t numRelocateInputs,
                       CUdeviceptr targetAccel,
                       size_t targetAccelSizeInBytes,
                       OptixTraversableHandle * targetHandle
                       )


    OptixResult optixCheckRelocationCompatibility(OptixDeviceContext context,
                                           const OptixRelocationInfo * info,
                                           int * compatible
                                           )


    OptixResult optixAccelGetRelocationInfo(OptixDeviceContext context,
                                OptixTraversableHandle handle,
                                OptixRelocationInfo * info
                                )


    OptixResult optixConvertPointerToTraversableHandle(OptixDeviceContext onDevice,
                                           CUdeviceptr pointer,
                                           OptixTraversableType traversableType,
                                           OptixTraversableHandle * traversableHandle
                                           )


    OptixResult optixAccelEmitProperty (OptixDeviceContext context,
                                        CUstream stream,
                                        OptixTraversableHandle handle,
                                        const OptixAccelEmitDesc * emittedProperty
                                        )
    
    # cluster API
    OptixResult optixClusterAccelBuild(OptixDeviceContext context,
                                       CUstream stream,
                                       const OptixClusterAccelBuildModeDesc* buildModeDesc,
                                       const OptixClusterAccelBuildInput* buildInput,
                                       CUdeviceptr argsArray,
                                       CUdeviceptr argsCount,
                                       unsigned int argsStrideInBytes
                                       )


    OptixResult optixClusterAccelComputeMemoryUsage(OptixDeviceContext context,
		                                            OptixClusterAccelBuildMode buildMode,
		                                            const OptixClusterAccelBuildInput* buildInput,
		                                            OptixAccelBufferSizes* bufferSizes
                                                    ) 	
                            


cdef class BuildInputArray(OptixObject):
    cdef OptixBuildInputType build_input_type
    cdef void prepare_build_input(self, OptixBuildInput* build_input) except *
    cdef size_t num_elements(self)


cdef class BuildInputTriangleArray(BuildInputArray):
    cdef OptixBuildInputTriangleArray build_input
    cdef list _d_vertex_buffers
    cdef vector[CUdeviceptr] _d_vertex_buffer_ptrs
    cdef object _d_index_buffer
    cdef object _d_sbt_offset_buffer
    cdef object _d_pre_transform
    cdef vector[unsigned int] _flags
    cdef BuildInputOpacityMicromap c_opacity_micromap


cdef class BuildInputCustomPrimitiveArray(BuildInputArray):
    cdef OptixBuildInputCustomPrimitiveArray build_input
    cdef list _d_aabb_buffers
    cdef vector[CUdeviceptr] _d_aabb_buffer_ptrs
    cdef object _d_sbt_offset_buffer
    cdef object _d_pre_transform
    cdef vector[unsigned int] _flags


cdef class BuildInputCurveArray(BuildInputArray):
    cdef OptixBuildInputCurveArray build_input
    cdef list _d_vertex_buffers
    cdef vector[CUdeviceptr] _d_vertex_buffer_ptrs
    cdef list _d_width_buffers
    cdef vector[CUdeviceptr] _d_width_buffer_ptrs
    cdef list _d_normal_buffers
    cdef vector[CUdeviceptr] _d_normal_buffer_ptrs
    cdef object _d_index_buffer


cdef class BuildInputSphereArray(BuildInputArray):
    cdef OptixBuildInputSphereArray build_input
    cdef list _d_vertex_buffers
    cdef vector[CUdeviceptr] _d_vertex_buffer_ptrs
    cdef list _d_radius_buffers
    cdef vector[CUdeviceptr] _d_radius_buffer_ptrs
    cdef object _d_sbt_offset_buffer
    cdef vector[unsigned int] _flags


cdef class Instance(OptixObject):
    cdef OptixInstance instance
    cdef AccelerationStructure _traversable


cdef class BuildInputInstanceArray(BuildInputArray):
    cdef OptixBuildInputInstanceArray build_input
    cdef object instances
    cdef object _d_instances


cdef class MotionOptions(OptixObject):
    cdef OptixMotionOptions options


cdef class AccelerationStructure(OptixContextObject):
    cdef unsigned int _build_flags
    cdef object _gas_buffer
    cdef size_t _num_elements
    cdef OptixAccelBufferSizes _buffer_sizes
    cdef object _instances
    cdef OptixTraversableHandle _handle
    cdef list _relocate_deps

    cdef void _init_build_inputs(self, build_inputs, vector[OptixBuildInput]& ret)
    cdef void _init_accel_options(self, size_t num_build_inputs, unsigned int build_flags, OptixBuildOperation operation, vector[OptixAccelBuildOptions]& ret)
    cdef void build(self, build_inputs, stream=*)
