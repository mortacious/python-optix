# distutils: language = c++

from .common cimport optix_check_return, optix_init
from .context cimport DeviceContext
import cupy as cp
import numpy as np
from enum import IntEnum, IntFlag
from libc.string cimport memcpy, memset
from libcpp.vector cimport vector
from .common import round_up

optix_init()

__all__ = ['GeometryFlags',
           'PrimitiveType',
           'InstanceFlags',
           'BuildInputTriangleArray',
           'BuildInputCustomPrimitiveArray',
           'BuildInputCurveArray',
           'BuildInputInstanceArray',
           'Instance',
           'AccelerationStructure'
           ]

class GeometryFlags(IntEnum):
    NONE = OPTIX_GEOMETRY_FLAG_NONE,
    DISABLE_ANYHIT = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
    REQUIRE_SINGLE_ANYHIT_CALL = OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL


class PrimitiveType(IntEnum):
    CUSTOM = OPTIX_PRIMITIVE_TYPE_CUSTOM,
    ROUND_QUADRATIC_BSPLINE = OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE,
    ROUND_CUBIC_BSPLINE = OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE,
    ROUND_LINEAR = OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR,
    TRIANGLE = OPTIX_PRIMITIVE_TYPE_TRIANGLE

class InstanceFlags(IntFlag):
    NONE = OPTIX_INSTANCE_FLAG_NONE,
    DISABLE_TRIANGLE_FACE_CULLING = OPTIX_INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_CULLING,
    FLIP_TRIANGLE_FACING = OPTIX_INSTANCE_FLAG_FLIP_TRIANGLE_FACING,
    DISABLE_ANYHIT = OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT,
    ENFORCE_ANYHIT = OPTIX_INSTANCE_FLAG_ENFORCE_ANYHIT,
    DISABLE_TRANSFORM = OPTIX_INSTANCE_FLAG_DISABLE_TRANSFORM


cdef class BuildInputArray:
    cdef void prepare_build_input(self, OptixBuildInput* build_input) except *:
        pass


cdef class BuildInputTriangleArray(BuildInputArray):
    def __init__(self,
                 vertex_buffers,
                 index_buffer = None,
                 num_sbt_records = 1,
                 flags = None,
                 sbt_record_offset_buffer = None,
                 pre_transform = None,
                 primitive_index_offset = 0
                 ):
        self._d_vertex_buffers = [cp.asarray(vb) for vb in vertex_buffers]
        self._d_vertex_buffer_ptrs.reserve(len(self._d_vertex_buffers))

        if len(self._d_vertex_buffers) > 0:
            dtype = self._d_vertex_buffers[0].dtype
            shape = self._d_vertex_buffers[0].shape
            strides = self._d_vertex_buffers[0].strides
            for vb in self._d_vertex_buffers:
                if vb.dtype != dtype or vb.shape != shape or vb.strides != strides:
                    raise ValueError("All vertex buffers must have the same size and dtype")
                self._d_vertex_buffer_ptrs.push_back(vb.data.ptr)

            self._build_input.vertexBuffers = self._d_vertex_buffer_ptrs.const_data()
            self._build_input.vertexFormat = self._vertex_format(dtype, shape)
            self._build_input.vertexStrideInBytes = self._d_vertex_buffers[0].strides[0]
            self._build_input.numVertices = shape[0]
        else:
            self._build_input.vertexBuffers = NULL
            self._build_input.vertexFormat = OPTIX_VERTEX_FORMAT_NONE
            self._build_input.vertexStrideInBytes = 0
            self._build_input.numVertices = 0

        if index_buffer is not None:
            self._d_index_buffer = cp.asarray(index_buffer)
            self._build_input.indexFormat = self._index_format(self._d_index_buffer.dtype, self._d_index_buffer.shape)
            self._build_input.indexStrideInBytes = self._d_index_buffer.strides[0]
            self._build_input.numIndexTriplets = self._d_index_buffer.shape[0]
            self._build_input.indexBuffer = self._d_index_buffer.data.ptr
        else:
            self._build_input.indexFormat = OPTIX_INDICES_FORMAT_NONE
            self._build_input.indexStrideInBytes = 0
            self._build_input.numIndexTriplets = 0
            self._build_input.indexBuffer = 0

        self._build_input.numSbtRecords = num_sbt_records

        self._flags.resize(num_sbt_records)

        if flags is None:
            for i in range(num_sbt_records):
                self._flags[i] = OPTIX_GEOMETRY_FLAG_NONE
        else:
            for i in range(num_sbt_records):
                self._flags[i] = flags[i].value

        self._build_input.flags = self._flags.data()

        if sbt_record_offset_buffer is not None:
            self._d_sbt_offset_buffer = cp.asarray(sbt_record_offset_buffer).ravel()
            self._build_input.sbtIndexOffsetBuffer = self._d_sbt_offset_buffer.data.ptr
            itemsize = self._d_sbt_offset_buffer.itemsize
            if itemsize > 4:
                raise ValueError("Only 32 bit allowed at max")
            self._build_input.sbtIndexOffsetSizeInBytes = itemsize
            self._build_input.sbtIndexOffsetStrideInBytes = self._d_sbt_offset_buffer.strides[0]
        else:
            self._build_input.sbtIndexOffsetBuffer = 0
            self._build_input.sbtIndexOffsetStrideInBytes = 0
            self._build_input.sbtIndexOffsetSizeInBytes = 0

        self._build_input.primitiveIndexOffset = primitive_index_offset

        if pre_transform is not None:
            self._d_pre_transform = cp.asarray(pre_transform, dtype=np.float32).reshape(3, 4)
            self._build_input.preTransform = self._d_pre_transform.data.ptr
            self._build_input.transformFormat = OPTIX_TRANSFORM_FORMAT_MATRIX_FLOAT12
        else:
            self._build_input.preTransform = 0
            self._build_input.transformFormat = OPTIX_TRANSFORM_FORMAT_NONE

    def __dealloc__(self):
        pass

    cdef void prepare_build_input(self, OptixBuildInput* build_input) except *:
        build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES
        build_input.triangleArray = self._build_input

    def _vertex_format(self, dtype, shape):
        if dtype == np.float32:
            if shape[-1] == 3:
                return OPTIX_VERTEX_FORMAT_FLOAT3
            elif shape[-1] == 2:
                return OPTIX_VERTEX_FORMAT_FLOAT2
            else:
                raise ValueError("Unsupported shape")
        if dtype == np.float16:
            if shape[-1] == 3:
                return OPTIX_VERTEX_FORMAT_HALF3
            elif shape[-1] == 2:
                return OPTIX_VERTEX_FORMAT_HALF2
            else:
                raise ValueError("Unsupported shape")
        else:
            raise ValueError("Unsupported dtype")

    def _index_format(self, dtype, shape):
        if shape[-1] != 3:
            raise ValueError("Unsupported shape")
        if dtype == np.uint32:
            return OPTIX_INDICES_FORMAT_UNSIGNED_INT3
        elif dtype == np.uint16:
            return OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3
        else:
            raise ValueError("Unsupported dtype")


cdef class BuildInputCustomPrimitiveArray(BuildInputArray):
    def __init__(self,
                 aabb_buffers,
                 num_sbt_records = 1,
                 flags = None,
                 sbt_record_offset_buffer = None,
                 primitive_index_offset = 0
                 ):

        self._d_aabb_buffers = [cp.asarray(ab, dtype=np.float32).reshape(-1, 6) for ab in aabb_buffers]
        self._d_aabb_buffer_ptrs.reserve(len(self._d_aabb_buffers))

        dtype = self._d_aabb_buffers[0].dtype
        shape = self._d_aabb_buffers[0].shape
        strides = self._d_aabb_buffers[0].strides
        for ab in self._d_aabb_buffers:
            if ab.shape[-1] != 6:
                raise ValueError("Invalid shape of aabb buffer")
            if ab.dtype != dtype or ab.shape != shape or ab.strides != strides:
                raise ValueError("All aabb buffers must have the same size and dtype")
            self._d_aabb_buffer_ptrs.push_back(ab.data.ptr)

        self._build_input.aabbBuffers = self._d_aabb_buffer_ptrs.const_data()
        self._build_input.numPrimitives = shape[0]
        self._build_input.strideInBytes = 0#self._d_aabb_buffers[0].strides[0]

        self._flags.resize(num_sbt_records)
        if flags is None:
            for i in range(num_sbt_records):
                self._flags[i] = OPTIX_GEOMETRY_FLAG_NONE
        else:
            for i in range(num_sbt_records):
                self._flags[i] = flags[i].value

        self._build_input.flags = self._flags.const_data()

        self._build_input.numSbtRecords = num_sbt_records

        if sbt_record_offset_buffer is not None:
            self._d_sbt_offset_buffer = cp.asarray(sbt_record_offset_buffer).ravel()
            if not np.issubdtype(self._d_sbt_offset_buffer.dtype, np.unsignedinteger):
                self._d_sbt_offset_buffer = self._d_sbt_offset_buffer.astype(np.uint32)
            itemsize = self._d_sbt_offset_buffer.itemsize
            if itemsize > 4:
                raise ValueError("Only 32 bit allowed at max")

            self._build_input.sbtIndexOffsetBuffer = self._d_sbt_offset_buffer.data.ptr
            self._build_input.sbtIndexOffsetSizeInBytes = itemsize
            self._build_input.sbtIndexOffsetStrideInBytes = self._d_sbt_offset_buffer.strides[0]
        else:
            self._build_input.sbtIndexOffsetBuffer = 0
            self._build_input.sbtIndexOffsetStrideInBytes = 0
            self._build_input.sbtIndexOffsetSizeInBytes = 0

        self._build_input.primitiveIndexOffset = primitive_index_offset

    cdef void prepare_build_input(self, OptixBuildInput * build_input) except *:
        build_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES
        build_input.customPrimitiveArray = self._build_input


cdef class BuildInputCurveArray(BuildInputArray):
    def __init__(self, curve_type, vertex_buffers, width_buffers, normal_buffers, index_buffer, flags=None, primitive_index_offset=0):
        self._d_vertex_buffers = [cp.asarray(vb, np.float32) for vb in vertex_buffers]
        self._d_vertex_buffer_ptrs.reserve(len(self._d_vertex_buffers))

        shape = self._d_vertex_buffers[0].shape
        strides = self._d_vertex_buffers[0].strides
        for vb in self._d_vertex_buffers:
            if vb.shape != shape or vb.strides != strides:
                raise ValueError("All vertex buffers must have the same size and strides")
            self._d_vertex_buffer_ptrs.push_back(vb.data.ptr)

        self._build_input.vertexBuffers = self._d_vertex_buffer_ptrs.const_data()
        self._build_input.vertexStrideInBytes = self._d_vertex_buffers[0].strides[0]
        self._build_input.numVertices = shape[0]


        self._d_width_buffers = [cp.asarray(wb, np.uint32) for wb in width_buffers]
        assert len(self._d_width_buffers) == len(self._d_vertex_buffers)
        self._d_width_buffer_ptrs.reserve(len(self._d_width_buffers))

        shape = self._d_width_buffers[0].shape
        strides = self._d_width_buffers[0].strides
        for wb in self._d_width_buffers:
            if wb.shape != shape or wb.strides != strides:
                raise ValueError("All width buffers must have the same size and strides")
            self._d_width_buffer_ptrs.push_back(wb.data.ptr)

        self._build_input.widthBuffers = self._d_width_buffer_ptrs.const_data()
        self._build_input.widthStrideInBytes = self._d_width_buffers[0].strides[0]


        self._d_normal_buffers = [cp.asarray(nb, np.uint32) for nb in normal_buffers]
        assert len(self._d_normal_buffers) == len(self._d_vertex_buffers)
        self._d_normal_buffer_ptrs.reserve(len(self._d_normal_buffers))

        shape = self._d_normal_buffers[0].shape
        strides = self._d_normal_buffers[0].strides
        for nb in self._d_normal_buffers:
            if nb.shape != shape or nb.strides != strides:
                raise ValueError("All normal buffers must have the same size and strides")
            self._d_normal_buffer_ptrs.push_back(nb.data.ptr)

        self._build_input.normalBuffers = self._d_normal_buffer_ptrs.const_data()
        self._build_input.normalStrideInBytes = self._d_normal_buffers[0].strides[0]

        self._d_index_buffer = cp.asarray(index_buffer, dtype=np.uint32)
        self._build_input.indexStrideInBytes = self._d_index_buffer.strides[0]
        self._build_input.indexBuffer = self._d_index_buffer.data.ptr

        if flags is None:
            self._build_input.flag = OPTIX_GEOMETRY_FLAG_NONE
        else:
            self._build_input.flag = flags.valuesbtIndexOffsetBuffer

        self._build_input.primitiveIndexOffset = primitive_index_offset

    cdef void prepare_build_input(self, OptixBuildInput * build_input) except *:
        build_input.type = OPTIX_BUILD_INPUT_TYPE_CURVES
        build_input.curveArray = self._build_input


cdef class Instance:
    def __init__(self, AccelerationStructure traversable, unsigned int instance_id, flags = InstanceFlags.NONE, unsigned int sbt_offset = 0, transform=None, visibility_mask=None):
        if transform is None:
            transform = np.eye(3, 4, dtype=np.float32)
        transform = np.asarray(transform, dtype=np.float32).reshape(3,4)
        cdef float[:, ::1] c_transform = transform
        memcpy(&self._instance.transform, &c_transform[0, 0], sizeof(float)*12)
        self._traversable = traversable
        self._instance.traversableHandle = self._traversable._handle
        self._instance.instanceId = instance_id
        self._instance.flags = flags.value
        self._instance.sbtOffset = sbt_offset
        visibility_mask = int(visibility_mask)
        if visibility_mask.bit_length() > self._traversable.context.num_bits_instances_visibility_mask:
            raise ValueError(f"Too many entries in visibility mask. Got {visibility_mask.bit_length()} but supported are only {self._traversable.context.num_bits_instances_visibility_mask}")
        self._instance.visibilityMask = visibility_mask


cdef class BuildInputInstanceArray(BuildInputArray):
    def __init__(self, instances):
        cdef ssize_t size = sizeof(OptixInstance)*len(instances)
        self._d_instances = cp.cuda.alloc(size)
        cdef vector[OptixInstance] c_instances
        c_instances.reserve(len(instances))
        for inst in instances:
            c_instances.push_back(inst._instance)

        cp.cuda.runtime.memcpy(self._d_instances.ptr, <size_t>c_instances.data(), size, cp.cuda.runtime.memcpyHostToDevice)
        #cudaMemcpy(<void*>self._d_instances.ptr, c_instances.const_data(), size, cudaMemcpyHostToDevice)
        self._build_input.instances = self._d_instances.ptr
        self._build_input.numInstances = len(instances)


    cdef void prepare_build_input(self, OptixBuildInput * build_input) except *:
        build_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES
        build_input.instanceArray = self.build_input


cdef class AccelerationStructure(OptixObject):
    def __init__(self, DeviceContext context, build_inputs, compact=True, allow_update=False, prefer_fast_build=False, random_vertex_access=False, random_instance_access=False):
        super().__init__(context)
        self._build_flags = OPTIX_BUILD_FLAG_NONE
        if compact:
            self._build_flags |= OPTIX_BUILD_FLAG_ALLOW_COMPACTION
        # if allow_update:
        #     self._build_flags |= OPTIX_BUILD_FLAG_ALLOW_UPDATE
        # if prefer_fast_build:
        #     self._build_flags |= OPTIX_BUILD_FLAG_PREFER_FAST_BUILD
        #     self._build_flags &= ~OPTIX_BUILD_FLAG_PREFER_FAST_TRACE
        # else:
        #     self._build_flags |= OPTIX_BUILD_FLAG_PREFER_FAST_TRACE
        #     self._build_flags &= ~OPTIX_BUILD_FLAG_PREFER_FAST_BUILD
        # if random_vertex_access:
        #     self._build_flags |= OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS
        # if random_instance_access:
        #     self._build_flags |= OPTIX_BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS

        self._gas_buffer = None
        self.build(build_inputs)

    @property
    def compact(self):
        return self._build_flags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION

    @property
    def allow_update(self):
        return self._build_flags & OPTIX_BUILD_FLAG_ALLOW_UPDATE

    def build(self, build_inputs):
        # build a single vector from all the build inputs
        cdef size_t inputs_size = len(build_inputs)
        cdef vector[OptixBuildInput] inputs = vector[OptixBuildInput](inputs_size)

        print("build inputs", build_inputs, inputs_size)
        cdef OptixAccelBuildOptions accel_option
        memset(&accel_option, 0, sizeof(OptixAccelBuildOptions))
        accel_option.buildFlags = self._build_flags
        accel_option.operation = OPTIX_BUILD_OPERATION_BUILD

        for i, build_input in enumerate(build_inputs):
            (<BuildInputArray>build_input).prepare_build_input(&inputs[i])

        print("inputs", hex(inputs[0].type), inputs[0].customPrimitiveArray.numPrimitives, inputs[0].customPrimitiveArray.flags[0])
        cdef vector[OptixAccelBuildOptions] accel_options = vector[OptixAccelBuildOptions](inputs_size)
        for i in range(inputs_size):
            accel_options[i] = accel_option

        cdef OptixAccelBufferSizes gas_buffer_sizes = []
        cdef OptixAccelEmitDesc compacted_size_property = []

        cdef CUdeviceptr gas_buffer_ptr
        cdef CUdeviceptr tmp_gas_buffer_ptr
        cdef unsigned int num_properties = 0

        optix_check_return(optixAccelComputeMemoryUsage(self.context.c_context,
                                                        accel_options.data(),
                                                        inputs.data(),
                                                        inputs_size,
                                                        &gas_buffer_sizes))

        print("memory sizes", gas_buffer_sizes.tempSizeInBytes, gas_buffer_sizes.outputSizeInBytes, gas_buffer_sizes.tempUpdateSizeInBytes)

        d_tmp_gas_buffer = cp.cuda.alloc(gas_buffer_sizes.tempSizeInBytes * 2)
        self._gas_buffer = cp.cuda.alloc(round_up(gas_buffer_sizes.outputSizeInBytes, 8) + 8)

        d_compacted_size = None
        cdef OptixAccelEmitDesc* property_ptr = NULL

        if self.compact:
            print("compacting...")
            d_compacted_size = cp.zeros(1, dtype=np.uint64)
            compacted_size_property.result = d_compacted_size.data.ptr
            compacted_size_property.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE
            num_properties = 1
            property_ptr = &compacted_size_property

        #cdef const OptixAccelBuildOptions* build_options_ptr = (<vector[OptixAccelBuildOptions]>accel_options_vector).const_data()
        gas_buffer_ptr = self._gas_buffer.ptr
        tmp_gas_buffer_ptr = d_tmp_gas_buffer.ptr

        stream = cp.cuda.Stream()

        print("building acceleration structure")
        #print("stream", stream.ptr)
        print("accel options", accel_options[0].buildFlags, accel_options[0].operation,
              accel_options[0].motionOptions.flags, accel_options[0].motionOptions.numKeys, accel_options[0].motionOptions.timeBegin, accel_options[0].motionOptions.timeEnd)

        print("inputs",
              hex(inputs[0].type),
              <size_t>inputs[0].customPrimitiveArray.aabbBuffers,
              inputs[0].customPrimitiveArray.numPrimitives,
              inputs[0].customPrimitiveArray.strideInBytes,
              <size_t>inputs[0].customPrimitiveArray.flags,
              inputs[0].customPrimitiveArray.flags[0],
              inputs[0].customPrimitiveArray.numSbtRecords,
              inputs[0].customPrimitiveArray.sbtIndexOffsetBuffer,
              inputs[0].customPrimitiveArray.sbtIndexOffsetSizeInBytes,
              inputs[0].customPrimitiveArray.sbtIndexOffsetStrideInBytes,
              inputs[0].customPrimitiveArray.primitiveIndexOffset)
        print("inputs size", inputs_size)
        print("temp Buffer", tmp_gas_buffer_ptr)
        print("temp size", gas_buffer_sizes.tempSizeInBytes)
        print("gas output buffer", gas_buffer_ptr)
        print("output size", gas_buffer_sizes.outputSizeInBytes)
        print("handle", self._handle)
        print("property ptr", <size_t>property_ptr)
        print("property type", compacted_size_property.result, compacted_size_property.type)
        print("num properties", num_properties)
        # print("building acceleration structure",
        #       <size_t>accel_options.data(),
        #       <size_t>inputs.data(),
        #       inputs_size,
        #       tmp_gas_buffer_ptr,
        #       gas_buffer_sizes.tempSizeInBytes * 2,
        #       gas_buffer_ptr,
        #       gas_buffer_sizes.outputSizeInBytes,
        #       self._handle,
        #       <size_t>property_ptr,
        #       compacted_size_property.result, compacted_size_property.type, num_properties)

        cdef size_t c_stream = stream.ptr

        # build acceleration structure without the gil
        optix_check_return(optixAccelBuild(self.context.c_context,
                                          <CUstream>c_stream,
                                          accel_options.data(),
                                          inputs.data(),
                                          inputs_size,
                                          tmp_gas_buffer_ptr,
                                          gas_buffer_sizes.tempSizeInBytes,
                                          gas_buffer_ptr,
                                          gas_buffer_sizes.outputSizeInBytes,
                                          &self._handle,
                                          property_ptr,
                                          num_properties))
        stream.synchronize()
        print("done")
        cdef size_t compacted_gas_size = 0

        if self.compact:
            compacted_gas_size = cp.asnumpy(d_compacted_size)[0]
            if compacted_gas_size < gas_buffer_sizes.outputSizeInBytes:
                # compact the acceleration structure
                d_gas_output_buffer = cp.cuda.alloc(compacted_gas_size)
                gas_buffer_ptr = d_gas_output_buffer.ptr
                with nogil:
                    optix_check_return(optixAccelCompact(self.context.c_context,
                                                         0,  # TODO use actual cuda stream
                                                         self._handle,
                                                         gas_buffer_ptr,
                                                         compacted_gas_size,
                                                         &self._handle))
                self._gas_buffer = d_gas_output_buffer # keep the new buffer instead of the old one

    def update(self, build_input):
        raise NotImplementedError()

    @property
    def handle(self):
        return <OptixTraversableHandle>self._handle

