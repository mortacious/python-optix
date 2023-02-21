# distutils: language = c++

from .common cimport optix_check_return, optix_init
from .context cimport DeviceContext
import cupy as cp
import numpy as np
from enum import IntEnum, IntFlag
from libc.string cimport memcpy, memset
from libcpp.vector cimport vector
from .common import round_up, ensure_iterable
import typing as typ
from .micromap cimport BuildInputOpacityMicromap, OpacityMicromapArray

optix_init()

__all__ = ['GeometryFlags',
           'PrimitiveType',
           'InstanceFlags',
           'BuildInputTriangleArray',
           'BuildInputCustomPrimitiveArray',
           'BuildInputCurveArray',
           'BuildInputSphereArray',
           'BuildInputInstanceArray',
           'Instance',
           'AccelerationStructure',
           'CurveEndcapFlags',
           ]


class GeometryFlags(IntEnum):
    """
    Wraps the OptixGeometryFlags enum.
    """
    NONE = OPTIX_GEOMETRY_FLAG_NONE,
    DISABLE_ANYHIT = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
    REQUIRE_SINGLE_ANYHIT_CALL = OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL
    DISABLE_TRIANGLE_FACE_CULLING = OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING


class BuildFlags(IntFlag):
    """
    Wraps the OptixBuildFlags enum
    """
    NONE = OPTIX_BUILD_FLAG_NONE,
    ALLOW_UPDATE = OPTIX_BUILD_FLAG_ALLOW_UPDATE,
    ALLOW_COMPACTION = OPTIX_BUILD_FLAG_ALLOW_COMPACTION,
    PREFER_FAST_TRACE = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE,
    PREFER_FAST_BUILD = OPTIX_BUILD_FLAG_PREFER_FAST_BUILD,
    ALLOW_RANDOM_VERTEX_ACCESS = OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS,
    ALLOW_RANDOM_INSTANCE_ACCESS = OPTIX_BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS,


class PrimitiveType(IntEnum):
    """
    Wraps the OptixPrimitiveType enum.
    """
    CUSTOM = OPTIX_PRIMITIVE_TYPE_CUSTOM,
    ROUND_QUADRATIC_BSPLINE = OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE,
    ROUND_CUBIC_BSPLINE = OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE,
    ROUND_LINEAR = OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR
    ROUND_CATMULLROM = OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM
    SPHERE = OPTIX_PRIMITIVE_TYPE_SPHERE
    TRIANGLE = OPTIX_PRIMITIVE_TYPE_TRIANGLE


class CurveEndcapFlags(IntEnum):
    DEFAULT = OPTIX_CURVE_ENDCAP_DEFAULT,
    ON = OPTIX_CURVE_ENDCAP_ON


class InstanceFlags(IntFlag):
    """
    Wraps the OptixInstanceFlags enum.
    """
    NONE = OPTIX_INSTANCE_FLAG_NONE,
    DISABLE_TRIANGLE_FACE_CULLING = OPTIX_INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_CULLING,
    FLIP_TRIANGLE_FACING = OPTIX_INSTANCE_FLAG_FLIP_TRIANGLE_FACING,
    DISABLE_ANYHIT = OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT,
    ENFORCE_ANYHIT = OPTIX_INSTANCE_FLAG_ENFORCE_ANYHIT,
    FORCE_OPACITY_MICROMAP_2_STATE = OPTIX_INSTANCE_FLAG_FORCE_OPACITY_MICROMAP_2_STATE,
    DISABLE_OPACITY_MICROMAPS = OPTIX_INSTANCE_FLAG_DISABLE_OPACITY_MICROMAPS


class BuildInputType(IntEnum):
    TRIANGLES = OPTIX_BUILD_INPUT_TYPE_TRIANGLES,
    CUSTOM_PRIMITIVES = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES,
    INSTANCES = OPTIX_BUILD_INPUT_TYPE_INSTANCES,
    INSTANCE_POINTERS = OPTIX_BUILD_INPUT_TYPE_INSTANCE_POINTERS,
    CURVES = OPTIX_BUILD_INPUT_TYPE_CURVES,
    SPHERES = OPTIX_BUILD_INPUT_TYPE_SPHERES


cdef class BuildInputArray(OptixObject):
    """
    Base class for all BuildInput Arrays. This is an internal class.
    """
    def __init__(self, type):
        self.build_input_type = <OptixBuildInputType><int>(BuildInputType(type).value)

    cdef void prepare_build_input(self, OptixBuildInput* build_input) except *:
        pass

    cdef size_t num_elements(self):
        return 0

    @property
    def type(self):
        return BuildInputType(self.build_input_type)


cdef class BuildInputTriangleArray(BuildInputArray):
    """
    BuildInputArray for a triangle mesh. This class wraps the OptixBuildInputTriangleArray struct.
    In Contrast to the behavior of the Optix C++ API, this Python class will automatically convert all numpy.ndarrays
    to cupy.ndarrays and keep track of them.

    Parameters
    ----------
    vertex_buffers:
        List of vertex buffers (one for each motion step) or a single array.
        All arrays will be converted to cupy.ndarrays before any further processing.
    index_buffer: ndarray, optional
        A single 2d array containing the indices of all triangles or None
    num_sbt_records: int
        The number of records in the ShaderBindingTable for this geometry
    flags: GeometryFlags
        Flags to use in this input for each motionstep
    sbt_record_offset_buffer: ndarray, optional
        Offsets into the ShaderBindingTable record for each primitive (index) or None
    pre_transform: ndarray(3,4) or None
        A transform to apply prior to processing
    primitive_index_offset: int
        The offset applied to the primitive index in device code
    """
    def __init__(self,
                 vertex_buffers,
                 index_buffer = None,
                 num_sbt_records = 1,
                 flags = None,
                 sbt_record_offset_buffer = None,
                 pre_transform = None,
                 primitive_index_offset = 0,
                 opacity_micromap: typ.Optional[BuildInputOpacityMicromap] = None):
        super().__init__(BuildInputType.TRIANGLES)
        self._d_vertex_buffers = [cp.asarray(vb) for vb in ensure_iterable(vertex_buffers)]
        self._d_vertex_buffer_ptrs.reserve(len(self._d_vertex_buffers))

        if len(self._d_vertex_buffers) > 0:
            dtype = self._d_vertex_buffers[0].dtype
            shape = self._d_vertex_buffers[0].shape
            strides = self._d_vertex_buffers[0].strides
            for vb in self._d_vertex_buffers:
                if vb.dtype != dtype or vb.shape != shape or vb.strides != strides:
                    raise ValueError("All vertex buffers must have the same size and dtype")
                self._d_vertex_buffer_ptrs.push_back(vb.data.ptr)

            self.build_input.vertexBuffers = self._d_vertex_buffer_ptrs.const_data()
            self.build_input.vertexFormat = self._vertex_format(dtype, shape)
            self.build_input.vertexStrideInBytes = self._d_vertex_buffers[0].strides[0]
            self.build_input.numVertices = shape[0]
        else:
            self.build_input.vertexBuffers = NULL
            self.build_input.vertexFormat = OPTIX_VERTEX_FORMAT_NONE
            self.build_input.vertexStrideInBytes = 0
            self.build_input.numVertices = 0

        if index_buffer is not None:
            self._d_index_buffer = cp.asarray(index_buffer)
            self.build_input.indexFormat = self._index_format(self._d_index_buffer.dtype, self._d_index_buffer.shape)
            self.build_input.indexStrideInBytes = self._d_index_buffer.strides[0]
            self.build_input.numIndexTriplets = self._d_index_buffer.shape[0]
            self.build_input.indexBuffer = self._d_index_buffer.data.ptr
        else:
            self.build_input.indexFormat = OPTIX_INDICES_FORMAT_NONE
            self.build_input.indexStrideInBytes = 0
            self.build_input.numIndexTriplets = 0
            self.build_input.indexBuffer = 0
            self._d_index_buffer = None

        self.build_input.numSbtRecords = num_sbt_records

        self._flags.resize(num_sbt_records)

        if flags is None:
            for i in range(num_sbt_records):
                self._flags[i] = OPTIX_GEOMETRY_FLAG_NONE
        else:
            for i in range(num_sbt_records):
                self._flags[i] = flags[i].value

        self.build_input.flags = self._flags.data()

        if sbt_record_offset_buffer is not None:
            self._d_sbt_offset_buffer = cp.asarray(sbt_record_offset_buffer).ravel()
            self.build_input.sbtIndexOffsetBuffer = self._d_sbt_offset_buffer.data.ptr
            itemsize = self._d_sbt_offset_buffer.itemsize
            if itemsize > 4:
                raise ValueError("Only 32 bit allowed at max")
            self.build_input.sbtIndexOffsetSizeInBytes = itemsize
            self.build_input.sbtIndexOffsetStrideInBytes = self._d_sbt_offset_buffer.strides[0]
        else:
            self.build_input.sbtIndexOffsetBuffer = 0
            self.build_input.sbtIndexOffsetStrideInBytes = 0
            self.build_input.sbtIndexOffsetSizeInBytes = 0

        self.build_input.primitiveIndexOffset = primitive_index_offset

        if pre_transform is not None:
            self._d_pre_transform = cp.asarray(pre_transform, dtype=np.float32).reshape(3, 4)
            self.build_input.preTransform = self._d_pre_transform.data.ptr
            self.build_input.transformFormat = OPTIX_TRANSFORM_FORMAT_MATRIX_FLOAT12
        else:
            self.build_input.preTransform = 0
            self.build_input.transformFormat = OPTIX_TRANSFORM_FORMAT_NONE

        self.c_opacity_micromap = opacity_micromap
        if self.c_opacity_micromap is not None:
            self.build_input.opacityMicromap = self.c_opacity_micromap.build_input


    def __dealloc__(self):
        pass

    cdef void prepare_build_input(self, OptixBuildInput* build_input) except *:
        build_input.type = self.build_input_type
        build_input.triangleArray = self.build_input

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

    cdef size_t num_elements(self):
        return self.build_input.numVertices

    @property
    def micromap(self):
        return self.c_opacity_micromap

    @property
    def num_sbt_records(self):
        return self.build_input.numSbtRecords


    def _repr_details(self):
        return f"nvertices={self.num_elements()}, " \
               f"ntriangles={self._d_index_buffer.shape[0] if self._d_index_buffer is not None else self.num_elements() // 3}, " \
               f"n_sbt_records={self.build_input.numSbtRecords}"

cdef class BuildInputCustomPrimitiveArray(BuildInputArray):
    """
    BuildInputArray for custom primitives. This class wraps the OptixBuildInputCustomPrimitiveArray struct.
    In Contrast to the behavior of the Optix C++ API, this Python class will automatically convert all numpy.ndarrays
    to cupy.ndarrays and keep track of them.

    Parameters
    ----------
    aabb_buffers:
        List of buffers (one for each motion step) or a single array specifiiing the bounding boxes of the primitives.
        The shape is therefore expected to be [n, 6] with the bounding box defined as [min_x, min_y, min_z, max_x, max_y, max_z].
        All arrays will be converted to cupy.ndarrays before any further processing.
    num_sbt_records: int
        The number of records in the ShaderBindingTable for this geometry
    flags: GeometryFlags
        Flags to use in this input
    sbt_record_offset_buffer: ndarray, optional
        Offsets into the ShaderBindingTable record for each primitive (index) or None
    primitive_index_offset: int
        The offset applied to the primitive index in device code
    """
    def __init__(self,
                 aabb_buffers,
                 num_sbt_records = 1,
                 flags = None,
                 sbt_record_offset_buffer = None,
                 primitive_index_offset = 0
                 ):
        super().__init__(BuildInputType.CUSTOM_PRIMITIVES)
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

        self.build_input.aabbBuffers = self._d_aabb_buffer_ptrs.const_data()
        self.build_input.numPrimitives = shape[0]
        
        # https://github.com/cupy/cupy/issues/5897
        self.build_input.strideInBytes = 6*np.float32().itemsize

        self._flags.resize(num_sbt_records)
        if flags is None:
            for i in range(num_sbt_records):
                self._flags[i] = OPTIX_GEOMETRY_FLAG_NONE
        else:
            for i in range(num_sbt_records):
                self._flags[i] = flags[i].value

        self.build_input.flags = self._flags.const_data()

        self.build_input.numSbtRecords = num_sbt_records

        if sbt_record_offset_buffer is not None:
            self._d_sbt_offset_buffer = cp.asarray(sbt_record_offset_buffer).ravel()
            if not np.issubdtype(self._d_sbt_offset_buffer.dtype, np.unsignedinteger):
                self._d_sbt_offset_buffer = self._d_sbt_offset_buffer.astype(np.uint32)
            itemsize = self._d_sbt_offset_buffer.itemsize
            if itemsize > 4:
                raise ValueError("Only 32 bit allowed at max")

            self.build_input.sbtIndexOffsetBuffer = self._d_sbt_offset_buffer.data.ptr
            self.build_input.sbtIndexOffsetSizeInBytes = itemsize
            self.build_input.sbtIndexOffsetStrideInBytes = self._d_sbt_offset_buffer.strides[0]
        else:
            self.build_input.sbtIndexOffsetBuffer = 0
            self.build_input.sbtIndexOffsetStrideInBytes = 0
            self.build_input.sbtIndexOffsetSizeInBytes = 0

        self.build_input.primitiveIndexOffset = primitive_index_offset

    cdef void prepare_build_input(self, OptixBuildInput * build_input) except *:
        build_input.type = self.build_input_type
        build_input.customPrimitiveArray = self.build_input

    cdef size_t num_elements(self):
        return self.build_input.numPrimitives


cdef class BuildInputCurveArray(BuildInputArray):
    """
    BuildInputArray for curve inputs. This class wraps the OptixBuildInputCurveArray struct.
    In Contrast to the behavior of the Optix C++ API, this Python class will automatically convert all numpy.ndarrays
    to cupy.ndarrays and keep track of them.

    Parameters
    ----------
    curve_type: PrimitiveType
        The curve degree and basis
    vertex_buffers:
        List of vertex buffers (one for each motion step) or a single array.
        All arrays will be converted to cupy.ndarrays before any further processing.
    width_buffers:
        List of width buffers (one for each motion step, see vertex_buffers) or a single array specifying the curve width (radius)
        for each vertex. All arrays will be converted to cupy.ndarrays before any further processing.
    index_buffer:
        Array of unsigned int, one per curve segment.
    normal_buffers: None
        reserved for future use.
    flags: GeometryFlags
        Flags to use in this input
    primitive_index_offset: int
        The offset applied to the primitive index in device code
    """
    def __init__(self,
                 curve_type,
                 vertex_buffers,
                 width_buffers,
                 index_buffer,
                 normal_buffers = None,
                 flags=None,
                 primitive_index_offset=0,
                 endcap_flags=CurveEndcapFlags.DEFAULT):
        super().__init__(BuildInputType.CURVES)
        self.build_input.curveType = curve_type.value
        self._d_vertex_buffers = [cp.asarray(vb, np.float32) for vb in ensure_iterable(vertex_buffers)]
        self._d_vertex_buffer_ptrs.reserve(len(self._d_vertex_buffers))

        shape = self._d_vertex_buffers[0].shape
        strides = self._d_vertex_buffers[0].strides
        for vb in self._d_vertex_buffers:
            if vb.shape != shape or vb.strides != strides:
                raise ValueError("All vertex buffers must have the same size and strides")
            self._d_vertex_buffer_ptrs.push_back(vb.data.ptr)

        self.build_input.vertexBuffers = self._d_vertex_buffer_ptrs.const_data()
        self.build_input.vertexStrideInBytes = self._d_vertex_buffers[0].strides[0]
        self.build_input.numVertices = shape[0]

        self._d_width_buffers = [cp.asarray(wb, np.uint32) for wb in ensure_iterable(width_buffers)]
        assert len(self._d_width_buffers) == len(self._d_vertex_buffers)
        self._d_width_buffer_ptrs.reserve(len(self._d_width_buffers))

        shape = self._d_width_buffers[0].shape
        strides = self._d_width_buffers[0].strides
        for wb in self._d_width_buffers:
            if wb.shape != shape or wb.strides != strides:
                raise ValueError("All width buffers must have the same size and strides")
            self._d_width_buffer_ptrs.push_back(wb.data.ptr)

        self.build_input.widthBuffers = self._d_width_buffer_ptrs.const_data()
        self.build_input.widthStrideInBytes = self._d_width_buffers[0].strides[0]

        if normal_buffers is not None:
            self._d_normal_buffers = [cp.asarray(nb, np.uint32) for nb in ensure_iterable(normal_buffers)]
            assert len(self._d_normal_buffers) == len(self._d_vertex_buffers)
            self._d_normal_buffer_ptrs.reserve(len(self._d_normal_buffers))

            shape = self._d_normal_buffers[0].shape
            strides = self._d_normal_buffers[0].strides
            for nb in self._d_normal_buffers:
                if nb.shape != shape or nb.strides != strides:
                    raise ValueError("All normal buffers must have the same size and strides")
                self._d_normal_buffer_ptrs.push_back(nb.data.ptr)

            self.build_input.normalBuffers = self._d_normal_buffer_ptrs.const_data()
            self.build_input.normalStrideInBytes = self._d_normal_buffers[0].strides[0]
        else:
            self.build_input.normalBuffers = NULL
            self.build_input.normalStrideInBytes = 0

        self._d_index_buffer = cp.asarray(index_buffer, dtype=np.uint32)
        self.build_input.indexStrideInBytes = self._d_index_buffer.strides[0]
        self.build_input.indexBuffer = self._d_index_buffer.data.ptr

        if flags is None:
            self.build_input.flag = OPTIX_GEOMETRY_FLAG_NONE
        else:
            self.build_input.flag = flags.valuesbtIndexOffsetBuffer

        self.build_input.primitiveIndexOffset = primitive_index_offset

        self.build_input.endcapFlags = endcap_flags

    cdef void prepare_build_input(self, OptixBuildInput * build_input) except *:
        build_input.type = self.build_input_type
        build_input.curveArray = self.build_input

    cdef size_t num_elements(self):
        return self.build_input.numPrimitives


cdef class BuildInputSphereArray(BuildInputArray):
    """
        BuildInputArray for a sphere. This class wraps the OptixBuildInputSphereArray struct.
        In Contrast to the behavior of the Optix C++ API, this Python class will automatically convert all numpy.ndarrays
        to cupy.ndarrays and keep track of them.

        Parameters
        ----------
        vertex_buffers:
            List of vertex buffers (one for each motion step) or a single array.
            All arrays will be converted to cupy.ndarrays before any further processing.
        index_buffer: ndarray, optional
            A single 2d array containing the indices of all triangles or None
        num_sbt_records: int
            The number of records in the ShaderBindingTable for this geometry
        flags: GeometryFlags
            Flags to use in this input for each motionstep
        sbt_record_offset_buffer: ndarray, optional
            Offsets into the ShaderBindingTable record for each primitive (index) or None
        pre_transform: ndarray(3,4) or None
            A transform to apply prior to processing
        primitive_index_offset: int
            The offset applied to the primitive index in device code
        """
    def __init__(self,
                 vertex_buffers,
                 radius_buffers,
                 num_sbt_records = 1,
                 flags = None,
                 sbt_record_offset_buffer = None,
                 pre_transform = None,
                 primitive_index_offset = 0
                 ):
        super().__init__(BuildInputType.SPHERES)
        self._d_vertex_buffers = [cp.asarray(vb) for vb in ensure_iterable(vertex_buffers)]
        self._d_vertex_buffer_ptrs.reserve(len(self._d_vertex_buffers))

        self._d_radius_buffers = [cp.asarray(vb) for vb in ensure_iterable(radius_buffers)]
        self._d_radius_buffer_ptrs.reserve(len(self._d_radius_buffers))

        if len(self._d_radius_buffers) != len(self._d_vertex_buffers):
            raise ValueError("Argument radius_buffers must have the same number of arrays as vertex_buffers.")

        if len(self._d_vertex_buffers) == 0:
            raise ValueError("BuildInputSphereArray cannot be empty.")

        dtype = self._d_vertex_buffers[0].dtype
        shape = self._d_vertex_buffers[0].shape
        strides = self._d_vertex_buffers[0].strides

        radius_dtype = self._d_radius_buffers[0].dtype
        radius_shape = self._d_radius_buffers[0].shape
        strides = self._d_radius_buffers[0].strides

        for vb, rb in zip(self._d_vertex_buffers, self._d_radius_buffers):
            if vb.dtype != dtype or vb.shape != shape or vb.strides != strides:
                raise ValueError("All vertex buffers must have the same size and dtype.")
            self._d_vertex_buffer_ptrs.push_back(vb.data.ptr)

            if rb.dtype != dtype or rb.shape != shape or rb.strides != strides:
                raise ValueError("All radius buffers must have the same size and dtype.")
            self._d_radius_buffer_ptrs.push_back(rb.data.ptr)

        self.build_input.vertexBuffers = self._d_vertex_buffer_ptrs.const_data()
        self.build_input.radiusBuffers = self._d_radius_buffer_ptrs.const_data()

        self.build_input.vertexStrideInBytes = self._d_vertex_buffers[0].strides[0]
        self.build_input.radiusStrideInBytes = self._d_radius_buffers[0].strides[0]

        self.build_input.numVertices = shape[0]
        self.build_input.singleRadius = 1 if self._d_radius_buffers[0].shape[0] == 1 else 0

        self.build_input.numSbtRecords = num_sbt_records
        self._flags.resize(num_sbt_records)

        if flags is None:
            for i in range(num_sbt_records):
                self._flags[i] = OPTIX_GEOMETRY_FLAG_NONE
        else:
            for i in range(num_sbt_records):
                self._flags[i] = flags[i].value

        self.build_input.flags = self._flags.data()


        if sbt_record_offset_buffer is not None:
            self._d_sbt_offset_buffer = cp.asarray(sbt_record_offset_buffer).ravel()
            self.build_input.sbtIndexOffsetBuffer = self._d_sbt_offset_buffer.data.ptr
            itemsize = self._d_sbt_offset_buffer.itemsize
            if itemsize > 4:
                raise ValueError("Only 32 bit allowed at max")
            self.build_input.sbtIndexOffsetSizeInBytes = itemsize
            self.build_input.sbtIndexOffsetStrideInBytes = self._d_sbt_offset_buffer.strides[0]
        else:
            self.build_input.sbtIndexOffsetBuffer = 0
            self.build_input.sbtIndexOffsetStrideInBytes = 0
            self.build_input.sbtIndexOffsetSizeInBytes = 0

        self.build_input.primitiveIndexOffset = primitive_index_offset

    cdef void prepare_build_input(self, OptixBuildInput * build_input) except *:
        build_input.type = self.build_input_type
        build_input.sphereArray = self.build_input

    cdef size_t num_elements(self):
        return self.build_input.numVertices


cdef class Instance(OptixObject):
    """
    Class representing a single instance (another AccelerationStructure) for use in a Instance level AccelerationStructure.
    This wraps the OptixInstance struct.

    Parameters
    ----------
    traversable: AccelerationStructure
        The AccelerationStructure (Geometry or Instance level) to represent
    instance_id:
        Application supplied id.
    flags: InstanceFlags
        Instance Flags to use
    sbt_offset:
        SBT record offset. This will only be used if the Instance wraps a Geometry AccelerationStructure. Otherwise this
        must be 0.
    transform: ndarray(3,4) or None
        A transform to apply prior to processing.
    visibility_mask: int
        The visibility mask used for culling.
    """
    def __init__(self,
                 AccelerationStructure traversable,
                 unsigned int instance_id,
                 flags = InstanceFlags.NONE,
                 unsigned int sbt_offset = 0,
                 transform=None,
                 visibility_mask=None):
        if transform is None:
            transform = np.eye(3, 4, dtype=np.float32)
        self.transform = transform
        self.traversable = traversable

        self.instance.instanceId = instance_id
        self.instance.flags = flags.value
        self.instance.sbtOffset = sbt_offset

        max_visibility_mask_bits = self.traversable.context.num_bits_instances_visibility_mask
        visibility_mask = int(visibility_mask) if visibility_mask is not None else (2**max_visibility_mask_bits - 1)
        if visibility_mask.bit_length() > self.traversable.context.num_bits_instances_visibility_mask:
            raise ValueError(f"Too many entries in visibility mask. Got {visibility_mask.bit_length()} but supported are only {max_visibility_mask_bits}")
        self.instance.visibilityMask = visibility_mask

    @property
    def traversable(self):
        return self._traversable

    @traversable.setter
    def traversable(self, AccelerationStructure traversable):
        self._traversable = traversable
        # update the handle as well
        self.instance.traversableHandle = self.traversable.handle

    def relocate(self,
                 device: typ.Optional[DeviceContext] = None,
                 stream: typ.Optional[cp.cuda.Stream] = None):
        cls = self.__class__
        result = cls.__new__(cls)
        result.instance = self.instance
        result.traversable = self.traversable.relocate(device=device, stream=stream)

    def __deepcopy__(self, memo):
        result = self.relocate()
        memo[id(self)] = result
        return result

    @property
    def transform(self):
        cdef float [:] transform_view = self.instance.transform
        return np.asarray(transform_view).reshape(3,4)

    @transform.setter
    def transform(self, tf):
        transform = np.ascontiguousarray(np.asarray(tf, dtype=np.float32).reshape(3,4))
        cdef float[:, ::1] c_transform = transform
        memcpy(&self.instance.transform, &c_transform[0, 0], sizeof(float) * 12)


cdef class BuildInputInstanceArray(BuildInputArray):
    """
    BuildInputArray for instance inputs. This class wraps the OptixBuildInputInstanceArray struct. This class provides the
    build input in the form of an array of instances.

    Parameters
    ----------
    instances: list[Instance]
        A list of the Instances to use as input
    """
    def __init__(self, instances):
        super().__init__(BuildInputType.INSTANCES)
        instances = ensure_iterable(instances)
        self.instances = instances

        cdef ssize_t size = sizeof(OptixInstance)*len(instances)
        self._d_instances = cp.cuda.alloc(size)
        cdef vector[OptixInstance] c_instances
        c_instances.reserve(len(instances))
        for inst in instances:
            c_instances.push_back((<Instance>inst).instance)

        cp.cuda.runtime.memcpy(self._d_instances.ptr, <size_t>c_instances.data(), size, cp.cuda.runtime.memcpyHostToDevice)

        self.build_input.instances = self._d_instances.ptr
        self.build_input.numInstances = len(instances)

    cdef void prepare_build_input(self, OptixBuildInput * build_input) except *:
        build_input.type = self.build_input_type
        build_input.instanceArray = self.build_input

    cdef size_t num_elements(self):
        return self.build_input.numInstances

    def __getitem__(self, index):
        return self.instances[index]

    def __setitem__(self, index, instance):
        if not isinstance(instance, Instance):
            raise TypeError("Only instance objects.")
        self.instances[index] = instance
        self.update_instance(index)

    def update_instance(self, index):
        """
        Update the instance at index in gpu memory from the instances list in host memory.

        Parameters
        ----------
        index: int
            The index to update
        """
        # update the value in the cuda buffer
        src_ptr = <size_t>&(<Instance>(self.instances[index])).instance
        dst_ptr = self._d_instances.ptr + index*sizeof(OptixInstance)
        cp.cuda.runtime.memcpy(dst_ptr, src_ptr, sizeof(OptixInstance), cp.cuda.runtime.memcpyHostToDevice)

    # TODO: still thinking of a better way to acomplish the transform access in an OO way.
    def view_instance_transform(self, index):
        """
        Obtain a view of the transform parameter at index in gpu memory as a cupy array for direct modification

        Parameters
        ----------
        index: int
            The index of the transform this view should point to

        Returns
        -------
        transform_view: cp.ndarray of shape (3, 4)
            A view of the transform matrix

        """
        if index < 0 or index >=len(self.instances):
            raise IndexError(f"Invalid index {index} for list of length {len(self.instances)}.")
        device_ptr = cp.cuda.MemoryPointer(mem=self._d_instances.mem, offset=<int>index*sizeof(OptixInstance))
        return cp.ndarray(shape=(3,4), dtype=np.float32, memptr=device_ptr)


cdef class RelocationDependency:
    cdef OptixBuildInputType _type

    def __init__(self, type):
        self._type = <OptixBuildInputType><int>(BuildInputType(type).value)

    @property
    def type(self):
        return BuildInputType(self._type)

    cdef RelocationDependency relocate(self, device, stream):
        return self

    cdef void fill_relocation_input(self, OptixRelocateInput& input):
        input.type = self._type

    cdef void finalize_relocation_input(self):
        pass

cdef class RelocationInstanceDependency(RelocationDependency):
    cdef object instances
    cdef object d_instances

    def __init__(self, instances):
        super().__init__(BuildInputType.INSTANCES)
        self.instances = instances
        self.d_instances = None

    cdef RelocationInstanceDependency relocate(self, device, stream):
        relocated_instances = [inst.relocate(device=device, stream=stream) for inst in self.instances]
        result = self.__class__(relocated_instances)
        return result

    cdef void fill_relocation_input(self, OptixRelocateInput& input):
        cdef vector[OptixTraversableHandle] c_instance_handles
        cdef ssize_t c_instance_handles_size = 0
        cdef object d_instances
        cdef size_t i
        cdef CUdeviceptr d_instances_ptr = 0
        cdef vector[OptixRelocateInput] c_relocate_inputs
        cdef size_t num_relocate_inputs

        input.type = self.type
        input.instanceArray.numInstances = len(self.instances)

        # prepare the new instance handles for relocation by copiing them into a temporary device buffer
        c_instance_handles.resize(len(self.instances))
        c_instance_handles_size = sizeof(OptixTraversableHandle) * c_instance_handles.size()

        self.d_instances = cp.cuda.alloc(c_instance_handles_size)
        for i in range(c_instance_handles.size()):
            c_instance_handles[i] = self.instances[i].traversable.handle

        d_instances_ptr = self.d_instances.ptr
        cp.cuda.runtime.memcpy(d_instances_ptr, <size_t> c_instance_handles.data(), c_instance_handles_size,
                               cp.cuda.runtime.memcpyHostToDevice)
        c_relocate_inputs[0].instanceArray.traversableHandles = d_instances_ptr
        input.instanceArray.traversableHandles = d_instances_ptr

    cdef void finalize_relocation_input(self):
        self.d_instances = None # remove the temporary cuda buffer again


cdef class RelocationTriangleDependency(RelocationDependency):
    cdef unsigned int num_sbt_records
    cdef OpacityMicromapArray micromap

    def __init__(self, num_sbt_records, micromap=None):
        super().__init__(BuildInputType.INSTANCES)
        self.num_sbt_records = num_sbt_records
        self.micromap = micromap

    cdef RelocationInstanceDependency relocate(self, device, stream):
        if self.micromap is not None:
            relocated_micromap = self.micromap.relocate(device=device, stream=stream)
        else:
            relocated_micromap = None
        result = self.__class__(self.num_sbt_records, relocated_micromap)
        return result

    cdef void fill_relocation_input(self, OptixRelocateInput& input):
        input.type = self.type
        input.triangleArray.numSbtRecords = self.num_sbt_records
        input.triangleArray.opacityMicromap.opacityMicromapArray = self.micromap.d_micromap_array_buffer.ptr


cdef class AccelerationStructure(OptixContextObject):
    """
    Class representing a Geometry Acceleration Structure (GAS) or Instance Acceleration Structure (IAS). This wraps the OptixTraversableHandle internally and manages the ressources like
    temporary buffers automatically.

    Parameters
    ----------
    context: DeviceContext
        The context to use
    build_inputs: list of BuildInputArray subclasses or a single instance
        The build inputs to use
    compact: bool
        Compact the AccelerationStructure in a second step or not. Usually this can be left enabled to
        reduce memory consumption.
    allow_update: bool
        Allow for updating the generated AccelerationStructure
    prefer_fast_build: bool
        Prefer a fast build over a fast access.
    random_vertex_access: bool
        Allow for random access of the vertices in triangle geometry
    random_instance_access: bool
        Allow for random access of the instances if an IAS is built
    allow_opacity_micromap_update: bool
        Allows to update the opacity micromaps in this structure
    allow_disable_opacity_micromaps: bool
        Allows to disable the opacity micromaps for instances in this structure
    stream: cupy.cuda.Stream, optional
        Cuda stream to use. If None the default stream is used
    """
    def __init__(self,
                 DeviceContext context,
                 build_inputs,
                 compact=True,
                 allow_update=False,
                 prefer_fast_build=False,
                 random_vertex_access=False,
                 random_instance_access=False,
                 allow_opacity_micromap_update=False,
                 allow_disable_opacity_micromaps=False,
                 stream=None):

        super().__init__(context)
        self._build_flags = OPTIX_BUILD_FLAG_NONE
        if compact:
            self._build_flags |= OPTIX_BUILD_FLAG_ALLOW_COMPACTION
        if allow_update:
            self._build_flags |= OPTIX_BUILD_FLAG_ALLOW_UPDATE
        if prefer_fast_build:
            self._build_flags |= OPTIX_BUILD_FLAG_PREFER_FAST_BUILD
            self._build_flags &= ~OPTIX_BUILD_FLAG_PREFER_FAST_TRACE
        else:
            self._build_flags |= OPTIX_BUILD_FLAG_PREFER_FAST_TRACE
            self._build_flags &= ~OPTIX_BUILD_FLAG_PREFER_FAST_BUILD
        if random_vertex_access:
            self._build_flags |= OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS
        if random_instance_access:
            self._build_flags |= OPTIX_BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS
        if allow_opacity_micromap_update:
            self._build_flags |= OPTIX_BUILD_FLAG_ALLOW_OPACITY_MICROMAP_UPDATE
        if allow_disable_opacity_micromaps:
            self._build_flags |= OPTIX_BUILD_FLAG_ALLOW_DISABLE_OPACITY_MICROMAPS


        self._gas_buffer = None
        self._instances = None
        self._relocate_deps = []
        build_inputs = ensure_iterable(build_inputs)
        self.build(build_inputs, stream=stream)

    @property
    def compact(self):
        return self._build_flags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION

    @property
    def allow_update(self):
        return self._build_flags & OPTIX_BUILD_FLAG_ALLOW_UPDATE

    cdef void _init_build_inputs(self, build_inputs, vector[OptixBuildInput]& ret):
        ret.resize(len(build_inputs))
        self._num_elements = 0
        for i, build_input in enumerate(build_inputs):
            if self._num_elements == 0:
                self._num_elements = (<BuildInputArray>build_input).num_elements()
            else:
                if self._num_elements != (<BuildInputArray>build_input).num_elements():
                    raise ValueError("All build inputs must have the same number of elements")

            (<BuildInputArray>build_input).prepare_build_input(&ret[i])


    cdef void _init_accel_options(self, size_t num_build_inputs, unsigned int build_flags, OptixBuildOperation operation, vector[OptixAccelBuildOptions]& ret):
        cdef OptixAccelBuildOptions accel_option
        memset(&accel_option, 0, sizeof(OptixAccelBuildOptions)) # init struct to 0

        cdef size_t i

        accel_option.buildFlags = build_flags
        accel_option.operation = operation

        ret.resize(num_build_inputs)
        for i in range(num_build_inputs):
            ret[i] = accel_option

    cdef void build(self, build_inputs, stream=None):
        # build a single vector from all the build inputs
        cdef size_t inputs_size = len(build_inputs)
        cdef vector[OptixBuildInput] c_inputs #= vector[OptixBuildInput](inputs_size)
        cdef size_t i

        self._init_build_inputs(build_inputs, c_inputs)

        for build_input in build_inputs:
            if isinstance(build_input, BuildInputInstanceArray):
                relocation_dep = RelocationInstanceDependency((<BuildInputInstanceArray>build_input).instances)
                if inputs_size > 1:
                    raise ValueError("Only a single build input allowed for instance builds")
            elif isinstance(build_input, BuildInputTriangleArray):
                micromap = <BuildInputTriangleArray>build_input.micromap
                micromap_array = micromap.micromap_array if micromap is not None else None
                relocation_dep = RelocationTriangleDependency(build_input.num_sbt_records, micromap=micromap_array)
            else:
                relocation_dep = RelocationDependency(build_input.type)
            self._relocate_deps.append(relocation_dep)

        cdef vector[OptixAccelBuildOptions] accel_options# = vector[OptixAccelBuildOptions](inputs_size)
        self._init_accel_options(inputs_size, self._build_flags, OPTIX_BUILD_OPERATION_BUILD, accel_options)

        cdef OptixAccelEmitDesc compacted_size_property

        cdef CUdeviceptr gas_buffer_ptr
        cdef CUdeviceptr tmp_gas_buffer_ptr
        cdef unsigned int num_properties = 0

        optix_check_return(optixAccelComputeMemoryUsage(self.context.c_context,
                                                        accel_options.data(),
                                                        c_inputs.data(),
                                                        inputs_size,
                                                        &self._buffer_sizes))

        d_tmp_gas_buffer = cp.cuda.alloc(round_up(self._buffer_sizes.tempSizeInBytes, 8) + 8)
        self._gas_buffer = cp.cuda.alloc(round_up(self._buffer_sizes.outputSizeInBytes, 8) + 8)

        d_compacted_size = None
        cdef OptixAccelEmitDesc* property_ptr = NULL

        if self.compact:
            d_compacted_size = cp.zeros(1, dtype=np.uint64)
            compacted_size_property.result = d_compacted_size.data.ptr
            compacted_size_property.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE
            num_properties = 1
            property_ptr = &compacted_size_property

        gas_buffer_ptr = self._gas_buffer.ptr
        tmp_gas_buffer_ptr = d_tmp_gas_buffer.ptr

        cdef uintptr_t c_stream = 0

        if stream is not None:
            c_stream = stream.ptr

        with nogil:
            optix_check_return(optixAccelBuild(self.context.c_context,
                                               <CUstream>c_stream,
                                               accel_options.data(),
                                               c_inputs.data(),
                                               inputs_size,
                                               tmp_gas_buffer_ptr,
                                               self._buffer_sizes.tempSizeInBytes,
                                               gas_buffer_ptr,
                                               self._buffer_sizes.outputSizeInBytes,
                                               &self._handle,
                                               property_ptr,
                                               num_properties))

        cdef size_t compacted_gas_size = 0

        if self.compact:
            compacted_gas_size = cp.asnumpy(d_compacted_size)[0]
            if compacted_gas_size < self._buffer_sizes.outputSizeInBytes:
                # compact the acceleration structure
                d_gas_output_buffer = cp.cuda.alloc(compacted_gas_size)
                gas_buffer_ptr = d_gas_output_buffer.ptr
                with nogil:
                    optix_check_return(optixAccelCompact(self.context.c_context,
                                                         <CUstream>c_stream,
                                                         self._handle,
                                                         gas_buffer_ptr,
                                                         compacted_gas_size,
                                                         &self._handle))
                self._gas_buffer = d_gas_output_buffer # keep the new buffer instead of the old one

    def update(self, build_inputs, stream=None):
        """
        Update the AccelerationStructure with updated build inputs. Refer to the OptiX documention for the restrictions.

        Parameters
        ----------
        build_inputs: list[BuildInputArray] or single instance
        stream: cupy.cuda.Stream, optional
            The cuda stream to use for the processing. If None the default stream is used.
        """
        if not self.allow_update:
            raise ValueError("Updates are not enabled for this AccelerationStructure")

        build_inputs = ensure_iterable(build_inputs)

        cdef size_t inputs_size = len(build_inputs)

        if inputs_size != len(self._relocate_deps):
            raise ValueError("Number of build inputs given to update() must be the same as the one used to build this GAS")
        cdef vector[OptixBuildInput] inputs #= vector[OptixBuildInput](inputs_size)
        self._init_build_inputs(build_inputs, inputs)

        cdef vector[OptixAccelBuildOptions] accel_options# = vector[OptixAccelBuildOptions](inputs_size)
        self._init_accel_options(inputs_size, self._build_flags, OPTIX_BUILD_OPERATION_UPDATE, accel_options)

        d_tmp_update_gas_buffer = cp.cuda.alloc(round_up(self._buffer_sizes.tempUpdateSizeInBytes, 8) + 8)
        cdef CUdeviceptr tmp_update_gas_buffer_ptr
        tmp_update_gas_buffer_ptr = d_tmp_update_gas_buffer.ptr

        cdef CUdeviceptr gas_buffer_ptr
        gas_buffer_ptr = self._gas_buffer.ptr

        cdef uintptr_t c_stream = 0

        if stream is not None:
            c_stream = stream.ptr

        with nogil:
            optix_check_return(optixAccelBuild(self.context.c_context,
                                               <CUstream> c_stream,
                                               accel_options.data(),
                                               inputs.data(),
                                               inputs_size,
                                               tmp_update_gas_buffer_ptr,
                                               self._buffer_sizes.tempUpdateSizeInBytes,
                                               gas_buffer_ptr,
                                               self._buffer_sizes.outputSizeInBytes,
                                               &self._handle,
                                               NULL,
                                               0))

    def __deepcopy__(self, memo):
        """
        Perform a deep copy of the AccelerationStructure by using the standard python copy.deepcopy function.
        """
        # relocate on the same device to perform a regular deep copy
        result = self.relocate(device=None)
        memo[id(self)] = result
        return result


    def relocate(self,
                 device: typ.Optional[DeviceContext] = None,
                 stream: typ.Optional[cp.cuda.Stream] = None):
        """
        Relocate this acceleration structure into another copy. Usually this is equivalent to a deep copy.
        Additionally, the accleration structure can be copied to a different defice by specifying the device
        parameter with a different DeviceContext.

        Parameters
        ----------
        device:
            An optional DeviceContext. The resulting copy of the acceleration structure will be copied
            to this device. If None, the acceleration structure's current device is used.
        stream:
            The stream to use for the relocation. If None, the default stream (0) is used.
        Returns
        -------
        copy: AccelerationStructure
            The copy of the AccelerationStructure on the new device

        """
        from copy import deepcopy
        # relocate the optix structure
        cdef OptixRelocationInfo gas_info
        memset(&gas_info, 0, sizeof(OptixRelocationInfo)) # init struct to 0

        optix_check_return(optixAccelGetRelocationInfo(self.context.c_context, self._handle, &gas_info))

        if device is None:
            device = self.context

        # check if the new device is compatible with this acceleration structure
        cdef int compatible = 0
        optix_check_return(optixCheckRelocationCompatibility(<OptixDeviceContext>(<DeviceContext>device).c_context,
                                                                  &gas_info,
                                                                  &compatible))
        if compatible != 1:
            raise RuntimeError("Device is not compatible for relocation of acceleration structure")

        # do the relocation
        cls = self.__class__
        cdef AccelerationStructure result = cls.__new__(cls)

        result.context = device
        result._build_flags = self._build_flags
        result._buffer_sizes = self._buffer_sizes

        #if self._instances is not None:
        #    result._instances = [inst.relocate(device=device, stream=stream) for inst in self._instances] # copy all instances and their AccelerationStructures first

        buffer_size = round_up(self._buffer_sizes.outputSizeInBytes, 8) + 8
        result._gas_buffer = cp.cuda.alloc(buffer_size)
        cp.cuda.runtime.memcpy(result._gas_buffer.ptr, self._gas_buffer.ptr, buffer_size, cp.cuda.runtime.memcpyDeviceToDevice)

        #cdef vector[OptixTraversableHandle] c_instance_handles
        #cdef ssize_t c_instance_handles_size = 0
        #cdef object d_instances
        #cdef size_t i
        #cdef CUdeviceptr d_instances_ptr = 0
        cdef vector[OptixRelocateInput] c_relocate_inputs
        #cdef size_t num_relocate_inputs
        c_relocate_inputs.resize(len(self._relocate_deps))

        # prepare to relocate the dependencies (micromaps and instances)
        result._relocate_deps = []
        for i, dep in enumerate(self._relocate_deps):
            relocated_dep = dep.relocate(device, stream)
            result._relocate_deps.append(relocated_dep)
            relocated_dep.fill_relocation_input(c_relocate_inputs[i])

        result._handle = 0
        cdef uintptr_t c_stream = 0

        if stream is not None:
            c_stream = stream.ptr

        cdef OptixTraversableHandle c_handle = 0
        optix_check_return(optixAccelRelocate(result.context.c_context,
                                              <CUstream>c_stream,
                                              &gas_info,
                                              &c_relocate_inputs[0],
                                              c_relocate_inputs.size(),
                                              result._gas_buffer,
                                              self._buffer_sizes.outputSizeInBytes,
                                              &c_handle))
        result._handle = c_handle

        for dep in result._relocate_deps:
            dep.finalize_relocation_input()

        return result

    @property
    def handle(self):
        """
        Return the internal OptixTraversableHandle for this AccelerationStructure
        """
        return <OptixTraversableHandle>self._handle

    def _repr_details(self):
        return f"{self._num_elements} elements in {self._buffer_sizes.outputSizeInBytes} bytes"

    @property
    def build_flags(self):
        return BuildFlags(self.build_flags)





