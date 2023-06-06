# distutils: language = c++

import numpy as np
cimport numpy as np
np.import_array()

import cupy as cp
from .common cimport optix_check_return, optix_init

cimport cython
from cython.operator import dereference
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uintptr_t
from libcpp cimport bool
from libcpp.vector cimport vector
from libc.string cimport memset
from collections import defaultdict, namedtuple
from collections.abc import Sequence
from enum import IntEnum, IntFlag
import typing as typ
from .common import ensure_iterable
from .context cimport DeviceContext

optix_init()

__all__ = ['DisplacementMicromapArrayIndexingMode', 
           'DisplacementMicromapFormat',
           'DisplacementMicromapTriangleFlags',
           'DisplacementMicromapInput',
           'DisplacementMicromapArray',
           'BuildInputDisplacementMicromap']


class DisplacementMicromapArrayIndexingMode(IntEnum):
    NONE = OPTIX_DISPLACEMENT_MICROMAP_ARRAY_INDEXING_MODE_NONE
    LINEAR = OPTIX_DISPLACEMENT_MICROMAP_ARRAY_INDEXING_MODE_LINEAR
    INDEXED = OPTIX_DISPLACEMENT_MICROMAP_ARRAY_INDEXING_MODE_INDEXED


class DisplacementMicromapFormat(IntEnum):
    FORMAT_64_MICRO_TRIS_64_BYTES = OPTIX_DISPLACEMENT_MICROMAP_FORMAT_64_MICRO_TRIS_64_BYTES
    FORMAT_256_MICRO_TRIS_128_BYTES  = OPTIX_DISPLACEMENT_MICROMAP_FORMAT_256_MICRO_TRIS_128_BYTES
    FORMAT_1024_MICRO_TRIS_128_BYTES = OPTIX_DISPLACEMENT_MICROMAP_FORMAT_1024_MICRO_TRIS_128_BYTES


class DisplacementMicromapTriangleFlags(IntFlag):
    NONE = OPTIX_DISPLACEMENT_MICROMAP_TRIANGLE_FLAG_NONE
    DECIMATE_EDGE_01 = OPTIX_DISPLACEMENT_MICROMAP_TRIANGLE_FLAG_DECIMATE_EDGE_01
    DECIMATE_EDGE_12 = OPTIX_DISPLACEMENT_MICROMAP_TRIANGLE_FLAG_DECIMATE_EDGE_12
    DECIMATE_EDGE_20 = OPTIX_DISPLACEMENT_MICROMAP_TRIANGLE_FLAG_DECIMATE_EDGE_20

_n_vertices_per_subdivision_level = (3, 6, 15, 45)

# Offset into vertex index LUT (u major to hierarchical order) for subdivision levels 0 to 3
# 6  values for subdiv lvl 1
# 15 values for subdiv lvl 2
# 45 values for subdiv lvl 3
# cdef uint16_t *UMAJOR_TO_HIERARCHICAL_VTX_IDX_LUT_OFFSET = [0, 3, 9, 24, 69]
# cdef uint16_t *UMAJOR_TO_HIERARCHICAL_VTX_IDX_LUT = [
#     # level 0
#     0, 2, 1,
#     # level 1
#     0, 3, 2, 5, 4, 1,
#     # level 2
#     0, 6, 3, 12, 2, 8, 7, 14, 13, 5, 9, 4, 11, 10, 1,
#     # level 3
#     0, 15, 6, 21, 3, 39, 12, 42, 2, 17, 16, 23, 22, 41, 40, 44, 43, 8, 18, 7, 24, 14, 36, 13, 20, 19, 26, 25,
#     38, 37, 5, 27, 9, 33, 4, 29, 28, 35, 34, 11, 30, 10, 32, 31, 1 
# ]

DisplacementMicromapType = namedtuple("DisplacementMicromapType", ["format", "subdivision_level"])


cdef class DisplacementMicromapInput(OptixObject):
    """
    This class is a wrapper around an uint8-numpy array that will convert convert it into
    the format required by the optix displacement micromesh of the requested format and subdivision level.
    The class currently only supports inputs in a baked for all formats.

    Parameters
    ----------
    dmm_data: The input array in baked format. must be an array of type uint8 with the shape (num_triangles, num_subtriangles, num_bytes)
    fmt: Format specifier.
    """
    def __init__(self,
                 displacement,
                 subdivision_level: int,
                 format: DisplacementMicromapFormat):

        if not np.issubdtype(displacement, np.uint8):
            raise TypeError("Expected dtype=np.uint8.")

        fmt = DisplacementMicromapFormat(format)
        if fmt == DisplacementMicromapFormat.FORMAT_64_MICRO_TRIS_64_BYTES:
            dmm_subdivision_level_sub_triangles = max(0, int(subdivision_level) - 3)
            num_bytes_per_subtriangle = 64
        else:
            #if not baked:
            #    raise ValueError("Only baked input is supported for compressed formats")
            num_bytes_per_subtriangle = 128
            if fmt == DisplacementMicromapFormat.FORMAT_256_MICRO_TRIS_128_BYTES:
                dmm_subdivision_level_sub_triangles = max(0, int(subdivision_level) - 4)
            elif fmt == DisplacementMicromapFormat.FORMAT_1024_MICRO_TRIS_128_BYTES:
                dmm_subdivision_level_sub_triangles = 0
            else:
                raise ValueError("Invalid DisplacementMicroMapFormat.")

        n_subtriangles = 1 << (2 * dmm_subdivision_level_sub_triangles)

        #if baked:
        # reshape baked input into n_triangles, n_subtriangles, n_format_bytes for easier encoding
        displacement = displacement.reshape(-1, n_subtriangles, num_bytes_per_subtriangle)
        # TODO bake raw inputs
        #else:
        #    # reshape unbaked input into n_triangles, n_subtriangles, n_vertices
        #    displacement = displacement.reshape(-1, n_subtriangles, _n_vertices_per_subdivision_level[subdivision_level])
        #    
        self.buffer = displacement
        self.c_format = <OptixDisplacementMicromapFormat>fmt.value
        self.c_subdivision_level = subdivision_level

    @property
    def format(self):
        return DisplacementMicromapFormat(self.c_format)

    @property
    def subdivision_level(self):
        return self.c_subdivision_level

    @property
    def ntriangles(self):
        return self.buffer.shape[0]

    @property
    def nsubtriangles(self):
        return self.buffer.shape[1]

    @property
    def nbytes(self):
        return self.buffer.size * self.buffer.itemsize

    def _repr_details(self):
        return f"ntriangles={self.ntriangles}, nsubtriangles={self.nsubtriangles}, format={self.format.name}, subdivision_level={self.subdivision_level}"

    
cdef class DisplacementMicromapArray(OptixContextObject):
    """
    Class representing an array of displaced micromaps on the optix device.
    This class wraps the internal GPU buffer containing the micromap data and serves to build the structure from
    one or multiple DisplacementMicromapInput inputs

    Parameters
    ----------
    context:
        The device context to use.
    inputs:
        An iterable of DisplacementMicromapInput.
    prefer_fast_build:
        If True, it prefers a fast built of the array over a fast trace.
    stream:
        Cuda stream to use for building this micromap array. If None the default stream is used.
    """
    def __init__(self,
                 context: DeviceContext,
                 inputs: typ.Iterable[DisplacementMicromapInput],
                 prefer_fast_build: bool = False,
                 stream: typ.Optional[cp.cuda.Stream] = None):
        super().__init__(context)
        self.d_micromap_array_buffer = None
        self._micromap_types = None
        self._build_flags = OPTIX_DISPLACEMENT_MICROMAP_FLAG_NONE
        if prefer_fast_build:
            self._build_flags = OPTIX_DISPLACEMENT_MICROMAP_FLAG_PREFER_FAST_BUILD
        else:
            self._build_flags = OPTIX_DISPLACEMENT_MICROMAP_FLAG_PREFER_FAST_TRACE
        self.build(inputs, stream=stream)

    cdef void build(self, inputs, stream=None):
        # convert all inputs into the correct format first
        inputs = ensure_iterable(inputs)

        cdef OptixDisplacementMicromapArrayBuildInput build_input
        build_input.flags = self._build_flags

        cdef size_t inputs_size_in_bytes = 0
        micromap_counts = defaultdict(lambda: 0)
        micromap_types = []

        self.c_num_micromaps = 0
        # build the histogram from the input specifications and convert it into a cpp vector to pass it to the build input
        for i in inputs:
            omm_type = DisplacementMicromapType(i.format, i.subdivision_level)
            micromap_counts[omm_type] += i.ntriangles
            micromap_types.append(omm_type)
            inputs_size_in_bytes += i.nbytes
            self.c_num_micromaps += i.ntriangles
        self._micromap_types = tuple(micromap_types)

        cdef vector[OptixDisplacementMicromapHistogramEntry] histogram_entries
        histogram_entries.resize(len(micromap_counts))
        build_input.numDisplacementMicromapHistogramEntries = histogram_entries.size()

        for i, (k, v) in enumerate(micromap_counts.items()):
            histogram_entries[i].count = v
            histogram_entries[i].format = <OptixDisplacementMicromapFormat>k.format.value
            histogram_entries[i].subdivisionLevel = k.subdivision_level

        build_input.displacementMicromapHistogramEntries = histogram_entries.data()
        del micromap_counts

        # allocate a buffer to hold all input micromaps and put it's pointer in the build input
        d_displacement_values = cp.cuda.alloc(inputs_size_in_bytes)
        build_input.displacementValuesBuffer = d_displacement_values.ptr

        cdef unsigned int offset = 0
        cdef vector[OptixDisplacementMicromapDesc] descs
        cdef uint16_t[:, :] buffer_view
        cdef unsigned int t
        cdef unsigned int desc_i = 0;

        descs.resize(self.c_num_micromaps)
        #TODO use the actual triangles in the input array here!
        # copy all input data into the device buffer
        for i, inp in enumerate(inputs):
            buffer_view = (<DisplacementMicromapInput>inp).buffer
            num_bytes = inp.nbytes
            cp.cuda.runtime.memcpy(d_displacement_values.ptr + offset,
                                   <uintptr_t>&buffer_view[0,0],
                                   <size_t>num_bytes,
                                   cp.cuda.runtime.memcpyHostToDevice)
            for t in range(inp.ntriangles):
                # fill the descriptor array at the same time with to information in input
                descs[desc_i].byteOffset = offset
                offset += buffer_view.shape[1] * sizeof(uint16_t)
                descs[desc_i].subdivisionLevel = <unsigned short>inp.subdivision_level
                descs[desc_i].format = <unsigned short>inp.format.value
                desc_i += 1

        # copy the descriptor array onto the device
        cdef size_t desc_size_in_bytes = descs.size() * sizeof(OptixDisplacementMicromapDesc)

        d_desc_buffer = cp.cuda.alloc(desc_size_in_bytes)
        cp.cuda.runtime.memcpy(d_desc_buffer.ptr, <uintptr_t>descs.data(), desc_size_in_bytes, cp.cuda.runtime.memcpyHostToDevice)

        build_input.perDisplacementMicromapDescBuffer = d_desc_buffer.ptr
        build_input.perDisplacementMicromapDescStrideInBytes = 0

        cdef OptixMicromapBufferSizes build_sizes

        optix_check_return(optixDisplacementMicromapArrayComputeMemoryUsage(self.context.c_context,
                                                                       &build_input,
                                                                       &build_sizes))
        # TODO: do we have to align this buffer?
        self.d_micromap_array_buffer = cp.cuda.alloc(build_sizes.outputSizeInBytes)
        self._buffer_size = build_sizes.outputSizeInBytes

        d_temp_buffer = cp.cuda.alloc(build_sizes.tempSizeInBytes)

        cdef OptixMicromapBuffers micromap_buffers

        micromap_buffers.tempSizeInBytes = build_sizes.tempSizeInBytes
        micromap_buffers.temp = d_temp_buffer.ptr

        micromap_buffers.outputSizeInBytes = build_sizes.outputSizeInBytes
        micromap_buffers.output = self.d_micromap_array_buffer.ptr

        cdef uintptr_t c_stream = 0

        if stream is not None:
            c_stream = stream.ptr
        with nogil:
            optix_check_return(optixDisplacementMicromapArrayBuild(self.context.c_context,
                                                              <CUstream>c_stream,
                                                              &build_input,
                                                              &micromap_buffers))
        # all temporary buffers will be freed automatically here

    @property
    def types(self):
        return self._micromap_types

    def __deepcopy__(self, memo):
        """
        Perform a deep copy of the OpactiyMicromap by using the standard python copy.deepcopy function.
        """
        # relocate on the same device to perform a regular deep copy
        result = self.relocate(device=None)
        memo[id(self)] = result
        return result

    def _repr_details(self):
        return f"size={self._buffer_size}, nmicromaps={self.c_num_micromaps}"

    def relocate(self,
                 device: typ.Optional[DeviceContext] = None,
                 stream: typ.Optional[cp.cuda.Stream] = None) -> DisplacementMicromapArray:
        """
        Relocate this displacement micromap array into another copy. Usually this is equivalent to a deep copy.
        Additionally, the micromap array can be copied to a different device by specifying the device
        parameter with a different DeviceContext.

        THIS OPERATION IS CURRENTLY NOT SUPPORTED BY OPTIX!

        Parameters
        ----------
        device:
            An optional DeviceContext. The resulting copy of the micromap array will be copied
            to this device. If None, the micromap array's current device is used.
        stream:
            The stream to use for the relocation. If None, the default stream (0) is used.

        Returns
        -------
        copy: OpacityMicromapArray
            The copy of the OpacityMicromapArray on the new device
        """

        raise RuntimeError("Relocation of displacement micromap arrays is currently not supported.")


cdef class BuildInputDisplacementMicromap(OptixObject):
    """
    Build input for an array of micromaps. Inputs of this type can optionally be passed to a
    BuildInputTriangleArray to use micromaps for it's triangles. Additionally an array of the usage_counts
    for the OMMs in the Array needs to be passed as a list.
    If the indexing mode is specified as INDEXED, an additional index buffer containing an index into the omm array or one of
    the values in OpacityMicromapPredefinedIndex is required.

    Parameters
    ----------
    omm_array:
        The DisplacementMicromapArray to use by the triangles.
    usage_counts:
        The number of times each omm in the OpacityMicromapArray is used.
    indexing_mode:
        The indexing mode the omms should use. Must be one of the values in OpacityMicromapArrayIndexingMode.
        By default NONE is used, disabling the use of OMMs
    index_buffer:
        If the indexing_mode is INDEXED, this additional index buffer, specifiing an omm to use or the default value
        for each triangle in the geometry is required.
    """
    def __init__(self,
                 DisplacementMicromapArray omm_array,
                 displacement_directions: np.ndarray,
                 usage_counts: Sequence[int],
                 indexing_mode: DisplacementMicromapArrayIndexingMode = DisplacementMicromapArrayIndexingMode.NONE,
                 index_buffer = None,
                 bias_and_scale = None):
        indexing_mode = DisplacementMicromapArrayIndexingMode(indexing_mode)

        self.build_input.indexingMode = <OptixDisplacementMicromapArrayIndexingMode>indexing_mode.value

        if indexing_mode == DisplacementMicromapArrayIndexingMode.INDEXED:
            if index_buffer is None:
                raise ValueError("index_buffer is required for indexing_mode=INDEXED.")
            if not any(np.issubdtype(index_buffer.dtype, dt) for dt in (np.uint16, np.uint32)):
                raise ValueError("index_buffer must be of dtype np.uint16 or np.uint32")
            self._d_index_buffer = cp.asarray(index_buffer).ravel()
            # enable the index buffer in the build_input struct
            self.build_input.displacementMicromapIndexBuffer = self._d_index_buffer.data.ptr
            self.build_input.displacementMicromapIndexOffset = 0
            self.build_input.displacementMicromapIndexSizeInBytes = self._d_index_buffer.itemsize
            self.build_input.displacementMicromapIndexStrideInBytes = self._d_index_buffer.strides[0]
        else:
            self.build_input.displacementMicromapIndexBuffer = 0
            self.build_input.displacementMicromapIndexOffset = 0
            self.build_input.displacementMicromapIndexSizeInBytes = 0
            self.build_input.displacementMicromapIndexStrideInBytes = 0

        self.c_micromap_array = omm_array
        self.build_input.displacementMicromapArray = self.c_micromap_array.d_micromap_array_buffer.data.ptr
        
        # must be shape num_triangles, 3, 3
        displacement_directions = cp.asarray(displacement_directions).reshape(-1, 3, 3)
        if np.issubdtype(displacement_directions.dtype, np.float16):
            self.build_input.vertexDirectionFormat = OPTIX_DISPLACEMENT_MICROMAP_DIRECTION_FORMAT_HALF3
        else:
            displacement_directions = displacement_directions.astype(np.float32)
            self.build_input.vertexDirectionFormat = OPTIX_DISPLACEMENT_MICROMAP_DIRECTION_FORMAT_FLOAT3
        self._d_displacement_directions = displacement_directions
        self.build_input.vertexDirectionsBuffer = self._d_displacement_directions.data.ptr
        self.build_input.vertexDirectionStrideInBytes = self._d_displacement_directions.strides[1]
        
        if bias_and_scale is not None:
            bias_and_scale = cp.asarray(displacement_directions).reshape(-1, 3, 2)
            if bias_and_scale.shape[0] != displacement_directions.shape[0]:
                raise ValueError("Invalid shape of bias_and_scale array")
            if np.issubdtype(bias_and_scale.dtype, np.float16):
                self.build_input.vertexBiasAndScaleFormat = OPTIX_DISPLACEMENT_MICROMAP_BIAS_AND_SCALE_FORMAT_HALF2
            else:
                bias_and_scale = bias_and_scale.astype(np.float32)
                self.build_input.vertexBiasAndScaleFormat = OPTIX_DISPLACEMENT_MICROMAP_BIAS_AND_SCALE_FORMAT_FLOAT2

            self._d_bias_and_scale = displacement_directions
            self.build_input.vertexBiasAndScaleBuffer = self._d_bias_and_scale.data.ptr
            self.build_input.vertexBiasAndScaleStrideInBytes = self._d_bias_and_scale.strides[1]
        else:
            self.build_input.vertexBiasAndScaleBuffer = 0
            self.build_input.vertexBiasAndScaleStrideInBytes = 0
            self.build_input.vertexBiasAndScaleFormat = OPTIX_DISPLACEMENT_MICROMAP_BIAS_AND_SCALE_FORMAT_NONE


        # fill the usage counts vector from the specified usage array
        # TODO: is there a way to determine the usage count automatically?
        micromap_types = self.c_micromap_array.types
        if len(usage_counts) != len(micromap_types):
            raise IndexError(f"Number of entries in usage_count must be equal to the number of omms in micromap_array. "
                             f"Expected {len(micromap_types)}), got ({len(usage_counts)})")
        usage_count_hist = defaultdict(lambda: 0)

        for type, usage_count in zip(micromap_types, usage_counts):
            usage_count_hist[type] += usage_count

        self._usage_counts = usage_count_hist
        self.c_usage_counts.resize(len(usage_counts))
        for i, (k, v) in enumerate(usage_count_hist.items()):
            self.c_usage_counts[i].count = v
            self.c_usage_counts[i].format = <OptixDisplacementMicromapFormat>k.format.value
            self.c_usage_counts[i].subdivisionLevel = k.subdivision_level

        self.build_input.displacementMicromapUsageCounts = self.c_usage_counts.data()
        self.build_input.numDisplacementMicromapUsageCounts = self.c_usage_counts.size()

    @property
    def usage_counts(self):
        return self._usage_counts

    @property
    def types(self):
        return self.c_micromap_array.types

    @property
    def micromap_array(self):
        return self.c_micromap_array

    @property
    def index_buffer(self):
        return self._d_index_buffer

    @property
    def displacement_directions(self):
        return self._d_displacement_directions

    @property
    def bias_and_scale(self):
        return self._d_bias_and_scale