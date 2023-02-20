# distutils: language = c++

import numpy as np
cimport numpy as np
np.import_array()

import cupy as cp
from .common cimport optix_check_return, optix_init

cimport cython
from cython.operator import dereference
from libc.stdint cimport uint8_t, uint32_t, uintptr_t
from libcpp cimport bool
from libcpp.vector cimport vector
from libc.string cimport memset
from collections import defaultdict
from enum import IntEnum, IntFlag
import typing as typ
from .common import ensure_iterable
from .context cimport DeviceContext

optix_init()

__all__ = ['micromap_indices_to_base_barycentrics',
           'OpacityMicromapFormat',
           'OpacityMicromapState',
           'OpacityMicromapInput',
           'OpacityMicromapArray']


cdef bool valid_subdivision_level(uint8_t[:, :] opacity):
    return (np.log2(opacity.shape[1]) / 2).is_integer()

cdef bool is_baked(uint8_t[:, :] opacity):
    return opacity[0, 0] > 3


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def micromap_indices_to_base_barycentrics(uint32_t[:] indices, uint32_t subdivision_level = 0):
    """
    Converts micromap triangle indices to three base-triangle barycentric coordinates of the micro triangle vertices.
    The base-triangle is the triangle that the micromap is applied to.

    Parameters
    ----------
    indices: Indices of the micro triangles within a micromap.
    subdivision_level: Subdivision level of the micromap.

    Returns
    -------
    base_barycentrics_0: Barycentric coordinates in the space of the base triangle of vertex 0 of the micro triangle.
    base_barycentrics_1: Barycentric coordinates in the space of the base triangle of vertex 1 of the micro triangle.
    base_barycentrics_2: Barycentric coordinates in the space of the base triangle of vertex 2 of the micro triangle.
    """
    cdef Py_ssize_t num_indices = indices.shape[0]

    barycentrics_0 = np.empty((num_indices, 2), dtype=np.float32)
    barycentrics_1 = np.empty((num_indices, 2), dtype=np.float32)
    barycentrics_2 = np.empty((num_indices, 2), dtype=np.float32)

    cdef float[:, ::1] barycentrics_0_view = barycentrics_0
    cdef float[:, ::1] barycentrics_1_view = barycentrics_1
    cdef float[:, ::1] barycentrics_2_view = barycentrics_2

    cdef unsigned int i
    with nogil:
        for i in range(num_indices):
            optixMicromapIndexToBaseBarycentrics(indices[i],
                                                 subdivision_level,
                                                 dereference(<float2*>&barycentrics_0_view[i, 0]),
                                                 dereference(<float2*>&barycentrics_1_view[i, 0]),
                                                 dereference(<float2*>&barycentrics_2_view[i, 0]))

    return barycentrics_0, barycentrics_1, barycentrics_2


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)
def bake_opacity_micromap(uint8_t[:, :] opacity, format = None):
    cdef Py_ssize_t num_tris = opacity.shape[0]
    cdef Py_ssize_t num_micro_tris = opacity.shape[1]

    if not valid_subdivision_level(opacity):
        raise ValueError(f"Shape of input ({opacity.shape[1]}) does "
                         f"not correspond to a valid subdivision level")

    cdef uint8_t bits_per_state

    if format is None:
        if np.any(np.ravel(opacity) > 1):
            bits_per_state = 2
        else:
            bits_per_state = 1
    else:
        bits_per_state = OpacityMicromapFormat(format).value

    # create the array to bake the opacities into
    opacity_baked = np.zeros((opacity.shape[0], opacity.shape[1] // 8 * bits_per_state), dtype=np.uint8)

    cdef unsigned int bake_stride = 8 // bits_per_state

    cdef uint8_t[:, :] opacity_baked_view = opacity_baked
    cdef unsigned int baked_index
    cdef unsigned int i, j
    with nogil:
        for i in range(num_tris):
            for j in range(0, num_micro_tris):
                baked_index = j // bake_stride
                opacity_baked_view[i, baked_index] |= opacity[i, j] << ((j%bake_stride) * bits_per_state)

    return opacity_baked, OpacityMicromapFormat(bits_per_state)




class OpacityMicromapArrayIndexingMode(IntEnum):
    NONE = OPTIX_OPACITY_MICROMAP_ARRAY_INDEXING_MODE_NONE,
    LINEAR = OPTIX_OPACITY_MICROMAP_ARRAY_INDEXING_MODE_LINEAR,
    INDEXED = OPTIX_OPACITY_MICROMAP_ARRAY_INDEXING_MODE_INDEXED


class OpacityMicromapFormat(IntEnum):
    NONE = OPTIX_OPACITY_MICROMAP_FORMAT_NONE, # invalid format
    TWO_STATE = OPTIX_OPACITY_MICROMAP_FORMAT_2_STATE  # 0: Transparent, 1: Opaque
    FOUR_STATE = OPTIX_OPACITY_MICROMAP_FORMAT_4_STATE  # 0: Transparent, 1: Opaque, 2: Unknown-Transparent, 3: Unknown-Opaque


class OpacityMicromapState(IntEnum):
    """
    This enum wraps the OPTIX_OPACITY_MICROMAP_STATE_* defines from optix
    """
    TRANSPARENT = OPTIX_OPACITY_MICROMAP_STATE_TRANSPARENT_DEFINE
    OPAQUE = OPTIX_OPACITY_MICROMAP_STATE_OPAQUE_DEFINE
    UNKNOWN_TRANSPARENT = OPTIX_OPACITY_MICROMAP_STATE_UNKNOWN_TRANSPARENT_DEFINE
    UNKNOWN_OPAQUE = OPTIX_OPACITY_MICROMAP_STATE_UNKNOWN_OPAQUE_DEFINE

    def bits_per_state(self):
        """
        Returns the number of bits (either 1 or 2) required to encode this micromap state.
        """
        if self < 2:
            return 1
        return 2

    def format(self):
        """
        Returns the OpacityMicromapFormat to encode this state. This is either
        OpacityMicromapFormat.TWO_STATE or OpacityMicromapFormat.FOUR_STATE.
        """
        if self < 2:
            return OpacityMicromapFormat.TWO_STATE
        return OpacityMicromapFormat.FOUR_STATE


cdef class OpacityMicromapInput(OptixObject):
    """
    This class is a simple wrapper around an uint8-numpy array that will convert convert it into
    the format required by the optix opacity micromaps while keeping track of the bit encoding and subdivision
    level. The class supports inputs in an unbaked format (uint8 array with the values in OpacityMicromapFormat) as
    well as baked formats (values encoded in either 1 or 2 bits).

    Parameters
    ----------
    opacity: The input array in either unbaked or baked format.
    format: Optional format specifier. Required for baked format, optional for unbaked. Invalid formats are not checked
            for unbaked inputs.
    """
    def __init__(self,
                 uint8_t[:, :] opacity,
                 format: typ.Optional[OpacityMicromapFormat] = None):
        """


        """
        buffer = np.atleast_2d(opacity)

        format = OpacityMicromapFormat(format) if format is not None else None
        if is_baked(buffer):
            if not format:
                raise ValueError("Baked input requires a format specification")
            shape_unbaked = (buffer.dtype.itemsize * 8 * buffer.shape[1]) / format.value
            subdivision_level = (np.log2(shape_unbaked) / 2)

            if not subdivision_level.is_integer():
                ValueError(f"Shape of baked input ({opacity.shape[1]}) does "
                           f"not correspond to a valid subdivision level in given format ({format}).")
        else:
            subdivision_level = (np.log2(buffer.shape[1]) / 2)
            buffer, fmt = bake_opacity_micromap(buffer, format)

            # elif format != fmt:
            #     raise ValueError(f"Attempt to bake the micromap input resulted in a different format than the given one. "
            #                      f"{format} != {fmt}.")

        self.buffer = buffer
        self.c_format = <OptixOpacityMicromapFormat>format.value
        self.c_subdivision_level = subdivision_level

    @property
    def format(self):
        return OpacityMicromapFormat(self.c_format)

    @property
    def subdivision_level(self):
        return self.c_subdivision_level

    @property
    def nbytes(self):
        return self.buffer.size * self.buffer.itemsize


# ctypedef pair[OptixOpacityMicromapFormat, int] histogram_entry
#
# cdef bint histogram_entry_hash(const histogram_entry& s) nogil:
#     return std_hash[OptixOpacityMicromapFormat]()(s.first) ^ std_hash[int]()(s.second)


cdef class OpacityMicromapArray(OptixContextObject):
    """
    Class representing an array of opacity micromaps on the optix device.
    This class wraps the internal GPU buffer containing the micromap data and serves to build the structure from
    one or multiple OpactiyMircomap inputs

    Parameters
    ----------
    context:
        The device context to use.
    inputs:
        An iterable of OpacityMicromapInput or numpy ndarrays. All numpy arrays will be converted into
        OpacityMicroMapInput automatically in this class.
    flags:
        A set of OpacityMicromapFlags to use for building this array. If None, the default OpacityMicromapFlags.NONE
        is used.
    stream:
        Cuda stream to use for building this micromap array. If None the default stream is used.
    """
    def __init__(self,
                 context: DeviceContext,
                 inputs: typ.Iterable[typ.Union[np.ndarray, OpacityMicromapInput]],
                 prefer_fast_build: bool = False,
                 stream: typ.Optional[cp.cuda.Stream] = None):
        super().__init__(context)
        self._d_micromap_array_buffer = None
        self._build_flags = OPTIX_OPACITY_MICROMAP_FLAG_NONE
        if prefer_fast_build:
            self._build_flags = OPTIX_OPACITY_MICROMAP_FLAG_PREFER_FAST_BUILD
        else:
            self._build_flags = OPTIX_OPACITY_MICROMAP_FLAG_PREFER_FAST_TRACE
        self.build(inputs, stream=stream)

    cdef void build(self, inputs, stream=None):
        # convert all inputs into the correct format first
        inputs = [OpacityMicromapInput(inp) if not isinstance(inp, OpacityMicromapInput)
                  else inp for inp in ensure_iterable(inputs)]

        cdef OptixOpacityMicromapArrayBuildInput build_input
        build_input.flags = self._build_flags

        cdef size_t inputs_size_in_bytes = 0
        # build the histogram from the input specifications and convert it into a cpp vector to pass it to the build input
        counts = defaultdict(lambda: 0)
        for i in inputs:
            counts[(i.format, i.subdivision_level)] += 1
            inputs_size_in_bytes += i.nbytes

        cdef vector[OptixOpacityMicromapHistogramEntry] histogram_entries
        histogram_entries.resize(len(counts))
        build_input.numMicromapHistogramEntries = histogram_entries.size()

        for i, (k, v) in enumerate(counts.items()):
            histogram_entries[i].count = v
            histogram_entries[i].format = <OptixOpacityMicromapFormat>k[0].value
            histogram_entries[i].subdivisionLevel = k[1]

        build_input.micromapHistogramEntries = histogram_entries.data()

        # allocate a buffer to hold all input micromaps and put it's pointer in the build input
        d_input_buffer = cp.cuda.alloc(inputs_size_in_bytes)
        build_input.inputBuffer = d_input_buffer.ptr

        cdef unsigned int offset = 0
        cdef vector[OptixOpacityMicromapDesc] descs
        cdef uint8_t[:, :] buffer_view

        descs.resize(len(inputs))

        # copy all input data into the device buffer
        for i, inp in enumerate(inputs):
            buffer_view = inp.buffer
            num_bytes = inp.nbytes
            cp.cuda.runtime.memcpy(d_input_buffer.ptr + offset,
                                   &buffer_view[0,0],
                                   num_bytes,
                                   cp.cuda.runtime.memcpyHostToDevice)
            # fill the descriptor array at the same time with to information in input
            descs[i].byteOffset = offset
            descs[i].subdivisionLevel = <unsigned short>inp.subdivision_level
            descs[i].format = <unsigned short>inp.format.value

            offset += num_bytes

        # copy the descriptor array onto the device
        cdef size_t desc_size_in_bytes = descs.size() * sizeof(OptixOpacityMicromapDesc)
        d_desc_buffer = cp.cuda.alloc(desc_size_in_bytes)
        cp.cuda.runtime.memcpy(d_desc_buffer.ptr, <uintptr_t>descs.data(), desc_size_in_bytes, cp.cuda.runtime.memcpyHostToDevice)

        build_input.perMicromapDescBuffer = d_desc_buffer.ptr
        build_input.perMicromapDescStrideInBytes = 0

        cdef OptixMicromapBufferSizes build_sizes

        optix_check_return(optixOpacityMicromapArrayComputeMemoryUsage(self.context.c_context,
                                                                       &build_input,
                                                                       &build_sizes))

        # TODO: do we have to align this buffer?
        self._d_micromap_array_buffer = cp.cuda.alloc(build_sizes.outputSizeInBytes)
        d_temp_buffer = cp.cuda.alloc(build_sizes.tempSizeInBytes)

        cdef OptixMicromapBuffers micromap_buffers

        micromap_buffers.tempSizeInBytes = build_sizes.tempSizeInBytes
        micromap_buffers.temp = d_temp_buffer.ptr

        micromap_buffers.outputSizeInBytes = build_sizes.outputSizeInBytes
        micromap_buffers.output = self._d_micromap_array_buffer.ptr

        cdef uintptr_t c_stream = 0

        if stream is not None:
            c_stream = stream.ptr
        with nogil:
            optix_check_return(optixOpacityMicromapArrayBuild(self.context.c_context,
                                                              <CUstream>c_stream,
                                                              &build_input,
                                                              &micromap_buffers))
        # all temporary buffers will be freed automatically here

    def __deepcopy__(self, memo):
        """
        Perform a deep copy of the OpactiyMicromap by using the standard python copy.deepcopy function.
        """
        # relocate on the same device to perform a regular deep copy
        result = self.relocate(device=None)
        memo[id(self)] = result
        return result

    def relocate(self,
                 device: typ.Optional[DeviceContext] = None,
                 stream: typ.Optional[cp.cuda.Stream] = None) -> OpacityMicromapArray:
        """
        Relocate this opacity micromap array into another copy. Usually this is equivalent to a deep copy.
        Additionally, the micromap array can be copied to a different device by specifying the device
        parameter with a different DeviceContext.

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
        # first determine the relocation info for this micromap array
        cdef OptixRelocationInfo micromap_info
        memset(&micromap_info, 0, sizeof(OptixRelocationInfo))  # init struct to 0

        optix_check_return(optixOpacityMicromapArrayGetRelocationInfo(self.context.c_context,
                                                                      self._d_micromap_array_buffer, &micromap_info))

        if device is None:
            device = self.context

        # check if the new device is compatible with this micromap array
        cdef int compatible = 0
        optix_check_return(optixCheckRelocationCompatibility(<OptixDeviceContext>(<DeviceContext>device).c_context,
                                                                  &micromap_info,
                                                                  &compatible))

        if compatible != 1:
            raise RuntimeError("Device is not compatible for relocation of opacity micromap array")

        # do the relocation
        cls = self.__class__
        cdef OpacityMicromapArray result = cls.__new__(cls)

        result.context = device
        result._build_flags = self._build_flags
        result._buffer_size = self._buffer_size
        # TODO: align this?
        result._gas_buffer = cp.cuda.alloc(result._buffer_size)
        cp.cuda.runtime.memcpy(result._d_micromap_array_buffer.ptr,
                               self._d_micromap_array_buffer.ptr,
                               result._buffer_size,
                               cp.cuda.runtime.memcpyDeviceToDevice)

        cdef uintptr_t c_stream = 0
        if stream is not None:
            c_stream = stream.ptr

        optix_check_return(optixOpacityMicromapArrayRelocate(result.context.c_context,
                                                             <CUstream>c_stream,
                                                             &micromap_info,
                                                             result._d_micromap_array_buffer.ptr,
                                                             result._buffer_size))

        return result


