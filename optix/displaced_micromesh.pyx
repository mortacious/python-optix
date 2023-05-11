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

__all__ = []


class DisplacementMicromapArrayIndexingMode(IntEnum):
    NONE = OPTIX_DISPLACEMENT_MICROMAP_ARRAY_INDEXING_MODE_NONE
    LINEAR = OPTIX_DISPLACEMENT_MICROMAP_ARRAY_INDEXING_MODE_LINEAR
    INDEXED = OPTIX_DISPLACEMENT_MICROMAP_ARRAY_INDEXING_MODE_INDEXED


class DisplacementMicromapBiasAndScaleFormat(IntEnum):
    NONE = OPTIX_DISPLACEMENT_MICROMAP_BIAS_AND_SCALE_FORMAT_NONE
    FLOAT2 = OPTIX_DISPLACEMENT_MICROMAP_BIAS_AND_SCALE_FORMAT_FLOAT2
    HALF2 = OPTIX_DISPLACEMENT_MICROMAP_BIAS_AND_SCALE_FORMAT_HALF2


class DisplacementMicromapDirectionFormat(IntEnum):
    NONE = OPTIX_DISPLACEMENT_MICROMAP_DIRECTION_FORMAT_NONE
    FLOAT3 = OPTIX_DISPLACEMENT_MICROMAP_DIRECTION_FORMAT_FLOAT3
    HALF3 = OPTIX_DISPLACEMENT_MICROMAP_DIRECTION_FORMAT_HALF3


class DisplacementMicromapFormat(IntEnum):
    NONE = OPTIX_DISPLACEMENT_MICROMAP_FORMAT_NONE
    FORMAT_64_MICRO_TRIS_64_BYTES = OPTIX_DISPLACEMENT_MICROMAP_FORMAT_64_MICRO_TRIS_64_BYTES
    FORMAT_256_MICRO_TRIS_128_BYTES  = OPTIX_DISPLACEMENT_MICROMAP_FORMAT_256_MICRO_TRIS_128_BYTES
    FORMAT_1024_MICRO_TRIS_128_BYTES = OPTIX_DISPLACEMENT_MICROMAP_FORMAT_1024_MICRO_TRIS_128_BYTES


class DisplacementMicromapTriangleFlags(IntFlag):
    NONE = OPTIX_DISPLACEMENT_MICROMAP_TRIANGLE_FLAG_NONE
    DECIMATE_EDGE_01 = OPTIX_DISPLACEMENT_MICROMAP_TRIANGLE_FLAG_DECIMATE_EDGE_01
    DECIMATE_EDGE_12 = OPTIX_DISPLACEMENT_MICROMAP_TRIANGLE_FLAG_DECIMATE_EDGE_12
    DECIMATE_EDGE_20 = OPTIX_DISPLACEMENT_MICROMAP_TRIANGLE_FLAG_DECIMATE_EDGE_20


cdef class DisplacementMicromapInput(OptixObject):
    """

    Parameters
    ----------
    dmm_data: The input array in unbaked format. must be an array of type uint8 with the shape (num_triangles, num_subtriangles,
    format: Optional format specifier. Required for baked format, optional for unbaked. Invalid formats are not checked
            for unbaked inputs.
    """
    def __init__(self,
                 displacement,
                 subdivision_level: int,
                 format: DisplacementMicromapFormat):

        format = DisplacementMicromapFormat(format)
        if format == DisplacementMicromapFormat.FORMAT_64_MICRO_TRIS_64_BYTES:
            dmm_subdivision_level_sub_triangles = max(0, int(subdivision_level) - 3)
            num_bytes_per_subtriangle = 64
        elif format == DisplacementMicromapFormat.FORMAT_256_MICRO_TRIS_128_BYTES:
            dmm_subdivision_level_sub_triangles = max(0, int(subdivision_level) - 4)
            num_bytes_per_subtriangle = 128
        elif format == DisplacementMicromapFormat.FORMAT_1024_MICRO_TRIS_128_BYTES:
            dmm_subdivision_level_sub_triangles = 0
        else:
            raise ValueError("DisplacementMicroMapFormat cannot be None")

        n_subtriangles = 1 << (2 * dmm_subdivision_level_sub_triangles)
        dmm_data = dmm_data.reshape(-1, n_subtriangles, )
        buffer = np.atleast_2d(dmm_data)

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
    def ntriangles(self):
        return self.buffer.shape[0]

    @property
    def nbytes(self):
        return self.buffer.size * self.buffer.itemsize

    def _repr_details(self):
        return f"ntriangles={self.ntriangles}, format={self.format.name}, subdivision_level={self.subdivision_level}"