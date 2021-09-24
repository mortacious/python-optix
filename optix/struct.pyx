# distutils: language = c++

import numpy as np
import ctypes
import cupy as cp
from .common import round_up
from .common cimport optix_init
cimport cython
#cimport numpy as np

optix_init()

def  _aligned_itemsize( formats, alignment ):
    names = []
    for i in range( len(formats ) ):
        names.append( 'x'+str(i) )

    temp_dtype = np.dtype( {
        'names'   : names,
        'formats' : formats,
        'align'   : True
        } )
    return round_up( temp_dtype.itemsize, alignment )


def array_to_device_memory(numpy_array, stream=None):
    if stream is None:
        stream = cp.cuda.Stream()

    byte_size = numpy_array.size*numpy_array.dtype.itemsize

    h_ptr = ctypes.c_void_p( numpy_array.ctypes.data )
    d_mem = cp.cuda.memory.alloc( byte_size )
    d_mem.copy_from_async( h_ptr, byte_size, stream )

    return d_mem

SBT_RECORD_ALIGNMENT = OPTIX_SBT_RECORD_ALIGNMENT
SBT_RECORD_HEADER_SIZE = OPTIX_SBT_RECORD_HEADER_SIZE


cdef class _StructHelper(object):
    def __init__(self, names=(), formats=(), size=1, alignment=1):
        self.array_values = {} # init dict
        self.dtype = self._convert_to_aligned_dtype(names, formats, alignment)
        self._array = self._create(size)

    @property
    def array(self):
        return self._array

    @classmethod
    def from_dtype(cls, dtype, size=1, alignment=1):
        try:
            descr = dtype.descr
        except AttributeError:
            raise ValueError("Not a structured dtype")
        names, formats = zip(*descr)
        return cls(names=names, formats=formats, size=1, alignment=1)

    def _convert_to_aligned_dtype(self, names, formats, alignment):
        """
        Construct an aligned dtype from the names and formats

        Parameters
        ----------
        names
        formats

        Returns
        -------

        """
        itemsize = _aligned_itemsize(formats, alignment)
        dtype = np.dtype({
            'names': names,
            'formats': formats,
            'itemsize': itemsize,
            'align': True
        })

        return dtype

    def _prepare_array(self, array):
        return array

    def _create(self, size):
        array = np.zeros(size, dtype=self.dtype)
        array = self._prepare_array(array)
        return array

    def __setitem__(self, key, value):
        value = (value,) if not isinstance(value, (list, tuple)) else value
        # special hook for cupy arrays if a pointer is to be stored
        if np.issubdtype(self.dtype.fields[key][0], 'u8') and all(isinstance(v, cp.ndarray) for v in value):
            self.array_values[key] = value
            value = [v.data.ptr for v in value]
        self.array[key] = value

    def __getitem__(self, key):
        return self.array[key]

    def to_gpu(self, stream=None):
        size = self.array.shape[0]
        mem = array_to_device_memory(self.array, stream=stream)
        array = cp.ndarray(size, dtype=self.dtype, memptr=mem)
        return array

    @property
    def itemsize(self):
        return self.dtype.itemsize

    @property
    def size(self):
        return self._array.shape[0]


cdef class SbtRecord(_StructHelper):
    def __init__(self, ProgramGroup program_group, names=(), formats=(), size=1):
        self.program_group = program_group
        header_format = '{}B'.format(OPTIX_SBT_RECORD_HEADER_SIZE)
        names = ('header',) + names
        formats = (header_format,) + formats
        super().__init__(names, formats, size=size, alignment=OPTIX_SBT_RECORD_ALIGNMENT)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _prepare_array(self, array):
        cdef size_t itemsize = array.dtype.itemsize
        cdef size_t i
        cdef size_t size = array.shape[0]
        cdef unsigned char[:, ::1] buffer =  array.view('B').reshape(-1, itemsize)
        for i in range(size):
            optixSbtRecordPackHeader(self.program_group._program_group, <void *>(&buffer[i, 0]))
        return array
        #insert_array_sbt_header(, array)
        # cdef size_t itemsize = self.dtype.itemsize
        # cdef size_t i
        # cdef size_t size = array.shape[0]
        # cdef unsigned char[:, ::1] buffer =  array.view('B').reshape(-1, itemsize)
        # # TODO make sure this loop is all in C
        # for i in range(size):
        #     optixSbtRecordPackHeader(self.program_group._program_group, <void*>(&buffer[i, 0]))
        #return array


cdef class LaunchParamsRecord(_StructHelper):
    def __init__(self, names=(), formats=(), size=1):
        # init with 8 bytes alignment #TODO is there a constant for that?
        super().__init__(names, formats, size=size, alignment=8)


