# distutils: language = c++

import numpy as np
import ctypes
import cupy as cp
from .common import round_up, ensure_iterable
from .common cimport optix_init
cimport cython
from collections.abc import Mapping

optix_init()

def  _aligned_itemsize( formats, alignment ):
    names = []
    for i in range( len(formats ) ):
        names.append( 'x'+str(i) )

    temp_dtype = np.dtype( {
        'names'   : names,
        'formats' : formats,
        }, align=True)
    return round_up( temp_dtype.itemsize, alignment )

def array_to_device_memory(numpy_array, stream=None):
    """
    Transfer a numpy array to cuda device memory. This does not generate a full cupy.ndarray, but an
    opaque cupy.cuda.MemoryPointer to support structured arrays which cupy currently does not
    (see https://github.com/cupy/cupy/issues/2031).

    Parameters
    ----------
    numpy_array: numpy.ndarray
        The array to transfer to the GPU
    stream: cupy.cuda.Stream, optional
        The stream to use. If None, the default stream is used.

    Returns
    -------
    d_mem: cupy.cuda.MemoryPointer
        Pointer to the memory on the GPU

    """
    byte_size = numpy_array.size*numpy_array.dtype.itemsize

    h_ptr = ctypes.c_void_p( numpy_array.ctypes.data )
    d_mem = cp.cuda.memory.alloc( byte_size )
    d_mem.copy_from_async( h_ptr, byte_size, stream )

    return d_mem

SBT_RECORD_ALIGNMENT = OPTIX_SBT_RECORD_ALIGNMENT
SBT_RECORD_HEADER_SIZE = OPTIX_SBT_RECORD_HEADER_SIZE


cdef class _StructHelper:
    """
    Helper class to create and manage a struct on the GPU. This class will construct a
    numpy structured array using the names and formats provided and transfer it to the GPU on demand.
    In order to access and modify the data the standard mapping syntax is used.

    This class is meant for internal use und should not be instanced directly.
    Parameters
    ----------
    names: tuple[str]
        The names of the struct's fields
    formats: tuple[str or numpy.dtype]
        The data types of the struct's fields
    values: tuple[scalar or array], optional
        Optional values for the fields. If not provided, they will be initialized to zero.
    size: int, optional
        The number of Records to generate. By default a single Record is created.
    alignment: int
        Byte alignment to use.
    """


    def __init__(self, names=(), formats=(), values=None, size=1, alignment=1):
        names = ensure_iterable(names)
        formats = ensure_iterable(formats)

        self.array_values = {} # init dict
        self._dtype = self._convert_to_aligneddtype(names, formats, alignment)
        self._array = self._create(size)

        if values is not None:
            for name, value in names, values:
                if value is not None:
                    self.__setitem__(name, value)

    @property
    def dtype(self):
        return self._dtype

    @property
    def array(self):
        return self._array

    @classmethod
    def fromdtype(cls, dtype, values=None, size=1, alignment=1):
        """
        Create the struct from a numpy structured dtype.

        Parameters
        ----------
        dtype: numpy.dtype
            The structured dtype to use
        values: tuple[scalar or array], optional
            The values to use for initialization.
        size: int
            The number of records to generate.
        alignment: int
            Byte alignment to use.

        Returns
        -------
        obj: _StructHelper
            The created struct.
        """
        try:
            descr = dtype.descr
        except AttributeError:
            raise ValueError("Not a structured dtype")
        names, formats = zip(*descr)
        return cls(names=names, formats=formats, values=values, size=1, alignment=1)

    @classmethod
    def from_dict(cls, dict_data: Mapping, size=1, alignment=1):
        """
        Create a struct from a dictionary. The following format is expected:
        {
            field_name: (format, (optional values))
        }

        Parameters
        ----------
        dict_data: dict or Mapping
            The dict data to build the struct in the format specified above
        size: int
            The number of records to generate
        alignment: int
            Byte alignment to use.

        Returns
        -------
        obj: _StructHelper
            The created struct.
        """
        names = []
        formats = []
        values = []
        for k, v in dict_data.items():
            names.append(k)
            v = ensure_iterable(v)
            formats.append(v[0])
            if len(v) > 1:
                values.append(v[1])
        return cls(names=names, formats=formats, values=values, size=size, alignment=1)

    def _convert_to_aligneddtype(self, names, formats, alignment):
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
        }, align=True)
        assert dtype.isalignedstruct
        return dtype

    def _prepare_array(self, array):
        return array

    def _create(self, size):
        array = np.zeros(size, dtype=self._dtype)
        array = self._prepare_array(array)
        return array

    def __setitem__(self, key, value):
        value = (value,) if not isinstance(value, (list, tuple)) else value
        # special hook for cupy arrays if a pointer is to be stored
        if np.issubdtype(self._dtype.fields[key][0], 'u8') and all(isinstance(v, cp.ndarray) for v in value):
            self.array_values[key] = value
            value = [v.data.ptr for v in value]
        self.array[key] = value

    def __getitem__(self, key):
        return self.array[key]

    def __len__(self):
        return len(self._dtype.fields)

    def __iter__(self):
        yield from self.keys()

    def __contains__(self, item):
        return item in self._dtype.fields

    def get(self, item, value=None):
        try:
            return self.array[item]
        except KeyError:
            return value

    def keys(self):
        yield from self._dtype.fields

    def values(self):
        for k in self.keys():
            yield self.array[k]

    def items(self):
        for k in self.keys():
            yield k, self.array[k]

    def to_gpu(self, stream=None):
        """
        Transfer the struct to the gpu and return a cupy.cuda.MemoryPointer.

        Parameters
        ----------
        stream: cupy.cuda.Stream, optional
            The stream to use for the transfer. If None the default stream is used.
        Returns
        -------
        pointer: cupy.cuda.MemoryPointer
            Pointer to the transferrred data.
        """
        size = self.array.shape[0]
        mem = array_to_device_memory(self.array, stream=stream)
        array = cp.ndarray(size, dtype=self._dtype, memptr=mem)
        return array

    @property
    def itemsize(self):
        return self._dtype.itemsize

    @property
    def size(self):
        return self._array.shape[0]

    def _repr_details(self):
        return f"size {self.size}, dtype {self._dtype}"


cdef class SbtRecord(_StructHelper):
    """
    Subclass to generate a valid Record for the ShaderBindingTable.
    All options are the same as in the base class.
.   The alignment parameter is ignored though and only present for the interface.
    """
    def __init__(self, program_groups, names=(), formats=(), values=None):
        program_groups = list(ensure_iterable(program_groups))
        names = ensure_iterable(names)
        formats = ensure_iterable(formats)
        
        if not all(isinstance(p, ProgramGroup) for p in program_groups):
            raise TypeError("Only program groups")
        
        cdef unsigned int num_program_groups = len(program_groups)
        
        self.program_groups = program_groups

        header_format = '{}B'.format(OPTIX_SBT_RECORD_HEADER_SIZE)
        names = ('header',) + names
        formats = (header_format,) + formats
        super().__init__(names, formats, values=values, size=num_program_groups, alignment=OPTIX_SBT_RECORD_ALIGNMENT)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _prepare_array(self, array):
        cdef size_t itemsize = array.dtype.itemsize
        cdef size_t i
        cdef size_t size = array.shape[0]
        cdef unsigned char[:, ::1] buffer =  array.view('B').reshape(-1, itemsize)
        for i in range(size):
            optixSbtRecordPackHeader((<ProgramGroup>self.program_groups[i]).program_group, <void *>(&buffer[i, 0]))
        return array

    def update_program_group(self, i, program_group):
        if not isinstance(program_group, ProgramGroup):
            raise TypeError("Expected a program group as second argument.")
        self.program_groups[i] = program_group
        
        cdef size_t itemsize = self._array.dtype.itemsize
        cdef unsigned char[:, ::1] buffer = self._array.view('B').reshape(-1, itemsize)
        optixSbtRecordPackHeader((<ProgramGroup>self.program_groups[<size_t>i]).program_group, <void *>(&buffer[<size_t>i, 0]))


cdef class LaunchParamsRecord(_StructHelper):
    """
    Subclass to generate a valid Record for the Pipeline launch parameters.
    All options are the same as in the base class.
    The alignment parameter is ignored though and only present for the interface.
    """
    def __init__(self, names=(), formats=(), values=None, size=1, alignment=8):
        # init with 8 bytes alignment #TODO is there a constant for that?
        super().__init__(names, formats, values=None, size=size, alignment=8)



