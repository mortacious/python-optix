# distutils: language = c++

from .struct cimport SbtRecord
from .struct import SbtRecord, array_to_device_memory
from .common cimport optix_init
import cupy as cp
import numpy as np

optix_init()

cdef class ShaderBindingTable:
    def __init__(self,
                 stream=None,
                 raygen_record=None,
                 exception_record=None,
                 miss_records=None,
                 miss_records_count=0,
                 miss_records_stride=0,
                 hitgroup_records=None,
                 hitgroup_records_count=0,
                 hitgroup_records_stride=0,
                 callables_records=None,
                 callables_records_count=0,
                 callables_records_strides=0
                 ):
        if raygen_record is not None:
            self._d_raygen_record, _, _ = self._process_record(raygen_record, None, None, stream)
            self.sbt.raygenRecord = self._d_raygen_record.ptr

        if exception_record is not None:
            self._d_exception_record, _, _ = self._process_record(exception_record, None, None, stream)
            self.sbt.exceptionRecord = self._d_exception_record.ptr

        if miss_records is not None:
            self._d_miss_records, self.sbt.missRecordCount, self.sbt.missRecordStrideInBytes = self._process_record(miss_records,
                                                                                                                    miss_records_count,
                                                                                                                    miss_records_stride,
                                                                                                                    stream)
            self.sbt.missRecordBase = self._d_miss_records.ptr


        if hitgroup_records is not None:
            self._d_hitgroup_records, self.sbt.hitgroupRecordCount, self.sbt.hitgroupRecordStrideInBytes = self._process_record(hitgroup_records,
                                                                                                                                hitgroup_records_count,
                                                                                                                                hitgroup_records_stride,
                                                                                                                                stream)

            self.sbt.hitgroupRecordBase = self._d_hitgroup_records.ptr

        if callables_records is not None:
            self._d_callables_records, self.sbt.callablesRecordCount, self.sbt.callablesRecordStrideInBytes = self._process_record(callables_records,
                                                                                                                                   callables_records_count,
                                                                                                                                   callables_records_strides,
                                                                                                                                   stream)
            self.sbt.callablesRecordBase = self._d_callables_records.ptr

    def _process_record(self, record, count, strides, stream=None):
        if isinstance(record, SbtRecord):
            record_buffer = record.to_gpu(stream=stream)
            return record_buffer.data, record.size, record.itemsize
        elif isinstance(record, cp.cuda.MemoryPointer):
            if count <= 0 or strides <= 0:
                raise ValueError("Got Memorypointer for record but strides and count are not defined")
            return record, count, strides # do nothing
        elif isinstance(record, np.ndarray):
            if "header" not in record.dtype.fields:
                raise ValueError("Malformed numpy array for SBT")
            record = record.ravel()
            return array_to_device_memory(record, stream=stream), record.shape[0], record.strides[0]
        else:
            raise ValueError(f"Unsupported record type '{type(record)}'")

