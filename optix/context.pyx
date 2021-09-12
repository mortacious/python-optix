# distutils: language = c++

from .common cimport optix_check_return, optix_init
from libc.stdint cimport uintptr_t, int32_t
import cupy as cp

optix_init()

cdef void context_log_cb(unsigned int level, const char* tag, const char* message, void* cbdata) with gil:
    cdef object cb_object = <object>cbdata # cast the cbdata to a pyobject
    cb_object(level, tag, message)


#cdef void init_optix() with gil:#
#
#    optix_check_return(optixInit())


cdef class DeviceContext:
    def __init__(self, object log_callback_function = None, int32_t log_callback_level = 1, bint validation_mode = False):
        cdef OptixDeviceContextOptions options
        cp.cuda.runtime.free(0)

        self._log_callback_function = log_callback_function

        if self._log_callback_function is not None:
            options.logCallbackFunction = context_log_cb
            options.logCallbackData = <void*>self._log_callback_function # cast to pointer here and keep the python object stored in this class

        self._log_callback_level = log_callback_level
        options.logCallbackLevel = self._log_callback_level
        self._validation_mode = validation_mode
        options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL if validation_mode else OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF
        with nogil:
            optix_check_return(optixDeviceContextCreate(<CUcontext>0, &options, &self.device_context))

    def __dealloc__(self):
        optix_check_return(optixDeviceContextDestroy(self.device_context))

    def __eq__(self, DeviceContext other):
        return self.device_context == other.device_context

    @property
    def cache_database_sizes(self):
        cdef size_t low_water_mark, high_water_mark
        optix_check_return(optixDeviceContextGetCacheDatabaseSizes(self.device_context, &low_water_mark, &high_water_mark))
        return low_water_mark, high_water_mark

    @cache_database_sizes.setter
    def cache_database_sizes(self, tuple sizes):
        cdef size_t low_water_mark, high_water_mark
        low_water_mark, high_water_mark = sizes
        optix_check_return(optixDeviceContextSetCacheDatabaseSizes(self.device_context, low_water_mark, high_water_mark))

    @property
    def cache_enabled(self):
        cdef bint enabled
        optix_check_return(optixDeviceContextGetCacheEnabled(self.device_context, <int*>&enabled))
        return enabled

    @cache_enabled.setter
    def cache_enabled(self, bint enabled):
        optix_check_return(optixDeviceContextSetCacheEnabled(self.device_context, <int>enabled))

    @property
    def cache_location(self):
        cdef char[2048] location
        cdef bytes py_location
        optix_check_return(optixDeviceContextGetCacheLocation(self.device_context, location, 2048))
        py_location = location
        return py_location.decode()

    @cache_location.setter
    def cache_location(self, str location):
        cdef bytes py_location = location.encode('ascii')
        optix_check_return(optixDeviceContextSetCacheLocation(self.device_context, py_location))

    cdef unsigned int _get_property(self, OptixDeviceProperty property):
        cdef unsigned int retval
        optix_check_return(optixDeviceContextGetProperty(self.device_context, property, &retval, sizeof(unsigned int)))
        return retval

    @property
    def log_callback(self):
        return self._log_callback_function

    @log_callback.setter
    def log_callback(self, object log_callback_function):
        self._log_callback_function = log_callback_function
        if self._log_callback_function is not None:
            optix_check_return(optixDeviceContextSetLogCallback(self.device_context, context_log_cb, <void*>self._log_callback_function, self._log_callback_level))

    @property
    def log_callback_level(self):
        return self._log_callback_level

    @log_callback_level.setter
    def log_callback_level(self, unsigned int callback_level):
        self._log_callback_level = callback_level
        if self._log_callback_function is not None:
            optix_check_return(optixDeviceContextSetLogCallback(self.device_context, context_log_cb, <void*>self._log_callback_function, self._log_callback_level))

    @property
    def validation_mode(self):
        return self._validation_mode

    @property
    def max_trace_depth(self):
        return self._get_property(OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRACE_DEPTH)

    @property
    def max_traversable_graph_depth(self):
        return self._get_property(OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRAVERSABLE_GRAPH_DEPTH)

    @property
    def max_primitives_per_gas(self):
        return self._get_property(OPTIX_DEVICE_PROPERTY_LIMIT_MAX_PRIMITIVES_PER_GAS)

    @property
    def max_instances_per_ias(self):
        return self._get_property(OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCES_PER_IAS)

    @property
    def rtcore_version(self):
        return self._get_property(OPTIX_DEVICE_PROPERTY_RTCORE_VERSION)

    @property
    def max_instance_id(self):
        return self._get_property(OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID)

    @property
    def num_bits_instances_visibility_mask(self):
        return self._get_property(OPTIX_DEVICE_PROPERTY_LIMIT_NUM_BITS_INSTANCE_VISIBILITY_MASK)

    @property
    def max_sbt_records_per_gas(self):
        return self._get_property(OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_RECORDS_PER_GAS)

    @property
    def max_sbt_offset(self):
        return self._get_property(OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_OFFSET)
