from .common cimport OptixResult, CUcontext
from .base cimport OptixObject

cdef extern from "optix_includes.h" nogil:
    ctypedef struct OptixDeviceContext:
        pass

    ctypedef void (*OptixLogCallback)(unsigned int, const char *, const char *, void *)

    cdef enum OptixDeviceContextValidationMode:
        OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF,
        OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL

    cdef struct OptixDeviceContextOptions:
        OptixLogCallback logCallbackFunction
        void * logCallbackData
        int logCallbackLevel
        OptixDeviceContextValidationMode validationMode

    cdef enum OptixDeviceProperty:
        OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRACE_DEPTH,
        OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRAVERSABLE_GRAPH_DEPTH,
        OPTIX_DEVICE_PROPERTY_LIMIT_MAX_PRIMITIVES_PER_GAS,
        OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCES_PER_IAS,
        OPTIX_DEVICE_PROPERTY_RTCORE_VERSION,
        OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID,
        OPTIX_DEVICE_PROPERTY_LIMIT_NUM_BITS_INSTANCE_VISIBILITY_MASK,
        OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_RECORDS_PER_GAS,
        OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_OFFSET

    cdef OptixResult optixDeviceContextCreate(CUcontext fromContext, const OptixDeviceContextOptions *options,
                                              OptixDeviceContext *context)
    cdef OptixResult optixDeviceContextDestroy(OptixDeviceContext context)
    cdef OptixResult optixDeviceContextGetCacheDatabaseSizes(OptixDeviceContext context, size_t * lowWaterMark,
                                                             size_t * highWaterMark)
    cdef OptixResult optixDeviceContextGetCacheEnabled(OptixDeviceContext context, int * enabled)
    cdef OptixResult optixDeviceContextGetCacheLocation(OptixDeviceContext    context, char * location, size_t locationSize)
    cdef OptixResult optixDeviceContextGetProperty(OptixDeviceContext context, OptixDeviceProperty property, void * value,
                                                   size_t sizeInBytes)
    cdef OptixResult optixDeviceContextSetCacheDatabaseSizes(OptixDeviceContext context, size_t lowWaterMark,
                                                             size_t highWaterMark)
    cdef OptixResult optixDeviceContextSetCacheEnabled(OptixDeviceContext context, int enabled)
    cdef OptixResult optixDeviceContextSetCacheLocation(OptixDeviceContext context, const char *location)
    cdef OptixResult optixDeviceContextSetLogCallback(OptixDeviceContext context, OptixLogCallback callbackFunction,
                                                      void *callbackData, unsigned int callbackLevel)

cdef class _LogWrapper:
    cdef object log_function
    cdef bint enabled


cdef class DeviceContext(OptixObject):
    cdef OptixDeviceContext c_context
    cdef object _device
    cdef _LogWrapper _log_callback
    cdef unsigned int _log_callback_level
    cdef unsigned int _get_property(self, OptixDeviceProperty property)
    cdef bint _validation_mode


cdef class OptixContextObject(OptixObject):
    cdef public DeviceContext context