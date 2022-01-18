from .common cimport optix_check_return, optix_init
from .context cimport DeviceContext
import cupy as cp
import numpy as np
from enum import IntEnum, IntFlag
from libc.string cimport memcpy, memset
from libcpp.vector cimport vector
from .common import ensure_iterable

optix_init()

class DenoiserModelKind(IntEnum):
    """
    Wraps the OptixDenoiserModelKind enum.
    """
    LHR = OPTIX_DENOISER_MODEL_KIND_LDR
    HDR = OPTIX_DENOISER_MODEL_KIND_HDR
    AOV = OPTIX_DENOISER_MODEL_KIND_AOV
    TEMPORAL = OPTIX_DENOISER_MODEL_KIND_TEMPORAL
    TEMPORAL_AOV = OPTIX_DENOISER_MODEL_KIND_TEMPORAL_AOV

class Image2D(OptixObject):
    _dtype_to_pixeltype = {
        (np.float16, 2): OPTIX_PIXEL_FORMAT_HALF2,
        (np.float16, 3): OPTIX_PIXEL_FORMAT_HALF3,
        (np.float16, 4): OPTIX_PIXEL_FORMAT_HALF4,
        (np.float32, 2): OPTIX_PIXEL_FORMAT_FLOAT2,
        (np.float32, 3): OPTIX_PIXEL_FORMAT_FLOAT3,
        (np.float32, 4): OPTIX_PIXEL_FORMAT_FLOAT4,
    }
    def __init__(self, data):
        self._d_data = cp.asarray(data) # push data to the gpu

        if len(self._d_data.shape) != 3:
            raise ValueError("Invalid shape of input array. Expected a 3D (height, width, channels) array.")
        try:
            pixel_type = self._dtype_to_pixeltype[(self._d_data.dtype, self._d_data.shape[2])]
        except KeyError:
            raise ValueError(f"Invalid dtype {self._d_data.dtype} of data. Only float32 and float16 are supported for Image2D")

        # fill the underlying struct values
        self.image.data = self._d_data.data.ptr
        self.image.height = self._d_data.shape[0]
        self.image.width = self._d_data.shape[1]
        self.image.rowStrideInBytes = self._d_data.strides[0]
        self.image.pixelStrideInBytes = self._d_data.strides[1]
        self.image.format = pixel_type

    @property
    def dtype(self):
        return self._d_data.dtype

    @property
    def shape(self):
        return self._d_data.shape


class DenoiserLayer(OptixObject):
    def __init__(self, input, previous_output=None, output=None):
        self.input = Image2D(input) if not isinstance(input, Image2D) else input

        if previous_output is not None:
            self.previous_output = Image2D(previous_output) if not isinstance(previous_output, Image2D) else previous_output

        if output is not None:
            self.output = Image2D(output) if not isinstance(output, Image2D) else output

        self.layer.input = (<Image2D>self.input).image
        self.layer.previousOutput = (<Image2D>self.previous_output).image
        self.layer.output = (<Image2D>self.output).image


class Denoiser(OptixContextObject):
    def __init__(self, DeviceContext context, DenoiserModelKind model_kind=DenoiserModelKind.LHR, enable_albedo=False, enable_normals=False):
        super().__init__(context)
        cdef OptixDenoiserOptions options

        if model_kind is not None:
            options.guideAlbedo = 1 if enable_albedo else 0
            options.guideNormal = 1 if enable_normals else 0
            self.guide_albedo = enable_albedo
            self.guide_normals = enable_normals

            optix_check_return(optixDenoiserCreate(self.context.c_context,
                                                   model_kind.value,
                                                   &options,
                                                   &self.denoiser))

    @classmethod
    def create_with_user_model(cls, DeviceContext context, user_model):
        raise NotImplementedError()
        #obj = cls(context, model_kind=None)
        #optix_check_return(optixDenoiserCreateWithUserModel(obj.context.c_context,
        #                                                    user_model, #TODO
        #                                                    len(user_model), #TODO
        #                                                    &obj.denoiser))
        #return obj

    def __dealloc__(self):
        optix_check_return(optixDenoiserDestroy(self.denoiser))


