# distutils: language = c++
import enum

from .common cimport optix_check_return, optix_init
from .context cimport DeviceContext
import cupy as cp
import numpy as np
from enum import IntEnum
from libcpp.vector cimport vector
from .common import ensure_iterable

optix_init()

__all__ = ['DenoiserModelKind',
           'Denoiser'
           ]

IF _OPTIX_VERSION > 70400:
    class DenoiserAlphaMode(enum.IntEnum):
        COPY = OPTIX_DENOISER_ALPHA_MODE_COPY
        ALPHA_AS_AOV = OPTIX_DENOISER_ALPHA_MODE_ALPHA_AS_AOV
        FULL_DENOISE_PASS = OPTIX_DENOISER_ALPHA_MODE_FULL_DENOISE_PASS

    __all__.append('DenoiserAlphaMode')

class DenoiserModelKind(IntEnum):
    """
    Wraps the OptixDenoiserModelKind enum.
    """
    LHR = OPTIX_DENOISER_MODEL_KIND_LDR
    HDR = OPTIX_DENOISER_MODEL_KIND_HDR
    AOV = OPTIX_DENOISER_MODEL_KIND_AOV
    TEMPORAL = OPTIX_DENOISER_MODEL_KIND_TEMPORAL

    IF _OPTIX_VERSION > 70300:
        TEMPORAL_AOV = OPTIX_DENOISER_MODEL_KIND_TEMPORAL_AOV
    IF _OPTIX_VERSION > 70400:
        UPSCALE2X = OPTIX_DENOISER_MODEL_KIND_UPSCALE2X
        TEMPORAL_UPSCALE2X = OPTIX_DENOISER_MODEL_KIND_TEMPORAL_UPSCALE2X

    def temporal_mode(self):
        IF _OPTIX_VERSION > 70400:
            return self == self.TEMPORAL or self==self.TEMPORAL_AOV or self == self.TEMPORAL_UPSCALE2X
        ELIF _OPTIX_VERSION > 70300:
            return self == self.TEMPORAL or self == self.TEMPORAL_AOV
        ELSE:
            return self == self.TEMPORAL


class PixelFormat(IntEnum):
    HALF2 = OPTIX_PIXEL_FORMAT_HALF2
    HALF3 = OPTIX_PIXEL_FORMAT_HALF3
    HALF4 = OPTIX_PIXEL_FORMAT_HALF4
    FLOAT2 = OPTIX_PIXEL_FORMAT_FLOAT2
    FLOAT3 = OPTIX_PIXEL_FORMAT_FLOAT3
    FLOAT4 = OPTIX_PIXEL_FORMAT_FLOAT4



    @classmethod
    def from_dtype_size(cls, dtype, size):
        try:
            return _dtype_to_pixeltype[(dtype, size)]
        except KeyError:
            raise ValueError(f"Invalid dtype {dtype} or size {size} for PixelFormat. Only float32 and float16 with either 2,3 or 4 elements are supported.")

    @property
    def dtype_size(self):
        return _pixeltype_to_dtype[self]

_dtype_to_pixeltype = {
    (np.dtype(np.float16), 2): PixelFormat.HALF2,
    (np.dtype(np.float16), 3): PixelFormat.HALF3,
    (np.dtype(np.float16), 4): PixelFormat.HALF4,
    (np.dtype(np.float32), 2): PixelFormat.FLOAT2,
    (np.dtype(np.float32), 3): PixelFormat.FLOAT3,
    (np.dtype(np.float32), 4): PixelFormat.FLOAT4,
}

_pixeltype_to_dtype = {v: k for k,v in _dtype_to_pixeltype.items()}

cdef class Image2D(OptixObject):
    def __init__(self, data, require_type=None):

        if len(data.shape) != 3:
            raise ValueError("Invalid shape of input array. Expected a 3D (height, width, channels) array.")
        pixel_type = PixelFormat.from_dtype_size(data.dtype, data.shape[2])

        if require_type is not None:
            require_type = ensure_iterable(require_type)
            if pixel_type not in require_type:
                raise ValueError(f"Invalid array given for required pixel type {require_type}")
        self._d_data = cp.asarray(data) # push data to the gpu

        # fill the underlying struct values
        self.image.data = self._d_data.data.ptr
        self.image.height = self._d_data.shape[0]
        self.image.width = self._d_data.shape[1]
        self.image.rowStrideInBytes = self._d_data.strides[0]
        self.image.pixelStrideInBytes = self._d_data.strides[1]
        self.image.format = pixel_type.value

    @classmethod
    def empty(cls, shape, dtype_or_pixeltype):
        if isinstance(dtype_or_pixeltype, PixelFormat):
            dtype_or_pixeltype, size = dtype_or_pixeltype.dtype_size
            shape = shape[:2] + (size,)
        elif len(shape) < 3:
            raise ValueError("Shape of length 3 is required if dtype is not a PixelType.")

        data = cp.empty(shape, dtype=dtype_or_pixeltype)
        return cls(data)

    @property
    def dtype(self):
        return self._d_data.dtype

    @property
    def shape(self):
        return self._d_data.shape

    @property
    def data(self):
        return self._d_data

    @property
    def pixel_format(self):
        return PixelFormat(self.image.format)


# cdef class DenoiserLayer(OptixObject):
#     def __init__(self, input, previous_output=None, output=None):
#         self.input = Image2D(input) if not isinstance(input, Image2D) else input
#
#         if previous_output is not None:
#             self.previous_output = Image2D(previous_output) if not isinstance(previous_output, Image2D) else previous_output
#
#         if output is not None:
#             self.output = Image2D(output) if not isinstance(output, Image2D) else output
#
#         self.layer.input = (<Image2D>self.input).image
#         self.layer.previousOutput = (<Image2D>self.previous_output).image
#         self.layer.output = (<Image2D>self.output).image


cdef class Denoiser(OptixContextObject):
    def __init__(self,
                 DeviceContext context,
                 model_kind=DenoiserModelKind.LHR,
                 guide_albedo=False,
                 guide_normals=False,
                 kp_mode=False,
                 tile_size=None):
        super().__init__(context)
        cdef OptixDenoiserOptions options
        self.guide_albedo = guide_albedo
        self.guide_normals = guide_normals
        self.kp_mode = kp_mode
        self.tile_size = tile_size
        self._scratch_size = 0

        self._guide_layer_scratch_size = 0
        self._average_color_scratch_size = 0
        self._state_size = 0

        if model_kind is not None:
            self.model_kind = DenoiserModelKind(model_kind)
            options.guideAlbedo = 1 if guide_albedo else 0
            options.guideNormal = 1 if guide_normals else 0

            optix_check_return(optixDenoiserCreate(self.context.c_context,
                                                   model_kind.value,
                                                   &options,
                                                   &self.denoiser))

    def _init_denoiser(self, num_inputs, input_size, stream=None):
        cdef OptixDenoiserSizes return_sizes

        #input_size = <DenoiserLayer>(inputs[0]).input.shape[:2]

        tile_size = self.tile_size
        if tile_size is None:
            tile_size = input_size

        optix_check_return(optixDenoiserComputeMemoryResources(self.denoiser, tile_size[1], tile_size[0], &return_sizes))

        if any(t <= 0 for t in tile_size):
            scratch_size = return_sizes.withoutOverlapScratchSizeInBytes
            self._overlap = 0
        else:
            scratch_size = return_sizes.withOverlapScratchSizeInBytes
            self._overlap = return_sizes.overlapWindowSizeInPixels

        if num_inputs == 1 and not self.kp_mode:
            self._d_intensity = cp.cuda.alloc(np.float32().itemsize)
            self._d_avg_color = None
        else:
            self._d_avg_color = cp.cuda.alloc(3 * np.float32().itemsize)
            self._d_intensity = None

        if self._scratch_size != scratch_size:
            self._scratch_size = scratch_size
            self._d_scratch = cp.cuda.alloc(scratch_size)
        if self._state_size != return_sizes.stateSizeInBytes:
            self._state_size = return_sizes.stateSizeInBytes
            self._d_state = cp.cuda.alloc(return_sizes.stateSizeInBytes)

        IF _OPTIX_VERSION > 70400:
            self._intensity_scratch_size = return_sizes.computeIntensitySizeInBytes
            self._average_color_scratch_size = return_sizes.computeAverageColorSizeInBytes
        ELSE:
            self._intensity_scratch_size = self._scratch_size
            self._average_color_scratch_size = self._scratch_size

        cdef uintptr_t c_stream = 0

        if stream is not None:
            c_stream = stream.ptr

        optix_check_return(optixDenoiserSetup(
            self.denoiser,
            <CUstream>c_stream,
            input_size[1] + 2 * self._overlap,
            input_size[0] + 2 * self._overlap,
            self._d_state.ptr,
            self._state_size,
            self._d_scratch.ptr,
            self._scratch_size))



    @classmethod
    def create_with_user_model(cls, DeviceContext context, unsigned char[::1] user_model not None):
        obj = cls(context, model_kind=None)
        optix_check_return(optixDenoiserCreateWithUserModel((<Denoiser>obj).context.c_context,
                                                            &user_model[0],
                                                            user_model.nbytes,
                                                            &(<Denoiser>obj).denoiser))
        return obj

    def invoke(self,
               inputs,
               prev_outputs=None,
               albedo=None,
               normals=None,
               flow=None,
               outputs=None,
               denoise_alpha=None,
               blend_factor=0.0,
               stream=None,
               temporal_use_previous_layer=False):

        accepted_input_types = (PixelFormat.FLOAT3, PixelFormat.FLOAT3, PixelFormat.HALF3, PixelFormat.HALF4)
        inputs = [Image2D(inp, require_type=accepted_input_types) for inp in ensure_iterable(inputs)]

        input_size = inputs[0].shape[:2]

        if self.guide_albedo:
            if albedo is None:
                raise ValueError("Albedo is expected.")
            albedo = Image2D(albedo)

        if self.guide_normals:
            if normals is None:
                raise ValueError("Normals is expected.")
            normals = Image2D(normals)


        cdef OptixDenoiserGuideLayer guide_layer
        guide_layer.albedo = (<Image2D>albedo).image
        guide_layer.normal = (<Image2D>normals).image

        temporal_mode = self.model_kind.temporal_mode()
        if temporal_mode:
            if flow is None:
                raise ValueError("Flow is expected for temporal mode.")
            flow = Image2D(flow, require_type=(PixelFormat.HALF2, PixelFormat.FLOAT2))
            guide_layer.flow = <Image2D>flow.image

            if prev_outputs is None:
                raise ValueError("Previous outputs are expected for temporal mode. For the first frame use a list of None values.")
            prev_outputs = ensure_iterable(prev_outputs)
            if len(prev_outputs) != len(inputs):
                raise ValueError("Number of previous outputs must be the same than number of inputs for temporal mode.")

            prev_outputs = [inputs[i] if prev is None else Image2D(prev, require_type=accepted_input_types) for i, prev in enumerate(prev_outputs)]

        # prepare the outputs
        if outputs is not None:
            outputs = ensure_iterable(outputs)
            if len(outputs) != len(inputs):
                raise ValueError("Number of outputs must be the same then number of inputs if given. Use None as list elements if not all outputs are specified.")
            outputs = [Image2D(outp) if outp is not None else Image2D.empty(input_size, inp.pixel_format) for inp, outp in zip(inputs, outputs)]
        else:
            outputs = [Image2D.empty(input_size, inp.pixel_format) for inp in inputs]


        # prepare the layers
        cdef vector[OptixDenoiserLayer] layers
        layers.resize(len(inputs))

        for i in range(len(inputs)):
            layers[i].input = (<Image2D>inputs[i]).image
            layers[i].output = (<Image2D>outputs[i]).image

            if temporal_mode:
                layers[i].previousOutput = (<Image2D>prev_outputs[i]).image

        self._init_denoiser(len(inputs), input_size, stream=stream)

        cdef OptixDenoiserParams params
        params.hdrIntensity = <CUdeviceptr>self._d_intensity.ptr if self._d_intensity is not None else 0
        params.hdrAverageColor = <CUdeviceptr>self._d_avg_color.ptr if self._d_avg_color is not None else 0
        params.blendFactor = blend_factor

        IF _OPTIX_VERSION > 70400:
            params.temporalModeUsePreviousLayers = 1 if temporal_use_previous_layer and temporal_mode else 0
            if denoise_alpha is None:
                denoise_alpha = DenoiserAlphaMode.COPY

            assert isinstance(denoise_alpha, DenoiserAlphaMode), "Optix >7.5 changed this from a boolean variable into an enum"
            params.denoiseAlpha = <OptixDenoiserAlphaMode>denoise_alpha.value
        ELSE:
            params.denoiseAlpha = 1 if denoise_alpha else 0


        cdef uintptr_t c_stream = 0

        if stream is not None:
            c_stream = stream.ptr

        # determinhe intensity and avg color if needed
        if self._d_intensity is not None:

            optix_check_return(optixDenoiserComputeIntensity(
                self.denoiser,
                <CUstream>c_stream,
                &layers[0].input,
                self._d_intensity.ptr,
                self._d_scratch.ptr,
                self._intensity_scratch_size))

        if self._d_avg_color is not None:
            optix_check_return(optixDenoiserComputeAverageColor(
                self.denoiser,
                <CUstream>c_stream,
                &layers[0].input,
                self._d_avg_color,
                self._d_scratch.ptr,
                self._average_color_scratch_size))


        if self.tile_size is None:
            # do not use tiling mode
            optix_check_return(optixDenoiserInvoke(
                self.denoiser,
                <CUstream>c_stream,
                &params,
                self._d_state.ptr,
                self._state_size,
                &guide_layer,
                layers.data(),
                <unsigned int>layers.size(),
                0,
                0,
                self._d_scratch.ptr,
                self._scratch_size))
        else:
            # do use tiling mode
            #TODO manual tiling for parallel processing
            optix_check_return(optixUtilDenoiserInvokeTiled(
                self.denoiser,
                <CUstream>c_stream,
                &params,
                self._d_state.ptr,
                self._state_size,
                &guide_layer,
                layers.data(),
                <unsigned int>layers.size(),
                self._d_scratch.ptr,
                self._scratch_size,
                self._overlap,
                self.tile_size[1],
                self.tile_size[0]))

        outputs = tuple(outp.data for outp in outputs)  # return the denoised images
        if len(outputs) == 1:
            return outputs[0]
        return outputs


    def __dealloc__(self):
        optix_check_return(optixDenoiserDestroy(self.denoiser))


