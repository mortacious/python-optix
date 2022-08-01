from .common cimport OptixResult, CUstream, CUdeviceptr
from .context cimport OptixDeviceContext, OptixContextObject
from libcpp.vector cimport vector
from .base cimport OptixObject
from libc.stdint cimport uintptr_t
from libcpp cimport bool

cdef extern from "optix_includes.h" nogil:
    IF _OPTIX_VERSION > 70400:
        cdef enum OptixDenoiserModelKind:
            OPTIX_DENOISER_MODEL_KIND_LDR
            OPTIX_DENOISER_MODEL_KIND_HDR
            OPTIX_DENOISER_MODEL_KIND_AOV
            OPTIX_DENOISER_MODEL_KIND_TEMPORAL
            OPTIX_DENOISER_MODEL_KIND_TEMPORAL_AOV
            OPTIX_DENOISER_MODEL_KIND_UPSCALE2X
            OPTIX_DENOISER_MODEL_KIND_TEMPORAL_UPSCALE2X

    ELIF _OPTIX_VERSION > 70300:
        cdef enum OptixDenoiserModelKind:
            OPTIX_DENOISER_MODEL_KIND_LDR
            OPTIX_DENOISER_MODEL_KIND_HDR
            OPTIX_DENOISER_MODEL_KIND_AOV
            OPTIX_DENOISER_MODEL_KIND_TEMPORAL
            OPTIX_DENOISER_MODEL_KIND_TEMPORAL_AOV
    ELSE:
        cdef enum OptixDenoiserModelKind:
            OPTIX_DENOISER_MODEL_KIND_LDR
            OPTIX_DENOISER_MODEL_KIND_HDR
            OPTIX_DENOISER_MODEL_KIND_AOV
            OPTIX_DENOISER_MODEL_KIND_TEMPORAL


    cdef struct OptixDenoiserOptions:
        unsigned int guideAlbedo
        unsigned int guideNormal

    IF _OPTIX_VERSION > 70400:
        cdef struct OptixDenoiserSizes:
            size_t stateSizeInBytes
            size_t  withOverlapScratchSizeInBytes
            size_t  withoutOverlapScratchSizeInBytes
            unsigned int overlapWindowSizeInPixels
            size_t    computeAverageColorSizeInBytes
            size_t    computeIntensitySizeInBytes
            size_t    internalGuideLayerPixelSizeInBytes

        cdef enum OptixDenoiserAlphaMode:
            OPTIX_DENOISER_ALPHA_MODE_COPY,
            OPTIX_DENOISER_ALPHA_MODE_ALPHA_AS_AOV,
            OPTIX_DENOISER_ALPHA_MODE_FULL_DENOISE_PASS

        cdef struct OptixDenoiserParams:
            OptixDenoiserAlphaMode denoiseAlpha
            CUdeviceptr  hdrIntensity
            float        blendFactor
            CUdeviceptr  hdrAverageColor
            unsigned int temporalModeUsePreviousLayers
    ELSE:
        cdef struct OptixDenoiserSizes:
            size_t stateSizeInBytes
            size_t  withOverlapScratchSizeInBytes
            size_t  withoutOverlapScratchSizeInBytes
            unsigned int overlapWindowSizeInPixels

        cdef struct OptixDenoiserParams:
            unsigned int denoiseAlpha
            CUdeviceptr  hdrIntensity
            float        blendFactor
            CUdeviceptr  hdrAverageColor


    cdef enum OptixPixelFormat:
        OPTIX_PIXEL_FORMAT_HALF2
        OPTIX_PIXEL_FORMAT_HALF3
        OPTIX_PIXEL_FORMAT_HALF4
        OPTIX_PIXEL_FORMAT_FLOAT2
        OPTIX_PIXEL_FORMAT_FLOAT3
        OPTIX_PIXEL_FORMAT_FLOAT4
        OPTIX_PIXEL_FORMAT_UCHAR3
        OPTIX_PIXEL_FORMAT_UCHAR4

    cdef struct OptixImage2D:
        CUdeviceptr data
        unsigned int width
        unsigned int height
        unsigned int rowStrideInBytes
        unsigned int pixelStrideInBytes
        OptixPixelFormat format

    cdef struct OptixDenoiserLayer:
        OptixImage2D input
        OptixImage2D previousOutput
        OptixImage2D output

    IF _OPTIX_VERSION > 70400:
        cdef struct OptixDenoiserGuideLayer:
            OptixImage2D albedo
            OptixImage2D normal
            OptixImage2D flow
            OptixImage2D previousOutputInternalGuideLayer
            OptixImage2D outputInternalGuideLayer
    ELSE:
        cdef struct OptixDenoiserGuideLayer:
            OptixImage2D albedo
            OptixImage2D normal
            OptixImage2D flow

    ctypedef struct OptixDenoiser:
        pass

    OptixResult optixDenoiserCreate(OptixDeviceContext context,
                                    OptixDenoiserModelKind modelKind,
                                    const OptixDenoiserOptions* options,
                                    OptixDenoiser* denoiser
                                    )

    OptixResult optixDenoiserCreateWithUserModel(OptixDeviceContext context,
                                                 const void* userData,
                                                 size_t userDataSizeInBytes,
                                                 OptixDenoiser* denoiser
                                                 )

    OptixResult optixDenoiserDestroy(OptixDenoiser denoiser)

    OptixResult optixDenoiserComputeMemoryResources(const OptixDenoiser denoiser,
                                                    unsigned int inputWidth,
                                                    unsigned int inputHeight,
                                                    OptixDenoiserSizes* returnSizes)

    OptixResult optixDenoiserSetup(
            OptixDenoiser denoiser,
            CUstream      stream,
            unsigned int  inputWidth,
            unsigned int  inputHeight,
            CUdeviceptr   denoiserState,
            size_t        denoiserStateSizeInBytes,
            CUdeviceptr   scratch,
            size_t        scratchSizeInBytes)

    OptixResult optixDenoiserInvoke(
            OptixDenoiser                  denoiser,
            CUstream                       stream,
            const OptixDenoiserParams *     params,
            CUdeviceptr                    denoiserState,
            size_t                         denoiserStateSizeInBytes,
            const OptixDenoiserGuideLayer * guideLayer,
            const OptixDenoiserLayer *      layers,
            unsigned int                   numLayers,
            unsigned int                   inputOffsetX,
            unsigned int                   inputOffsetY,
            CUdeviceptr                    scratch,
            size_t                         scratchSizeInBytes)

    OptixResult optixDenoiserComputeAverageColor(
            OptixDenoiser       denoiser,
            CUstream            stream,
            const OptixImage2D * inputImage,
            CUdeviceptr         outputAverageColor,
            CUdeviceptr         scratch,
            size_t              scratchSizeInBytes)

    OptixResult optixDenoiserComputeIntensity(
            OptixDenoiser       denoiser,
            CUstream            stream,
            const OptixImage2D * inputImage,
            CUdeviceptr         outputIntensity,
            CUdeviceptr         scratch,
            size_t              scratchSizeInBytes)

cdef extern from "optix_denoiser_tiling.h" nogil:
    OptixResult optixUtilDenoiserInvokeTiled(OptixDenoiser denoiser,
                                             CUstream stream,
                                             const OptixDenoiserParams *params,
                                             CUdeviceptr denoiserState,
                                             size_t denoiserStateSizeInBytes,
                                             const OptixDenoiserGuideLayer *guideLayer,
                                             const OptixDenoiserLayer *layers,
                                             unsigned int numLayers,
                                             CUdeviceptr scratch,
                                             size_t scratchSizeInBytes,
                                             unsigned int overlapWindowSizeInPixels,
                                             unsigned int tileWidth,
                                             unsigned int tileHeight)

    cdef struct OptixUtilDenoiserImageTile:
        OptixImage2D input
        OptixImage2D output
        unsigned int inputOffsetX
        unsigned int inputOffsetY

    OptixResult optixUtilDenoiserSplitImage(const OptixImage2D & input,
                                            const OptixImage2D & output,
                                            unsigned int overlapWindowSizeInPixels, unsigned int tileWidth,
                                            unsigned int tileHeight,
                                            vector[OptixUtilDenoiserImageTile]& tiles)

    unsigned int optixUtilGetPixelStride(const OptixImage2D & image)



cdef class Image2D(OptixObject):
    cdef OptixImage2D image
    cdef object _d_data

cdef class DenoiserLayer(OptixObject):
    cdef OptixDenoiserLayer layer
    cdef Image2D input
    cdef Image2D previous_output
    cdef Image2D output

cdef class DenoiserGuideLayer(OptixObject):
    cdef OptixDenoiserGuideLayer layer
    cdef Image2D albedo
    cdef Image2D normal
    cdef Image2D flow

cdef class Denoiser(OptixContextObject):
    cdef OptixDenoiser denoiser
    cdef object model_kind
    cdef bool guide_albedo
    cdef bool guide_normals
    cdef bool kp_mode
    cdef tuple tile_size
    cdef object _d_state
    cdef size_t _state_size
    cdef object _d_scratch
    cdef size_t _scratch_size
    cdef object _d_window
    cdef size_t _window_size
    cdef object _d_intensity
    cdef object _d_avg_color
    cdef unsigned int _overlap
