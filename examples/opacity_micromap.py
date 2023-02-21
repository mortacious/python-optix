# 
#  Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# 
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions
#  are met:
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#   * Neither the name of NVIDIA CORPORATION nor the names of its
#     contributors may be used to endorse or promote products derived
#     from this software without specific prior written permission.
# 
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
#  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
#  PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
#  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
#  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
#  OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
# 
# -----------------------------------------------------------------------------
# 
#  A simple demonstration of opacity micromaps.
# 
#  * A single quad, made of two triangles ABC and ACD is rendered with a
#    transparent circular cutout at its center.
#  * OMMs are applied to the two triangles to accelerate the evaluation of the
#    opacity function during traversal.
#  * As a preproces, OMM microtriangles are marked as either completely
#    transparent, completely opaque, or unknown.
#  * During traversal, rays that hit opaque or transparent regions of the OMM
#    can skip the anyhit function.
#  * Rays that hit 'unknown' regions of the OMM evaluate the anyhit to get
#    accurate evaluation of the opacity function.
#  * Regions of the micromap which are unknown are tinted a lighter color to
#    visualize the regions which required anyhit evaluation.
# 
# -----------------------------------------------------------------------------


import os, sys, logging, collections

import cupy as cp
import numpy as np
import optix as ox
from sutil.camera import Camera
from PIL import Image, ImageOps

script_dir = os.path.dirname(os.path.abspath(__file__))
cuda_src = os.path.join(script_dir, "cuda", "opacity_micromap.cu")


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger()

DEBUG = False

OMM_SUBDIV_LEVEL = 4
NUM_TRIS         = 2
DEFAULT_WIDTH    = 1024
DEFAULT_HEIGHT   = 768
CIRCLE_RADIUS = 0.75

g_uvs = np.array([[[1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0]],
                  [[1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]]], dtype=np.float32)

d_uvs = cp.asarray(g_uvs)

vertices = np.array([[-0.5, -0.5, 0.0],
                     [ 0.5, -0.5, 0.0],
                     [ 0.5,  0.5, 0.0],

                     [-0.5, -0.5, 0.0],
                     [ 0.5,  0.5, 0.0],
                     [-0.5,  0.5, 0.0]], dtype=np.float32)


class Params:
    _params = collections.OrderedDict([
            ('image',        'u8'),
            ('image_width',  'u4'),
            ('image_height', 'u4'),
            ('cam_eye',      '3f4'),
            ('camera_u',     '3f4'),
            ('camera_v',     '3f4'),
            ('camera_w',     '3f4'),
            ('trav_handle', 'u8'),
        ])

    def __init__(self):
        self.handle = ox.LaunchParamsRecord(names=tuple(self._params.keys()),
                                            formats=tuple(self._params.values()))

    def __getattribute__(self, name):
        if name in Params._params.keys():
            return self.__dict__['handle'][name]
        else:
            return super().__getattribute__(name)

    def __setattr__(self, name, value):
        if name in Params._params.keys():
            self.handle[name] = value
        elif name in {'handle'}:
            super().__setattr__(name, value)
        else:
            raise AttributeError(name)

    def __str__(self):
        return '\n'.join(f'{k}:  {self.handle[k]}' for k in self._params)


##------------------------------------------------------------------------------
##
## Helper Functions
##
##------------------------------------------------------------------------------


def init_camera(params):
    camera = Camera()
    camera.eye = (0, 0, 1.5)
    camera.look_at = (0, 0, 0)
    camera.up = (0, 1, 3)
    camera.fov_y = 45
    camera.aspect_ratio = params.image_width / params.image_height

    u, v, w = camera.uvw_frame()
    params.camera_u = u
    params.camera_v = v
    params.camera_w = w
    params.cam_eye = camera.eye


def compute_uv(bary, uv0, uv1, uv2):
    bary = bary[np.newaxis, :, np.newaxis]
    return (1.0 - bary[..., 0] - bary[..., 1]) * uv0[:, np.newaxis] + bary[..., 0] * uv1[:, np.newaxis] + bary[..., 1] * uv2[:, np.newaxis]


def in_circle(uv, radius):
    return (uv[..., 0] * uv[..., 0] + uv[..., 1] * uv[..., 1]) < (radius * radius)


def evaluate_opacity(bary0, bary1, bary2, uvs, radius):
    """
    Calculate the texture coordinate at the micromesh vertices of the triangle and
    determine if the triangle is inside, outside, or spanning the boundary of the circle.
    Note that the tex coords are in [-1, 1] and the circle is centered at uv=(0,0).
    """

    uv0 = compute_uv(bary0, uvs[:, 0, :], uvs[:, 1, :], uvs[:, 2, :])
    uv1 = compute_uv(bary1, uvs[:, 0, :], uvs[:, 1, :], uvs[:, 2, :])
    uv2 = compute_uv(bary2, uvs[:, 0, :], uvs[:, 1, :], uvs[:, 2, :])

    in_circle0 = in_circle(uv0, radius)
    in_circle1 = in_circle(uv1, radius)
    in_circle2 = in_circle(uv2, radius)

    opacity = np.full_like(in_circle0, dtype=np.uint8, fill_value=ox.OpacityMicromapState.UNKNOWN_OPAQUE)

    transparent = np.logical_and.reduce((in_circle0, in_circle1, in_circle2))
    opaque = np.logical_and.reduce((~in_circle0, ~in_circle1, ~in_circle2))
    opacity[transparent] = ox.OpacityMicromapState.TRANSPARENT
    opacity[opaque] = ox.OpacityMicromapState.OPAQUE

    return opacity


def create_opacity_micromap(ctx):
    NUM_MICRO_TRIS = 1 << (OMM_SUBDIV_LEVEL * 2)

    # this has to be compressed later
    uTriIs = np.arange(0, NUM_MICRO_TRIS, dtype=np.uint32)

    bary0, bary1, bary2 = ox.micromap_indices_to_base_barycentrics(uTriIs, OMM_SUBDIV_LEVEL)
    omm_input_data = evaluate_opacity(bary0, bary1, bary2, g_uvs, CIRCLE_RADIUS)

    # construct the omm input from the data array (this will also bake the data into the format required by optix)
    omm_input = ox.OpacityMicromapInput(omm_input_data, format=ox.OpacityMicromapFormat.FOUR_STATE)
    omm = ox.OpacityMicromapArray(ctx, omm_input)
    return omm


def create_acceleration_structure(ctx, vertices, omm):
    usage_counts = [2]
    index_buffer = np.array([0, 1], dtype=np.uint16)
    omm_build_input = ox.BuildInputOpacityMicromap(omm, usage_counts,
                                                   ox.OpacityMicromapArrayIndexingMode.INDEXED,
                                                   index_buffer=index_buffer)
    triangle_input = ox.BuildInputTriangleArray(vertices,
                                                flags=[ox.GeometryFlags.NONE],
                                                opacity_micromap=omm_build_input)
    print("triangle input", triangle_input)
    gas = ox.AccelerationStructure(ctx, triangle_input, compact=True)
    return gas


def create_module(ctx, pipeline_opts):
    compile_opts = ox.ModuleCompileOptions(debug_level=ox.CompileDebugLevel.FULL, opt_level=ox.CompileOptimizationLevel.LEVEL_0)
    module = ox.Module(ctx, cuda_src, compile_opts, pipeline_opts)
    return module


def create_program_groups(ctx, module):
    raygen_grp = ox.ProgramGroup.create_raygen(ctx, module, "__raygen__rg")
    miss_grp = ox.ProgramGroup.create_miss(ctx, module, "__miss__ms")
    hit_grp = ox.ProgramGroup.create_hitgroup(ctx, module,
                                              entry_function_CH="__closesthit__ch",
                                              entry_function_AH="__anyhit__opacity")
    return raygen_grp, miss_grp, hit_grp


def create_pipeline(ctx, program_grps, pipeline_options):
    link_opts = ox.PipelineLinkOptions(max_trace_depth=1,
                                       debug_level=ox.CompileDebugLevel.FULL)

    pipeline = ox.Pipeline(ctx,
                           compile_options=pipeline_options,
                           link_options=link_opts,
                           program_groups=program_grps)

    pipeline.compute_stack_sizes(1,  # max_trace_depth
                                 0,  # max_cc_depth
                                 0)  # max_dc_depth

    return pipeline


def create_sbt(program_grps):
    raygen_grp, miss_grp, hit_grp = program_grps

    raygen_sbt = ox.SbtRecord(raygen_grp)
    miss_sbt = ox.SbtRecord(miss_grp, names=('bg_color',), formats=('3f4',))
    miss_sbt['bg_color'] = [0.01, 0.01, 0.01]

    hit_sbt = ox.SbtRecord(hit_grp, names=('uvs',), formats=('u8',))
    hit_sbt["uvs"] = d_uvs.data.ptr

    sbt = ox.ShaderBindingTable(raygen_record=raygen_sbt, miss_records=miss_sbt, hitgroup_records=hit_sbt)

    return sbt


def launch_pipeline(pipeline: ox.Pipeline, sbt, params):
    img_size = (params.image_width[0], params.image_height[0])
    output_image = np.zeros(img_size + (4, ), 'B')
    #output_image[:, :, :] = [255, 128, 0, 255]
    output_image = cp.asarray(output_image)

    params.image = output_image.data.ptr

    stream = cp.cuda.Stream()

    pipeline.launch(sbt, dimensions=img_size, params=params.handle, stream=stream)

    stream.synchronize()

    return cp.asnumpy(output_image)


if __name__ == "__main__":
    logger = ox.Logger(log)
    ctx = ox.DeviceContext(validation_mode=True,
                           log_callback_function=logger,
                           log_callback_level=4)
    ctx.cache_enabled = False

    params = Params()
    params.image_width = DEFAULT_WIDTH
    params.image_height = DEFAULT_HEIGHT

    init_camera(params)

    opacity_micromap = create_opacity_micromap(ctx)
    print("omm", opacity_micromap)
    gas = create_acceleration_structure(ctx, vertices, opacity_micromap)
    params.trav_handle = gas.handle

    pipeline_options = ox.PipelineCompileOptions(traversable_graph_flags=ox.TraversableGraphFlags.ALLOW_SINGLE_GAS,
                                                 num_payload_values=4,
                                                 num_attribute_values=2,
                                                 uses_motion_blur=False,
                                                 exception_flags=ox.ExceptionFlags.NONE,
                                                 uses_primitive_type_flags=ox.PrimitiveTypeFlags.TRIANGLE,
                                                 pipeline_launch_params_variable_name="params",
                                                 allow_opacity_micromaps=True)

    module = create_module(ctx, pipeline_options)
    program_grps = create_program_groups(ctx, module)
    pipeline = create_pipeline(ctx, program_grps, pipeline_options)

    sbt = create_sbt(program_grps)

    img = launch_pipeline(pipeline, sbt, params)

    img = img.reshape(params.image_height[0], params.image_width[0], 4)
    img = ImageOps.flip(Image.fromarray(img, 'RGBA'))
    img.show()









