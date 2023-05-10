#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#
# -----------------------------------------------------------------------------
#
#  A simple hello-world like example showcasing the most basic usage of this
#  package.
#
#  * The code initializes a valid OptiX context
#  * A ray-generation kernel is launched for each pixel in the output buffer
#    that will just paint it in a solid color without performing any actual
#    ray-tracing operations on the GPU
#  * The result is transferred from the GPU and displayed
#
# -----------------------------------------------------------------------------

import os, sys, logging
import optix as ox
import cupy as cp
import numpy as np
from PIL import Image

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger()

script_dir = os.path.dirname(__file__)
cuda_src = os.path.join(script_dir, "cuda", "hello.cu")


def create_module(ctx, pipeline_opts):
    compile_opts = ox.ModuleCompileOptions(debug_level=ox.CompileDebugLevel.FULL, opt_level=ox.CompileOptimizationLevel.LEVEL_0)
    module = ox.Module(ctx, cuda_src, compile_opts, pipeline_opts)
    return module


def create_program_groups(ctx, module):
    raygen_grp = ox.ProgramGroup.create_raygen(ctx, module, "__raygen__draw_solid_color")
    miss_grp = ox.ProgramGroup.create_miss(ctx, None, None)
    return raygen_grp, miss_grp


def create_pipeline(ctx, program_grps, pipeline_options):
    link_opts = ox.PipelineLinkOptions(max_trace_depth=0)

    pipeline = ox.Pipeline(ctx, compile_options=pipeline_options, link_options=link_opts,
            program_groups=program_grps, max_traversable_graph_depth=1)

    pipeline.compute_stack_sizes(0,  # max_trace_depth
                                 0,  # max_cc_depth
                                 0)  # max_dc_depth
    return pipeline


def create_sbt(program_grps):
    raygen_grp, miss_grp = program_grps

    raygen_sbt = ox.SbtRecord(raygen_grp, names=('r','g','b'), formats=('f4',)*3)
    raygen_sbt['r'] = 0.462
    raygen_sbt['g'] = 0.725
    raygen_sbt['b'] = 0.0

    miss_sbt = ox.SbtRecord(miss_grp)

    sbt = ox.ShaderBindingTable(raygen_record=raygen_sbt, miss_records=miss_sbt)

    return sbt


def launch_pipeline(pipeline : ox.Pipeline, sbt):
    width = 512
    height = 384

    output_image = cp.empty(shape=(height,width,4), dtype=np.uint8)

    params = ox.LaunchParamsRecord(names=('image', 'image_width'),
                                   formats=('u8', 'u4'))
    params['image'] = output_image.data.ptr
    params['image_width'] = width

    stream = cp.cuda.Stream()

    pipeline.launch(sbt, dimensions=(width, height), params=params, stream=stream)

    stream.synchronize()

    return cp.asnumpy(output_image)


if __name__ == "__main__":
    logger = ox.Logger(log)
    ctx = ox.DeviceContext(validation_mode=True, log_callback_function=logger, log_callback_level=4)
    ctx.cache_enabled = False

    tgf = ox.TraversableGraphFlags
    pipeline_options = ox.PipelineCompileOptions(traversable_graph_flags=tgf.ALLOW_SINGLE_LEVEL_INSTANCING | tgf.ALLOW_SINGLE_GAS,
                                                 num_payload_values=0,
                                                 num_attribute_values=0,
                                                 exception_flags=ox.ExceptionFlags.NONE,
                                                 pipeline_launch_params_variable_name="params")

    module = create_module(ctx, pipeline_options)
    program_grps = create_program_groups(ctx, module)
    pipeline = create_pipeline(ctx, program_grps, pipeline_options)
    sbt = create_sbt(program_grps)
    img = launch_pipeline(pipeline, sbt)
    Image.fromarray(img, 'RGBA').show()
