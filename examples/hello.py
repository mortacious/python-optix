import os, sys, logging
import optix as ox
import cupy as cp
import numpy as np
from PIL import Image, ImageOps

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
    link_opts = ox.PipelineLinkOptions(max_trace_depth=0, debug_level=ox.CompileDebugLevel.FULL)

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
