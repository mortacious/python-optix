import os, sys, logging
import optix as ox
import cupy as cp
import numpy as np
from PIL import Image, ImageOps

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger()

script_dir = os.path.dirname(__file__)
cuda_src = os.path.join(script_dir, "cuda", "spheres.cu")

img_size = (1024, 768)

def compute_spheres_bbox(centers, radii):
    out = cp.empty((centers.shape[0], 6), dtype='f4')
    out[:, :3] = centers - radii.reshape(-1, 1)
    out[:, 3:] = centers + radii.reshape(-1, 1)
    return out


def create_acceleration_structure(ctx, bboxes):
    build_input = ox.BuildInputCustomPrimitiveArray([bboxes], num_sbt_records=1, flags=[ox.GeometryFlags.NONE])
    gas = ox.AccelerationStructure(ctx, [build_input], compact=True)
    return gas


def create_module(ctx, pipeline_opts):
    compile_opts = ox.ModuleCompileOptions(debug_level=ox.CompileDebugLevel.FULL, opt_level=ox.CompileOptimizationLevel.LEVEL_0)
    module = ox.Module(ctx, cuda_src, compile_opts, pipeline_opts)
    return module


def create_program_groups(ctx, module):
    raygen_grp = ox.ProgramGroup.create_raygen(ctx, module, "__raygen__rg")
    miss_grp = ox.ProgramGroup.create_miss(ctx, module, "__miss__ms")
    hit_grp = ox.ProgramGroup.create_hitgroup(ctx, module,
                                              entry_function_IS="__intersection__sphere",
                                              entry_function_CH="__closesthit__ch")
    return raygen_grp, miss_grp, hit_grp


def create_pipeline(ctx, program_grps, pipeline_options):
    link_opts = ox.PipelineLinkOptions(max_trace_depth=1, debug_level=ox.CompileDebugLevel.FULL)

    pipeline = ox.Pipeline(ctx, compile_options=pipeline_options, link_options=link_opts, program_groups=program_grps)
    pipeline.compute_stack_sizes(1,  # max_trace_depth
                                 0,  # max_cc_depth
                                 0)  # max_dc_depth
    return pipeline


def create_sbt(program_grps, centers, radii):
    raygen_grp, miss_grp, hit_grp = program_grps

    raygen_sbt = ox.SbtRecord(raygen_grp)
    miss_sbt = ox.SbtRecord(miss_grp, names=('rgb',), formats=('3f4',))
    miss_sbt['rgb'] = [0.3, 0.1, 0.2]
    hit_sbt = ox.SbtRecord(hit_grp, names=('centers', 'radii'), formats=('u8', 'u8'))
    hit_sbt['centers'] = centers.data.ptr
    hit_sbt['radii'] = radii.data.ptr

    sbt = ox.ShaderBindingTable(raygen_record=raygen_sbt, miss_records=miss_sbt, hitgroup_records=hit_sbt)

    return sbt


def launch_pipeline(pipeline : ox.Pipeline, sbt, gas):

    output_image = np.zeros(img_size + (4, ), 'B')
    output_image[:, :, :] = [255, 128, 0, 255]
    output_image = cp.asarray(output_image)
    params_tmp = [
        ( 'u8', 'image'),
        ( 'u4', 'image_width'),
        ( 'u4', 'image_height'),
        ( '3f4', 'cam_eye'),
        ( '3f4', 'cam_U'),
        ( '3f4', 'cam_V'),
        ( '3f4', 'cam_W'),
        ( 'u8', 'trav_handle')
    ]

    params = ox.LaunchParamsRecord(names=[p[1] for p in params_tmp],
                                   formats=[p[0] for p in params_tmp])
    params['image'] = output_image.data.ptr
    params['image_width'] = img_size[0]
    params['image_height'] = img_size[1]
    params['cam_eye'] = [0, 0, 2.0]
    params['cam_U'] = [1.10457, 0, 0]
    params['cam_V'] = [0, 0.828427, 0]
    params['cam_W'] = [0, 0, -2.0]
    params['trav_handle'] = gas.handle

    stream = cp.cuda.Stream()

    pipeline.launch(sbt, dimensions=img_size, params=params, stream=stream)

    stream.synchronize()

    return cp.asnumpy(output_image)


if __name__ == "__main__":
    logger = ox.Logger(log)
    ctx = ox.DeviceContext(validation_mode=True, log_callback_function=logger, log_callback_level=4)
    ctx.cache_enabled = False

    centers = cp.array([[0.0, 0.0, 0.0], [-0.5, -0.5, -0.5], [0.5, 0.5, 0]], dtype=np.float32)
    radii = cp.array([0.25, 0.1, 0.25], dtype=np.float32)
    bboxes = compute_spheres_bbox(centers, radii)
    gas = create_acceleration_structure(ctx, bboxes)

    pipeline_options = ox.PipelineCompileOptions(traversable_graph_flags=ox.TraversableGraphFlags.ALLOW_SINGLE_GAS,
                                                 num_payload_values=3,
                                                 num_attribute_values=4,
                                                 exception_flags=ox.ExceptionFlags.NONE,
                                                 pipeline_launch_params_variable_name="params")
    module = create_module(ctx, pipeline_options)
    program_grps = create_program_groups(ctx, module)
    pipeline = create_pipeline(ctx, program_grps, pipeline_options)
    sbt = create_sbt(program_grps, centers, radii)
    img = launch_pipeline(pipeline, sbt, gas)

    img = img.reshape(img_size[1], img_size[0], 4)
    img = ImageOps.flip(Image.fromarray(img, 'RGBA'))
    img.show()


