import optix
import optix as ox
import cupy as cp
import numpy as np

img_size = (800, 600)


def compute_spheres_bbox(centers, radii):
    out = cp.empty((centers.shape[0], 6), dtype='f4')
    out[:, :3] = centers - radii.reshape(-1, 1)
    out[:, 3:] = centers + radii.reshape(-1, 1)
    return out


def create_acceleration_structure(ctx, centers, radii):
    bboxes = compute_spheres_bbox(centers, radii)
    print(centers, bboxes)
    build_input = ox.BuildInputCustomPrimitiveArray([bboxes], num_sbt_records=1, flags=[ox.GeometryFlags.NONE])
    gas = ox.AccelerationStructure(ctx, [build_input], compact=True)
    return gas


def create_module(ctx, pipeline_opts):
    compile_opts = ox.ModuleCompileOptions(debug_level=ox.CompileDebugLevel.LINEINFO)
    module = ox.Module(ctx, 'spheres.cu', compile_opts, pipeline_opts)
    return module


def create_program_groups(ctx, module):
    print("raygen")
    raygen_grp = ox.ProgramGroup.create_raygen(ctx, module, "__raygen__rg")
    print("miss")
    miss_grp = ox.ProgramGroup.create_miss(ctx, module, "__miss__ms")
    print("hit")
    hit_grp = ox.ProgramGroup.create_hitgroup(ctx, module,
                                              entry_function_IS="__intersection__sphere")

    return raygen_grp, miss_grp, hit_grp


def create_pipeline(ctx, program_grps, pipeline_options):
    link_opts = ox.PipelineLinkOptions(max_trace_depth=1, debug_level=ox.CompileDebugLevel.FULL)

    pipeline = ox.Pipeline(ctx, compile_options=pipeline_options, link_options=link_opts, program_groups=program_grps)
    pipeline.compute_stack_sizes(1, 0, 1) # set the stack sizes

    return pipeline


def create_sbt(program_grps, centers, radii):
    raygen_grp, miss_grp, hit_grp = program_grps

    raygen_sbt = ox.SbtRecord(raygen_grp)
    miss_sbt = ox.SbtRecord(miss_grp)

    hit_sbt = ox.SbtRecord(hit_grp, names=('centers', 'radii'), formats=('u8', 'u8'))

    hit_sbt['centers'] = centers.data.ptr
    hit_sbt['radii'] = radii.data.ptr

    sbt = ox.ShaderBindingTable(raygen_record=raygen_sbt, miss_records=[miss_sbt], hitgroup_records=[hit_sbt])

    return sbt


def launch_pipeline(pipeline : ox.Pipeline, sbt, gas, centers, ray_direction):
    centers = cp.asarray(centers, dtype='f4')
    visible = cp.zeros(centers.shape[0], dtype=np.uint32)

    num_rays = centers.shape[0]

    params = ox.LaunchParamsRecord(names=('points', 'visible', 'ray_direction', 'tolerance', 'trav_handle'),
                                   formats=('u8', 'u8', '3f4', 'f4', 'u8'))
    params['points'] = centers.data.ptr
    params['visible'] = visible.data.ptr
    params['ray_direction'] = ray_direction
    params['tolerance'] = 0.15
    params['trav_handle'] = gas.c_obj

    stream = cp.cuda.Stream()

    pipeline.launch(stream, sbt, dimensions=(num_rays,), params=params)

    stream.synchronize()

    visible = cp.asnumpy(visible)
    return visible


def log_callback(level, tag, msg):
    #print("[{:>2}][{:>12}]: {}".format(level, tag, msg))
    pass


if __name__ == "__main__":
    input()
    ctx = ox.DeviceContext(validation_mode=True, log_callback_function=log_callback, log_callback_level=4)
    ctx.cache_enabled = False

    centers = cp.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1,1,1]], dtype=np.float32)
    radii = cp.array([0.25, 0.25, 0.25], dtype=np.float32)
    print("creating gas")
    gas = create_acceleration_structure(ctx, centers, radii)
    print("creating pipeline compile options")
    pipeline_options = ox.PipelineCompileOptions(traversable_graph_flags=ox.TraversableGraphFlags.ALLOW_SINGLE_GAS,
                                                 num_payload_values=2,
                                                 num_attribute_values=4,
                                                 exception_flags=ox.ExceptionFlags.NONE,
                                                 pipeline_launch_params_variable_name="params")
    print("creating module")
    module = create_module(ctx, pipeline_options)
    print("creating program groups")
    program_grps = create_program_groups(ctx, module)
    print("creating pipeline")
    pipeline = create_pipeline(ctx, program_grps, pipeline_options)
    print("creating sbt")
    sbt = create_sbt(program_grps, centers, radii)
    print("launching")
    visible = launch_pipeline(pipeline, sbt, gas, centers, (0,1,1))

    visible = visible.astype(np.float32)
    print(visible)


