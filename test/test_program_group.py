import pytest
import optix




def log_function(level, tag, msg):
    print("[{:>2}][{:>12}]: {}".format(level, tag.decode(), msg.decode()))


def test_program_group():
    ctx = optix.DeviceContext(log_callback_function=log_function, log_callback_level=4, validation_mode=True)
    #ctx.cache_enabled = False
    #assert not ctx.cache_enabled

    module_opts = optix.ModuleCompileOptions(opt_level=optix.CompileOptimizationLevel.LEVEL_3)
    pipeline_opts = optix.PipelineCompileOptions(traversable_graph_flags=optix.TraversableGraphFlags.ALLOW_SINGLE_LEVEL_INSTANCING,
                                                 num_payload_values=2,
                                                 num_attribute_values=1,
                                                 exception_flags=optix.ExceptionFlags.NONE,
                                                 pipeline_launch_params_variable_name="params",
                                                 uses_primitive_type_flags=optix.PrimitiveTypeFlags.CUSTOM)
    module = optix.Module(ctx, 'discs.cu', module_compile_options=module_opts, pipeline_compile_options=pipeline_opts)

    program_group_raygen = optix.ProgramGroup(ctx, raygen_module=module, raygen_entry_function_name="__raygen__rg")


    program_group_miss = optix.ProgramGroup.create_miss(ctx, module=module, entry_function_name="__miss__ms")



    # with pytest.raises(RuntimeError):
    #     program_group_invalid = optix.ProgramGroup(ctx, raygen_module=module, raygen_entry_function_name="foo")


    #module = optix.Module(ctx, example_cuda_program)

    #program_group_hit = optix.ProgramGroup.create_hitgroup(ctx, module_AH=module,
    #                                                       entry_function_AH="__anyhit__noop")