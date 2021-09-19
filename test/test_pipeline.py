import pytest
import optix


def test_pipeline_compile_options_default():
    opts = optix.PipelineCompileOptions()

    assert not opts.uses_motion_blur
    assert opts.traversable_graph_flags == optix.TraversableGraphFlags.ALLOW_ANY
    assert opts.num_payload_values == 0
    assert opts.num_attribute_values == 0
    assert opts.exception_flags == optix.ExceptionFlags.NONE
    assert opts.pipeline_launch_params_variable_name == "params"
    assert opts.uses_primitive_type_flags == optix.PrimitiveTypeFlags.DEFAULT


def test_pipeline_compile_options():
    opts = optix.PipelineCompileOptions(traversable_graph_flags=optix.TraversableGraphFlags.ALLOW_SINGLE_GAS,
                                        num_payload_values=2, num_attribute_values=5,
                                        exception_flags=optix.ExceptionFlags.STACK_OVERFLOW, pipeline_launch_params_variable_name="params",
                                        uses_primitive_type_flags=optix.PrimitiveTypeFlags.CUSTOM)

    assert opts.pipeline_launch_params_variable_name == "params"
    assert opts.traversable_graph_flags == optix.TraversableGraphFlags.ALLOW_SINGLE_GAS
    assert opts.num_payload_values == 2
    assert opts.num_attribute_values == 5
    assert opts.exception_flags == optix.ExceptionFlags.STACK_OVERFLOW
    assert opts.uses_primitive_type_flags == optix.PrimitiveTypeFlags.CUSTOM


def test_pipeline_link_options():
    opts = optix.PipelineLinkOptions(max_trace_depth=2, debug_level=optix.CompileDebugLevel.LINEINFO)
    assert opts.max_trace_depth == 2
    assert opts.debug_level == optix.CompileDebugLevel.LINEINFO


def test_pipeline_link_options_default():
    opts = optix.PipelineLinkOptions()
    assert opts.max_trace_depth == 1
    assert opts.debug_level == optix.CompileDebugLevel.DEFAULT
