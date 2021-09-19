import pytest
import optix


def test_module_compile_options_default():
    opts = optix.ModuleCompileOptions()
    assert opts.max_register_count == optix.ModuleCompileOptions.DEFAULT_MAX_REGISTER_COUNT
    assert opts.opt_level == optix.CompileOptimizationLevel.DEFAULT
    assert opts.debug_level == optix.CompileDebugLevel.DEFAULT


def test_module_compile_options():
    opts = optix.ModuleCompileOptions(opt_level=optix.CompileOptimizationLevel.LEVEL_3,
                                      debug_level=optix.CompileDebugLevel.LINEINFO)

    assert opts.opt_level == optix.CompileOptimizationLevel.LEVEL_3
    assert opts.debug_level == optix.CompileDebugLevel.LINEINFO
    assert opts.max_register_count == optix.ModuleCompileOptions.DEFAULT_MAX_REGISTER_COUNT


def log_function(level, tag, msg):
    #print(level, tag, msg)
    print("[{:>2}][{:>12}]: {}".format(level, tag, msg))


def test_module_create():
    ctx = optix.DeviceContext(validation_mode=True, log_callback_function=log_function, log_callback_level=4)

    module_opts = optix.ModuleCompileOptions()
    pipeline_opts = optix.PipelineCompileOptions()

    print(module_opts.c_obj, pipeline_opts.c_obj)

    module = optix.Module(ctx, 'discs.cu', module_opts, pipeline_opts)

if __name__ == "__main__":
    test_module_create()
