import optix


def log_function(level, tag, msg):
    print("[{:>2}][{:>12}]: {}".format(level, tag, msg))


def create_default_context():
    ctx = optix.DeviceContext(log_callback_function=log_function, log_callback_level=3, validation_mode=True)
    return ctx


def create_default_module(ctx=None):
    if ctx is None:
        ctx = create_default_context()
    module_opts = optix.ModuleCompileOptions()
    pipeline_opts = optix.PipelineCompileOptions()

    module = optix.Module(ctx, 'discs.cu', module_opts, pipeline_opts)

    return module

