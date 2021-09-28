# distutils: language = c++

from enum import IntEnum
import os
from .path_utility import get_cuda_include_path, get_optix_include_path
from .common cimport optix_check_return, optix_init
from .context cimport DeviceContext
from .pipeline cimport PipelineCompileOptions
from .pipeline import CompileDebugLevel

optix_init()

class CompileOptimizationLevel(IntEnum):
    """
    Wraps the OptixCompileOptimizationLevel enum
    """
    DEFAULT = OPTIX_COMPILE_OPTIMIZATION_DEFAULT,
    LEVEL_0 = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0,
    LEVEL_1 = OPTIX_COMPILE_OPTIMIZATION_LEVEL_1,
    LEVEL_2 = OPTIX_COMPILE_OPTIMIZATION_LEVEL_2,
    LEVEL_3 = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3,


cdef class ModuleCompileOptions(OptixObject):
    """
    Wraps the OptixModuleCompileOptions struct.
    """
    DEFAULT_MAX_REGISTER_COUNT = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT
    def __init__(self,
                 max_register_count=OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
                 opt_level=CompileOptimizationLevel.DEFAULT,
                 debug_level= CompileDebugLevel.DEFAULT): #TODO add bound values
        self.compile_options.maxRegisterCount = max_register_count
        self.compile_options.optLevel = opt_level.value
        self.compile_options.debugLevel = debug_level.value
        self.compile_options.numBoundValues = 0
        self.compile_options.boundValues = NULL # currently not supported

    @property
    def max_register_count(self):
        return self.compile_options.maxRegisterCount

    @max_register_count.setter
    def max_register_count(self, count):
        self.compile_options.maxRegisterCount = count

    @property
    def opt_level(self):
        return CompileOptimizationLevel(self.compile_options.optLevel)

    @opt_level.setter
    def opt_level(self, level):
        self.compile_options.optLevel = level.value

    @property
    def debug_level(self):
        return CompileDebugLevel(self.compile_options.debugLevel)

    @debug_level.setter
    def debug_level(self, level):
        self.compile_options.debugLevel = level.value


cdef tuple _nvrtc_compile_flags_default = ('-use_fast_math', '-lineinfo', '-default-device', '-std=c++11', '-rdc', 'true')


cdef _is_ptx(src):
    if not isinstance(src, (bytes, bytearray)):
        return False
    for line in src.splitlines():
        print(line)
        if len(line) == 0 or line.startswith(b'//') or line.startswith(b'\n'):
            continue
        return line.startswith(b'.version')


cdef class Module(OptixContextObject):
    """
    Class representing a Optix Cuda program that will be called during pipeline execution. Wraps the OptixModule struct.

    Parameters
    ----------
    context: DeviceContext
        The context to use for this module
    src: str
        Either a string containing the module's source code or the path to a file containing it.
    module_compile_options: ModuleCompileOptions
        Compile options of this module
    pipeline_compile_options: PipelineCompileOptions
        Compile options of the pipeline the module will be used in
    compile_flags: list[str], optional
        List of compiler flags to use. If omitted, the default flags are used.
    program_name: str, optional
        The name the program is given internally. Of omitted either the filename is used if given or a default name is used.
    """
    def __init__(self,
                 DeviceContext context,
                 src,
                 ModuleCompileOptions module_compile_options = ModuleCompileOptions(),
                 PipelineCompileOptions pipeline_compile_options = PipelineCompileOptions(),
                 compile_flags=_nvrtc_compile_flags_default,
                 program_name=None):
        super().__init__(context)
        self._compile_flags = list(compile_flags)
        if not _is_ptx(src):
            ptx = self._compile_cuda_ptx(src, name=program_name)
        else:
            ptx = src
        cdef const char* c_ptx = ptx
        optixModuleCreateFromPTX(self.context.c_context,
                                 &module_compile_options.compile_options,
                                 &pipeline_compile_options.compile_options,
                                 c_ptx,
                                 len(ptx) + 1,
                                 NULL,
                                 NULL,
                                 &self.module)

    def _compile_cuda_ptx(self, src, name=None, **kwargs):
        if os.path.exists(src):
            name = src
            with open(src, 'r') as f:
                src = f.read()

        elif name is None:
            name = "default_program"

        # TODO is there a public API for that?
        from cupy.cuda.compiler import _NVRTCProgram as NVRTCProgram
        prog = NVRTCProgram(src, name, **kwargs)
        flags = self._compile_flags

        # get cuda and optix_include_paths
        cuda_include_path = get_cuda_include_path()
        optix_include_path = get_optix_include_path()

        flags.extend([f'-I{cuda_include_path}', f'-I{optix_include_path}'])
        ptx, _ = prog.compile(flags)
        return ptx

