# distutils: language = c++

from enum import IntEnum, IntFlag
import os
import warnings
from .path_utility import get_cuda_include_path, get_optix_include_path, get_local_optix_include_path
from .common cimport optix_check_return, optix_init
from .context cimport DeviceContext
from .pipeline cimport PipelineCompileOptions
from .pipeline import CompileDebugLevel
from .build import PrimitiveType, BuildFlags, CurveEndcapFlags
from .common import ensure_iterable
from libc.stdint cimport uintptr_t
from libcpp.vector cimport vector

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

IF _OPTIX_VERSION > 70300:
    class PayloadSemantics(IntFlag):
        """
        Wraps the PayloadSemantics enum.
        """

        DEFAULT = OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_AH_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_IS_READ_WRITE # allow everything as default
        TRACE_CALLER_NONE = OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_NONE,
        TRACE_CALLER_READ = OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ,
        TRACE_CALLER_WRITE = OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE,
        TRACE_CALLER_READ_WRITE = OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE,
        CH_NONE = OPTIX_PAYLOAD_SEMANTICS_CH_NONE,
        CH_READ = OPTIX_PAYLOAD_SEMANTICS_CH_READ,
        CH_WRITE = OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
        CH_READ_WRITE = OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE,
        MS_NONE = OPTIX_PAYLOAD_SEMANTICS_MS_NONE,
        MS_READ = OPTIX_PAYLOAD_SEMANTICS_MS_READ,
        MS_WRITE = OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
        MS_READ_WRITE = OPTIX_PAYLOAD_SEMANTICS_MS_READ_WRITE,
        AH_NONE = OPTIX_PAYLOAD_SEMANTICS_AH_NONE,
        AH_READ = OPTIX_PAYLOAD_SEMANTICS_AH_READ,
        AH_WRITE = OPTIX_PAYLOAD_SEMANTICS_AH_WRITE,
        AH_READ_WRITE = OPTIX_PAYLOAD_SEMANTICS_AH_READ_WRITE,
        IS_NONE = OPTIX_PAYLOAD_SEMANTICS_IS_NONE,
        IS_READ = OPTIX_PAYLOAD_SEMANTICS_IS_READ,
        IS_WRITE = OPTIX_PAYLOAD_SEMANTICS_IS_WRITE,
        IS_READ_WRITE = OPTIX_PAYLOAD_SEMANTICS_IS_READ_WRITE

    class ModuleCompileState(IntFlag):
        NOT_STARTED = OPTIX_MODULE_COMPILE_STATE_NOT_STARTED,
        STARTED = OPTIX_MODULE_COMPILE_STATE_STARTED,
        IMPENDING_FAILURE = OPTIX_MODULE_COMPILE_STATE_IMPENDING_FAILURE,
        FAILED = OPTIX_MODULE_COMPILE_STATE_FAILED,
        COMPLETED = OPTIX_MODULE_COMPILE_STATE_COMPLETED,


    cdef class Task(OptixObject):
        """
        Class to represent a parallel Task to compile an OptiX module.
        A Task can be executed in parallel by e.g. a thread pool to handle lots of module compilations concurrently.
        It is only valid as long as the corresponding module exists, therefore in this wrapper a reference to the module
        if stored.

        Note, that a Task is not supposed to be created by the user directly, but provided by the create_as_task method
        of the Module class.

        Parameters
        ----------
        module: Module
            The module this Task belongs to.
        """
        def __init__(self, Module module):
            self.module = module
            self.task = <OptixTask>NULL

        def execute(self, max_additional_tasks=2):
            """
            Execute the Task. If more parallel work is found, it will be returned as a new list of Task objects.
            The list has a maximum size of max_additional_tasks.

            Node, that each Task can only be executed by a single thread.

            Parameters
            ----------
            max_additional_tasks: int
                The maximum number of new Tasks to create from this one

            Returns
            -------
            tasks: List[Task]
                The newly created tasks if any
            """
            cdef vector[OptixTask] additional_tasks
            cdef unsigned int i
            cdef unsigned int additional_tasks_created = 0
            cdef unsigned int max_num_additional_tasks = max_additional_tasks

            with nogil:
                additional_tasks.resize(max_num_additional_tasks)
                optix_check_return(optixTaskExecute(self.task, additional_tasks.data(), max_num_additional_tasks, &additional_tasks_created))

            cdef list tasks = []
            for i in range(additional_tasks_created):
                t = Task(self.module)
                t.task = additional_tasks[i]
                tasks.append(t)
            return tasks


cdef class ModuleCompileOptions(OptixObject):
    """
    Wraps the OptixModuleCompileOptions struct.
    """
    DEFAULT_MAX_REGISTER_COUNT = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT
    DEFAULT_MAX_PAYLOAD_TYPE_COUNT = OPTIX_COMPILE_DEFAULT_MAX_PAYLOAD_TYPE_COUNT
    DEFAULT_MAX_PAYLOAD_VALUE_COUNT = OPTIX_COMPILE_DEFAULT_MAX_PAYLOAD_VALUE_COUNT

    def __init__(self,
                 max_register_count=DEFAULT_MAX_REGISTER_COUNT,
                 opt_level=CompileOptimizationLevel.DEFAULT,
                 debug_level= CompileDebugLevel.DEFAULT,
                 payload_types=None): #TODO add bound values
        self.compile_options.maxRegisterCount = max_register_count
        self.compile_options.optLevel = opt_level.value
        self.compile_options.debugLevel = debug_level.value
        self.compile_options.numBoundValues = 0
        self.compile_options.boundValues = NULL # currently not supported

        IF _OPTIX_VERSION > 70300:
            if payload_types is None:
                self.compile_options.numPayloadTypes = 0
                self.compile_options.payloadTypes = NULL
            else:
                # set the payload types for these compile options (this is horrible, i know ;))
                payload_types = [ensure_iterable(pt) for pt in ensure_iterable(payload_types)] # list of lists
                self.payload_types.resize(len(payload_types)) # the number of different payload types
                self.payload_values.resize(self.payload_types.size()) # a vector of semantics for each payload type
                self.compile_options.numPayloadTypes = self.payload_types.size()
                for i, payload_values in enumerate(payload_types):
                    self.payload_types[i].numPayloadValues = len(payload_values)
                    self.payload_values[i].resize(self.payload_types[i].numPayloadValues)
                    for j, payload_semantics in enumerate(payload_values):
                        self.payload_values[i][j] = payload_semantics.value
                    self.payload_types[i].payloadSemantics = self.payload_values[i].data()
                self.compile_options.payloadTypes = self.payload_types.data()



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

def get_default_nvrtc_compile_flags(std=None, rdc=False):
    flags = list(_nvrtc_compile_flags_default[:-3])
    flags.append('-std=c++11' if std is None else f'-std=c++{std}')
    if rdc:
        flags.extend(['-rdc', 'true'])
    return tuple(flags)

cdef _is_ptx(src):
    if not isinstance(src, (bytes, bytearray)):
        return False
    for line in src.splitlines():
        if len(line) == 0 or line.startswith(b'//') or line.startswith(b'\n'):
            continue
        return line.startswith(b'.version')


cdef class BuiltinISOptions(OptixObject):
    def __init__(self,
                 primitive_type,
                 build_flags=None,
                 uses_motion_blur=False,
                 curve_endcap_flags=None):
        self.options.builtinISModuleType = primitive_type.value
        self.options.usesMotionBlur = uses_motion_blur

        IF _OPTIX_VERSION > 70300:
            if build_flags is None:
                raise ValueError("Parameter build_flags is required for OptiX versions >= 7.4.")
            self.options.buildFlags = build_flags.value
            if curve_endcap_flags is None:
                curve_endcap_flags = CurveEndcapFlags.DEFAULT
            self.options.curveEndcapFlags = curve_endcap_flags.value


cdef class Module(OptixContextObject):
    """
    Class representing a Optix Cuda program that will be called during pipeline execution. Wraps the OptixModule struct.

    TODO: support creating modules through nvcc instead of nvrtc as well to support the new optix-ir format in 7.5

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
                 compile_flags=None,
                 program_name=None):
        super().__init__(context)
        cdef const char * c_ptx
        cdef unsigned int pipeline_payload_values, i

        if compile_flags is None:
            compile_flags = _nvrtc_compile_flags_default

        self._compile_flags = list(compile_flags)

        if module_compile_options.debug_level != CompileDebugLevel.NONE:
            self._compile_flags.append("-G")
        if src is not None:
            ptx = self.compile_cuda_ptx(src, compile_flags, name=program_name)
            c_ptx = ptx
            #IF _OPTIX_VERSION > 70300:
            #    self._check_payload_values(module_compile_options, pipeline_compile_options)

            optix_check_return(optixModuleCreateFromPTX(self.context.c_context,
                                     &module_compile_options.compile_options,
                                     &pipeline_compile_options.compile_options,
                                     c_ptx,
                                     len(ptx) + 1,
                                     NULL,
                                     NULL,
                                     &self.module))

    def __dealloc__(self):
        if <uintptr_t> self.module != 0:
            optix_check_return(optixModuleDestroy(self.module))

    IF _OPTIX_VERSION > 70300:
        @property
        def compile_state(self):
            cdef OptixModuleCompileState state
            with nogil:
                optix_check_return(optixModuleGetCompilationState(self.module, &state))
            return ModuleCompileState(state)

        # @staticmethod
        # def _check_payload_values(ModuleCompileOptions module_compile_options, PipelineCompileOptions pipeline_compile_options):
        #     IF _OPTIX_VERSION > 70300:
        #         # check if the payload values match between the module and pipeline compile options
        #         pipeline_payload_values = <unsigned int> pipeline_compile_options.compile_options.numPayloadValues
        #         if module_compile_options.payload_types.size() > 0:
        #             for i in range(module_compile_options.compile_options.numPayloadTypes):
        #                 if pipeline_payload_values != module_compile_options.compile_options.payloadTypes[
        #                     i].numPayloadValues:
        #                     raise ValueError(
        #                         f"number of payload values in module compile options at index {i} does not match the num_payload_values in the pipeline_compile_options.")
        #     return

        @classmethod
        def create_as_task(cls,
                             DeviceContext context,
                             src,
                             ModuleCompileOptions module_compile_options = ModuleCompileOptions(),
                             PipelineCompileOptions pipeline_compile_options = PipelineCompileOptions(),
                             compile_flags=_nvrtc_compile_flags_default,
                             program_name=None):
            """
            Create a module associated with a parallel task.
            The function will perform just enough work to instantiate the module.
            Everything else will be done by the task on request.

            Parameters
            ----------
            context: DeviceContext
                The current OptiX context
            src: str
                Either a string containing the module's source code or PTX or the path to a file containing it.
            module_compile_options: ModuleCompileOptions
                Compile options of this module
            pipeline_compile_options: PipelineCompileOptions
                Compile options of the pipeline the module will be used in
            compile_flags: list[str], optional
                List of compiler flags to use. If omitted, the default flags are used.
            program_name: str, optional
                The name the program is given internally. Of omitted either the filename is used if given or a default name is used.

            Returns
            -------

            module: Module
                The created module
            task: Task
                The task associated with this module

            """
            cdef Module module = Module(context, None, compile_flags=compile_flags)
            cdef const char * c_ptx
            cdef unsigned int pipeline_payload_values, i
            #cls._check_payload_values(module_compile_options, pipeline_compile_options)

            ptx = cls.compile_cuda_ptx(src, compile_flags, name=program_name)
            c_ptx = ptx

            cdef Task task = Task(module)

            optix_check_return(optixModuleCreateFromPTXWithTasks(context.c_context,
                                                                 &module_compile_options.compile_options,
                                                                 &pipeline_compile_options.compile_options,
                                                                 c_ptx,
                                                                 len(ptx) + 1,
                                                                 NULL,
                                                                 NULL,
                                                                 &module.module,
                                                                 &task.task))
            return module, task


    @classmethod
    def create_builtin_is_module(cls,
                          DeviceContext context,
                          ModuleCompileOptions module_compile_options,
                          PipelineCompileOptions pipeline_compile_options,
                          BuiltinISOptions builtin_is_options):
        """
        Return a module containing the builtin intersection program for the given primitive

        Parameters
        ----------
        context: DeviceContext
            The current optix context
        module_compile_options: ModuleCompileOptions
            The compile options for the module
        pipeline_compile_options: PipelineCompileOptions
            The compile options of the pipeline
        builtin_is_options: BuiltinISOptions
            Special options for the intersection program like the endcap type for curves

        Returns
        -------
        module: Module
            The Module containing the intersection program
        """
        cdef Module module = cls(context, None)

        IF _OPTIX_VERSION > 70300:
            cls._check_payload_values(module_compile_options, pipeline_compile_options)
        optix_check_return(optixBuiltinISModuleGet(context.c_context,
                                                   &module_compile_options.compile_options,
                                                   &pipeline_compile_options.compile_options,
                                                   &builtin_is_options.options, &module.module))
        return module

    @staticmethod
    def get_default_nvrtc_compile_flags(std=None, rdc=False):
        return get_default_nvrtc_compile_flags(std, rdc)

    @staticmethod
    def compile_cuda_ptx(src, compile_flags=_nvrtc_compile_flags_default, name=None, **kwargs):
        if os.path.exists(src):
            name = src
            with open(src, 'r') as f:
                src = f.read()
        if _is_ptx(src):
            return src

        elif name is None:
            name = "default_program"

        # TODO is there a public API for that?
        from cupy.cuda.compiler import _NVRTCProgram as NVRTCProgram
        prog = NVRTCProgram(src, name, **kwargs)
        flags = list(compile_flags)
        # get cuda and optix_include_paths
        cuda_include_path = get_cuda_include_path()
        optix_include_path = get_local_optix_include_path()
        if not os.path.exists(optix_include_path):
            warnings.warn("Local optix not found. This usually indicates some installation issue. Attempting"
                          " to load the global optix includes instead.", RuntimeWarning)
            optix_include_path = get_optix_include_path()
        flags.extend([f'-I{cuda_include_path}', f'-I{optix_include_path}'])
        ptx, _ = prog.compile(flags)
        return ptx

