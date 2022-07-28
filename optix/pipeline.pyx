# distutils: language = c++

from .common cimport optix_check_return, optix_init
from .common import ensure_iterable
from .context cimport DeviceContext
from .program_group cimport OptixProgramGroup, OptixProgramGroupOptions, optixProgramGroupCreate, OptixProgramGroupDesc
from enum import IntEnum, IntFlag
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector
from libc.stdint cimport uintptr_t

from .struct cimport LaunchParamsRecord
from .struct import LaunchParamsRecord
from .shader_binding_table cimport ShaderBindingTable

optix_init()


class ExceptionFlags(IntFlag):
    """
    Wraps the OptixExceptionFlags enum.
    """
    NONE = OPTIX_EXCEPTION_FLAG_NONE,
    STACK_OVERFLOW = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW,
    TRACE_DEPTH = OPTIX_EXCEPTION_FLAG_TRACE_DEPTH,
    USER = OPTIX_EXCEPTION_FLAG_USER,
    DEBUG = OPTIX_EXCEPTION_FLAG_DEBUG


class TraversableGraphFlags(IntFlag):
    """
    Wraps the OptixTraversableGraphFlags enum.
    """
    ALLOW_ANY = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY,
    ALLOW_SINGLE_GAS = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS,
    ALLOW_SINGLE_LEVEL_INSTANCING = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING


class PrimitiveTypeFlags(IntFlag):
    """
    Wraps the OptixPrimitiveTypeFlags enum.
    """
    DEFAULT = 0, # corresponds to CUSTOM | TRIANGLE
    CUSTOM = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM,
    ROUND_QUADRATIC_BSPLINE = OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_QUADRATIC_BSPLINE,
    ROUND_CUBIC_BSPLINE = OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BSPLINE,
    ROUND_LINEAR = OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_LINEAR

    # switch to new primitive type flags
    IF _OPTIX_VERSION > 70300:  # switch to new compile debug level flags
        ROUND_CATMULLROM = OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CATMULLROM
    IF _OPTIX_VERSION > 70400:
        SPHERE = OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE
    TRIANGLE = <unsigned int>OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE # fixes negative number error


IF _OPTIX_VERSION > 70300:  # switch to new compile debug level flags
    class CompileDebugLevel(IntEnum):
        """
        Wraps the OptixCompileDebugLevel enum.
        """
        DEFAULT = OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT,
        NONE = OPTIX_COMPILE_DEBUG_LEVEL_NONE,
        MINIMAL = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL,
        MODERATE = OPTIX_COMPILE_DEBUG_LEVEL_MODERATE,
        FULL = OPTIX_COMPILE_DEBUG_LEVEL_FULL
ELSE:
    class CompileDebugLevel(IntEnum):
        """
        Wraps the OptixCompileDebugLevel enum.
        """
        DEFAULT = OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT,
        NONE = OPTIX_COMPILE_DEBUG_LEVEL_NONE,
        LINEINFO = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO,
        FULL = OPTIX_COMPILE_DEBUG_LEVEL_FULL


cdef class PipelineCompileOptions(OptixObject):
    """
    Class wrapping the OptixPipelineCompileOptions struct.
    """
    DEFAULT_MAX_PAYLOAD_VALUE_COUNT = OPTIX_COMPILE_DEFAULT_MAX_PAYLOAD_VALUE_COUNT

    def __init__(self,
                 uses_motion_blur=False,
                 traversable_graph_flags = TraversableGraphFlags.ALLOW_ANY,
                 num_payload_values = 0,
                 num_attribute_values = 0,
                 exception_flags = ExceptionFlags.NONE,
                 pipeline_launch_params_variable_name = "params",
                 uses_primitive_type_flags = PrimitiveTypeFlags.DEFAULT):
        self.uses_motion_blur = uses_motion_blur
        self.traversable_graph_flags = traversable_graph_flags
        self.num_payload_values = num_payload_values
        self.num_attribute_values = num_attribute_values
        self.exception_flags = exception_flags
        self.pipeline_launch_params_variable_name = pipeline_launch_params_variable_name
        self.uses_primitive_type_flags = uses_primitive_type_flags

    @property
    def uses_motion_blur(self):
        return self.compile_options.usesMotionBlur

    @uses_motion_blur.setter
    def uses_motion_blur(self, motion_blur):
        self.compile_options.usesMotionBlur = motion_blur

    @property
    def traversable_graph_flags(self):
        return TraversableGraphFlags(self.compile_options.traversableGraphFlags)

    @traversable_graph_flags.setter
    def traversable_graph_flags(self, flags):
        self.compile_options.traversableGraphFlags = flags.value

    @property
    def num_payload_values(self):
        return self.compile_options.numPayloadValues

    @num_payload_values.setter
    def num_payload_values(self, num_payload_values):
        if num_payload_values > self.DEFAULT_MAX_PAYLOAD_VALUE_COUNT:
            raise ValueError(f"A maximum of {self.DEFAULT_MAX_PAYLOAD_VALUE_COUNT} payload values is allowed.")
        self.compile_options.numPayloadValues = num_payload_values

    @property
    def num_attribute_values(self):
        return self.compile_options.numAttributeValues

    @num_attribute_values.setter
    def num_attribute_values(self, num_attribute_values):
        self.compile_options.numAttributeValues = num_attribute_values

    @property
    def exception_flags(self):
        return ExceptionFlags(self.compile_options.exceptionFlags)

    @exception_flags.setter
    def exception_flags(self, flags):
        self.compile_options.exceptionFlags = flags.value

    @property
    def pipeline_launch_params_variable_name(self):
        return self._pipeline_launch_params_variable_name.decode('ascii')

    @pipeline_launch_params_variable_name.setter
    def pipeline_launch_params_variable_name(self, name):
        self._pipeline_launch_params_variable_name = name.encode('ascii')
        self.compile_options.pipelineLaunchParamsVariableName = <const char*>self._pipeline_launch_params_variable_name

    @property
    def uses_primitive_type_flags(self):
        return PrimitiveTypeFlags(self.compile_options.usesPrimitiveTypeFlags)

    @uses_primitive_type_flags.setter
    def uses_primitive_type_flags(self, flags):
        self.compile_options.usesPrimitiveTypeFlags = flags.value

    @property
    def c_obj(self):
        return <size_t>&self.compile_options


cdef class PipelineLinkOptions(OptixObject):
    """
    Class wrapping the OptixPipelineLinkOptions struct.
    """
    def __init__(self, unsigned int max_trace_depth = 1, debug_level = CompileDebugLevel.DEFAULT):
        self.link_options.maxTraceDepth = max_trace_depth
        self.link_options.debugLevel = debug_level.value

    @property
    def max_trace_depth(self):
        return self.link_options.maxTraceDepth

    @max_trace_depth.setter
    def max_trace_depth(self, unsigned int max_trace_depth):
        self.link_options.maxTraceDepth = max_trace_depth

    @property
    def debug_level(self):
        return CompileDebugLevel(self.link_options.debugLevel)

    @debug_level.setter
    def debug_level(self, debug_level):
        self.link_options.debugLevel = debug_level.value


ctypedef vector[unsigned int] uint_vector

cdef class Pipeline(OptixContextObject):
    """
    The pipeline is the main entry point into a OptiX program. I combines several program groups containing different parts of the program
    intro a single object that can be launched afterwards.

    Parameters
    ----------
    context: DeviceContext
        The context to use for this pipeline
    compile_options: PipelineCompileOptions
        Compile options of this pipeline
    link_options: PipelineLinkOptions
        Link options of this pipeline
    program_groups: list[ProgramGroup] or single ProgramGroup
        The program groups to use in this pipeline
    max_traversable_graph_depth: int, optional
        The maximum traversable graph depth in this pipeline. If omitted, a default value of 1 is used
    """
    def __init__(self,
                 DeviceContext context,
                 PipelineCompileOptions compile_options,
                 PipelineLinkOptions link_options,
                 program_groups,
                 max_traversable_graph_depth=1):
        super().__init__(context)
        program_groups = ensure_iterable(program_groups)

        if not all(isinstance(p, ProgramGroup) for p in program_groups):
            raise TypeError("Only program groups")
        cdef unsigned int num_program_groups = len(program_groups)
        cdef vector[OptixProgramGroup] c_program_groups = vector[OptixProgramGroup](num_program_groups)
        cdef unsigned int i
        cdef OptixProgramGroupOptions options
        cdef ProgramGroup grp
        for i in range(num_program_groups):
            grp = <ProgramGroup>program_groups[i]
            c_program_groups[i] = grp.program_group

        for i in range(len(program_groups)):
            optix_check_return(optixUtilAccumulateStackSizes(c_program_groups[i], &self._stack_sizes))
        optix_check_return(optixPipelineCreate(self.context.c_context,
                                               &compile_options.compile_options,
                                               &link_options.link_options,
                                               c_program_groups.const_data(),
                                               len(program_groups),
                                               NULL,
                                               NULL,
                                               &self.pipeline))

        self._max_traversable_graph_depth = max_traversable_graph_depth

    cpdef set_stack_sizes(self,
                          unsigned int direct_callable_stack_size_from_traversal,
                          unsigned int direct_callable_stack_size_from_state,
                          unsigned int continuation_stack_size):
        """
        Set the stack sizes manually. Usually this will not be used but one of the compute_stack_sizes functions.
        
        Parameters
        ----------
        direct_callable_stack_size_from_traversal: int
        direct_callable_stack_size_from_state: int
        continuation_stack_size: int
        """

        optix_check_return(optixPipelineSetStackSize(self.pipeline,
                                                     direct_callable_stack_size_from_traversal,
                                                     direct_callable_stack_size_from_state,
                                                     continuation_stack_size,
                                                     self._max_traversable_graph_depth))

    cpdef compute_stack_sizes(self,
                              unsigned int max_trace_depth,
                              unsigned int max_cc_depth,
                              unsigned int max_dc_depth):
        """
        Compute the stack sizes using a conservative default algorithm. For documentation of the used algorithm refer to
        the OptiX documentation at https://raytracing-docs.nvidia.com/optix7/guide/index.html#program_pipeline_creation#pipeline-stack-size.
        
        Parameters
        ----------
        max_trace_depth: int
        max_cc_depth: int
        max_dc_depth: int
        """
        cdef unsigned int direct_callable_stack_size_from_traversal, direct_callable_stack_size_from_state, continuation_stack_size
        optix_check_return(optixUtilComputeStackSizes(&self._stack_sizes,
                                                      max_trace_depth,
                                                      max_cc_depth,
                                                      max_dc_depth,
                                                      &direct_callable_stack_size_from_traversal,
                                                      &direct_callable_stack_size_from_state,
                                                      &continuation_stack_size))

        self.set_stack_sizes(direct_callable_stack_size_from_traversal, direct_callable_stack_size_from_state, continuation_stack_size)

    cpdef compute_stack_sizes_css_cc_tree(self,
                                          unsigned int css_cc_tree,
                                          unsigned int max_trace_depth,
                                          unsigned int max_dc_depth):
        """
        Compute optimized stack sizes. Refer to the OptiX documentation at 
        https://raytracing-docs.nvidia.com/optix7/guide/index.html#program_pipeline_creation#pipeline-stack-size
        for details.
        
        Parameters
        ----------
        css_cc_tree: int
        max_trace_depth: int
        max_dc_depth: int
        """
        cdef unsigned int direct_callable_stack_size_from_traversal, direct_callable_stack_size_from_state, continuation_stack_size
        optix_check_return(optixUtilComputeStackSizesCssCCTree(&self._stack_sizes,
                                                               css_cc_tree,
                                                               max_trace_depth,
                                                               max_dc_depth,
                                                               &direct_callable_stack_size_from_traversal,
                                                               &direct_callable_stack_size_from_state,
                                                               &continuation_stack_size))
        self.set_stack_sizes(direct_callable_stack_size_from_traversal, direct_callable_stack_size_from_state, continuation_stack_size)

    cpdef compute_stack_sizes_dc_split(self,
                                       unsigned int dss_dc_from_traversal,
                                       unsigned int dss_dc_from_state,
                                       unsigned int max_trace_depth,
                                       unsigned int max_cc_depth,
                                       unsigned int max_dc_depth_from_traversal,
                                       unsigned int max_dc_depth_from_state):
        """
        Compute optimized stack sizes. Refer to the OptiX documentation at 
        https://raytracing-docs.nvidia.com/optix7/guide/index.html#program_pipeline_creation#pipeline-stack-size
        for details.
        
        Parameters
        ----------
        dss_dc_from_traversal: int
        dss_dc_from_state: int
        max_trace_depth: int
        max_cc_depth: int
        max_dc_depth_from_traversal: int
        max_dc_depth_from_state: int
        """
        cdef unsigned int direct_callable_stack_size_from_traversal, direct_callable_stack_size_from_state, continuation_stack_size
        optix_check_return(optixUtilComputeStackSizesDCSplit(&self._stack_sizes,
                                                             dss_dc_from_traversal,
                                                             dss_dc_from_state,
                                                             max_trace_depth,
                                                             max_cc_depth,
                                                             max_dc_depth_from_traversal,
                                                             max_dc_depth_from_state,
                                                             &direct_callable_stack_size_from_traversal,
                                                             &direct_callable_stack_size_from_state,
                                                             &continuation_stack_size))
        self.set_stack_sizes(direct_callable_stack_size_from_traversal, direct_callable_stack_size_from_state, continuation_stack_size)

    cpdef compute_stack_sizes_simple_path_tracer(self,
                                                 ProgramGroup program_group_raygen,
                                                 ProgramGroup program_group_miss_1,
                                                 object program_groups_closesthit_1,
                                                 ProgramGroup program_group_miss_2,
                                                 object program_groups_closesthit_2):
        """
        Compute optimized stack sizes for simple path tracers (camera and shadow rays). Refer to the OptiX documentation at 
        https://raytracing-docs.nvidia.com/optix7/guide/index.html#program_pipeline_creation#pipeline-stack-size
        for details.
        
        Parameters
        ----------
        program_group_raygen: ProgramGroup
            The ProgramGroup containing the raygen program
        program_group_miss_1: ProgramGroup 
            The ProgramGroup invoked for the camera rays on miss
        program_groups_closesthit_1: list[ProgramGroup]
            The ProgramGroups invoked on closest hits of camera rays
        program_group_miss_2: ProgramGroup
            The ProgramGroup invoked for the shadow rays on miss
        program_groups_closesthit_2: list[ProgramGroup]
            The ProgramGroups invoked on closest hits of shadow rays
        """

        cdef OptixProgramGroup* c_program_groups_closesthit[2]
        cdef bint must_free[2]
        cdef i
        for j, ch_group in enumerate((program_groups_closesthit_1, program_groups_closesthit_2)):
            size = len(ch_group)
            if isinstance(ch_group, (tuple, list)):
                c_program_groups_closesthit[j] = <OptixProgramGroup*>malloc(size * sizeof(OptixProgramGroup))
                must_free[j] = True
                for i in range(size):
                    c_program_groups_closesthit[j][i] = (<ProgramGroup>ch_group[i]).program_group
            else:
                c_program_groups_closesthit[j][i] = (<ProgramGroup>ch_group).program_group
                must_free[j] = False

        cdef unsigned int direct_callable_stack_size_from_traversal = 0, direct_callable_stack_size_from_state = 0, continuation_stack_size = 0
        try:
            optix_check_return(optixUtilComputeStackSizesSimplePathTracer(program_group_raygen.program_group,
                                                                          program_group_miss_1.program_group,
                                                                          c_program_groups_closesthit[0],
                                                                          len(program_groups_closesthit_1),
                                                                          program_group_miss_2.program_group,
                                                                          c_program_groups_closesthit[1],
                                                                          len(program_groups_closesthit_2),
                                                                          &direct_callable_stack_size_from_traversal,
                                                                          &direct_callable_stack_size_from_state,
                                                                          &continuation_stack_size))
            self.set_stack_sizes(direct_callable_stack_size_from_traversal, direct_callable_stack_size_from_state, continuation_stack_size)
        finally:
            for i in range(2):
                if must_free[i]:
                    free(c_program_groups_closesthit[i])

    def __dealloc__(self):
        if <uintptr_t>self.pipeline != 0:
            optix_check_return(optixPipelineDestroy(self.pipeline))

    def launch(self, ShaderBindingTable sbt, tuple dimensions, LaunchParamsRecord params=None, stream=None):
        """
        Launches the pipeline using the specified dimensions and Data structs.

        Parameters
        ----------
        sbt: ShaderBindingTable
            The ShaderbindingTable to use on this launch.
        dimensions: tuple(3)
            The launch dimensions. This has to be a tuple with a maximum of 3 elements specifying the x, y, and z dimensions
            of the launch. If dimensions are omitted, they are assumed to be 1.
        params: LaunchParamsRecord, optional
            The launch params that will accessible under their configured name in the device code.
        stream: cupy.cuda.Stream, optional
            The cuda stream to use for the launch. If None the default stream (0) will be used.
        """

        cdef uint_vector c_dims = uint_vector(3, 1)
        cdef int i
        for i in range(len(dimensions)):
            c_dims[i] = dimensions[i]

        cdef size_t total_launch_size = c_dims[0] * c_dims[1] * c_dims[2]
        if total_launch_size > self.context.maximum_launch_size:
            raise ValueError(f"Requested launch size of {total_launch_size} is larger than the limit of {self.context.maximum_launch_size} set by OptiX.")

        cdef size_t c_stream = 0
        if stream is not None:
            c_stream = stream.ptr

        d_params = params.to_gpu(stream=stream)

        cdef CUdeviceptr d_params_ptr = d_params.data.ptr
        cdef size_t c_itemsize = params.itemsize
        cdef const OptixShaderBindingTable* c_sbt = &sbt.sbt

        with nogil:
            optix_check_return(optixLaunch(self.pipeline, <CUstream>c_stream, d_params_ptr, c_itemsize, c_sbt, c_dims[0], c_dims[1], c_dims[2]))
