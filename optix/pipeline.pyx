# distutils: language = c++

from .common cimport optix_check_return, optix_init
from .common import ensure_iterable
from .context cimport DeviceContext
from .program_group cimport OptixProgramGroup, OptixProgramGroupOptions, optixProgramGroupCreate, OptixProgramGroupDesc
from enum import IntEnum, IntFlag
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
from .struct cimport LaunchParamsRecord
from .struct import LaunchParamsRecord
from .shader_binding_table cimport ShaderBindingTable
from cython.operator cimport dereference as deref
import cupy as cp

optix_init()


class CompileDebugLevel(IntEnum):
    DEFAULT = OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT,
    NONE = OPTIX_COMPILE_DEBUG_LEVEL_NONE,
    LINEINFO = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO,
    FULL = OPTIX_COMPILE_DEBUG_LEVEL_FULL


class ExceptionFlags(IntFlag):
    NONE = OPTIX_EXCEPTION_FLAG_NONE,
    STACK_OVERFLOW = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW,
    TRACE_DEPTH = OPTIX_EXCEPTION_FLAG_TRACE_DEPTH,
    USER = OPTIX_EXCEPTION_FLAG_USER,
    DEBUG = OPTIX_EXCEPTION_FLAG_DEBUG


class TraversableGraphFlags(IntFlag):
    ALLOW_ANY = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY,
    ALLOW_SINGLE_GAS = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS,
    ALLOW_SINGLE_LEVEL_INSTANCING = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING


class PrimitiveTypeFlags(IntFlag):
    DEFAULT = 0, # corresponds to CUSTOM | TRIANGLE
    CUSTOM = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM,
    ROUND_QUADRATIC_BSPLINE = OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_QUADRATIC_BSPLINE,
    ROUND_CUBIC_BSPLINE = OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BSPLINE,
    ROUND_LINEAR = OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_LINEAR,
    TRIANGLE = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE


cdef class PipelineCompileOptions:
    def __init__(self,
                 uses_motion_blur=False,
                 traversable_graph_flags = TraversableGraphFlags.ALLOW_ANY,
                 num_payload_values = 0,
                 num_attribute_values = 0,
                 exception_flags = ExceptionFlags.NONE,
                 pipeline_launch_params_variable_name = "params",
                 uses_primitive_type_flags = PrimitiveTypeFlags.DEFAULT):
        #self._compile_options = []
        self._compile_options.usesMotionBlur = uses_motion_blur
        self._compile_options.traversableGraphFlags = traversable_graph_flags.value
        self._compile_options.numPayloadValues = num_payload_values
        self._compile_options.numAttributeValues = num_attribute_values
        self._compile_options.exceptionFlags = exception_flags.value
        self.pipeline_launch_params_variable_name = pipeline_launch_params_variable_name
        self._compile_options.usesPrimitiveTypeFlags = uses_primitive_type_flags.value

    @property
    def uses_motion_blur(self):
        return self._compile_options.usesMotionBlur

    @uses_motion_blur.setter
    def uses_motion_blur(self, motion_blur):
        self._compile_options.usesMotionBlur = motion_blur

    @property
    def traversable_graph_flags(self):
        return TraversableGraphFlags(self._compile_options.traversableGraphFlags)

    @traversable_graph_flags.setter
    def traversable_graph_flags(self, flags):
        self._compile_options.traversableGraphFlags = flags.value

    @property
    def num_payload_values(self):
        return self._compile_options.numPayloadValues

    @num_payload_values.setter
    def num_payload_values(self, num_payload_values):
        self._compile_options.numPayloadValues = num_payload_values

    @property
    def num_attribute_values(self):
        return self._compile_options.numAttributeValues

    @num_attribute_values.setter
    def num_attribute_values(self, num_attribute_values):
        self._compile_options.numAttributeValues = num_attribute_values

    @property
    def exception_flags(self):
        return ExceptionFlags(self._compile_options.exceptionFlags)

    @exception_flags.setter
    def exception_flags(self, flags):
        self._compile_options.exceptionFlags = flags.value

    @property
    def pipeline_launch_params_variable_name(self):
        return self._pipeline_launch_params_variable_name.decode('ascii')

    @pipeline_launch_params_variable_name.setter
    def pipeline_launch_params_variable_name(self, name):
        self._pipeline_launch_params_variable_name = name.encode('ascii')
        self._compile_options.pipelineLaunchParamsVariableName = <const char*>self._pipeline_launch_params_variable_name

    @property
    def uses_primitive_type_flags(self):
        return PrimitiveTypeFlags(self._compile_options.usesPrimitiveTypeFlags)

    @uses_primitive_type_flags.setter
    def uses_primitive_type_flags(self, flags):
        self._compile_options.usesPrimitiveTypeFlags = flags.value

    @property
    def c_obj(self):
        return <size_t>&self._compile_options


cdef class PipelineLinkOptions:
    def __init__(self, unsigned int max_trace_depth = 1, debug_level = CompileDebugLevel.DEFAULT):
        self._link_options.maxTraceDepth = max_trace_depth
        self._link_options.debugLevel = debug_level.value

    @property
    def max_trace_depth(self):
        return self._link_options.maxTraceDepth

    @max_trace_depth.setter
    def max_trace_depth(self, unsigned int max_trace_depth):
        self._link_options.maxTraceDepth = max_trace_depth

    @property
    def debug_level(self):
        return CompileDebugLevel(self._link_options.debugLevel)

    @debug_level.setter
    def debug_level(self, debug_level):
        self._link_options.debugLevel = debug_level.value

    @property
    def c_obj(self):
        return <size_t>&self._link_options

ctypedef vector[unsigned int] uint_vector

cdef class Pipeline:
    def __init__(self, DeviceContext context, PipelineCompileOptions compile_options, PipelineLinkOptions link_options, program_groups, max_traversable_graph_depth=1):
        program_groups = ensure_iterable(program_groups)

        if not all(isinstance(p, ProgramGroup) for p in program_groups):
            raise TypeError("Only program groups")
        cdef unsigned int num_program_groups = len(program_groups)
        cdef vector[OptixProgramGroup] c_program_groups = vector[OptixProgramGroup](num_program_groups)
        cdef int i
        cdef OptixProgramGroupOptions options
        cdef ProgramGroup grp
        for i in range(num_program_groups):
            grp = <ProgramGroup>program_groups[i]
            c_program_groups[i] = grp._program_group

        print("computing stack sizes")
        for i in range(len(program_groups)):
            optix_check_return(optixUtilAccumulateStackSizes(c_program_groups[i], &self._stack_sizes))
        print("creating pipeline ...")
        optix_check_return(optixPipelineCreate(context.device_context,
                                               &compile_options._compile_options,
                                               &link_options._link_options,
                                               c_program_groups.const_data(),
                                               len(program_groups),
                                               NULL,
                                               NULL,
                                               &self._pipeline))

        self._max_traversable_graph_depth = max_traversable_graph_depth

    cpdef set_stack_sizes(self,
                          unsigned int direct_callable_stack_size_from_traversal,
                          unsigned int direct_callable_stack_size_from_state,
                          unsigned int continuation_stack_size):
        optix_check_return(optixPipelineSetStackSize(self._pipeline,
                                                     direct_callable_stack_size_from_traversal,
                                                     direct_callable_stack_size_from_state,
                                                     continuation_stack_size,
                                                     self._max_traversable_graph_depth))

    cpdef compute_stack_sizes(self,
                              unsigned int max_trace_depth,
                              unsigned int max_cc_depth,
                              unsigned int max_dc_depth):
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
        cdef OptixProgramGroup* c_program_groups_closesthit[2]
        cdef bint must_free[2]
        cdef i
        for j, ch_group in enumerate((program_groups_closesthit_1, program_groups_closesthit_2)):
            size = len(ch_group)
            if isinstance(ch_group, (tuple, list)):
                c_program_groups_closesthit[j] = <OptixProgramGroup*>malloc(size * sizeof(OptixProgramGroup))
                must_free[j] = True
                for i in range(size):
                    c_program_groups_closesthit[j][i] = (<ProgramGroup>ch_group[i])._program_group
            else:
                c_program_groups_closesthit[j][i] = (<ProgramGroup>ch_group)._program_group
                must_free[j] = False

        cdef unsigned int direct_callable_stack_size_from_traversal = 0, direct_callable_stack_size_from_state = 0, continuation_stack_size = 0
        try:
            optix_check_return(optixUtilComputeStackSizesSimplePathTracer(program_group_raygen._program_group,
                                                                          program_group_miss_1._program_group,
                                                                          c_program_groups_closesthit[0],
                                                                          len(program_groups_closesthit_1),
                                                                          program_group_miss_2._program_group,
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
        if <size_t>self._pipeline != 0:
            optix_check_return(optixPipelineDestroy(self._pipeline))

    @property
    def c_obj(self):
        return <size_t>&self._pipeline

    def launch(self, stream, ShaderBindingTable sbt, tuple dimensions, LaunchParamsRecord params=None):
        cdef uint_vector c_dims = uint_vector(3, 1)
        cdef int i
        for i in range(len(dimensions)):
            c_dims[i] = dimensions[i]

        stream = cp.cuda.Stream()
        d_params = params.to_gpu(stream=stream)
        cdef size_t c_stream = stream.ptr
        optix_check_return(optixLaunch(self._pipeline, <CUstream>c_stream, d_params, params.itemsize, &sbt._shader_binding_table, c_dims[0], c_dims[1], c_dims[2]))