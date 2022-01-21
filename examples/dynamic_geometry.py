import os, sys, enum, logging, collections

import cupy as cp
import numpy as np
import optix as ox

import glfw, imgui

from sutil.gui import init_ui, display_stats
from sutil.gl_display import GLDisplay
from sutil.trackball import Trackball, TrackballViewMode
from sutil.cuda_output_buffer import CudaOutputBuffer, CudaOutputBufferType, BufferImageFormat

script_dir = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger()

DEBUG=False

if DEBUG:
    exception_flags=ox.ExceptionFlags.DEBUG | ox.ExceptionFlags.TRACE_DEPTH | ox.ExceptionFlags.STACK_OVERFLOW,
    debug_level = ox.CompileDebugLevel.FULL
    opt_level = ox.CompileOptimizationLevel.LEVEL_0
else:
    exception_flags=ox.ExceptionFlags.NONE
    debug_level = ox.CompileDebugLevel.MINIMAL
    opt_level = ox.CompileOptimizationLevel.LEVEL_3


#------------------------------------------------------------------------------
# Local types
#------------------------------------------------------------------------------

class Params:
    _params = collections.OrderedDict([
            ('frame_buffer',   'u8'),
            ('width',          'u4'),
            ('height',         'u4'),
            ('eye',            '3f4'),
            ('u',              '3f4'),
            ('v',              '3f4'),
            ('w',              '3f4'),
            ('trav_handle',    'u8'),
            ('subframe_index', 'i4'),
        ])

    def __init__(self):
        self.handle = ox.LaunchParamsRecord(names=tuple(self._params.keys()),
                                            formats=tuple(self._params.values()))

    def __getattribute__(self, name):
        if name in Params._params.keys():
            return self.__dict__['handle'][name]
        else:
            return super().__getattribute__(name)

    def __setattr__(self, name, value):
        if name in Params._params.keys():
            self.handle[name] = value
        elif name in {'handle'}:
            super().__setattr__(name, value)
        else:
            raise AttributeError(name)

    def __str__(self):
        return '\n'.join(f'{k}:  {self.handle[k]}' for k in self._params)


class DynamicGeometryState:
    __slots__ = ['params', 'time', 'ctx', 'module', 'pipeline', 'pipeline_opts',
            'raygen_grp', 'miss_grp', 'hit_grp', 'sbt',
            'generate_vertices_kernel', 'd_temp_vertices', 'last_exploding_sphere_rebuild_time',
            'gas_build_input', 'static_gas', 'deforming_gas', 'exploding_gas',
            'ias_build_input', 'ias',
            'trackball', 'camera_changed', 'mouse_button', 'resize_dirty', 'minimized']

    def __init__(self):
        for slot in self.__slots__:
            setattr(self, slot, None)
        self.params = Params()

        self.trackball = Trackball()
        self.camera_changed = True
        self.mouse_button = -1
        self.resize_dirty = False
        self.minimized = False

    @property
    def camera(self):
        return self.trackball.camera

    @property
    def launch_dimensions(self):
        return (int(self.params.width), int(self.params.height))


class AnimationMode(enum.Enum):
    NONE = 0
    DEFORM = 1
    EXPLODE = 2


#------------------------------------------------------------------------------
# Scene data
#------------------------------------------------------------------------------
g_tessellation_resolution = 128
g_exploding_gas_rebuild_frequency = 10.0

g_diffuse_colors = np.asarray([
    [0.7, 0.7, 0.7],
    [0.8, 0.8, 0.8],
    [0.9, 0.9, 0.9],
    [1.0, 1.0, 1.0],
], dtype=np.float32)

INST_COUNT = g_diffuse_colors.shape[0]

g_instances = np.asarray([
    [1, 0, 0, -4.5,
     0, 1, 0, 0,
     0, 0, 1, 0],
    [1, 0, 0, -1.5,
     0, 1, 0, 0,
     0, 0, 1, 0],
    [1, 0, 0, 1.5,
     0, 1, 0, 0,
     0, 0, 1, 0],
    [1, 0, 0, 4.5,
     0, 1, 0, 0,
     0, 0, 1, 0],
], dtype=np.float32).reshape(INST_COUNT, 3, 4)


#------------------------------------------------------------------------------
# GLFW callbacks
#------------------------------------------------------------------------------
def mouse_button_callback(window, button, action, mods):
    state = glfw.get_window_user_pointer(window)
    (x, y) = glfw.get_cursor_pos(window)
    if action is glfw.PRESS:
        state.mouse_button = button
        state.trackball.start_tracking(x, y)
    else:
        state.mouse_button = -1

def cursor_position_callback(window, x, y):
    state = glfw.get_window_user_pointer(window)
    if state.mouse_button is glfw.MOUSE_BUTTON_LEFT:
        state.trackball.view_mode = TrackballViewMode.LookAtFixed
        state.trackball.update_tracking(x, y, state.params.width, state.params.height)
        state.camera_changed = True
    elif state.mouse_button is glfw.MOUSE_BUTTON_RIGHT:
        state.trackball.view_mode = TrackballViewMode.EyeFixed
        state.trackball.update_tracking(x, y, state.params.width, state.params.height)
        state.camera_changed = True

def window_size_callback(window, res_x, res_y):
    state = glfw.get_window_user_pointer(window)
    if state.minimized:
        return

    res_x = max(res_x, 1)
    res_y = max(res_y, 1)

    state.params.width = res_x
    state.params.height = res_y
    state.camera_changed = True
    state.resize_dirty = True

def window_iconify_callback(window, iconified):
    state = glfw.get_window_user_pointer(window)
    state.minimized = (iconified > 0)

def key_callback(window, key, scancode, action, mods):
    if action is glfw.PRESS:
        if key in {glfw.KEY_Q, glfw.KEY_ESCAPE}:
            glfw.set_window_should_close(window, True)

def scroll_callback(window, xscroll, yscroll):
    state = glfw.get_window_user_pointer(window)
    if state.trackball.wheel_event(yscroll):
        state.camera_changed = True

#------------------------------------------------------------------------------
# Helper functions
#------------------------------------------------------------------------------
def init_launch_params(state):
    state.params.frame_buffer = 0
    state.params.subframe_index = 0

def handle_camera_update(state):
    if not state.camera_changed:
        return
    state.camera_changed = False

    camera = state.camera
    params = state.params

    camera.aspect_ratio = params.width / float(params.height)
    params.eye = camera.eye

    u,v,w = camera.uvw_frame()
    params.u = u
    params.v = v
    params.w = w

def handle_resize(output_buffer, state):
    if not state.resize_dirty:
        return
    state.resize_dirty = False

    output_buffer.resize(state.params.width, state.params.height)

def update_state(output_buffer, state):
    handle_camera_update(state)
    handle_resize(output_buffer, state)

def launch_subframe(output_buffer, state):
    state.params.frame_buffer = output_buffer.map()

    state.pipeline.launch(state.sbt, dimensions=state.launch_dimensions,
            params=state.params.handle, stream=output_buffer.stream)

    output_buffer.unmap()

def display_subframe(output_buffer, gl_display, window):
    (framebuf_res_x, framebuf_res_y) = glfw.get_framebuffer_size(window)
    gl_display.display( output_buffer.width, output_buffer.height,
                        framebuf_res_x, framebuf_res_y,
                        output_buffer.get_pbo() )


def init_camera_state(state):
    camera = state.camera
    camera.eye = (0, 1, -20)
    camera.look_at = (0, 0, 0)
    camera.up = (0, 1, 0)
    camera.fov_y = 35
    camera_changed = True

    trackball = state.trackball
    trackball.move_speed = 10.0
    trackball.set_reference_frame([1,0,0], [0,0,1], [0,1,0])
    trackball.reinitialize_orientation_from_camera()


def create_context(state):
    logger = ox.Logger(log)
    ctx = ox.DeviceContext(validation_mode=False, log_callback_function=logger, log_callback_level=4)
    ctx.cache_enabled = False
    state.ctx = ctx


def generate_animated_vertices(out_vertices, animation_mode, time, width, height):
    threads_per_block = 128
    num_blocks = (width*height + threads_per_block - 1) // threads_per_block

    args = (out_vertices, np.int32(animation_mode.value), np.float32(time), np.int32(width), np.int32(height))

    state.generate_vertices_kernel(grid=(num_blocks,1,1), block=(threads_per_block,1,1), args=args)


def launch_generate_animated_vertices(state, animation_mode):
    generate_animated_vertices(state.d_temp_vertices, animation_mode, state.time, g_tessellation_resolution, g_tessellation_resolution)


def update_mesh_accel(state):
    # first sphere is static

    # second sphere moves by updating its transform matrix
    transform = state.ias_build_input.view_instance_transform(1)
    transform[1, -1] = np.sin(4*state.time)

    # third sphere deforms
    launch_generate_animated_vertices(state, AnimationMode.DEFORM)
    state.deforming_gas.update(state.gas_build_input)

    # fourth sphere explodes
    launch_generate_animated_vertices(state, AnimationMode.EXPLODE)

    # we occasionally rebuild the exploding sphere to maintain AS quality
    if state.time - state.last_exploding_sphere_rebuild_time > 1 / g_exploding_gas_rebuild_frequency:
        state.last_exploding_sphere_rebuild_time = state.time
        state.exploding_gas = ox.AccelerationStructure(state.ctx, state.gas_build_input,
                compact=True, allow_update=True, random_vertex_access=True)
        state.ias_build_input[3].traversable = state.exploding_gas
        state.ias_build_input.update_instance(3)
    else:
        state.exploding_gas.update(state.gas_build_input)

    state.ias.update(state.ias_build_input)


def build_vertex_generation_kernel(state):
    cuda_source = os.path.join(script_dir, 'cuda', 'dynamic_geometry_vertex_generation.cu')
    example_include_path = os.path.dirname(cuda_source)

    build_flags = ox.module.get_default_nvrtc_compile_flags() + (f'-I{example_include_path}',)

    with open(cuda_source, 'r') as f:
        code = f.read()

    state.generate_vertices_kernel = cp.RawKernel(code=code, backend='nvrtc',
            options=build_flags, name='generate_vertices')


def build_mesh_accel(state):
    # Allocate temporary space for vertex generation.
    # The same memory space is reused for generating the deformed and exploding vertices before updates.
    num_vertices = g_tessellation_resolution * g_tessellation_resolution * 6
    state.d_temp_vertices = cp.empty(shape=(num_vertices,3), dtype=np.float32)

    # Build static triangulated sphere.
    build_vertex_generation_kernel(state)
    launch_generate_animated_vertices(state, AnimationMode.NONE)

    #V = cp.asnumpy(state.d_temp_vertices)
    #import trimesh
    #trimesh.Trimesh(vertices=V, faces=np.arange(V.shape[0]).reshape(-1,3)).show()

    # Build an AS over the triangles.
    # We use un-indexed triangles so we can explode the sphere per triangle.
    state.gas_build_input = ox.BuildInputTriangleArray(state.d_temp_vertices, flags=[ox.GeometryFlags.NONE])
    state.static_gas = ox.AccelerationStructure(state.ctx, state.gas_build_input,
            compact=True, allow_update=False, random_vertex_access=True)

    state.deforming_gas = ox.AccelerationStructure(state.ctx, state.gas_build_input,
            compact=True, allow_update=True, random_vertex_access=True)

    state.exploding_gas = ox.AccelerationStructure(state.ctx, state.gas_build_input,
            compact=True, allow_update=True, random_vertex_access=True)

    traversables = [state.static_gas, state.static_gas,
                    state.deforming_gas, state.exploding_gas]
    instances = []
    for i in range(INST_COUNT):
        instance = ox.Instance(traversable=traversables[i], instance_id=0, flags=ox.InstanceFlags.NONE,
                sbt_offset=i, transform=g_instances[i])
        instances.append(instance)

    state.ias_build_input = ox.BuildInputInstanceArray(instances)
    state.ias = ox.AccelerationStructure(context=state.ctx,
            build_inputs=state.ias_build_input, compact=True, allow_update=True)
    state.params.trav_handle = state.ias.handle


def create_module(state):
    pipeline_opts = ox.PipelineCompileOptions(
            uses_motion_blur=False,
            uses_primitive_type_flags=ox.PrimitiveTypeFlags.TRIANGLE,
            traversable_graph_flags=ox.TraversableGraphFlags.ALLOW_SINGLE_LEVEL_INSTANCING,
            exception_flags=exception_flags,
            num_payload_values=3,
            num_attribute_values=2,
            pipeline_launch_params_variable_name="params")

    compile_opts = ox.ModuleCompileOptions(
            max_register_count=ox.ModuleCompileOptions.DEFAULT_MAX_REGISTER_COUNT,
            opt_level=opt_level, debug_level=debug_level)

    cuda_source = os.path.join(script_dir, 'cuda', 'dynamic_geometry.cu')
    state.module = ox.Module(state.ctx, cuda_source, compile_opts, pipeline_opts)
    state.pipeline_opts = pipeline_opts

def create_program_groups(state):
    ctx, module = state.ctx, state.module

    state.raygen_grp = ox.ProgramGroup.create_raygen(ctx, module, "__raygen__rg")
    state.miss_grp = ox.ProgramGroup.create_miss(ctx, module, "__miss__ms")
    state.hit_grp = ox.ProgramGroup.create_hitgroup(ctx, module, entry_function_CH="__closesthit__ch")

def create_pipeline(state):
    program_grps = [state.raygen_grp, state.miss_grp, state.hit_grp]

    link_opts = ox.PipelineLinkOptions(max_trace_depth=1,
                                       debug_level=debug_level)

    pipeline = ox.Pipeline(state.ctx,
                           compile_options=state.pipeline_opts,
                           link_options=link_opts,
                           program_groups=program_grps,
                           max_traversable_graph_depth=2)

    pipeline.compute_stack_sizes(1,  # max_trace_depth
                                 0,  # max_cc_depth
                                 0)  # max_dc_depth

    state.pipeline = pipeline

def create_sbt(state):
    raygen_grp, miss_grp, hit_grp = state.raygen_grp, state.miss_grp, state.hit_grp

    raygen_sbt = ox.SbtRecord(raygen_grp)

    miss_sbt = ox.SbtRecord(miss_grp, names=('bg_color',), formats=('4f4',))
    miss_sbt['bg_color'] = [0.0, 0.0, 0.0, 0.0]

    hit_groups = [hit_grp]*INST_COUNT
    hit_sbts = ox.SbtRecord(hit_groups, names=('color',), formats=('3f4',))
    for i in range(INST_COUNT):
        hit_sbts['color'][i] = g_diffuse_colors[i]

    state.sbt = ox.ShaderBindingTable(raygen_record=raygen_sbt, miss_records=miss_sbt,
            hitgroup_records=hit_sbts)

#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
if __name__ == '__main__':
    state = DynamicGeometryState()
    state.params.width = 1024
    state.params.height = 768
    state.time = 0.0

    num_frames = 16
    animation_time = 1.0

    buffer_format = BufferImageFormat.UCHAR4
    output_buffer_type = CudaOutputBufferType.enable_gl_interop()

    init_camera_state(state)
    create_context(state)
    create_module(state)
    create_program_groups(state)
    create_pipeline(state)
    create_sbt(state)
    init_launch_params(state)
    build_mesh_accel(state)

    window, impl = init_ui("optixDynamicGeometry", state.params.width, state.params.height)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_position_callback)
    glfw.set_window_size_callback(window, window_size_callback)
    glfw.set_window_iconify_callback(window, window_iconify_callback)
    glfw.set_key_callback(window, key_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_window_user_pointer(window, state)

    output_buffer = CudaOutputBuffer(output_buffer_type, buffer_format,
            state.params.width, state.params.height)

    gl_display = GLDisplay(buffer_format)

    state_update_time = 0.0
    render_time = 0.0
    display_time = 0.0

    tstart = glfw.get_time()

    state.last_exploding_sphere_rebuild_time = 0.0

    while not glfw.window_should_close(window):
        t0 = glfw.get_time()
        glfw.poll_events()

        state.time = glfw.get_time() - tstart

        update_mesh_accel(state)

        update_state(output_buffer, state)

        t1 = glfw.get_time()
        state_update_time += t1 - t0
        t0 = t1

        launch_subframe(output_buffer, state)
        t1 = glfw.get_time()
        render_time += t1 - t0
        t0 = t1

        display_subframe(output_buffer, gl_display, window)
        display_time += t1 - t0

        if display_stats(state_update_time, render_time, display_time):
            state_update_time = 0.0
            render_time = 0.0
            display_time = 0.0

        imgui.render()
        impl.render(imgui.get_draw_data())

        glfw.swap_buffers(window)

        state.params.subframe_index = state.params.subframe_index.item() + 1

    impl.shutdown()
    glfw.terminate()
