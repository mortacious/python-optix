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

DEBUG = False

if DEBUG:
    exception_flags = ox.ExceptionFlags.DEBUG | ox.ExceptionFlags.TRACE_DEPTH | ox.ExceptionFlags.STACK_OVERFLOW,
    debug_level = ox.CompileDebugLevel.FULL
    opt_level = ox.CompileOptimizationLevel.LEVEL_0
else:
    exception_flags = ox.ExceptionFlags.NONE
    debug_level = ox.CompileDebugLevel.MINIMAL
    opt_level = ox.CompileOptimizationLevel.LEVEL_3



# ------------------------------------------------------------------------------
# Local types
# ------------------------------------------------------------------------------


class Params:
    _params = collections.OrderedDict([
        ('accum_buffer', 'u8'),
        ('frame_buffer', 'u8'),
        ('width', 'u4'),
        ('height', 'u4'),
        ('spp', 'u4'),
        ('eye', '3f4'),
        ('u', '3f4'),
        ('v', '3f4'),
        ('w', '3f4'),
        ('trav_handle', 'u8'),
        ('subframe_index', 'i4'),
        ('ao', '?')
    ])

    def __init__(self):
        self.handle = ox.LaunchParamsRecord(names=tuple(self._params.keys()),
                                            formats=tuple(self._params.values()))

    def __getattribute__(self, name):
        if name in Params._params.keys():
            item = self.__dict__['handle'][name]
            if isinstance(item, np.ndarray) and item.shape in ((0,), (1,)):
                return item.item()
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


class DisplacedMicromeshState:
    __slots__ = ['params', 'time', 'ctx', 'module', 'pipeline', 'pipeline_opts',
                 'raygen_grp', 'miss_grp', 'hit_grp', 'sbt',
                 'trackball', 'camera_changed', 'mouse_button', 'resize_dirty', 'minimized',
                 'enable_dmms',
                 'dmm_subdivision_level', 'displacement_scale', 'render_ao', 'accum_buffer']

    def __init__(self):
        for slot in self.__slots__:
            setattr(self, slot, None)
        self.params = Params()

        self.trackball = Trackball()
        self.camera_changed = True
        self.mouse_button = -1
        self.resize_dirty = False
        self.minimized = False

        self.enable_dmms = True
        self.dmm_subdivision_level = 3
        self.displacement_scale = 1.0

        self.render_ao = True

        self.accum_buffer = cp.zeros((self.params.height, self.params.width, 4), dtype=np.float32)

    @property
    def camera(self):
        return self.trackball.camera

    @property
    def launch_dimensions(self):
        return (int(self.params.width), int(self.params.height))


# ------------------------------------------------------------------------------
# GLFW callbacks
# ------------------------------------------------------------------------------

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
    state: DisplacedMicromeshState = glfw.get_window_user_pointer(window)
    if action is glfw.PRESS:
        if key in {glfw.KEY_Q, glfw.KEY_ESCAPE}:
            glfw.set_window_should_close(window, True)
    elif key == glfw.KEY_KP_1:
        state.dmm_subdivision_level = max(0, state.dmm_subdivision_level - 1)
        log.info("DMM subdivision levels: %d", state.dmm_subdivision_level)
        build_mesh_accel(state)
        update(state)
    elif key == glfw.KEY_KP_2:
        state.dmm_subdivision_level = min(5, state.dmm_subdivision_level + 1)
        log.info("DMM subdivision levels: %d", state.dmm_subdivision_level)
        build_mesh_accel(state)
        update(state)
    elif key == glfw.KEY_KP_4:
        state.displacement_scale /= 1.5
        log.info("displacement scale: %d", state.dmm_subdivision_level)
        build_mesh_accel(state)
        update(state)
    elif key == glfw.KEY_KP_5:
        state.displacement_scale *= 1.5
        log.info("displacement scale: %d", state.dmm_subdivision_level)
        build_mesh_accel(state)
        update(state)
    elif key == glfw.KEY_D:
        state.enable_dmms = not state.enable_dmms
        log.info("enable dmms: %s", state.enable_dmms)
        build_mesh_accel(state)
        update(state)
    elif key == glfw.KEY_A:
        state.render_ao = not state.render_ao
        create_module(state)
        create_program_groups(state)
        create_pipeline(state)
        create_sbt(state)

        state.params.ao = state.render_ao
        update(state)
    elif key == glfw.KEY_R:
        create_module(state)
        create_program_groups(state)
        create_pipeline(state)
        create_sbt(state)
        update(state)


def scroll_callback(window, xscroll, yscroll):
    state = glfw.get_window_user_pointer(window)
    if state.trackball.wheel_event(yscroll):
        state.camera_changed = True


# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------

def update(state: DisplacedMicromeshState):
    state.accum_buffer = state.accum_buffer.resize(state.params.height, state.params.width, 4)
    state.accum_buffer[:] = 0
    state.params.subframe_index = 0


def init_launch_params(state):
    state.params.frame_buffer = 0
    state.params.subframe_index = 0
    state.params.spp = 1
    state.params.ao = True

    update(state)


def handle_resize(output_buffer, state: DisplacedMicromeshState):
    if not state.resize_dirty:
        return
    state.resize_dirty = False

    output_buffer.resize(state.params.width, state.params.height)
    update(state)


def render_frame(output_buffer, state):
    state.params.frame_buffer = output_buffer.map()
    state.pipeline.launch(state.sbt, dimensions=state.launch_dimensions,
                          params=state.params.handle, stream=output_buffer.stream)
    output_buffer.unmap()


def display_subframe(output_buffer, gl_display, window):
    (framebuf_res_x, framebuf_res_y) = glfw.get_framebuffer_size(window)
    gl_display.display( output_buffer.width, output_buffer.height,
                        framebuf_res_x, framebuf_res_y,
                        output_buffer.get_pbo())


def init_camera_state(state: DisplacedMicromeshState):
    camera = state.camera
    camera.eye = (0, -100, 100)
    camera.look_at = (50, 50, 0)
    camera.up = (0, 0, 1)
    camera.fov_y = 35.0
    state.camera_changed = True

    trackball = state.trackball
    trackball.move_speed = 10.0
    trackball.set_reference_frame([0, 1, 0],
                                  [1, 0, 0],
                                  [0, 0, 1])
    trackball.gimbal_lock = True
    trackball.reinitialize_orientation_from_camera()


def handle_camera_update(state: DisplacedMicromeshState):
    if not state.camera_changed:
        return
    state.camera_changed = False

    camera = state.camera
    params = state.params

    camera.aspect_ratio = params.width / float(params.height)
    params.eye = camera.eye

    u, v, w = camera.uvw_frame()
    params.u = u
    params.v = v
    params.w = w
    update(state)


def create_context(state: DisplacedMicromeshState):
    logger = ox.Logger(log)
    ctx = ox.DeviceContext(validation_mode=False, log_callback_function=logger, log_callback_level=4)
    ctx.cache_enabled = False
    state.ctx = ctx


class DisplacementBlock64MicroTris64B:
    def __init__(self):
        self.data = np.zeros(64, dtype=np.uint8)


