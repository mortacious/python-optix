
import glfw
import OpenGL.GL as gl

import imgui
from imgui.integrations.glfw import GlfwRenderer


def static_vars(**kwargs):
    """
    Attach a static variables local to decorated function.
    """
    def decorate(f):
        for k in kwargs:
            setattr(f, k, kwargs[k])
        return f
    return decorate

def init_gl():
    gl.glClearColor(0.212, 0.271, 0.31, 1.0)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)

def init_imgui(window):
    imgui.create_context()
    impl = GlfwRenderer(window)
    impl.io.fonts.add_font_default()
    imgui.core.style_colors_dark();
    imgui.get_style().window_border_size = 0.0
    return impl

def init_ui(window_title, width, height):
    if not glfw.init():
        raise RuntimeError("Could not initialize OpenGL context")

    window = glfw.create_window(int(width), int(height), window_title, None, None)
    glfw.make_context_current(window)

    if not window:
        raise RuntimeError("Could not initialize Window")

    glfw.swap_interval(0)

    init_gl()
    impl = init_imgui(window)

    return window, impl

def display_text(text, x, y):
    imgui.set_next_window_bg_alpha(0.0)
    imgui.set_next_window_position(x, y)

    flags = (imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE |
             imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_NO_SAVED_SETTINGS | imgui.WINDOW_NO_INPUTS)

    imgui.begin("TextOverlayFG", None, flags)
    imgui.push_style_color(imgui.COLOR_TEXT, 0.7, 0.7, 0.7, 1.0)
    imgui.text(text)
    imgui.pop_style_color()
    imgui.end()

@static_vars(total_subframe_count=0, last_update_frames=0,
        last_update_time=None, display_text="")
def display_stats(state_update_time, render_time, display_time):
    display_update_min_interval_time = 0.5

    cur_time = glfw.get_time()

    display_stats.last_update_frames += 1
    last_update_time = display_stats.last_update_time or cur_time - 0.5
    last_update_frames = display_stats.last_update_frames
    total_subframe_count = display_stats.total_subframe_count

    dt = cur_time - last_update_time

    do_update = (dt > display_update_min_interval_time) or (total_subframe_count == 0)

    if do_update:
        fps = last_update_frames / dt
        state_ms = 1000.0 * state_update_time / last_update_frames
        render_ms = 1000.0 * render_time / last_update_frames
        display_ms = 1000.0 * display_time / last_update_frames

        display_stats.last_update_time = cur_time
        display_stats.last_update_frames = 0

        display_stats.display_text = \
f"""{fps:5.1f} fps

state update: {state_ms:8.1f} ms
render      : {render_ms:8.1f} ms
display     : {display_ms:8.1f} ms
"""

    imgui.new_frame()
    display_text(display_stats.display_text, 10.0, 10.0)
    imgui.end_frame()

    display_stats.total_subframe_count += 1

    return do_update
