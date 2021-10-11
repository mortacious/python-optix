
import glfw
import OpenGL.GL as gl

import imgui
from imgui.integrations.glfw import GlfwRenderer

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
