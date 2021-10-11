
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
    init_imgui(window)

    return window
