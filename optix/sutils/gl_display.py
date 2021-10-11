
import numpy as np

import OpenGL.GL as gl

class GLDisplay:
    vert_source = \
            r"""
#version 330 core

layout(location = 0) in vec3 vertexPosition_modelspace
out vec2 UV

void main()
{
        gl_Position =  vec4(vertexPosition_modelspace,1)
        UV = (vec2(vertexPosition_modelspace.x, vertexPosition_modelspace.y)+vec2(1,1))/2.0
}
"""

    frag_source = \
            r"""
#version 330 core

in vec2 UV
out vec3 color

uniform sampler2D render_tex
uniform bool correct_gamma

void main()
{
    color = texture(render_tex, UV).xyz
}
"""

    quad_vertex_buffer_data = np.asarray([
        [-1.0, -1.0, 0.0],
        [ 1.0, -1.0, 0.0],
        [-1.0,  1.0, 0.0],
        [-1.0,  1.0, 0.0],
        [ 1.0, -1.0, 0.0],
        [ 1.0,  1.0, 0.0],
        ], dtype=np.float32)

    __slots__ = ['_format', '_render_tex', '_program', '_render_tex_uniforloc', 
            '_quad_vertex_buffer', '_image_format']

    def __init__(self, fmt='uchar4'):
        self._format = fmt
        self._render_tex = 0
        self._program = 0
        self._render_tex_uniforloc = 0
        self._quad_vertex_buffer = 0

        vertex_array = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(vertex_array)

        program = gl.createGLProgram(s_vert_source, s_frag_source)
        render_tex_uniforloc = gl.getGLUniformLocation(program, "render_tex")

        render_tex = gl.glGenTextures(1)
        gl.glBindTexture(GL_TEXTURE_2D, render_tex)

        gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        quad_vertex_buffer = gl.glGenBuffers(1)
        gl.glBindBuffer(GL_ARRAY_BUFFER, quad_vertex_buffer)
        gl.glBufferData(GL_ARRAY_BUFFER,
            quad_vertex_buffer_data,
            GL_STATIC_DRAW)

    def display(screen_res_x, screen_res_y, framebuf_res_x, framebuf_res_y):
        GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, 0))
        GL_CHECK(glViewport(0, 0, framebuf_res_x, framebuf_res_y))

        GL_CHECK(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT))

        GL_CHECK(glUseProgram(program))

        # Bind our texture in Texture Unit 0
        GL_CHECK(glActiveTexture(GL_TEXTURE0))
        GL_CHECK(glBindTexture(GL_TEXTURE_2D, render_tex))
        GL_CHECK(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo))

        GL_CHECK(glPixelStorei(GL_UNPACK_ALIGNMENT, 4));

        elmt_size = pixelFormatSize(image_format)
        if      (elmt_size % 8 == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 8)
        else if (elmt_size % 4 == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 4)
    else if (elmt_size % 2 == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 2)
else                          glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

        bool convertToSrgb = true

        if(image_format == BufferImageFormat::UNSIGNED_BYTE4)
        {
                // input is assumed to be in srgb since it is only 1 byte per channel in size
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,   screen_res_x, screen_res_y, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr)
                convertToSrgb = false
                }
    else if(image_format == BufferImageFormat::FLOAT3)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F,  screen_res_x, screen_res_y, 0, GL_RGB,  GL_FLOAT,         nullptr)

        else if(image_format == BufferImageFormat::FLOAT4)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, screen_res_x, screen_res_y, 0, GL_RGBA, GL_FLOAT,         nullptr)

        else
            throw Exception("Unknown buffer format")

        GL_CHECK(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0))
        GL_CHECK(glUniform1i(render_tex_uniforloc , 0))

        // 1st attribute buffer : vertices
        GL_CHECK(glEnableVertexAttribArray(0))
        GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, quad_vertex_buffer))
        GL_CHECK(glVertexAttribPointer(
            0,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
            3,                  // size
            GL_FLOAT,           // type
            GL_FALSE,           // normalized?
            0,                  // stride
            (void*)0            // array buffer offset
            )
            )

        if(convertToSrgb)
            GL_CHECK(glEnable(GL_FRAMEBUFFER_SRGB))
        else 
            GL_CHECK(glDisable(GL_FRAMEBUFFER_SRGB))

        // Draw the triangles !
        GL_CHECK(glDrawArrays(GL_TRIANGLES, 0, 6)); // 2*3 indices starting at 0 -> 2 triangles

        GL_CHECK(glDisableVertexAttribArray(0))

        GL_CHECK(glDisable(GL_FRAMEBUFFER_SRGB))

        GL_CHECK_ERRORS()
