import numpy as np

from optix.sutils.vecmath import length, normalize, cross


def _get_member(varname):

    def getter(self, varname=varname):
        return getattr(self, varname, None)

    return getter


def _set_float(varname, default_value=None):

    def setter(self, value, varname=varname, default_value=default_value):
        if value is None:
            value = default_value
        value = np.float32(value)
        setattr(self, varname, value)

    return setter


def _set_float3(varname, default_value=None):

    def setter(self, value, varname=varname, default_value=default_value):
        if value is None:
            value = default_value
        
        if value is None:
            pass
        elif np.isscalar(value):
            value = np.full(shape=(3,), dtype=np.float32, fill_value=value)
        else:
            value = np.asarray(value, dtype=np.float32)
        setattr(self, varname, value)

    return setter


class Camera:
    """Implements a perspective camera."""

    __slots__ = ['_eye', '_look_at', '_up', '_fov_y', '_aspect_ratio']
    
    def __init__(self, eye=None, look_at=None, up=None, fov_y=None, aspect_ratio=None):
        self.eye = eye
        self.look_at = look_at
        self.up = up
        self.fov_y = fov_y
        self.aspect_ratio = aspect_ratio

    eye = property(_get_member("_eye"), _set_float3("_eye", 1.0))
    look_at = property(_get_member("_look_at"), _set_float3("_look_at", 0.0))
    up = property(_get_member("_up"), _set_float3("_up", [0.0,1.0,0.0]))
    fov_y = property(_get_member("_fov_y"), _set_float("_fov_y", 35.0))
    aspect_ratio = property(_get_member("_aspect_ratio"), _set_float("_aspect_ratio", 1.0))

    def _get_direction(self):
        return normalize(self.look_at - self.eye)
    def _set_direction(self, value):
        self.look_at = self.eye + length(self.look_at - self.eye)*value;
    direction = property(_get_direction, _set_direction)

    def uvw_frame(self):
        W = self.look_at - self.eye
        wlen = length(W)

        U = normalize(cross(W, self.up))
        V = normalize(cross(U, W))

        vlen = wlen * np.tan(0.5 * np.deg2rad(self.fov_y))
        V *= vlen

        ulen = vlen * self.aspect_ratio
        U *= ulen

        return (U,V,W)
