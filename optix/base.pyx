# distutils: language = c++

cdef class OptixObject:
    def __init__(self, DeviceContext context):
        self.context = context

    def _repr_details(self):
        return ""

    def __repr__(self):
        return f"<optix.{self.__class__.__name__}({self._repr_details()})>"