# distutils: language = c++

cdef class OptixObject:
    """
    Base class for all optix objects providing common utilities
    """

    def _repr_details(self):
        return ""

    def __repr__(self):
        return f"<optix.{self.__class__.__name__}({self._repr_details()})>"