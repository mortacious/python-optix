# distutils: language = c++

cdef class OptixObject:
    def __init__(self, DeviceContext context):
        self.context = context