import numpy as np
cimport cython

__all__ = ['micromap_indices_to_base_barycentrics']


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def micromap_indices_to_base_barycentrics(uint32_t[:] indices, uint32_t subdivision_level=0):
    """
    Converts micromap triangle indices to three base-triangle barycentric coordinates of the micro triangle vertices.
    The base triangle is the triangle that the micromap is applied to.

    Parameters
    ----------
    indices: Indices of the micro triangles within a micromap.
    subdivision_level: Subdivision level of the micromap.

    Returns
    -------
    base_barycentrics_0: Barycentric coordinates in the space of the base triangle of vertex 0 of the micro triangle.
    base_barycentrics_1: Barycentric coordinates in the space of the base triangle of vertex 1 of the micro triangle.
    base_barycentrics_2: Barycentric coordinates in the space of the base triangle of vertex 2 of the micro triangle.
    """
    cdef Py_ssize_t num_indices = indices.shape[0]

    barycentrics_0 = np.empty((num_indices, 2), dtype=np.float32)
    barycentrics_1 = np.empty((num_indices, 2), dtype=np.float32)
    barycentrics_2 = np.empty((num_indices, 2), dtype=np.float32)

    cdef float[:, ::1] barycentrics_0_view = barycentrics_0
    cdef float[:, ::1] barycentrics_1_view = barycentrics_1
    cdef float[:, ::1] barycentrics_2_view = barycentrics_2

    with nogil:
        for i in range(num_indices):
            optixMicromapIndexToBaseBarycentrics(indices[i],
                                                 subdivision_level,
                                                 *(<float2*>&barycentrics_0_view[i]),
                                                 *(<float2*>&barycentrics_1_view[i]),
                                                 *(<float2*>&barycentrics_2_view[i]))

    return barycentrics_0, barycentrics_1, barycentrics_2


