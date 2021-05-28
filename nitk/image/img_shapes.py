"""
Created on ven. 26 Feb. 2021 15:35:51 CET

@author: edouard.duchesnay@cea.fr
"""
import numpy as np

def make_sphere(box_shape, center, radius):
    """Make a boolean mask of sphere within arr at center.

    Parameters
    ----------
    box_shape: (int, int, int)
        the box shape that contains the sphere

    center: [float, float, float]
        center

    radius: float
        radius of the sphere

    Returns
    -------
    boolean array of the shape arr

    Examples
    --------
    >>> from nitk.image import make_sphere
    >>> make_sphere((3, 3, 3), [1, 1, 1], 1)
    array([[[False, False, False],
        [False,  True, False],
        [False, False, False]],

       [[False,  True, False],
        [ True,  True,  True],
        [False,  True, False]],

       [[False, False, False],
        [False,  True, False],
        [False, False, False]]])

    """
    # get 3 arrays representing indicies along each axis
    xx, yy, zz = np.ogrid[:box_shape[0], :box_shape[1], :box_shape[2]]
    # create 3d array with values that are the distance from the center squared
    d2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 + (zz - center[2]) ** 2
    mask = d2 <= radius ** 2
    return mask
