import numpy as np
import pandas as pd
from typing import Union
from sklearn.decomposition import PCA, base
from scipy.spatial.transform import Rotation


def compute_max_rescaling_factor(max_coord, min_coord):
    """
    Compute a single rescaling factor that is defined by
    the maximum range of any axis of data

    :return: rescaling factor
    """
    return 2 * np.max(max_coord - min_coord)


def compute_rescaling_factor(max_coord, min_coord):
    """
    Compute a rescaling factor for each axis that is defined by
    the maximum range of each axis

    :return: rescaling factor
    """
    return 2 * (max_coord - min_coord)

def compute_centers(max_coord, min_coord):
    """
    Compute the centers of the data from the min and max coords for each axis
    :param max_coord:
    :param min_coord:
    :return:
    """
    centers = ((max_coord + min_coord) / 2).astype(float).values
    return centers

def rescale_surface_points(XYZ: np.ndarray, centers: np.ndarray, rescaling_factor, offset: float = 0.5001):
    """
    Rescale the data to the min/max range and between 0 and 1 in all axes

    roughly equivalent to elliptical anisotropy correction / minmax rescaling

    Consider replacing with sklearn preprocessing function
    :param XYZ:
    :param centers:
    :param offset:
    :return:
    """
    new_coord_surface_points = (XYZ - centers) / rescaling_factor + offset
    return new_coord_surface_points


def apply_rotation_surface_points(XYZ: pd.DataFrame, rotation: Union[np.ndarray, Rotation])-> pd.DataFrame:
    """
    Apply a rotation to the the surface points

    rotations should only be provided in radians

    :param XYZ:
    :param rotation: can by a scipy rotation or a euler vector or a 3x3 matrix
    :return:
    """
    if isinstance(rotation, np.ndarray):
        if np.isnan(rotation).all() or not rotation.any():
            # no rotation needed
            return XYZ
        if rotation.shape == (3, 3):
            rotation = Rotation.from_matrix(rotation)
        elif rotation.shape == (1, 3) or rotation.shape == (3,):
            rotation = Rotation.from_euler('zyx', rotation)
        else:
            raise RuntimeError(f'Unsupported rotation matrix provided by user of shape: {rotation.shape}. Should be 3x3 or 1x3 or 3')
    if isinstance(rotation, Rotation):
        return pd.DataFrame(data=rotation.apply(XYZ), index=XYZ.index, columns=XYZ.columns)
    else:
        raise RuntimeError(f'Unsupported rotation parameter provided by user, should be a numpy array or scipy Rotation, not: {type(rotation)}')


def correct_anisotropy_from_pca(XYZ, pca_routine: base._BasePCA = PCA(n_components=3)):
    """
    Given a distribution of XYZ points, determine how to rotate the data to be more isotropic

    this method uses PCA to determine the primary axes of the data and will return the rotation
    matrix to be applied to the data. It is anticipated that other anisotropy methods will be
    implemented in gempy.

    :param pca_routine: scikit-learn based method to perform PCA
    :param XYZ:
    :return:
    """
    pca_routine.fit(XYZ)
    return Rotation.from_matrix(pca_routine.components_), pca_routine.transform(XYZ)