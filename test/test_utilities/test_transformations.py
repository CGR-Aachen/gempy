import pytest

import sys, os
sys.path.append("../..")
print(os.environ)
import numpy as np
import pandas.testing

import gempy as gp
import gempy.utils.transformations
from scipy.spatial.transform import Rotation

input_path = os.path.dirname(__file__)+'/../../notebooks/data'


def test_rescale_surface_points():
    geo_data = gp.create_data([0, 2000, 0, 2000, 0, 2000], [50, 50, 50],
                              path_o=input_path + '/input_data/tut_chapter1/simple_fault_model_orientations.csv',
                              path_i=input_path + '/input_data/tut_chapter1/simple_fault_model_points.csv')
    XYZ = geo_data.surface_points.df[['X', 'Y', 'Z']]
    rescaling_factor = geo_data.rescaling.compute_rescaling_factor(surface_points=geo_data.surface_points, orientations=geo_data.orientations, inplace=False)
    centers = geo_data.rescaling.compute_data_center(surface_points=geo_data.surface_points, orientations=geo_data.orientations, inplace=False)
    new_coords = gp.utils.transformations.rescale_surface_points(XYZ, centers, rescaling_factor)
    pytest.approx(geo_data.rescaling.get_rescaled_surface_points()[0].to_numpy(), new_coords.to_numpy())

def test_correct_anisotropy_from_pca():
    geo_data = gp.create_data([0, 2000, 0, 2000, 0, 2000], [50, 50, 50],
                              path_o=input_path + '/input_data/tut_chapter1/simple_fault_model_orientations.csv',
                              path_i=input_path + '/input_data/tut_chapter1/simple_fault_model_points.csv')
    XYZ = geo_data.surface_points.df[['X', 'Y', 'Z']]
    rotation, XYZt = gp.utils.transformations.correct_anisotropy_from_pca(XYZ)
    pass

def test_apply_rotation_surface_points():
    geo_data = gp.create_data([0, 2000, 0, 2000, 0, 2000], [50, 50, 50],
                              path_o=input_path + '/input_data/tut_chapter1/simple_fault_model_orientations.csv',
                              path_i=input_path + '/input_data/tut_chapter1/simple_fault_model_points.csv')
    XYZ = geo_data.surface_points.df[['X','Y','Z']]
    # create a fake rotation
    r = Rotation.from_euler('zyx', [30, 0, 20], degrees=True)
    # test using rotation object, todo: verify rotation is correct
    _ = gp.utils.transformations.apply_rotation_surface_points(XYZ, r)
    # test using euler rotation matrix (3x3 array)
    _ = gp.utils.transformations.apply_rotation_surface_points(XYZ, r.as_matrix())
    # test using euler rotation vector (3, array)
    _ = gp.utils.transformations.apply_rotation_surface_points(XYZ, r.as_euler('zyx', degrees=False))
