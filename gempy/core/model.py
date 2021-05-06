import os
import shutil
import sys
from abc import ABC

import numpy as np
import pandas as pn
from typing import Union, Iterable
import warnings

from gempy.core.data_modules.geometric_data import Orientations, SurfacePoints, \
    ScalingSystem, Surfaces, Grid
from gempy.core.data_modules.stack import Stack, Faults, Series
from gempy.core.data import AdditionalData, MetaData, Options, Structure, \
    KrigingParameters
from gempy.core.solution import Solution
from gempy.core.interpolator import InterpolatorModel, InterpolatorGravity
from gempy.utils.meta import _setdoc, _setdoc_pro
import gempy.utils.docstring as ds
from gempy.plot.decorators import *

pn.options.mode.chained_assignment = None


class RestrictingWrapper(object):
    def __init__(self, w, accepted_members=['__repr__', '_repr_html_', '__str__']):
        self._w = w
        self._accepted_members = accepted_members

    def __repr__(self):
        return self._w.__repr__()

    def __getattr__(self, item):
        if item in self._accepted_members:
            return getattr(self._w, item)
        else:
            raise AttributeError(item)


@_setdoc_pro([Grid.__doc__, Faults.__doc__, Series.__doc__, Surfaces.__doc__,
              SurfacePoints.__doc__,
              Orientations.__doc__, ScalingSystem.__doc__, AdditionalData.__doc__,
              InterpolatorModel.__doc__,
              Solution.__doc__])
class ImplicitCoKriging(object):
    """This class handles all the mutation of the data objects of the model involved on the
     implicit cokriging ensuring the synchronization of all the members.

    Attributes:
        _grid (:class:`gempy.core.data.Grid`): [s0]
        _faults (:class:`gempy.core.data.Grid`): [s1]
        _stack (:class:`gempy.core.data_modules.stack.Stack`): [s2]
        _surfaces (:class:`gempy.core.data.Surfaces`): [s3]
        _surface_points (:class:`gempy.core.data_modules.geometric_data.SurfacePoints`): [s4]
        _orientations (:class:`gempy.core.data_modules.geometric_data.Orientations`): [s5]
        _rescaling (:class:`gempy.core.data_modules.geometric_data.Rescaling`): [s6]
        _additional_data (:class:`gempy.core.data.AdditionalData`): [s7]
        _interpolator (:class:`gempy.core.interpolator.InterpolatorModel`): [s8]
        solutions (:class:`gempy.core.solutions.Solutions`): [s9]


     """

    def __init__(self):

        self._grid = Grid()
        # Old way
        self._faults = Faults()
        self._stack = Series(self._faults)

        # New way
        self._stack = Stack()
        self._faults = self._stack.faults

        self._series = self._stack
        self._surfaces = Surfaces(self._stack)
        self._surface_points = SurfacePoints(self._surfaces)
        self._orientations = Orientations(self._surfaces)

        self._rescaling = ScalingSystem(self._surface_points, self._orientations,
                                        self._grid)
        self._additional_data = AdditionalData(self._surface_points,
                                               self._orientations, self._grid,
                                               self._faults,
                                               self._surfaces, self._rescaling)

        self._interpolator = InterpolatorModel(self._surface_points,
                                               self._orientations, self._grid,
                                               self._surfaces,
                                               self._stack, self._faults,
                                               self._additional_data)

        self.solutions = Solution(self._grid, self._surfaces, self._stack)

        # Previous values of sfai.
        self._sfai_order_0 = None

    @_setdoc_pro(Grid.__doc__)
    @property
    def grid(self):
        """ :class:`gempy.core.data.Grid` [s0]

        """
        return RestrictingWrapper(
            self._grid,
            accepted_members=['__repr__', '__str__', 'values', 'regular_grid',
                              'sections', 'centered_grid'])

    @_setdoc_pro(Faults.__doc__)
    @property
    def faults(self):
        """:class:`gempy.core.data_modules.stack.Faults` [s0]"""
        return RestrictingWrapper(self._faults,
                                  accepted_members=['__repr__', '_repr_html_',
                                                    'faults_relations_df'])

    @_setdoc_pro(Stack.__doc__)
    @property
    def stack(self):
        """:class:`gempy.core.data_modules.stack.Stack` [s0]"""
        return RestrictingWrapper(self._stack,
                                  accepted_members=['__repr__', '_repr_html_', 'df'])

    @_setdoc_pro(Series.__doc__)
    @property
    def series(self):
        """"""
        # warnings.warn(DeprecationWarning, 'series will be deprecated in the future.'
        #                                  'Use stack instead.')
        return RestrictingWrapper(self._stack,
                                  accepted_members=['__repr__', '_repr_html_', 'df'])

    @_setdoc_pro(Surfaces.__doc__)
    @property
    def surfaces(self):
        """:class:`gempy.core.data.Surfaces` [s0]"""
        return RestrictingWrapper(self._surfaces,
                                  accepted_members=['__repr__', '_repr_html_',
                                                    'colors', 'df'])

    @_setdoc_pro(SurfacePoints.__doc__)
    @property
    def surface_points(self):
        """:class:`gempy.core.data_modules.geometric_data.SurfacePoints` [s0]"""
        return RestrictingWrapper(self._surface_points,
                                  accepted_members=['__repr__', '_repr_html_', 'df'])

    @_setdoc_pro(Orientations.__doc__)
    @property
    def orientations(self):
        """:class:`gempy.core.data_modules.geometric_data.Orientations` [s0]"""
        return RestrictingWrapper(self._orientations,
                                  accepted_members=['__repr__', '_repr_html_', 'df'])

    @_setdoc_pro(ScalingSystem.__doc__)
    @property
    def rescaling(self):
        """:class:`gempy.core.data_modules.geometric_data.Rescaling` [s0]"""
        return RestrictingWrapper(self._rescaling)

    @_setdoc_pro(AdditionalData.__doc__)
    @property
    def additional_data(self):
        """:class:`gempy.core.data.AdditionalData` [s0]"""
        return RestrictingWrapper(self._additional_data,
                                  accepted_members=['__repr__', '_repr_html_',
                                                    'structure_data', 'options',
                                                    'kriging_parameters',
                                                    'kriging_data',
                                                    'rescaling_data'])

    @_setdoc_pro(InterpolatorModel.__doc__)
    @property
    def interpolator(self):
        """:class:`gempy.core.interpolator.InterpolatorModel` [s0]"""
        return RestrictingWrapper(self._interpolator,
                                  accepted_members=['__repr__', '_repr_html_',
                                                    'theano_graph'])

    def _add_valid_idx_s(self, idx):
        if idx is None:
            idx = self._surface_points.df.index.max()
            if idx is np.nan:
                idx = 0
            else:
                idx += 1
        else:
            assert isinstance(idx, (
                int, list, np.ndarray)), 'idx must be an int or a list of ints'

        return idx

    def _add_valid_idx_o(self, idx):
        if idx is None:
            idx = self._orientations.df.index.max()
            if idx is np.nan:
                idx = 0
            else:
                idx += 1
        else:
            assert isinstance(idx, (
                int, list, np.ndarray)), 'idx must be an int or a list of ints'

        return idx

    @_setdoc_pro([AdditionalData.update_structure.__doc__,
                  InterpolatorModel.set_theano_shared_structure.__doc__,
                  InterpolatorModel.modify_results_matrices_pro.__doc__,
                  InterpolatorModel.modify_results_weights.__doc__])
    def update_structure(self, update_theano=None, update_series_is_active=True,
                         update_surface_is_active=True):
        """Update python and theano structure parameters.

        [s0]
        [s1]

        Args:
            update_theano (str{'matrices', 'weights'}):

                * matrices [s2]

                * weights [s3]

        Returns:
            :class:`gempy.core.data.Structure`
        """

        self._additional_data.update_structure()

        if update_series_is_active is True:
            len_series_i = self._additional_data.structure_data.df.loc[
                               'values', 'len series surface_points'] - \
                           self._additional_data.structure_data.df.loc[
                               'values', 'number surfaces per series']

            len_series_o = self._additional_data.structure_data.df.loc[
                'values', 'len series orientations'].astype(
                'int32')

            # Remove series without data
            non_zero_i = len_series_i.nonzero()[0]
            non_zero_o = len_series_o.nonzero()[0]
            non_zero = np.intersect1d(non_zero_i, non_zero_o)

            bool_vec = np.zeros_like(self._stack.df['isActive'], dtype=bool)
            bool_vec[non_zero] = True
            self._stack.df['isActive'] = bool_vec

        if update_surface_is_active is True:
            act_series = self._surfaces.df['series'].map(
                self._stack.df['isActive']).astype(bool)
            unique_surf_points = np.unique(self._surface_points.df['id'])
            if len(unique_surf_points) != 0:
                bool_surf_points = np.zeros_like(act_series, dtype=bool)
                bool_surf_points[unique_surf_points.astype('int') - 1] = True

                # This is necessary to find the intersection between orientations
                # (series) and  surface points
                self._surfaces.df['isActive'] = (
                    act_series & bool_surf_points) | self._surfaces.df['isBasement']
                self._surfaces.df['hasData'] = (
                        act_series | bool_surf_points)

        if update_theano == 'matrices':
            self._interpolator.modify_results_matrices_pro()
        elif update_theano == 'weights':
            self._interpolator.modify_results_weights()

        self._interpolator.set_theano_shared_structure()
        return self._additional_data.structure_data

    # region Grid
    def update_from_grid(self):
        """Update objects dependent from the grid.

        """
        self._rescaling.rescale_data()
        self._interpolator.set_initial_results_matrices()

        if 'gravity' in self._interpolator.theano_graph.output or 'magnetics' in self._interpolator.theano_graph.output:
            self._interpolator.set_theano_shared_l0_l1()

        # Check if grid is shared
        if hasattr(self._interpolator.theano_graph.grid_val_T, 'get_value'):
            self._interpolator.theano_graph.grid_val_T.set_value(
                self._grid.values_r.astype(self._interpolator.dtype))

    def set_active_grid(self, grid_name: Union[str, Iterable[str]], reset=False):
        """Set active a given or several grids.

        Args:
            grid_name (str, list[str]): Name of the grid you want to activate. Options are
             {regular, custom, topography, centered}
            reset (bool): If True set inactive all grids not in grid_name

        Returns:
            :class:`gempy.core.data.Grid`

        """
        if reset is True:
            self._grid.deactivate_all_grids()
        self._grid.set_active(grid_name)
        self.update_from_grid()
        print(f'Active grids: {self._grid.grid_types[self._grid.active_grids]}')

        return self._grid

    def get_active_grids(self):
        return self._grid.grid_types[self._grid.active_grids]

    def set_grid_object(self, grid: Grid, update_model=True):
        """Not implemented

        # TODO this should go to the api and let call all different grid types

        Args:
            grid:
            update_model:

        Returns:

        """
        raise NotImplementedError

    # @_setdoc(Grid.create_regular_grid.__doc__)
    @_setdoc_pro()
    def set_regular_grid(self, extent, resolution):
        """Set a regular grid, rescale data and initialize theano solutions.

        Args:
            extent: [s_extent]
            resolution: [s_resolution]

        Returns:
            :class:`gempy.core.data.Grid`

        See Also:
            :class:`gempy.core.data.Grid.create_regular_grid`

            :class:`gempy.core.data.grid_modules.grid_types.RegularGrid`

        """
        if self._grid.regular_grid is None:
            self._grid.create_regular_grid(extent=extent, resolution=resolution)
        else:
            self._grid.regular_grid.set_regular_grid(extent=extent,
                                                     resolution=resolution)
            self._grid.set_active('regular')

        if self._grid.topography is not None and self._grid.topography.values.shape[
            0] != 0:
            self._grid.regular_grid.set_topography_mask(self._grid.topography)

        self.update_from_grid()
        print(f'Active grids: {self._grid.grid_types[self._grid.active_grids]}')
        return self._grid

    @_setdoc_pro()
    def set_custom_grid(self, custom_grid):
        """Set custom grid, rescale gird and initialize theano solutions. foo

        Args:
            custom_grid: [s_coord]

        Returns:
            :class:`gempy.core.data.Grid`

        See Also:
            :class:`gempy.core.data.Grid.create_custom_grid`

            :class:`gempy.core.data.grid_modules.grid_types.CustomGrid`

        """
        if self._grid.custom_grid is None:
            self._grid.create_custom_grid(custom_grid)
        else:
            self._grid.custom_grid.set_custom_grid(custom_grid)
            self._grid.update_grid_values()

        self.update_from_grid()
        print(f'Active grids: {self._grid.grid_types[self._grid.active_grids]}')
        return self._grid

    @plot_set_topography
    def set_topography(self, source='random', set_mask=True, **kwargs):
        """Create a topography grid and activate it.

        Args:
            source:
                * 'gdal': Load topography from a raster file.
                * 'random': Generate random topography (based on a fractal grid).
                * 'saved': Load topography that was saved with the topography.save() function.
                  this is useful after loading and saving a heavy raster file with gdal once
                  or after saving a random topography with the save() function. This .npy file can then be set as
                  topography.

        Keyword Args:
            source = 'gdal':

                * filepath: path to raster file, e.g. '.tif', (for all file formats see
                  https://gdal.org/drivers/raster/index.html)

            source = 'random':

                * fd:         fractal dimension, defaults to 2.0

                * d_z:        maximum height difference. If none, last 20% of the model in z direction

                * extent:     extent in xy direction. If none, geo_model.grid.extent

                * resolution: desired resolution of the topography array. If none, geo_model.grid.resoution

            source = 'saved':

                * filepath:   path to the .npy file that was created using the topography.save() function

            source = 'numpy':
                * array: numpy array containing the data

        Returns:
             :class:`gempy.core.data.Grid`

        See Also:
            :class:`gempy.core.grid_modules.grid_types.Topography`

        """

        self._grid.create_topography(source, **kwargs)
        if set_mask is True:
            try:
                self._grid.regular_grid.set_topography_mask(self._grid.topography)
            except AttributeError:
                pass

        self.update_from_grid()
        print(f'Active grids: {self._grid.grid_types[self._grid.active_grids]}')
        return self._grid

    @_setdoc_pro(Grid.create_centered_grid.__doc__)
    def set_centered_grid(self, centers, radius, resolution=None):
        """[s0]

        Args:
            centers (numpy.ndarray[float, 3]): Location of the center of each kernel.
            radius (float): Distance from each center to create each XYZ point
            resolution (numpy.ndarray[3]): Number of voxels in each direction per kernel

        Returns:
            :class:`gempy.core.data.Grid`

        See Also:
            :class:`gempy.core.grid_modules.grid_types.CenteredGrid`

        """
        if self._grid.centered_grid is None:
            self._grid.create_centered_grid(centers, radius, resolution=resolution)
        else:
            self._grid.centered_grid.set_centered_grid(centers=centers,
                                                       radius=radius,
                                                       resolution=resolution)
            self._grid.update_grid_values()
        self.set_active_grid('centered')
        self.update_from_grid()
        # print(f'Active grids: {self._grid.grid_types[self._grid.active_grids]}')
        return self._grid

    @_setdoc_pro(Grid.create_section_grid.__doc__)
    def set_section_grid(self, section_dict):
        """[s0]

        Args:
            section_dict: [s_section_dict]

        Returns:
            :class:`gempy.core.grid_modules.grid_types.Sections`

        See Also:
            :class:`gempy.core.grid_modules.grid_types.Sections`
        """
        # TODO being able to change the regular grid associated to the section grid
        if self._grid.sections is None:
            self._grid.create_section_grid(section_dict=section_dict)
        else:
            self._grid.sections.set_sections(section_dict,
                                             regular_grid=self._grid.regular_grid)

        self.set_active_grid('sections')
        self.update_from_grid()
        return self._grid.sections

    # endregion

    # region Series
    def set_series_object(self):
        """
        Not implemented yet. Exchange the series object of the Model object.

        """
        raise NotImplementedError

    @_setdoc([Series.set_bottom_relation.__doc__], indent=False)
    def set_bottom_relation(self, series: Union[str, list],
                            bottom_relation: Union[str, list]):
        """"""
        self._stack.set_bottom_relation(series, bottom_relation)
        self._interpolator.set_theano_shared_relations()
        return self._stack

    @_setdoc_pro(Stack.reset_order_series.__doc__)
    def add_features(self, features_list: Union[str, list], reset_order_series=True):
        """ Add series, update the categories dependent on them and reset the flow control.

        Args:
            features_list: (str, list): name or list of names of the series to apply the functionality
            reset_order_series: if true [s0]

        Returns:
            :class:`gempy.core.data_modules.stack.Stack`

        """
        self._stack.add_series(features_list, reset_order_series)
        self._surfaces.df['series'].cat.add_categories(features_list, inplace=True)
        self._surface_points.df['series'].cat.add_categories(features_list,
                                                             inplace=True)
        self._orientations.df['series'].cat.add_categories(features_list,
                                                           inplace=True)

        self.update_structure()
        self._interpolator.set_flow_control()
        self._interpolator.set_theano_shared_kriging()
        return self._stack

    @_setdoc(Series.add_series.__doc__, indent=False)
    def add_series(self, series_list: Union[str, list], reset_order_series=True):
        warnings.warn('Series are getting renamed to Stack/features.'
                      'Please use add_features instead', DeprecationWarning, )
        return self.add_features(series_list, reset_order_series)

    @_setdoc_pro(Stack.reset_order_series.__doc__)
    def delete_features(self, indices: Union[str, list], reset_order_features=True,
                        remove_surfaces=False, remove_data=False):
        """ Delete series, update the categories dependent on them and reset the flow control.

        Args:
            indices (str, list): name or list of names of the series to apply the functionality
            reset_order_features: (bool): if true [s0]
            remove_surfaces (bool): if True remove the surfaces associated with the feature.
            remove_data (bool): if True remove the geometric data associated with the feature

        Returns:
             :class:`gempy.core.data_modules.stack.Stack`

        """
        indices = np.atleast_1d(indices)
        self._stack.delete_series(indices, reset_order_features)

        if remove_surfaces is True:
            for s in indices:
                self.delete_surfaces(
                    self._surfaces.df.groupby('series').get_group(s)['surface'],
                    remove_data=remove_data)

        self._surfaces.df['series'].cat.remove_categories(indices, inplace=True)
        self._surface_points.df['series'].cat.remove_categories(indices,
                                                                inplace=True)
        self._orientations.df['series'].cat.remove_categories(indices, inplace=True)
        self.map_geometric_data_df(self._surface_points.df)
        self.map_geometric_data_df(self._orientations.df)

        self.update_structure()
        self._interpolator.set_theano_shared_relations()
        self._interpolator.set_theano_shared_kriging()
        self._interpolator.set_flow_control()
        return self._stack

    @_setdoc(delete_features.__doc__, indent=False)
    def delete_series(self, indices: Union[str, list], refactor_order_series=True,
                      remove_surfaces=False, remove_data=False):

        warnings.warn(DeprecationWarning,
                      'Series are getting renamed to Stack/features.'
                      'Please use delete_features instead')
        return self.delete_features(indices, refactor_order_series,
                                    remove_surfaces, remove_data)

    @_setdoc(Series.rename_series.__doc__, indent=False)
    def rename_features(self, new_categories: Union[dict, list]):
        """Rename features and update the category dependent on them.

        Args:
            new_categories (list, dict):
                * list-like: all items must be unique and the number of items in the new
                  categories must match the existing number of categories.

                * dict-like: specifies a mapping from old categories to new. Categories
                  not contained in the mapping are passed through and extra categories in the mapping are ignored.

        Returns:
            :class:`gempy.core.data_modules.stack.Stack`


        """
        self._stack.rename_series(new_categories)
        self._surfaces.df['series'].cat.rename_categories(new_categories,
                                                          inplace=True)
        self._surface_points.df['series'].cat.rename_categories(new_categories,
                                                                inplace=True)
        self._orientations.df['series'].cat.rename_categories(new_categories,
                                                              inplace=True)
        return self._stack

    @_setdoc(rename_features.__doc__, indent=False)
    def rename_series(self, new_categories: Union[dict, list]):
        warnings.warn('Series are getting renamed to Stack/features.'
                      'Please use rename_features instead', DeprecationWarning)
        self.rename_features(new_categories)

    def modify_order_features(self, new_value: int, idx: str):
        """Modify order of the feature. Reorder categories of the link Surfaces, sort surface (reset the basement layer)
        remap the Stack and Surfaces to the corespondent dataframes, sort Geometric objects, update structure and
        reset the flow control objects.

        Args:
            new_value (int): New location
            idx (str): name of the feature to be moved

        Returns:
            :class:`gempy.core.data_modules.stack.Stack`

        """
        self._stack.modify_order_series(new_value, idx)

        self._surfaces.df['series'].cat.reorder_categories(
            np.asarray(self._stack.df.index),
            ordered=False, inplace=True)

        self._surfaces.sort_surfaces()
        self._surfaces.set_basement()

        self.map_geometric_data_df(self._surface_points.df)
        self._surface_points.sort_table()
        self.map_geometric_data_df(self._orientations.df)
        self._orientations.sort_table()

        self._interpolator.set_flow_control()
        self.update_structure()
        return self._stack

    @_setdoc(Series.modify_order_series.__doc__, indent=False)
    def modify_order_series(self, new_value: int, idx: str):
        warnings.warn('Series are getting renamed to Stack/features.'
                      'Please use modify_order_features instead',
                      DeprecationWarning, )
        return self.modify_order_features(new_value, idx)

    def reorder_features(self, new_categories: Iterable[str]):
        """Reorder series. Reorder categories of the link Surfaces, sort surface (reset the basement layer)
        remap the Series and Surfaces to the corespondent dataframes, sort Geometric objects, update structure and
        reset the flow control objects.

        Args:
           new_categories (list): list with all series names in the desired order.

        Returns:
            :class:`gempy.core.data_modules.stack.Stack`
        """
        self._stack.reorder_series(new_categories)
        self._surfaces.df['series'].cat.reorder_categories(
            np.asarray(self._stack.df.index),
            ordered=False, inplace=True)

        self._surfaces.sort_surfaces()
        self._surfaces.set_basement()

        self.map_geometric_data_df(self._surface_points.df)
        self._surface_points.sort_table()
        self.map_geometric_data_df(self._orientations.df)
        self._orientations.sort_table()

        self._interpolator.set_flow_control()
        self.update_structure(update_theano='weights')
        return self._stack

    @_setdoc(reorder_features.__doc__, indent=False)
    def reorder_series(self, new_categories: Iterable[str]):
        warnings.warn('Series are getting renamed to Stack/features.'
                      'Please use reorder_features instead', DeprecationWarning)
        return self.reorder_features(new_categories)

    # endregion

    # region Faults
    def set_fault_object(self):
        """Not implemented"""
        raise NotImplementedError

    @_setdoc([Faults.set_is_fault.__doc__], indent=False)
    def set_is_fault(self, feature_fault: Union[str, list] = None,
                     toggle: bool = False,
                     change_color: bool = True, twofins=False):
        """
        Set a feature to fault and update all dependent objects of the Model.

        Args:

            feature_fault(str, list[str]): Name of the series which are faults
            toggle (bool): if True, passing a name which is already True will set it False.
            twofins (bool): If True, it allows to set several surfaces of a given geological feature to fault.
             This is behaviour is not tested and could have unexpected behaviour.
            change_color (bool): If True faults surfaces get the default fault color (light gray)

        Returns:
            :class:`gempy.core.data_modules.stack.Faults`

        See Also:
            :class:`gempy.core.data_modules.stack.Faults.set_is_fault`

        """
        feature_fault = np.atleast_1d(feature_fault)
        if twofins is False:
            for fault in feature_fault:
                if self._surfaces.df.shape[0] == 0:
                    aux_assert = True
                elif np.sum(self._surfaces.df.groupby('isBasement').get_group(False)[
                                'series'] == fault) < 2:
                    aux_assert = True
                else:
                    aux_assert = False

                assert aux_assert, \
                    'Having more than one fault in a series is generally rather bad. Better go' \
                    ' back to the function map_series_to_surfaces and give each fault its own' \
                    ' series. If you are really sure what you are doing, you can set twofins to' \
                    ' True to suppress this error.'

        self._faults.set_is_fault(feature_fault, toggle=toggle)

        if toggle is True:
            already_fault = self._stack.df.loc[
                                feature_fault, 'BottomRelation'] == 'Fault'
            self._stack.df.loc[
                feature_fault[already_fault], 'BottomRelation'] = 'Erosion'
            self._stack.df.loc[
                feature_fault[~already_fault], 'BottomRelation'] = 'Fault'
        else:
            self._stack.df.loc[feature_fault, 'BottomRelation'] = 'Fault'

        self._additional_data.structure_data.set_number_of_faults()
        self._interpolator.set_theano_shared_relations()
        self._interpolator.set_theano_shared_loop()
        if change_color:
            print(
                'Fault colors changed. If you do not like this behavior, set change_color to False.')
            self._surfaces.colors.make_faults_black(feature_fault)
        self.update_from_series(False, False, False)
        self.update_structure(update_theano='matrices')
        return self._faults

    @_setdoc([Faults.set_is_finite_fault.__doc__], indent=False)
    def set_is_finite_fault(self, series_fault=None, toggle: bool = True):
        """"""
        s = self._faults.set_is_finite_fault(series_fault,
                                             toggle)  # change df in Fault obj
        # change shared theano variable for infinite factor
        self._interpolator.set_theano_shared_is_finite()
        return s

    @_setdoc([Faults.set_fault_relation.__doc__], indent=False)
    def set_fault_relation(self, rel_matrix):
        """"""
        self._faults.set_fault_relation(rel_matrix)

        # Updating
        self._interpolator.set_theano_shared_fault_relation()
        self._interpolator.set_theano_shared_weights()
        return self._faults.faults_relations_df

    # endregion

    # region Surfaces
    def set_surfaces_object(self):
        """
        Not implemented yet. Exchange the surface object of the Model object
        Returns:

        """
        raise NotImplementedError

    @_setdoc(Surfaces.add_surface.__doc__, indent=False)
    def add_surfaces(self, surface_list: Union[str, list], update_df=True):
        self._surfaces.add_surface(surface_list, update_df)
        self._surface_points.df['surface'].cat.add_categories(surface_list,
                                                              inplace=True)
        self._orientations.df['surface'].cat.add_categories(surface_list,
                                                            inplace=True)
        self.update_structure()
        return self._surfaces

    @_setdoc_pro([Surfaces.update_id.__doc__])
    def delete_surfaces(self, indices: Union[str, Iterable[str]], update_id=True,
                        remove_data=True):
        """
        @TODO When implemeted activate geometric data, change remove data to False by default
        Delete a surface and update all related object.

        Args:
            indices (str, list): name or list of names of the series to apply the functionality
            update_id (bool): if true [s0]
            remove_data (bool): if true delete all GeometricData labeled with the given surface.

        Returns:
            :class:`gempy.core.data.Surfaces`

        """
        indices = np.atleast_1d(indices)
        self._surfaces.delete_surface(indices, update_id)

        if indices.dtype == int:
            surfaces_names = self._surfaces.df.loc[indices, 'surface']
        else:
            surfaces_names = indices

        if remove_data:
            self._surface_points.del_surface_points(
                self._surface_points.df[
                    self._surface_points.df.surface.isin(surfaces_names)].index)
            self._orientations.del_orientation(
                self._orientations.df[
                    self._orientations.df.surface.isin(surfaces_names)].index)

        self._surface_points.df['surface'].cat.remove_categories(surfaces_names,
                                                                 inplace=True)
        self._orientations.df['surface'].cat.remove_categories(surfaces_names,
                                                               inplace=True)
        self.map_geometric_data_df(self._surface_points.df)
        self.map_geometric_data_df(self._orientations.df)
        self._surfaces.colors.delete_colors(surfaces_names)

        if remove_data:
            self.update_structure(update_theano='matrices')
            self.update_structure(update_theano='weights')

        return self._surfaces

    @_setdoc(Surfaces.rename_surfaces.__doc__, indent=False)
    def rename_surfaces(self, to_replace: Union[dict], **kwargs):

        self._surfaces.rename_surfaces(to_replace, **kwargs)
        self._surface_points.df['surface'].cat.rename_categories(to_replace,
                                                                 inplace=True)
        self._orientations.df['surface'].cat.rename_categories(to_replace,
                                                               inplace=True)
        return self._surfaces

    @_setdoc(Surfaces.modify_order_surfaces.__doc__, indent=False)
    def modify_order_surfaces(self, new_value: int, idx: int,
                              series_name: str = None):
        """"""

        self._surfaces.modify_order_surfaces(new_value, idx, series_name)

        self.map_geometric_data_df(self._surface_points.df)
        self._surface_points.sort_table()
        self.map_geometric_data_df(self._orientations.df)
        self._orientations.sort_table()

        self.update_structure()
        return self._surfaces

    @_setdoc(Surfaces.add_surfaces_values.__doc__, indent=False)
    def add_surface_values(self, values_array: Iterable,
                           properties_names: Iterable[str] = np.empty(0)):
        self._surfaces.add_surfaces_values(values_array, properties_names)
        self.update_structure(update_theano='matrices')
        return self._surfaces

    @_setdoc(Surfaces.delete_surface_values.__doc__, indent=False)
    def delete_surface_values(self, properties_names: list):
        self.delete_surface_values(properties_names)
        return self._surfaces

    @_setdoc(Surfaces.modify_surface_values.__doc__, indent=False)
    def modify_surface_values(self, idx, properties_names, values):
        self._surfaces.modify_surface_values(idx, properties_names, values)
        return self._surfaces

    @_setdoc(Surfaces.set_surfaces_values.__doc__, indent=False)
    def set_surface_values(self, values_array: Iterable,
                           properties_names: list = np.empty(0)):
        self._surfaces.set_surfaces_values(values_array, properties_names)
        return self._surfaces

    def map_stack_to_surfaces(self,
                              mapping_object: Union[dict, pn.Categorical] = None,
                              set_series=True, sort_geometric_data: bool = True,
                              remove_unused_series=True,
                              twofins=False
                              ):
        """Map series to surfaces and update all related objects accordingly to the following arguments:

        Args:
            mapping_object (dict, :class:`pandas.DataFrame`):

                * dict: keys are the series and values the surfaces belonging to that series

                * pandas.DataFrame: Dataframe with surfaces as index and a column series with the correspondent series
                  name of each surface

            set_series (bool): if True, if mapping object has non existing series they will be created.
            sort_geometric_data (bool): If true geometric data will be sorted accordingly to the new order of the
             series
            remove_unused_series (bool): if true, if an existing series is not assigned with a surface, it will get
             removed from the Series object.
            twofins (bool): If True, it allows to set several surfaces of a given geological feature to fault.
             This is behaviour is not tested and could have unexpected behaviour.

        Returns:
            :class:`gempy.core.data.Surfaces`

        """
        # Add New series to the series df
        if set_series is True:
            if type(mapping_object) is dict:
                series_list = list(mapping_object.keys())
                self._stack.add_series(series_list)
            elif isinstance(mapping_object, pn.Categorical):
                series_list = mapping_object['series'].values
                self._stack.add_series(series_list)
            else:
                raise AttributeError(
                    str(type(mapping_object)) + ' is not the right attribute type.')

        self._surfaces.map_series(mapping_object)

        # Here we remove the series that were not assigned to a surface
        if remove_unused_series is True:
            self._surfaces.df['series'].cat.remove_unused_categories(inplace=True)
            unused_cat = self._stack.df.index[~self._stack.df.index.isin(
                self._surfaces.df['series'].cat.categories)]
            self._stack.delete_series(unused_cat)

        self._stack.reset_order_series()

        self.update_from_surfaces()
        self.update_from_series()

        if sort_geometric_data is True:
            self._surface_points.sort_table()
            self._orientations.sort_table()

        if set_series is True and self._stack.df.index.isin(['Basement']).any():
            aux = self._stack.df.index.drop('Basement').array
            self.reorder_features(np.append(aux, 'Basement'))

        if twofins is False:  # assert if every fault has its own series
            for serie in list(
                    self._faults.df[self._faults.df['isFault'] == True].index):
                assert np.sum(self._surfaces.df['series'] == serie) < 2, \
                    'Having more than one fault in a series is generally rather bad. Better give each ' \
                    'fault its own series. If you are really sure what you are doing, you can set ' \
                    'twofins to True to suppress this error.'

        self.update_structure()

        return self.surfaces

    @_setdoc(map_stack_to_surfaces.__doc__, indent=False)
    def map_series_to_surfaces(self, *args, **kwargs):
        return self.map_stack_to_surfaces(*args, **kwargs)

    # endregion

    # region Surface_points

    def set_surface_points_object(self, surface_points: SurfacePoints,
                                  update_model=True):
        """Not Implemented"""
        raise NotImplementedError

    @staticmethod
    def _check_possible_column_names(table, possible_candidates):
        possible_candidates = np.array(possible_candidates)
        return possible_candidates[np.isin(possible_candidates, table.columns)][0]

    @_setdoc_pro(SurfacePoints.set_surface_points.__doc__)
    def set_surface_points(self, table: pn.DataFrame, **kwargs):
        """Set coordinates and surface columns on the df.

        Args:
            table (pandas.Dataframe): table with surface points data.

        Keyword Args:
            bool add_basement: add a basement surface to the df.

        See Also:
            :class:`gempy.core.data_modules.geometric_data.SurfacePoints`

        """

        try:
            coord_x_name = kwargs.get('coord_x_name') if 'coord_x_name' in kwargs \
                else self._check_possible_column_names(table, ['X', 'x'])
            coord_y_name = kwargs.get('coord_y_name') if 'coord_y_name' in kwargs \
                else self._check_possible_column_names(table, ['Y', 'y'])
        except IndexError:
            raise ValueError('X and Y coordinates columns/values missing')

        try:
            coord_z_name = kwargs.get('coord_z_name') if 'coord_z_name' in kwargs \
                else self._check_possible_column_names(table, ['Z', 'z'])
        except IndexError:
            raise ValueError('Z coordinates column/values missing')
        
        surface_name = kwargs.get('surface_name') if 'surface_name' in kwargs \
            else self._check_possible_column_names(table,
                                                   ['surface', 'Surface', 'surfaces',
                                                    'surfaces', 'formations',
                                                    'formation'])
        update_surfaces = kwargs.get('update_surfaces', True)

        if update_surfaces is True:
            self.add_surfaces(table[surface_name].unique())

        c = np.array(self._surface_points._columns_i_1)
        surface_points_table = table.assign(
            **dict.fromkeys(c[~np.in1d(c, table.columns)], np.nan))
        self._surface_points.set_surface_points(
            surface_points_table[[coord_x_name, coord_y_name, coord_z_name]],
            surface=surface_points_table[surface_name])

        if 'add_basement' in kwargs:
            if kwargs['add_basement'] is True:
                self._surfaces.add_surface(['basement'])
                self.map_stack_to_surfaces({'Basement': 'basement'}, set_series=True)

        self.map_geometric_data_df(self._surface_points.df)
        self._rescaling.rescale_data()
        self.update_structure()
        return self._surface_points

    @_setdoc(Orientations.set_orientations.__doc__, indent=False, position='beg')
    def set_orientations(self, table: pn.DataFrame, **kwargs):
        """ Set coordinates, surface and orientation data.

        If both are passed pole vector has priority over orientation

        Args:
            table (pn.Dataframe): table with surface points data.

        Returns:
            :class:`gempy.core.data_modules.geometric_data.Orientations`

        See Also:
            :class:`gempy.core.data_modules.geometric_data.Orientations`

        """
        g_x_name = kwargs.get('G_x_name', 'G_x')
        g_y_name = kwargs.get('G_y_name', 'G_y')
        g_z_name = kwargs.get('G_z_name', 'G_z')
        azimuth_name = kwargs.get('azimuth_name', 'azimuth')
        dip_name = kwargs.get('dip_name', 'dip')
        polarity_name = kwargs.get('polarity_name', 'polarity')
        update_surfaces = kwargs.get('update_surfaces', False)

        coord_x_name = kwargs.get('coord_x_name') if 'coord_x_name' in kwargs \
            else self._check_possible_column_names(table, ['X', 'x'])
        coord_y_name = kwargs.get('coord_y_name') if 'coord_y_name' in kwargs \
            else self._check_possible_column_names(table, ['Y', 'y'])
        coord_z_name = kwargs.get('coord_z_name') if 'coord_z_name' in kwargs \
            else self._check_possible_column_names(table, ['Z', 'z'])
        surface_name = kwargs.get('surface_name') if 'surface_name' in kwargs \
            else self._check_possible_column_names(table,
                                                   ['surface', 'Surface', 'surfaces',
                                                    'surfaces', 'formations',
                                                    'formation', 'Formation'])

        if update_surfaces is True:
            self.add_surfaces(table[surface_name].unique())

        c = np.array(self._orientations._columns_o_1)
        orientations_read = table.assign(
            **dict.fromkeys(c[~np.in1d(c, table.columns)], np.nan))
        self._orientations.set_orientations(
            coord=orientations_read[[coord_x_name, coord_y_name, coord_z_name]],
            pole_vector=orientations_read[[g_x_name, g_y_name, g_z_name]].values,
            orientation=orientations_read[
                [azimuth_name, dip_name, polarity_name]].values,
            surface=orientations_read[surface_name])

        self.map_geometric_data_df(self._orientations.df)
        self._rescaling.rescale_data()
        self.update_structure()

        return self._orientations

    @_setdoc_pro(ds.recompute_rf)
    @_setdoc(SurfacePoints.add_surface_points.__doc__, indent=False, position='beg')
    @plot_add_surface_points
    def add_surface_points(self, X, Y, Z, surface,
                           idx: Union[int, Iterable[int]] = None,
                           recompute_rescale_factor=False):
        """
        Args:
            X:
            Y:
            Z:
            surface (str):
            idx: Index of the point. If None, next available index will be used
            recompute_rescale_factor (bool): [s0].
        """

        surface = np.atleast_1d(surface)
        idx = self._add_valid_idx_s(idx)
        self._surface_points.add_surface_points(X, Y, Z, surface, idx)

        if recompute_rescale_factor is True or idx < 20:
            # This will rescale all data again
            self._rescaling.rescale_data()
            self._interpolator.set_theano_shared_kriging()
        else:
            # This branch only recompute the added point
            self._rescaling.set_rescaled_surface_points(idx)
        self.update_structure(update_theano='matrices')
        self._interpolator.set_theano_shared_nuggets()

        return self._surface_points, idx

    @_setdoc(SurfacePoints.del_surface_points.__doc__, indent=False, position='beg')
    @plot_delete_surface_points
    def delete_surface_points(self, idx: Union[int, Iterable[int]]):
        self._surface_points.del_surface_points(idx)
        self.update_structure(update_theano='matrices')
        return self._surface_points

    def delete_surface_points_basement(self):
        """Delete surface points belonging to the basement layer if any"""

        basement_name = \
            self._surfaces.df['surface'][self._surfaces.df['isBasement']].values[0]
        select = (self._surface_points.df['surface'] == basement_name)
        self.delete_surface_points(self._surface_points.df.index[select])
        return True

    @_setdoc_pro()
    @plot_move_surface_points
    def modify_surface_points(self, indices: Union[int, list],
                              recompute_rescale_factor=False, **kwargs):
        """Allows modification of the x,y and/or z-coordinates of an interface at specified dataframe index.

        Args:
            indices: [s_idx_sp]
            recompute_rescale_factor: [s_recompute_rf]

        Keyword Args:
                * X: [s_x]

                * Y: [s_y]

                * Z: [s_z]

                * surface: [s_surface_sp]

         Returns:
            :class:`gempy.core.data_modules.geometric_data.SurfacePoints`

         """

        keys = list(kwargs.keys())
        is_surface = np.isin('surface', keys).all()
        if is_surface:
            assert (
                ~self._surfaces.df[self._surfaces.df['isBasement']]['surface'].isin(
                    np.atleast_1d(kwargs['surface']))).any(), \
                'Surface points cannot belong to Basement. Add a new surface.'

        self._surface_points.modify_surface_points(indices, **kwargs)

        if recompute_rescale_factor is True or np.atleast_1d(indices).shape[0] < 20:
            # This will rescale all data again
            self._rescaling.rescale_data()
            self._interpolator.set_theano_shared_kriging()
        else:
            # This branch only recompute the added point
            self._rescaling.set_rescaled_surface_points(indices)

        keys = list(kwargs.keys())
        is_surface = np.isin('surface', keys).all()
        if is_surface == True:
            self.update_structure(update_theano='matrices')

        if 'smooth' in kwargs:
            self._interpolator.set_theano_shared_nuggets()

        return self._surface_points

    # endregion

    # region Orientation
    def set_orientations_object(self, orientations: Orientations, update_model=True):
        raise NotImplementedError

    @_setdoc_pro()
    @plot_add_orientation
    def add_orientations(self, X, Y, Z, surface, pole_vector: Iterable = None,
                         orientation: Iterable = None, idx=None,
                         recompute_rescale_factor=False):
        """Add orientation.

        Args:
            X: [s_x]
            Y: [s_y]
            Z: [s_z]
            surface: [s_surface_sp]
            pole_vector: [s_pole_vector]
            orientation: [s_orientations]
            idx: [s_idx_sp]
            recompute_rescale_factor: [s_recompute_rf]

        Returns:
            :class:`gempy.core.data_modules.geometric_data.Orientations`

        """

        surface = np.atleast_1d(surface)
        idx = self._add_valid_idx_o(idx)

        self._orientations.add_orientation(X, Y, Z, surface, pole_vector=pole_vector,
                                           orientation=orientation, idx=idx)
        if recompute_rescale_factor is True or idx < 5:
            # This will rescale all data again
            self._rescaling.rescale_data()
        else:
            # This branch only recompute the added point
            self._rescaling.set_rescaled_orientations(idx)
        self.update_structure(update_theano='weights')
        self._interpolator.set_theano_shared_nuggets()

        return self._orientations, idx

    @_setdoc(Orientations.del_orientation.__doc__, indent=False, position='beg')
    @plot_delete_orientations
    def delete_orientations(self, idx: Union[list, int]):
        self._orientations.del_orientation(idx)
        self.update_structure(update_theano='weights')
        return self._orientations

    @_setdoc(Orientations.modify_orientations.__doc__, indent=False, position='beg')
    @plot_move_orientations
    def modify_orientations(self, idx: list, **kwargs):

        idx = np.array(idx, ndmin=1)
        keys = list(kwargs.keys())
        is_surface = np.isin('surface', keys).all()
        self._orientations.modify_orientations(idx, **kwargs)
        self._rescaling.set_rescaled_orientations(idx)

        if is_surface:
            self.update_structure(update_theano='weights')
        if 'smooth' in kwargs:
            self._interpolator.set_theano_shared_nuggets()
        return self._orientations

    # endregion

    # region Options
    @_setdoc(Options.modify_options.__doc__, indent=False, position='beg')
    def modify_options(self, attribute, value):
        self._additional_data.options.modify_options(attribute, value)
        warnings.warn(
            'You need to recompile the Theano code to make it the changes in options.')

        return self._additional_data.options

    # endregion

    # region Kriging
    @_setdoc(KrigingParameters.modify_kriging_parameters.__doc__, indent=False,
             position='beg')
    def modify_kriging_parameters(self, attribute, value, **kwargs):
        self._additional_data.kriging_data.modify_kriging_parameters(attribute,
                                                                     value, **kwargs)
        self._interpolator.set_theano_shared_kriging()
        if attribute == 'drift equations':
            self._interpolator.set_initial_results()
            self.update_structure()
        return self._additional_data.kriging_data

    # endregion

    # region rescaling
    @_setdoc(ScalingSystem.modify_rescaling_parameters.__doc__, indent=False,
             position='beg')
    def modify_rescaling_parameters(self, attribute, value):
        self._additional_data.rescaling_data.modify_rescaling_parameters(attribute,
                                                                         value)
        self._additional_data.rescaling_data.rescale_data()
        self._additional_data.update_default_kriging()

        return self._additional_data.rescaling_data

    # endregion

    # ======================================
    # --------------------------------------
    # ======================================

    def set_default_surface_point(self, **kwargs):
        """Set a default surface point if the df is empty. This is necessary for some
        type of functionality such as qgrid.

        Args:
            **kwargs: Same as :func:`gempy.core.data_modules.geometric_data.SurfacePoints.add_surface_points`

        Returns:
            :class:`gempy.core.data_modules.geometric_data.SurfacePoints`

        """
        if self._surface_points.df.shape[0] == 0:
            self.add_surface_points(0.00001, 0.00001, 0.00001,
                                    self._surfaces.df['surface'].iloc[0],
                                    recompute_rescale_factor=True, **kwargs)
        return self._surface_points

    def set_default_orientation(self, **kwargs):
        """Set a default orientation if the df is empty. This is necessary for some type of functionality such as qgrid

         Args:
             **kwargs: Same as :func::class:`gempy.core.data_modules.geometric_data.Orientations.add_orientation`

         Returns:
             :class:`gempy.core.data_modules.geometric_data.Orientations`

         """
        if self._orientations.df.shape[0] == 0:
            # TODO DEBUG: I am not sure that surfaces always has at least one entry. Check it
            self.add_orientations(.00001, .00001, .00001,
                                  self._surfaces.df['surface'].iloc[0],
                                  [0, 0, 1], recompute_rescale_factor=True, **kwargs)

    def set_default_surfaces(self):
        """Set two default surfaces if the df is empty. This is necessary for some type of functionality such as qgrid

         Returns:
             :class:`gempy.core.data.Surfaces`

         """
        if len(self._surfaces.df['surface']) != 0:
            self.delete_surfaces(self._surfaces.df['surface'])

        if self._surfaces.df.shape[0] == 0:
            self.add_surfaces(['surface1', 'surface2'])
        self.update_from_surfaces()
        return self._surfaces

    @_setdoc_pro()
    def set_extent(self, extent: Iterable):
        """
        Set project extent

        Args:
            extent: [s_extent]

        Returns:
            :class:`gempy.core.data.Grid`

        """
        extent = np.atleast_1d(extent)
        self._grid.extent = extent
        self._rescaling.set_rescaled_grid()
        return self._grid

    def update_from_series(self, reorder_series=True, sort_geometric_data=True,
                           update_interpolator=True):
        """Update all objects dependent on series.

        This method is a bit of a legacy and has been substituted by :meth:`rename_series` and :meth:`reorder_series`,
        however is useful if you want to make sure all objects are up to date with the latest changes on series.

        Args:

            reorder_series (bool): if True reorder all pandas categories accordingly to the series.df
            sort_geometric_data (bool): It True sort the geometric data after mapping the new order
            update_interpolator (bool): If True update the theano shared variables dependent on the structure

        Returns:
            True
        """

        if reorder_series is True:
            self._surfaces.df['series'].cat.reorder_categories(
                np.asarray(self._stack.df.index),
                ordered=False, inplace=True)
            self._stack.df.index = self._stack.df.index.reorder_categories(
                self._stack.df.index.array,
                ordered=False)
            self._surfaces.sort_surfaces()
            self.update_from_surfaces(set_categories_from_series=False,
                                      set_categories_from_surfaces=True,
                                      map_surface_points=False,
                                      map_orientations=False,
                                      update_structural_data=False)

        # Update surface is active from series does not work because you can have only a subset of surfaces of a
        # series active
        self._surfaces.df['isFault'] = self._surfaces.df['series'].map(
            self._faults.df['isFault'])
        self._surfaces.set_basement()

        # Add categories from series
        self._surface_points.set_series_categories_from_series(self._stack)
        self._orientations.set_series_categories_from_series(self._stack)

        self._surface_points.map_data_from_series(self._stack, 'order_series')
        self._orientations.map_data_from_series(self._stack, 'order_series')

        if sort_geometric_data is True:
            self._surface_points.sort_table()
            self._orientations.sort_table()

        self._additional_data.update_structure()
        # For the drift equations.
        self._additional_data.update_default_kriging()

        if update_interpolator is True:
            self._interpolator.set_theano_shared_structure(reset_ctrl=True)

        return True

    def update_from_surfaces(self, set_categories_from_series=True,
                             set_categories_from_surfaces=True,
                             map_surface_points=True, map_orientations=True,
                             update_structural_data=True):
        """
        Update all objects dependt on surfaces.

        Args:
            set_categories_from_series (bool): If True update the pandas categories with the Series object
            set_categories_from_surfaces (bool): If True update the pandas categories with the surfaces object
            map_surface_points (bool): If True map the surface points fields with the Surfaces obejct
            map_orientations (bool): If True map the orientations fields with the Surfaces object
            update_structural_data (bool): If true update the Structure with the Surface object

        Returns:
            True
        """
        # Add categories from series
        if set_categories_from_series is True:
            self._surface_points.set_series_categories_from_series(
                self._surfaces.series)
            self._orientations.set_series_categories_from_series(
                self._surfaces.series)

        # Add categories from surfaces
        if set_categories_from_surfaces is True:
            self._surface_points.set_surface_categories_from_surfaces(self._surfaces)
            self._orientations.set_surface_categories_from_surfaces(self._surfaces)

        if map_surface_points is True:
            self._surface_points.map_data_from_surfaces(self._surfaces, 'series')
            self._surface_points.map_data_from_surfaces(self._surfaces, 'id')

        if map_orientations is True:
            self._orientations.map_data_from_surfaces(self._surfaces, 'series')
            self._orientations.map_data_from_surfaces(self._surfaces, 'id')

        if update_structural_data is True:
            self._additional_data.update_structure()

        return True

    # region Theano interface
    @_setdoc(InterpolatorModel.__doc__)
    def set_theano_graph(self, interpolator: InterpolatorModel,
                         update_structure=True, update_kriging=True):
        """ Pass a theano graph of a Interpolator instance other than the Model compose

        Use this method only if you know what are you doing!

        Args:
            interpolator (:class:`InterpolatorModel`): [s0]

        Returns:
             True """
        warnings.warn(
            'This function is going to be deprecated. Use Model.set_theano_function instead',
            DeprecationWarning)
        self._interpolator.theano_graph = interpolator.theano_graph
        self._interpolator.theano_function = interpolator.theano_function
        self.update_additional_data(update_structure=update_structure,
                                    update_kriging=update_kriging)
        self.update_to_interpolator()
        return True

    # @_setdoc(InterpolatorModel.__doc__)
    def set_theano_function(self, interpolator: InterpolatorModel,
                            update_structure=True, update_kriging=True):
        """Pass a theano function and its correspondent graph from an Interpolator
         instance other than the Model compose

        Args:
            interpolator (:class:`gempy.core.interpolator.InterpolatorModel`): interpolator object
             with the compile graph.
            update_kriging (bool): if True update kriging parameters
            update_structure (bool): if Ture update structure

        Returns:
            bool: True

        See Also:
            :class:`gempy.core.interpolator.InterpolatorModel`
        """

        self._interpolator.theano_graph = interpolator.theano_graph
        self._interpolator.theano_function = interpolator.theano_function
        self.update_additional_data(update_structure=update_structure,
                                    update_kriging=update_kriging)
        self.update_to_interpolator()

        return True

    def update_additional_data(self, update_structure=True, update_kriging=True):
        if update_structure is True:
            self.update_structure(update_theano='matrices')

        if update_kriging is True:
            print('Setting kriging parameters to their default values.')
            self._additional_data.update_default_kriging()

        return self._additional_data

    def update_to_interpolator(self, reset=True):
        """Update all shared parameters from the data objects

        Args:
            reset (bool): if True reset the flow control and initialize results arrays

        Returns:
            True
        """
        self._interpolator.set_all_shared_parameters()
        if reset is True:
            self._interpolator.reset_flow_control_initial_results()
        return True

    # endregion

    def map_geometric_data_df(self, d: pn.DataFrame):
        """
        Map a geometric data dataframe from the linked objects (at 07.2019 surfaces and series)

        Args:
            d (pn.DataFrame): Geometric data dataframe to be mapped

        Returns:
            DataFrame
        """
        d['series'] = d['surface'].map(
            self._surfaces.df.set_index('surface')['series'])
        d['id'] = d['surface'].map(
            self._surfaces.df.set_index('surface')['id']).astype(int)
        d['order_series'] = d['series'].map(self._stack.df['order_series']).astype(
            int)
        return d

    def set_surface_order_from_solution(self):
        """
        Order the surfaces respect the last computation. Therefore if you call this method,
        after sorting surface_points without recomputing you may get wrong results.

        Returns:
            Surfaces
        """

        sfai_order = self.solutions.scalar_field_at_surface_points.sum(axis=0)
        # Check if the order has changed
        if not np.array_equal(sfai_order, self._sfai_order_0):
            self._sfai_order_0 = sfai_order
            sel = self._surfaces.df['isActive'] & ~self._surfaces.df['isBasement']
            self._surfaces.df.loc[sel, 'sfai'] = sfai_order
            self._surfaces.df.sort_values(by=['series', 'sfai'], inplace=True,
                                          ascending=False)
            self._surfaces.reset_order_surfaces()
            self._surfaces.sort_surfaces()
            self._surfaces.set_basement()
            self._surface_points.df['id'] = self._surface_points.df['surface'].map(
                self._surfaces.df.set_index('surface')['id']).astype(int)
            self._orientations.df['id'] = self._orientations.df['surface'].map(
                self._surfaces.df.set_index('surface')['id']).astype(int)
            self._surface_points.sort_table()
            self._orientations.sort_table()
            self.update_structure()
        return self._surfaces


def Model(project_name='default_project'):
    """ Container class of all objects that constitute a GemPy model.

      In addition the class provides the methods that act in more than one of this class. Model is a child class of
      :class:`DataMutation` and :class:`MetaData`.

      """
    warnings.warn('This C;ass is going to be deprecated in GemPy 2.3. '
                  'Use Project instead.',
                  DeprecationWarning)
    return Project(project_name)


class Project(ImplicitCoKriging):
    """Container class of all objects that constitute a GemPy model.

    In addition the class provides all the methods you need to construct a geological
    model with :class:`ImplicitCoKriging`.

    See Also:
        :class:`MetaData`, :class:`ImplicitCoKriging`
    """

    def __init__(self, project_name='default_project'):

        self.meta = MetaData(project_name=project_name)
        super().__init__()

    def __repr__(self):
        return self.meta.project_name + ' ' + self.meta.date

    def new_model(self, name_project='default_project'):
        """Reset the model object."""
        self.__init__(name_project)

    def save_model_pickle(self, path=False):
        """
        Short term model storage. Object to a python pickle (serialization of python). Be aware that if the dependencies
        versions used to export and import the pickle differ it may give problems

        Args:
            path (str): path where save the pickle

        Returns:
            True
        """
        # Deleting qi attribute otherwise doesnt allow to pickle
        if hasattr(self, 'qi'):
            self.__delattr__('qi')

        sys.setrecursionlimit(10000)

        if not path:
            path = './' + self.meta.project_name
        import pickle
        with open(path + '.pickle', 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        return True

    @staticmethod
    def load_model_pickle(path):
        """
        Read InputData object from python pickle.

        Args:
           path (str): path where save the pickle

        Returns:
            :class:`gempy.core.model.Model`

        """
        import pickle
        with open(path, 'rb') as f:
            # The protocol version used is detected automatically, so we do not
            # have to specify it.
            model = pickle.load(f)
            return model

    def save_model(self, name=None, path=None, compress=True):
        """
        Save model in new folder. Input data is saved as csv files. Solutions, extent and resolutions are saved as npy.

        Args:
            name (str): name of the newly created folder and the part of the files name
            path (str): path where save the model folder.
            compress (bool): If true create a zip

        Returns:
            True
        """
        if name is None or path is None:
            from gempy.api_modules.io import default_path_and_name
            name, path = default_path_and_name(self, name, path)

        # save dataframes as csv
        self._surface_points.df.to_csv(f'{path}/{name}_surface_points.csv')
        self._surfaces.df.to_csv(f'{path}/{name}_surfaces.csv')
        self._orientations.df.to_csv(f'{path}/{name}_orientations.csv')
        self._stack.df.to_csv(f'{path}/{name}_series.csv')
        self._faults.df.to_csv(f'{path}/{name}_faults.csv')
        self._faults.faults_relations_df.to_csv(
            f'{path}/{name}_faults_relations.csv')
        self._additional_data.kriging_data.df.to_csv(
            f'{path}/{name}_kriging_data.csv')
        self._additional_data.rescaling_data.df.to_csv(
            f'{path}/{name}_rescaling_data.csv')
        self._additional_data.options.df.to_csv(f'{path}/{name}_options.csv')

        # # save resolution and extent as npy
        np.save(f'{path}/{name}_extent.npy', self._grid.regular_grid.extent)
        np.save(f'{path}/{name}_resolution.npy', self._grid.regular_grid.resolution)

        if self._grid.topography is not None:
            self._grid.topography.save(f'{path}/{name}_topography.npy')

        # if compress is True:
        #     shutil.make_archive(name, 'zip', path)
        #     shutil.rmtree(path)
        return True

    def save_solution(self):
        pass

    def read_data(self, source_i=None, source_o=None, add_basement=True, **kwargs):
        """
        Read data from a csv, or directly supplied dataframes

        Args:
            source_i: Path to the data bases of surface_points. Default os.getcwd(), or direct pandas data frame
            source_o: Path to the data bases of orientations. Default os.getcwd(), or direct pandas data frame
            add_basement (bool): if True add a basement surface. This wont be interpolated it just gives the values
            for the volume below the last surface.

        Keyword Args:
            update_surfaces (bool): True

        Returns:
            bool: True

        See Also:

            * :class:`gempy.core.data_modules.geometric_data.SurfacePoints.read_surface_points.`

            * :class:`gempy.core.data_modules.geometric_data.Orientations.read_orientations`
        """
        if 'update_surfaces' not in kwargs:
            kwargs['update_surfaces'] = True
        if 'path_i' in kwargs:
            source_i = kwargs['path_i']
        if 'path_o' in kwargs:
            source_o = kwargs['path_o']

        if isinstance(source_i, pn.DataFrame) or source_i:
            self._surface_points.read_surface_points(source_i, inplace=True,
                                                     **kwargs)
        if isinstance(source_o, pn.DataFrame) or source_o:
            self._orientations.read_orientations(source_o, inplace=True, **kwargs)
        if add_basement is True:
            self._surfaces.add_surface(['basement'])
            self.map_stack_to_surfaces({'Basement': 'basement'}, set_series=True)
        self._rescaling.rescale_data()

        self._additional_data.update_structure()
        self._additional_data.update_default_kriging()
        return True

    @_setdoc_pro()
    def get_data(self, itype='data', verbosity=0, numeric=False):
        """Method to return the data stored in :class:`panda.DataFrame` within a
        :class:`gempy.core.model.Project` data object.

        Args:
            itype: [s_itype]
            numeric(bool): Return only the numerical values of the dataframe. This is much lighter database for storing
                traces
            verbosity (int): Number of properties shown

        Returns:
            pandas.DataFrame: Data Object df.

        """

        if verbosity == 0:
            show_par_f = self._orientations._columns_rend
            show_par_i = self._surface_points._columns_rend
        elif verbosity == 1:
            show_par_f = self._orientations._columns_o_1
            show_par_i = self._surface_points._columns_i_1

        if numeric:
            show_par_f = self._orientations._columns_o_num
            show_par_i = self._surface_points._columns_i_num
            dtype = 'float'

        if itype == 'orientations':
            raw_data = self._orientations.df[show_par_f]  # .astype(dtype)
            # Be sure that the columns are in order when used for operations
            if numeric:
                raw_data = raw_data[
                    ['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z', 'dip', 'azimuth',
                     'polarity']]
        elif itype == 'surface_points' or itype == 'surface points':
            raw_data = self._surface_points.df[show_par_i]  # .astype(dtype)
            # Be sure that the columns are in order when used for operations
            if numeric:
                raw_data = raw_data[
                    ['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z', 'dip', 'azimuth',
                     'polarity']]
        elif itype == 'data':
            raw_data = pn.concat(
                [self._surface_points.df[show_par_i],  # .astype(dtype),
                 self._orientations.df[show_par_f]],  # .astype(dtype)],
                keys=['surface_points', 'orientations'],
                sort=False)
            # Be sure that the columns are in order when used for operations
            if numeric:
                raw_data = raw_data[
                    ['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z', 'dip', 'azimuth',
                     'polarity']]

        elif itype == 'surfaces':
            raw_data = self._surfaces
        elif itype == 'series':
            raw_data = self._stack
        elif itype == 'faults':
            raw_data = self._faults
        elif itype == 'faults_relations_df' or itype == 'faults_relations':
            raw_data = self._faults.faults_relations_df
        elif itype == 'additional data' or itype == 'additional_data':
            raw_data = self._additional_data
        elif itype == 'kriging':
            raw_data = self._additional_data.kriging_data
        else:
            raise AttributeError(
                'itype has to be \'data\', \'additional data\', \'surface_points\', \'orientations\','
                ' \'surfaces\',\'series\', \'faults\' or \'faults_relations_df\'')

        return raw_data

    def get_additional_data(self):
        return self._additional_data.get_additional_data()
