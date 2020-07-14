import sys
import warnings
from typing import Union, Iterable

import numpy as np
import pandas as pn

from gempy.core.data import Surfaces, Grid

from gempy.core.checkers import check_for_nans
from gempy.utils import docstring as ds
from gempy.utils.meta import _setdoc_pro, _setdoc
from gempy.utils.transformations import rescale_surface_points, compute_centers, compute_max_rescaling_factor, compute_rescaling_factor


@_setdoc_pro(Surfaces.__doc__)
class GeometricData(object):
    """
    Parent class of the objects which containing the input parameters: surface_points and orientations. This class
     contain the common methods for both types of data sets.

    Args:
        surfaces (:class:`Surfaces`): [s0]

    Attributes:
        surfaces (:class:`Surfaces`)
        df (:class:`pn.DataFrame`): Pandas DataFrame containing all the properties of each individual data point i.e.
        surface points and orientations
    """

    def __init__(self, surfaces: Surfaces):

        self.surfaces = surfaces
        self.df = pn.DataFrame()

    def __repr__(self):
        c_ = self._columns_rend
        return self.df[c_].to_string()

    def _repr_html_(self):
        c_ = self._columns_rend
        return self.df[c_].to_html()

    def init_dependent_properties(self):
        """Set the defaults values to the columns before gets mapped with the the :class:`Surfaces` attribute. This
        method will get invoked for example when we add a new point."""

        # series
        self.df['series'] = 'Default series'
        self.df['series'] = self.df['series'].astype('category', copy=True)
        #self.df['order_series'] = self.df['order_series'].astype('category', copy=True)

        self.df['series'].cat.set_categories(self.surfaces.df['series'].cat.categories, inplace=True)

        # id
        self.df['id'] = np.nan

        # order_series
        self.df['order_series'] = 1
        return self

    @staticmethod
    @_setdoc(pn.read_csv.__doc__, indent=False)
    def read_data(file_path, **kwargs):
        """"""
        if 'sep' not in kwargs:
            kwargs['sep'] = ','

        table = pn.read_csv(file_path, **kwargs)
        return table

    def sort_table(self):
        """
        First we sort the dataframes by the series age. Then we set a unique number for every surface and resort
        the surfaces. All inplace
        """

        # We order the pandas table by surface (also by series in case something weird happened)
        self.df.sort_values(by=['order_series', 'id'],
                            ascending=True, kind='mergesort',
                            inplace=True)
        return self.df

   # @_setdoc_pro(Series.__doc__)
    def set_series_categories_from_series(self, series):
        """set the series categorical columns with the series index of the passed :class:`Series`

        Args:
            series (:class:`Series`): [s0]
        """
        self.df['series'].cat.set_categories(series.df.index, inplace=True)
        return True

    def update_series_category(self):
        """Update the series categorical columns with the series categories of the :class:`Surfaces` attribute."""
        self.df['series'].cat.set_categories(self.surfaces.df['series'].cat.categories, inplace=True)

        return True

    @_setdoc_pro(Surfaces.__doc__)
    def set_surface_categories_from_surfaces(self, surfaces: Surfaces):
        """set the series categorical columns with the series index of the passed :class:`Series`.

        Args:
            surfaces (:class:`Surfaces`): [s0]

        """

        self.df['surface'].cat.set_categories(surfaces.df['surface'], inplace=True)
        return True

   # @_setdoc_pro(Series.__doc__)
    def map_data_from_series(self, series, attribute: str, idx=None):
        """
        Map columns from the :class:`Series` data frame to a :class:`GeometricData` data frame.

        Args:
            series (:class:`Series`): [s0]
            attribute (str): column to be mapped from the :class:`Series` to the :class:`GeometricData`.
            idx (Optional[int, list[int]): If passed, list of indices of the :class:`GeometricData` that will be mapped.

        Returns:
            :class:GeometricData
        """
        if idx is None:
            idx = self.df.index

        idx = np.atleast_1d(idx)
        if attribute in ['id', 'order_series']:
            self.df.loc[idx, attribute] = self.df['series'].map(series.df[attribute]).astype(int)

        else:
            self.df.loc[idx, attribute] = self.df['series'].map(series.df[attribute])

        if type(self.df['order_series'].dtype) is pn.CategoricalDtype:

            self.df['order_series'].cat.remove_unused_categories(inplace=True)
        return self

    @_setdoc_pro(Surfaces.__doc__)
    def map_data_from_surfaces(self, surfaces, attribute: str, idx=None):
        """
        Map columns from the :class:`Series` data frame to a :class:`GeometricData` data frame.
        Properties of surfaces: series, id, values.

        Args:
            surfaces (:class:`Surfaces`): [s0]
            attribute (str): column to be mapped from the :class:`Series` to the :class:`GeometricData`.
            idx (Optional[int, list[int]): If passed, list of indices of the :class:`GeometricData` that will be mapped.

        Returns:
            :class:GeometricData
        """

        if idx is None:
            idx = self.df.index
        idx = np.atleast_1d(idx)
        if attribute is 'series':
            if surfaces.df.loc[~surfaces.df['isBasement']]['series'].isna().sum() != 0:
                raise AttributeError('Surfaces does not have the correspondent series assigned. See'
                                     'Surfaces.map_series_from_series.')
            self.df.loc[idx, attribute] = self.df.loc[idx, 'surface'].map(surfaces.df.set_index('surface')[attribute])

        elif attribute in ['id', 'order_series']:
            self.df.loc[idx, attribute] = (self.df.loc[idx, 'surface'].map(surfaces.df.set_index('surface')[attribute])).astype(int)
        else:

            self.df.loc[idx, attribute] = self.df.loc[idx, 'surface'].map(surfaces.df.set_index('surface')[attribute])

    # def map_data_from_faults(self, faults, idx=None):
    #     """
    #     Method to map a df object into the data object on surfaces. Either if the surface is fault or not
    #     Args:
    #         faults (Faults):
    #
    #     Returns:
    #         pandas.core.frame.DataFrame: Data frame with the raw data
    #
    #     """
    #     if idx is None:
    #         idx = self.df.index
    #     idx = np.atleast_1d(idx)
    #     if any(self.df['series'].isna()):
    #         warnings.warn('Some points do not have series/fault')
    #
    #     self.df.loc[idx, 'isFault'] = self.df.loc[[idx], 'series'].map(faults.df['isFault'])


@_setdoc_pro([Surfaces.__doc__, ds.coord, ds.surface_sp])
class SurfacePoints(GeometricData):
    """
    Data child with specific methods to manipulate interface data. It is initialize without arguments to give
    flexibility to the origin of the data.

    Args:
        surfaces (:class:`Surfaces`): [s0]
        coord (np.ndarray): [s1]
        surface (list[str]): [s2]


    Attributes:
          df (:class:`pn.core.frame.DataFrames`): Pandas data frame containing the necessary information respect
          the surface points of the model
    """

    def __init__(self, surfaces: Surfaces, coord=None, surface=None):

        super().__init__(surfaces)
        self._columns_i_all = ['X', 'Y', 'Z', 'surface', 'series', 'X_std', 'Y_std', 'Z_std',
                               'order_series', 'surface_number']

        self._columns_i_1 = ['X', 'Y', 'Z', 'X_r', 'Y_r', 'Z_r', 'surface', 'series', 'id',
                             'order_series', 'isFault', 'smooth']

        self._columns_rep = ['X', 'Y', 'Z', 'surface', 'series']
        self._columns_i_num = ['X', 'Y', 'Z', 'X_r', 'Y_r', 'Z_r']
        self._columns_rend = ['X', 'Y', 'Z', 'smooth', 'surface']

        if (np.array(sys.version_info[:2]) <= np.array([3, 6])).all():
            self.df: pn.DataFrame

        self.set_surface_points(coord, surface)

    @_setdoc_pro([ds.coord, ds.surface_sp])
    def set_surface_points(self, coord: np.ndarray = None, surface: list = None):
        """
        Set coordinates and surface columns on the df.

        Args:
            coord (np.ndarray): [s0]
            surface (list[str]): [s1]

        Returns:
            :class:`SurfacePoints`
        """
        self.df = pn.DataFrame(columns=['X', 'Y', 'Z', 'X_r', 'Y_r', 'Z_r', 'surface'], dtype=float)

        if coord is not None and surface is not None:
            self.df[['X', 'Y', 'Z']] = pn.DataFrame(coord)
            self.df['surface'] = surface

        self.df['surface'] = self.df['surface'].astype('category', copy=True)
        self.df['surface'].cat.set_categories(self.surfaces.df['surface'].values, inplace=True)

        # Choose types
        self.init_dependent_properties()

        # Add nugget columns
        self.df['smooth'] = 2e-6

        assert ~self.df['surface'].isna().any(), 'Some of the surface passed does not exist in the Formation' \
                                                 'object. %s' % self.df['surface'][self.df['surface'].isna()]

        return self

    @_setdoc_pro([ds.x, ds.y, ds.z, ds.surface_sp, ds.idx_sp])
    def add_surface_points(self, x: Union[float, np.ndarray], y: Union[float, np.ndarray], z: Union[float, np.ndarray],
                           surface: Union[list, np.ndarray], idx: Union[int, list, np.ndarray] = None):
        """
        Add surface points.

        Args:
            x (float, np.ndarray): [s0]
            y (float, np.ndarray): [s1]
            z (float, np.ndarray): [s2]
            surface (list[str]): [s3]
            idx (Optional[int, list[int]): [s4]

        Returns:
           :class:`gempy.core.data_modules.geometric_data.SurfacePoints`

        """
        max_idx = self.df.index.max()

        if idx is None:
            idx = max_idx
            if idx is np.nan:
                idx = 0
            else:
                idx += 1

        if max_idx is not np.nan:
            self.df.loc[idx] = self.df.loc[max_idx]

        coord_array = np.array([x, y, z])
        assert coord_array.ndim == 1, 'Adding an interface only works one by one.'

        try:
            if self.surfaces.df.groupby('isBasement').get_group(True)['surface'].isin(surface).any():
                warnings.warn('Surface Points for the basement will not be used. Maybe you are missing an extra'
                              'layer at the bottom of the pile.')

            self.df.loc[idx, ['X', 'Y', 'Z']] = coord_array.astype('float64')
            self.df.loc[idx, 'surface'] = surface
        # ToDO test this
        except ValueError as error:
            self.del_surface_points(idx)
            print('The surface passed does not exist in the pandas categories. This may imply that'
                  'does not exist in the surface object either.')
            raise ValueError(error)

        self.df.loc[idx, ['smooth']] = 1e-6

        self.df['surface'] = self.df['surface'].astype('category', copy=True)
        self.df['surface'].cat.set_categories(self.surfaces.df['surface'].values, inplace=True)

        self.df['series'] = self.df['series'].astype('category', copy=True)
        self.df['series'].cat.set_categories(self.surfaces.df['series'].cat.categories, inplace=True)

        self.map_data_from_surfaces(self.surfaces, 'series', idx=idx)
        self.map_data_from_surfaces(self.surfaces, 'id', idx=idx)
        self.map_data_from_series(self.surfaces.series, 'order_series', idx=idx)

        self.sort_table()
        return self, idx

    @_setdoc_pro([ds.idx_sp])
    def del_surface_points(self, idx: Union[int, list, np.ndarray]):
        """Delete surface points.

        Args:
            idx (int, list[int]): [s0]

        Returns:
            :class:`gempy.core.data_modules.geometric_data.SurfacePoints`

        """
        self.df.drop(idx, inplace=True)
        return self

    @_setdoc_pro([ds.idx_sp, ds.x, ds.y, ds.z, ds.surface_sp])
    def modify_surface_points(self, idx: Union[int, list, np.ndarray], **kwargs):
        """Allows modification of the x,y and/or z-coordinates of an interface at specified dataframe index.

         Args:
             idx (int, list, np.ndarray): [s0]
             **kwargs:
                * X: [s1]
                * Y: [s2]
                * Z: [s3]
                * surface: [s4]

         Returns:
            :class:`gempy.core.data_modules.geometric_data.SurfacePoints`

         """
        idx = np.array(idx, ndmin=1)
        try:
            surface_names = kwargs.pop('surface')
            self.df.loc[idx, ['surface']] = surface_names
            self.map_data_from_surfaces(self.surfaces, 'series', idx=idx)
            self.map_data_from_surfaces(self.surfaces, 'id', idx=idx)
            self.map_data_from_series(self.surfaces.series, 'order_series', idx=idx)
            self.sort_table()
        except KeyError:
            pass

        # keys = list(kwargs.keys())
    #    is_surface = np.isin('surface', keys).all()

        # Check idx exist in the df
        assert np.isin(np.atleast_1d(idx), self.df.index).all(), 'Indices must exist in the dataframe to be modified.'

        # Check the properties are valid
        assert np.isin(list(kwargs.keys()), ['X', 'Y', 'Z', 'surface', 'smooth']).all(),\
            'Properties must be one or more of the following: \'X\', \'Y\', \'Z\', ' '\'surface\''
        # stack properties values
        values = np.array(list(kwargs.values()))

        # If we pass multiple index we need to transpose the numpy array
        if type(idx) is list or type(idx) is np.ndarray:
            values = values.T

        # Selecting the properties passed to be modified
        self.df.loc[idx, list(kwargs.keys())] = values

        return self

    @_setdoc_pro([ds.file_path, ds.debug, ds.inplace])
    def read_surface_points(self, table_source, debug=False, inplace=False,
                            kwargs_pandas: dict = None, **kwargs, ):
        """
        Read tabular using pandas tools and if inplace set it properly to the surface points object.

        Parameters:
            table_source (str, path object, file-like object or direct pandas data frame): [s0]
            debug (bool): [s1]
            inplace (bool): [s2]
            kwargs_pandas: kwargs for the panda function :func:`pn.read_csv`
            **kwargs:
                * update_surfaces (bool): If True add to the linked `Surfaces` object unique surface names read on
                  the csv file
                * coord_x_name (str): Name of the header on the csv for this attribute, e.g for coord_x. Default X
                * coord_y_name (str): Name of the header on the csv for this attribute. Default Y.
                * coord_z_name (str): Name of the header on the csv for this attribute. Default Z.
                * surface_name (str): Name of the header on the csv for this attribute. Default formation

        Returns:

        See Also:
            :meth:`GeometricData.read_data`
        """
        # TODO read by default either formation or surface

        if 'sep' not in kwargs:
            kwargs['sep'] = ','

        coord_x_name = kwargs.get('coord_x_name', "X")
        coord_y_name = kwargs.get('coord_y_name', "Y")
        coord_z_name = kwargs.get('coord_z_name', "Z")
        surface_name = kwargs.get('surface_name', "formation")

        if kwargs_pandas is None:
            kwargs_pandas = {}

        if 'sep' not in kwargs_pandas:
            kwargs_pandas['sep'] = ','

        if isinstance(table_source, pn.DataFrame):
            table = table_source
        else:
            table = pn.read_csv(table_source, **kwargs_pandas)

        if 'update_surfaces' in kwargs:
            if kwargs['update_surfaces'] is True:
                self.surfaces.add_surface(table[surface_name].unique())

        if debug is True:
            print('Debugging activated. Changes won\'t be saved.')
            return table
        else:
            assert {coord_x_name, coord_y_name, coord_z_name, surface_name}.issubset(table.columns), \
                "One or more columns do not match with the expected values " + str(table.columns)

            if inplace:
                c = np.array(self._columns_i_1)
                surface_points_read = table.assign(**dict.fromkeys(c[~np.in1d(c, table.columns)], np.nan))
                self.set_surface_points(surface_points_read[[coord_x_name, coord_y_name, coord_z_name]],
                                        surface=surface_points_read[surface_name])
            else:
                return table

    def set_default_surface_points(self):
        """
        Set a default point at the middle of the extent area to be able to start making the model
        """
        if self.df.shape[0] == 0:
            self.add_surface_points(0.00001, 0.00001, 0.00001, self.surfaces.df['surface'].iloc[0])
        return True

    def update_annotations(self):
        """
        Add a column in the Dataframes with latex names for each input_data paramenter.

        Returns:
            :class:`SurfacePoints`
        """
        point_num = self.df.groupby('id').cumcount()
        point_l = [r'${\bf{x}}_{\alpha \,{\bf{' + str(f) + '}},' + str(p) + '}$'
                   for p, f in zip(point_num, self.df['id'])]

        self.df['annotations'] = point_l
        return self


@_setdoc_pro([Surfaces.__doc__, ds.coord_ori, ds.surface_sp, ds.pole_vector, ds.orientations])
class Orientations(GeometricData):
    """
    Data child with specific methods to manipulate orientation data. It is initialize without arguments to give
    flexibility to the origin of the data.

    Args:
        surfaces (:class:`Surfaces`): [s0]
        coord (np.ndarray): [s1]
        pole_vector (np.ndarray): [s3]
        orientation (np.ndarray): [s4]
        surface (list[str]): [s2]
    Attributes:
        df (:class:`pn.core.frame.DataFrames`): Pandas data frame containing the necessary information respect
         the orientations of the model
    """

    def __init__(self, surfaces: Surfaces, coord=None, pole_vector=None, orientation=None, surface=None):
        super().__init__(surfaces)
        self._columns_o_all = ['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z', 'dip', 'azimuth', 'polarity',
                               'surface', 'series', 'id', 'order_series', 'surface_number']
        self._columns_o_1 = ['X', 'Y', 'Z', 'X_r', 'Y_r', 'Z_r', 'G_x', 'G_y', 'G_z', 'dip', 'azimuth', 'polarity',
                             'surface', 'series', 'id', 'order_series', 'isFault']
        self._columns_o_num = ['X', 'Y', 'Z', 'X_r', 'Y_r', 'Z_r', 'G_x', 'G_y', 'G_z', 'dip', 'azimuth', 'polarity']
        self._columns_rend = ['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z', 'smooth', 'surface']

        if (np.array(sys.version_info[:2]) <= np.array([3, 6])).all():
            self.df: pn.DataFrame

        self.set_orientations(coord, pole_vector, orientation, surface)

    @_setdoc_pro([ds.coord_ori, ds.surface_sp, ds.pole_vector, ds.orientations])
    def set_orientations(self, coord: np.ndarray = None, pole_vector: np.ndarray = None,
                         orientation: np.ndarray = None, surface: list = None):
        """
        Set coordinates, surface and orientation data.

        If both are passed pole vector has priority over orientation

        Args:
            coord (np.ndarray): [s0]
            pole_vector (np.ndarray): [s2]
            orientation (np.ndarray): [s3]
            surface (list[str]): [s1]

        Returns:

        """
        self.df = pn.DataFrame(columns=['X', 'Y', 'Z', 'X_r', 'Y_r', 'Z_r', 'G_x', 'G_y', 'G_z', 'dip',
                                        'azimuth', 'polarity', 'surface'], dtype=float)

        self.df['surface'] = self.df['surface'].astype('category', copy=True)
        self.df['surface'].cat.set_categories(self.surfaces.df['surface'].values, inplace=True)

        pole_vector = check_for_nans(pole_vector)
        orientation = check_for_nans(orientation)

        if coord is not None and ((pole_vector is not None) or (orientation is not None)) and surface is not None:

            self.df[['X', 'Y', 'Z']] = pn.DataFrame(coord)
            self.df['surface'] = surface
            if pole_vector is not None:
                self.df['G_x'] = pole_vector[:, 0]
                self.df['G_y'] = pole_vector[:, 1]
                self.df['G_z'] = pole_vector[:, 2]
                self.calculate_orientations()

                if orientation is not None:
                    warnings.warn('If pole_vector and orientation are passed pole_vector is used/')
            else:
                if orientation is not None:
                    self.df['azimuth'] = orientation[:, 0]
                    self.df['dip'] = orientation[:, 1]
                    self.df['polarity'] = orientation[:, 2]
                    self.calculate_gradient()
                else:
                    raise AttributeError('At least pole_vector or orientation should have been passed to reach'
                                         'this point. Check previous condition')

        self.df['surface'] = self.df['surface'].astype('category', copy=True)
        self.df['surface'].cat.set_categories(self.surfaces.df['surface'].values, inplace=True)

        self.init_dependent_properties()

        # Add nugget effect
        self.df['smooth'] = 0.01
        assert ~self.df['surface'].isna().any(), 'Some of the surface passed does not exist in the Formation' \
                                                 'object. %s' % self.df['surface'][self.df['surface'].isna()]

    @_setdoc_pro([ds.x, ds.y, ds.z, ds.surface_sp, ds.pole_vector, ds.orientations, ds.idx_sp])
    def add_orientation(self, x, y, z, surface, pole_vector: Union[list, tuple, np.ndarray] = None,
                        orientation: Union[list, np.ndarray] = None, idx=None):
        """
        Add orientation.

        Args:
            x (float, np.ndarray): [s0]
            y (float, np.ndarray): [s1]
            z (float, np.ndarray): [s2]
            surface (list[str], str): [s3]
            pole_vector (np.ndarray): [s4]
            orientation (np.ndarray): [s5]
            idx (Optional[int, list[int]): [s6]

        Returns:
            Orientations
        """
        if pole_vector is None and orientation is None:
            raise AttributeError('Either pole_vector or orientation must have a value. If both are passed pole_vector'
                                 'has preference')

        max_idx = self.df.index.max()

        if idx is None:
            idx = max_idx
            if idx is np.nan:
                idx = 0
            else:
                idx += 1

        if max_idx is not np.nan:
            self.df.loc[idx] = self.df.loc[max_idx]

        if pole_vector is not None:
            self.df.loc[idx, ['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z']] = np.array([x, y, z, *pole_vector], dtype=float)
            self.df.loc[idx, 'surface'] = surface

            self.calculate_orientations(idx)

            if orientation is not None:
                warnings.warn('If pole_vector and orientation are passed pole_vector is used/')
        else:
            if orientation is not None:
                self.df.loc[idx, ['X', 'Y', 'Z', ]] = np.array([x, y, z], dtype=float)
                self.df.loc[idx, ['azimuth', 'dip', 'polarity']] = np.array(orientation, dtype=float)
                self.df.loc[idx, 'surface'] = surface

                self.calculate_gradient(idx)
            else:
                raise AttributeError('At least pole_vector or orientation should have been passed to reach'
                                     'this point. Check previous condition')
        self.df.loc[idx, ['smooth']] = 0.01
        self.df['surface'] = self.df['surface'].astype('category', copy=True)
        self.df['surface'].cat.set_categories(self.surfaces.df['surface'].values, inplace=True)

        self.df['series'] = self.df['series'].astype('category', copy=True)
        self.df['series'].cat.set_categories(self.surfaces.df['series'].cat.categories, inplace=True)

        self.map_data_from_surfaces(self.surfaces, 'series', idx=idx)
        self.map_data_from_surfaces(self.surfaces, 'id', idx=idx)
        self.map_data_from_series(self.surfaces.series, 'order_series', idx=idx)

        self.sort_table()
        return self

    @_setdoc_pro()
    def del_orientation(self, idx):
        """Delete orientation

        Args:
            idx: [s_idx_sp]

        Returns:
            :class:`gempy.core.data_modules.geometric_data.Orientations`

        """
        self.df.drop(idx, inplace=True)

    @_setdoc_pro([ds.idx_sp, ds.surface_sp])
    def modify_orientations(self, idx, **kwargs):
        """Allows modification of any of an orientation column at a given index.

        Args:
            idx (int, list[int]): [s0]

        Keyword Args:
                * X
                * Y
                * Z
                * G_x
                * G_y
                * G_z
                * dip
                * azimuth
                * polarity
                * surface (str): [s1]

         Returns:
            :class:`gempy.core.data_modules.geometric_data.Orientations`

         """

        idx = np.array(idx, ndmin=1)
        try:
            surface_names = kwargs.pop('surface')
            self.df.loc[idx, ['surface']] = surface_names
            self.map_data_from_surfaces(self.surfaces, 'series', idx=idx)
            self.map_data_from_surfaces(self.surfaces, 'id', idx=idx)
            self.map_data_from_series(self.surfaces.series, 'order_series', idx=idx)
            self.sort_table()
        except KeyError:
            pass

        # TODO Dep
        keys = list(kwargs.keys())

        # Check idx exist in the df
        assert np.isin(np.atleast_1d(idx), self.df.index).all(), 'Indices must exist in the dataframe to be modified.'

        # Check the properties are valid
        assert np.isin(list(kwargs.keys()), ['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z', 'dip',
                                             'azimuth', 'polarity', 'surface', 'smooth']).all(),\
            'Properties must be one or more of the following: \'X\', \'Y\', \'Z\', \'G_x\', \'G_y\', \'G_z\', \'dip,\''\
            '\'azimuth\', \'polarity\', \'surface\''

        # stack properties values
        values = np.atleast_1d(list(kwargs.values()))

        # If we pass multiple index we need to transpose the numpy array
        if type(idx) is list or type(idx) is np.ndarray:
            values = values.T

        # Selecting the properties passed to be modified
        self.df.loc[idx, list(kwargs.keys())] = values.astype('float64')

        if np.isin(list(kwargs.keys()), ['G_x', 'G_y', 'G_z']).any():
            self.calculate_orientations(idx)
        else:
            if np.isin(list(kwargs.keys()), ['azimuth', 'dip', 'polarity']).any():
                self.calculate_gradient(idx)
        return self

    def calculate_gradient(self, idx=None):
        """
        Calculate the gradient vector of module 1 given dip and azimuth to be able to plot the orientations
        """
        if idx is None:
            self.df['G_x'] = np.sin(np.deg2rad(self.df["dip"].astype('float'))) * \
                np.sin(np.deg2rad(self.df["azimuth"].astype('float'))) * \
                self.df["polarity"].astype('float') + 1e-12
            self.df['G_y'] = np.sin(np.deg2rad(self.df["dip"].astype('float'))) * \
                np.cos(np.deg2rad(self.df["azimuth"].astype('float'))) * \
                self.df["polarity"].astype('float') + 1e-12
            self.df['G_z'] = np.cos(np.deg2rad(self.df["dip"].astype('float'))) * \
                self.df["polarity"].astype('float') + 1e-12
        else:
            self.df.loc[idx, 'G_x'] = np.sin(np.deg2rad(self.df.loc[idx, "dip"].astype('float'))) * \
                                      np.sin(np.deg2rad(self.df.loc[idx, "azimuth"].astype('float'))) * \
                                      self.df.loc[idx, "polarity"].astype('float') + 1e-12
            self.df.loc[idx, 'G_y'] = np.sin(np.deg2rad(self.df.loc[idx, "dip"].astype('float'))) * \
                np.cos(np.deg2rad(self.df.loc[idx, "azimuth"].astype('float'))) * \
                self.df.loc[idx, "polarity"].astype('float') + 1e-12
            self.df.loc[idx, 'G_z'] = np.cos(np.deg2rad(self.df.loc[idx, "dip"].astype('float'))) * \
                self.df.loc[idx, "polarity"].astype('float') + 1e-12
        return True

    def calculate_orientations(self, idx=None):
        """
        Calculate and update the orientation data (azimuth and dip) from gradients in the data frame.

        Authors: Elisa Heim, Miguel de la Varga
        """
        if idx is None:
            self.df['polarity'] = 1
            self.df["dip"] = np.rad2deg(np.nan_to_num(np.arccos(self.df["G_z"] / self.df["polarity"])))

            self.df["azimuth"] = np.rad2deg(np.nan_to_num(np.arctan2(self.df["G_x"] / self.df["polarity"],
                                                                     self.df["G_y"] / self.df["polarity"])))
            self.df["azimuth"][self.df["azimuth"] < 0] += 360  # shift values from [-pi, 0] to [pi,2*pi]
            self.df["azimuth"][self.df["dip"] < 0.001] = 0  # because if dip is zero azimuth is undefined

        else:

            self.df.loc[idx, 'polarity'] = 1
            self.df.loc[idx, "dip"] = np.rad2deg(np.nan_to_num(np.arccos(self.df.loc[idx, "G_z"] /
                                                                         self.df.loc[idx, "polarity"])))

            self.df.loc[idx, "azimuth"] = np.rad2deg(np.nan_to_num(
                np.arctan2(self.df.loc[idx, "G_x"] / self.df.loc[idx, "polarity"],
                           self.df.loc[idx, "G_y"] / self.df.loc[idx, "polarity"])))

            self.df["azimuth"][self.df["azimuth"] < 0] += 360  # shift values from [-pi, 0] to [pi,2*pi]
            self.df["azimuth"][self.df["dip"] < 0.001] = 0  # because if dip is zero azimuth is undefined

        return True

    @_setdoc_pro([SurfacePoints.__doc__])
    def create_orientation_from_surface_points(self, surface_points: SurfacePoints, indices):
        # TODO test!!!!
        """
        Create and set orientations from at least 3 points categories_df

        Args:
            surface_points (:class:`SurfacePoints`): [s0]
            indices (list[int]): indices of the surface point used to generate the orientation. At least
             3 independent points will need to be passed.
        """
        selected_points = surface_points.df[['X', 'Y', 'Z']].loc[indices].values.T

        center, normal = self.plane_fit(selected_points)
        orientation = self.get_orientation(normal)

        return np.array([*center, *orientation, *normal])

    def set_default_orientation(self):
        """
        Set a default point at the middle of the extent area to be able to start making the model
        """
        if self.df.shape[0] == 0:
            self.add_orientation(.00001, .00001, .00001,
                                 self.surfaces.df['surface'].iloc[0],
                                 [0, 0, 1],
                                 )

    @_setdoc_pro([ds.file_path, ds.debug, ds.inplace])
    def read_orientations(self, table_source, debug=False, inplace=True, kwargs_pandas: dict = None, **kwargs):
        """
        Read tabular using pandas tools and if inplace set it properly to the surface points object.

        Args:
            table_source (str, path object, file-like object, or direct data frame): [s0]
            debug (bool): [s1]
            inplace (bool): [s2]
            kwargs_pandas: kwargs for the panda function :func:`pn.read_csv`
            **kwargs:
                * update_surfaces (bool): If True add to the linked `Surfaces` object unique surface names read on
                  the csv file
                * coord_x_name (str): Name of the header on the csv for this attribute, e.g for coord_x. Default X
                * coord_y_name (str): Name of the header on the csv for this attribute. Default Y
                * coord_z_name (str): Name of the header on the csv for this attribute. Default Z
                * coord_x_name (str): Name of the header on the csv for this attribute. Default G_x
                * coord_y_name (str): Name of the header on the csv for this attribute. Default G_y
                * coord_z_name (str): Name of the header on the csv for this attribute. Default G_z
                * azimuth_name (str): Name of the header on the csv for this attribute. Default azimuth
                * dip_name     (str): Name of the header on the csv for this attribute. Default dip
                * polarity_name (str): Name of the header on the csv for this attribute. Default polarity
                * surface_name (str): Name of the header on the csv for this attribute. Default formation


        Returns:

        See Also:
            :meth:`GeometricData.read_data`
        """
        coord_x_name = kwargs.get('coord_x_name', "X")
        coord_y_name = kwargs.get('coord_y_name', "Y")
        coord_z_name = kwargs.get('coord_z_name', "Z")
        g_x_name = kwargs.get('G_x_name', 'G_x')
        g_y_name = kwargs.get('G_y_name', 'G_y')
        g_z_name = kwargs.get('G_z_name', 'G_z')
        azimuth_name = kwargs.get('azimuth_name', 'azimuth')
        dip_name = kwargs.get('dip_name', 'dip')
        polarity_name = kwargs.get('polarity_name', 'polarity')
        surface_name = kwargs.get('surface_name', "formation")

        if kwargs_pandas is None:
            kwargs_pandas = {}

        if 'sep' not in kwargs_pandas:
            kwargs_pandas['sep'] = ','

        if isinstance(table_source, pn.DataFrame):
            table = table_source
        else:
            table = pn.read_csv(table_source, **kwargs_pandas)

        if 'update_surfaces' in kwargs:
            if kwargs['update_surfaces'] is True:
                self.surfaces.add_surface(table[surface_name].unique())

        if debug is True:
            print('Debugging activated. Changes won\'t be saved.')
            return table

        else:
            assert np.logical_or({coord_x_name, coord_y_name, coord_z_name, dip_name, azimuth_name,
                    polarity_name, surface_name}.issubset(table.columns),
                 {coord_x_name, coord_y_name, coord_z_name, g_x_name, g_y_name, g_z_name,
                  polarity_name, surface_name}.issubset(table.columns)), \
                "One or more columns do not match with the expected values, which are: \n" +\
                "- the locations of the measurement points '{}','{}' and '{}' \n".format(coord_x_name,coord_y_name,
                                                                                         coord_z_name)+ \
                "- EITHER '{}' (trend direction indicated by an angle between 0 and 360 with North at 0 AND " \
                "'{}' (inclination angle, measured from horizontal plane downwards, between 0 and 90 degrees) \n".format(
                azimuth_name, dip_name) +\
                "- OR the pole vectors of the orientation in a cartesian system '{}','{}' and '{}' \n".format(g_x_name,
                                                                                                              g_y_name,
                                                                                                              g_z_name)+\
                "- the '{}' of the orientation, can be normal (1) or reversed (-1) \n".format(polarity_name)+\
                "- the name of the surface: '{}'\n".format(surface_name)+\
                "Your headers are "+str(list(table.columns))

            if inplace:
                # self.categories_df[table.columns] = table
                c = np.array(self._columns_o_1)
                orientations_read = table.assign(**dict.fromkeys(c[~np.in1d(c, table.columns)], np.nan))
                self.set_orientations(coord=orientations_read[[coord_x_name, coord_y_name, coord_z_name]],
                                      pole_vector=orientations_read[[g_x_name, g_y_name, g_z_name]].values,
                                      orientation=orientations_read[[azimuth_name, dip_name, polarity_name]].values,
                                      surface=orientations_read[surface_name])
            else:
                return table

    def update_annotations(self):
        """
        Add a column in the Dataframes with latex names for each input_data paramenter.

        Returns:

        """
        orientation_num = self.df.groupby('id').cumcount()
        foli_l = [r'${\bf{x}}_{\beta \,{\bf{' + str(f) + '}},' + str(p) + '}$'
                  for p, f in zip(orientation_num, self.df['id'])]

        self.df['annotations'] = foli_l
        return self

    @staticmethod
    def get_orientation(normal):
        """Get orientation (dip, azimuth, polarity ) for points in all point set"""

        # calculate dip
        dip = np.arccos(normal[2]) / np.pi * 180.

        # calculate dip direction
        # +/+
        if normal[0] >= 0 and normal[1] > 0:
            dip_direction = np.arctan(normal[0] / normal[1]) / np.pi * 180.
        # border cases where arctan not defined:
        elif normal[0] > 0 and normal[1] == 0:
            dip_direction = 90
        elif normal[0] < 0 and normal[1] == 0:
            dip_direction = 270
        # +-/-
        elif normal[1] < 0:
            dip_direction = 180 + np.arctan(normal[0] / normal[1]) / np.pi * 180.
        # -/-
        elif normal[0] < 0 >= normal[1]:
            dip_direction = 360 + np.arctan(normal[0] / normal[1]) / np.pi * 180.
        # if dip is just straight up vertical
        elif normal[0] == 0 and normal[1] == 0:
            dip_direction = 0

        else:
            raise ValueError('The values of normal are not valid.')

        if -90 < dip < 90:
            polarity = 1
        else:
            polarity = -1

        return dip, dip_direction, polarity

    @staticmethod
    def plane_fit(point_list):
        """
        Fit plane to points in PointSet
        Fit an d-dimensional plane to the points in a point set.
        adjusted from: http://stackoverflow.com/questions/12299540/plane-fitting-to-4-or-more-xyz-points

        Args:
            point_list (array_like): array of points XYZ

        Returns:
            Return a point, p, on the plane (the point-cloud centroid),
            and the normal, n.
        """

        points = point_list

        from numpy.linalg import svd
        points = np.reshape(points, (np.shape(points)[0], -1))  # Collapse trialing dimensions
        assert points.shape[0] <= points.shape[1], "There are only {} points in {} dimensions.".format(points.shape[1],
                                                                                                       points.shape[0])
        ctr = points.mean(axis=1)
        x = points - ctr[:, np.newaxis]
        M = np.dot(x, x.T)  # Could also use np.cov(x) here.

        # ctr = Point(x=ctr[0], y=ctr[1], z=ctr[2], type='utm', zone=self.points[0].zone)
        normal = svd(M)[0][:, -1]
        # return ctr, svd(M)[0][:, -1]
        if normal[2] < 0:
            normal = - normal

        return ctr, normal


@_setdoc_pro([SurfacePoints.__doc__, Orientations.__doc__, Grid.__doc__])
class RescaledData(object):
    """
    Auxiliary class to rescale the coordinates between 0 and 1 to increase float stability.

    Attributes:
        df (:class:`pn.DataFrame`): Data frame containing the rescaling factor and centers
        surface_points (:class:`SurfacePoints`): [s0]
        orientations (:class:`Orientations`): [s1]
        grid (:class:`Grid`): [s2]

    Args:
        surface_points (:class:`SurfacePoints`):
        orientations (:class:`Orientations`):
        grid (:class:`Grid`):
        rescaling_factor (float): value which divide all coordinates
        centers (list[float]): New center of the coordinates after shifting
    """

    def __init__(self, surface_points: SurfacePoints, orientations: Orientations, grid: Grid,
                 rescaling_factor: float = None, centers: Union[list, pn.DataFrame] = None):

        self.surface_points = surface_points
        self.orientations = orientations
        self.grid = grid

        self.df = pn.DataFrame(np.array([rescaling_factor, centers]).reshape(1, -1),
                               index=['values'],
                               columns=['rescaling factor', 'centers'])

        self.rescale_data(rescaling_factor=rescaling_factor, centers=centers)

    def __repr__(self):
        return self.df.T.to_string()

    def _repr_html_(self):
        return self.df.T.to_html()

    @_setdoc_pro([ds.centers, ds.rescaling_factor])
    def modify_rescaling_parameters(self, attribute, value):
        """
        Modify the parameters used to rescale data

        Args:
            attribute (str): Attribute to be modified. It can be: centers, rescaling factor
            value (float, list[float]):
                * centers: [s0]
                * rescaling factor: [s1]

        Returns:

        """
        assert np.isin(attribute, self.df.columns).all(), 'Valid attributes are: ' + np.array2string(self.df.columns)

        if attribute == 'centers':
            try:
                assert value.shape[0] is 3

                self.df.loc['values', attribute] = value

            except AssertionError:
                print('centers length must be 3: XYZ')

        else:
            self.df.loc['values', attribute] = value

    @_setdoc_pro([ds.rescaling_factor, ds.centers])
    def rescale_data(self, rescaling_factor=None, centers=None):
        """
        Rescale inplace: surface_points, orientations---adding columns in the categories_df---and grid---adding values_r
        attributes. The rescaled values will get stored on the linked objects.

        Args:
            rescaling_factor: [s1]
            centers: [s0]

        Returns:

        """
        max_coord, min_coord = self.max_min_coord(self.surface_points, self.orientations)
        if rescaling_factor is None:
            self.df['rescaling factor'] = self.compute_rescaling_factor(self.surface_points, self.orientations,
                                                                        max_coord, min_coord)
        else:
            self.df['rescaling factor'] = rescaling_factor
        if centers is None:
            self.df.at['values', 'centers'] = self.compute_data_center(self.surface_points, self.orientations,
                                                                       max_coord, min_coord)
        else:
            self.df.at['values', 'centers'] = centers

        self.set_rescaled_surface_points()
        self.set_rescaled_orientations()
        self.set_rescaled_grid()
        return True

    def get_rescaled_surface_points(self):
        """
        Get the rescaled coordinates. return an image of the interface and orientations categories_df with the X_r..
         columns

        Returns:
            :attr:`SurfacePoints.df[['X_r', 'Y_r', 'Z_r']]`
        """
        return self.surface_points.df[['X_r', 'Y_r', 'Z_r']],

    def get_rescaled_orientations(self):
        """
        Get the rescaled coordinates. return an image of the interface and orientations categories_df with the X_r..
         columns.

        Returns:
            :attr:`Orientations.df[['X_r', 'Y_r', 'Z_r']]`
        """
        return self.orientations.df[['X_r', 'Y_r', 'Z_r']]

    @staticmethod
    @_setdoc_pro([SurfacePoints.__doc__, Orientations.__doc__])
    def max_min_coord(surface_points=None, orientations=None):
        """
        Find the maximum and minimum location of any input data in each cartesian coordinate

        Args:
            surface_points (:class:`SurfacePoints`): [s0]
            orientations (:class:`Orientations`): [s1]

        Returns:
            tuple: max[XYZ], min[XYZ]
        """
        if surface_points is None:
            if orientations is None:
                raise AttributeError('You must pass at least one Data object')
            else:
                df = orientations.df
        else:
            if orientations is None:
                df = surface_points.df
            else:
                df = pn.concat([orientations.df, surface_points.df], sort=False)

        max_coord = df.max()[['X', 'Y', 'Z']]
        min_coord = df.min()[['X', 'Y', 'Z']]
        return max_coord, min_coord

    @_setdoc_pro([SurfacePoints.__doc__, Orientations.__doc__, ds.centers])
    def compute_data_center(self, surface_points=None, orientations=None,
                            max_coord=None, min_coord=None, inplace=True):
        """
        Calculate the center of the data once it is shifted between 0 and 1.

        Args:
            surface_points (:class:`SurfacePoints`): [s0]
            orientations (:class:`Orientations`): [s1]
            max_coord (float): Max XYZ coordinates of all GeometricData
            min_coord (float): Min XYZ coordinates of all GeometricData
            inplace (bool): if True modify the self.df rescaling factor attribute

        Returns:
            np.array: [s2]
        """

        if max_coord is None or min_coord is None:
            max_coord, min_coord = self.max_min_coord(surface_points, orientations)

        # Get the centers of every axis
        centers = compute_centers(max_coord, min_coord)
        if inplace is True:
            self.df.at['values', 'centers'] = centers
        return centers

    # def update_centers(self, surface_points=None, orientations=None, max_coord=None, min_coord=None):
    #     # TODO this should update the additional data
    #     self.compute_data_center(surface_points, orientations, max_coord, min_coord, inplace=True)

    @_setdoc_pro([SurfacePoints.__doc__, Orientations.__doc__, ds.rescaling_factor])
    def compute_rescaling_factor(self, surface_points=None, orientations=None,
                                 max_coord=None, min_coord=None, inplace=True):
        """
        Calculate the rescaling factor of the data to keep all coordinates between 0 and 1

        Args:
            surface_points (:class:`SurfacePoints`): [s0]
            orientations (:class:`Orientations`): [s1]
            max_coord (float): Max XYZ coordinates of all GeometricData
            min_coord (float): Min XYZ coordinates of all GeometricData
            inplace (bool): if True modify the self.df rescaling factor attribute

        Returns:
            float: [s2]
        """

        if max_coord is None or min_coord is None:
            max_coord, min_coord = self.max_min_coord(surface_points, orientations)
        rescaling_factor_val = compute_rescaling_factor(max_coord, min_coord)
        if inplace is True:
            self.df['rescaling factor'] = rescaling_factor_val
        return rescaling_factor_val

    # def update_rescaling_factor(self, surface_points=None, orientations=None,
    #                             max_coord=None, min_coord=None):
    #     self.compute_rescaling_factor(surface_points, orientations, max_coord, min_coord, inplace=True)

    @staticmethod
    @_setdoc_pro([SurfacePoints.__doc__, compute_data_center.__doc__, compute_rescaling_factor.__doc__, ds.idx_sp])
    def rescale_surface_points(surface_points, rescaling_factor, centers, idx: list = None, offset: float = 0.5001):
        """
        Rescale inplace: surface_points. The rescaled values will get stored on the linked objects.

        Args:
            surface_points (:class:`SurfacePoints`): [s0]
            rescaling_factor: [s2]
            centers: [s1]
            idx (int, list of int): [s3]
            offset: ???

        Returns:

        """

        if idx is None:
            idx = surface_points.df.index

        # Change the coordinates of surface_points
        XYZ = surface_points.df.loc[idx, ['X', 'Y', 'Z']]
        new_coord_surface_points = rescale_surface_points(XYZ, centers, rescaling_factor, offset=offset)
        new_coord_surface_points.rename(columns={"X": "X_r", "Y": "Y_r", "Z": 'Z_r'}, inplace=True)
        return new_coord_surface_points

    @_setdoc_pro(ds.idx_sp)
    def set_rescaled_surface_points(self, idx: Union[list, np.ndarray] = None):
        """
        Set the rescaled coordinates into the surface_points categories_df

        Args:
            idx (int, list of int): [s0]

        Returns:

        """
        if idx is None:
            idx = self.surface_points.df.index
        idx = np.atleast_1d(idx)

        self.surface_points.df.loc[idx, ['X_r', 'Y_r', 'Z_r']] = self.rescale_surface_points(
            self.surface_points,
            self.df.loc['values', 'rescaling factor'],
            self.df.loc['values', 'centers'],
            idx=idx)

        return self.surface_points

    def rescale_data_point(self, data_points: np.ndarray, rescaling_factor=None, centers=None, offset: float = 0.5001):
        """This method now is very similar to set_rescaled_surface_points passing an index"""
        if rescaling_factor is None:
            rescaling_factor = self.df.loc['values', 'rescaling factor']
        if centers is None:
            centers = self.df.loc['values', 'centers']

        rescaled_data_point = rescale_surface_points(data_points, centers, rescaling_factor, offset=offset)

        return rescaled_data_point

    @staticmethod
    @_setdoc_pro([Orientations.__doc__, compute_data_center.__doc__, compute_rescaling_factor.__doc__, ds.idx_sp])
    def rescale_orientations(orientations, rescaling_factor, centers, idx: list = None, offset: float = 0.5001):
        """
        Rescale inplace: surface_points. The rescaled values will get stored on the linked objects.

        Args:
            orientations (:class:`Orientations`): [s0]
            rescaling_factor: [s2]
            centers: [s1]
            idx (int, list of int): [s3]
            offset: ???

        Returns:

        """
        if idx is None:
            idx = orientations.df.index

        # Change the coordinates of orientations
        XYZ = orientations.df.loc[idx, ['X', 'Y', 'Z']]
        new_coord_orientations = rescale_surface_points(XYZ, centers, rescaling_factor, offset=offset)

        new_coord_orientations.rename(columns={"X": "X_r", "Y": "Y_r", "Z": 'Z_r'}, inplace=True)

        return new_coord_orientations

    @_setdoc_pro(ds.idx_sp)
    def set_rescaled_orientations(self, idx: Union[list, np.ndarray] = None):
        """
        Set the rescaled coordinates into the surface_points categories_df

        Args:
            idx (int, list of int): [s0]

        Returns:

        """
        if idx is None:
            idx = self.orientations.df.index
        idx = np.atleast_1d(idx)

        self.orientations.df.loc[idx, ['X_r', 'Y_r', 'Z_r']] = self.rescale_orientations(
            self.orientations, self.df.loc['values', 'rescaling factor'], self.df.loc['values', 'centers'], idx=idx)
        return True

    @staticmethod
    def rescale_grid(grid, rescaling_factor, centers, offset: float = 0.5001):
        """
        Rescale the grid to to match the transformations of the data

        Args:

            offset: ???

        """
        new_grid_extent = rescale_surface_points(grid.regular_grid.extent, np.repeat(centers, 2), rescaling_factor, offset=offset)
        new_grid_values = rescale_surface_points(grid.values, centers, rescaling_factor, offset=offset)
        return new_grid_extent, new_grid_values

    def set_rescaled_grid(self):
        """
        Set the rescaled coordinates and extent into a grid object
        """

        self.grid.extent_r, self.grid.values_r = self.rescale_grid(
            self.grid, self.df.loc['values', 'rescaling factor'], self.df.loc['values', 'centers'])

        self.grid.regular_grid.extent_r, self.grid.regular_grid.values_r = self.grid.extent_r, self.grid.values_r