"""
    This file is part of gempy.

    gempy is free software: you can redistribute it and/or modify it under the
    terms of the GNU General Public License as published by the Free Software
    Foundation, either version 3 of the License, or (at your option) any later
    version.

    gempy is distributed in the hope that it will be useful, but WITHOUT ANY
    WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
    FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
    details.

    You should have received a copy of the GNU General Public License along with
    gempy.  If not, see <http://www.gnu.org/licenses/>.


    Module with classes and methods to visualized structural geology data and
    potential fields of the regional modelling based on the potential field
    method. Tested on Windows 10

    Created on 08.04.2020

    @author: Miguel de la Varga, Bane Sullivan, Alexander Schaaf, Jan von Harten
"""
from __future__ import annotations

import warnings
from typing import Union, Dict, List, Iterable, Set, Tuple

import matplotlib.colors as mcolors
from matplotlib import cm
import numpy as np
import pandas as pd
import pyvista as pv
from pyvista.plotting import parse_color
# TODO Check if this is necessary if it is implemented in the API
try:
    import pyvistaqt as pvqt
    PYVISTA_IMPORT = True
except ImportError:
    PYVISTA_IMPORT = False

import gempy as gp
from gempy.plot.vista_aux import WidgetsCallbacks, RenderChanges
import matplotlib

warnings.filterwarnings("ignore",
                        message='.*Conversion of the second argument of issubdtype *.',
                        append=True)
try:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    VTK_IMPORT = True
except ImportError:
    VTK_IMPORT = False


class GemPyToVista(WidgetsCallbacks, RenderChanges):

    def __init__(self, model, plotter_type: str = 'basic', extent=None, lith_c=None, live_updating=False, **kwargs):
        """GemPy 3-D visualization using pyVista.

        Args:
            model (gp.Model): Geomodel instance with solutions.
            plotter_type (str): Set the plotter type. Defaults to 'basic'.
            extent (List[float], optional): Custom extent. Defaults to None.
            lith_c (pn.DataFrame, optional): Custom color scheme in the form of
                a look-up table. Defaults to None.
            live_updating (bool, optional): Toggles real-time updating of the
                plot. Defaults to False.
            **kwargs:

        """

        # Override default notebook value
        pv.set_plot_theme("document")
        kwargs['notebook'] = kwargs.get('notebook', False)

        # Model properties
        self.model = model
        self.extent = model._grid.regular_grid.extent if extent is None else extent

        # plotting options
        self.live_updating = live_updating

        # Choosing plotter
        if plotter_type == 'basic':
            self.p = pv.Plotter(**kwargs)
            self.p.view_isometric(negative=False)
        elif plotter_type == 'notebook':
            raise NotImplementedError
            # self.p = pv.PlotterITK()
        elif plotter_type == 'background':
            try:
                self.p = pv.BackgroundPlotter(**kwargs)
            except pv.QtDeprecationError:
                from pyvistaqt import BackgroundPlotter
                self.p = BackgroundPlotter(**kwargs)
            self.p.view_isometric(negative=False)
        else:
            raise AttributeError('Plotter type must be basic, background or notebook.')

        self.plotter_type = plotter_type
        # Default camera and bounds
        self.set_bounds()
        self.p.view_isometric(negative=False)

        # Actors containers
        self.surface_actors = {}
        self.surface_poly = {}

        self.regular_grid_actor = None
        self.regular_grid_mesh = None

        self.surface_points_actor = None
        self.surface_points_mesh = None
        self.surface_points_widgets = {}

        self.orientations_actor = None
        self.orientations_mesh = None
        self.orientations_widgets = {}

        # Private attributes
        self._grid_values = None
        col = matplotlib.cm.get_cmap('viridis')(np.linspace(0, 1, 255)) * 255
        nv = numpy_to_vtk(col, array_type=3)
        self._cmaps = {'viridis': nv}

        # Topology properties
        self.topo_edges = None
        self.topo_ctrs = None

    def _get_color_lot(self, lith_c: pd.DataFrame = None,
                       index='surface',
                       is_faults: bool = True,
                       is_basement: bool = False) -> \
            pd.Series:
        """Method to get the right color list depending on the type of plot.

        Args:
            lith_c (pd.DataFrame): Pandas series with index surface names and
                values hex strings with the colors
            is_faults (bool): Return the colors of the faults. This should be
                true for surfaces and input data and false for scalar values.
            is_basement (bool): Return or not the basement. This should be true
                for the lith block and false for surfaces and input data.
        """
        if lith_c is None:
            surf_df = self.model._surfaces.df.set_index(index)
            unique_surf_points = np.unique(self.model._surface_points.df['id']).astype(int)

            if len(unique_surf_points) != 0:
                bool_surf_points = np.zeros(surf_df.shape[0], dtype=bool)
                bool_surf_points[unique_surf_points.astype('int') - 1] = True

                surf_df['isActive'] = (surf_df['isActive'] | bool_surf_points)

                if is_faults is True and is_basement is True:
                    lith_c = surf_df.groupby('isActive').get_group(True)['color']
                elif is_faults is True and is_basement is False:
                    lith_c = surf_df.groupby(['isActive', 'isBasement']).get_group((True, False))['color']
                else:
                    lith_c = surf_df.groupby(['isActive', 'isFault']).get_group((True, False))[
                        'color']

        color_lot = lith_c
        return color_lot

    @property
    def scalar_bar_options(self):
        sargs = dict(
            title_font_size=20,
            label_font_size=16,
            shadow=True,
            italic=True,
            font_family="arial",
            height=0.25, vertical=True,
            position_x=0.05,
            position_y=0.35
        )

        return sargs

    def set_scalar_bar(self):
        n_labels = self.model._surfaces.df.shape[0]
        arr_ = self.model._surfaces.df['id']
        sargs = self.scalar_bar_options
        sargs['title'] = 'id'
        sargs['n_labels'] = n_labels
        sargs['position_y'] = 0.30
        sargs['height'] = -0.25
        sargs['fmt'] = "%.0f"
        self.p.add_scalar_bar(**sargs)
        self.p.update_scalar_bar_range((arr_.min(), arr_.max()))

    def set_bounds(
            self,
            extent: list = None,
            grid: bool = False,
            location: str = 'furthest',
            **kwargs
    ):
        """Set and toggle display of bounds of geomodel.

        Args:
            extent (list): [description]. Defaults to None.
            grid (bool): [description]. Defaults to False.
            location (str): [description]. Defaults to 'furthest'.
            **kwargs:
        """
        if self.plotter_type != 'notebook':
            if extent is None:
                extent = self.extent
            try:
                self.p.show_bounds(
                    bounds=extent, location=location, grid=grid, use_2d=False, **kwargs
                )
            except KeyError:
                pass

    def plot_data(self, surfaces='all', surface_points=None, orientations=None, **kwargs):
        """Plot all the geometric data

        Args:
            surfaces(str, List[str]): Name of the surface, or list of names of surfaces to plot.
             By default all will plot all surfaces.
            surface_points:
            orientations:
            **kwargs:
        """
        if self.model.surface_points.df.shape[0] != 0:
            self.plot_surface_points(surfaces=surfaces, surface_points=surface_points, **kwargs)
            self.set_scalar_bar()
        if self.model.orientations.df.shape[0] != 0:
            self.plot_orientations(surfaces=surfaces, orientations=orientations, **kwargs)

    @staticmethod
    def _select_surfaces_data(data_df: pd.core.frame.DataFrame,
                              surfaces: Union[str, List[str]] = 'all') -> \
            pd.core.frame.DataFrame:
        """Select the surfaces that has to be plot.

        Args:
            data_df (pd.core.frame.DataFrame): GemPy data df that contains
                surface property. E.g Surfaces, SurfacePoints or Orientations.
            surfaces: If 'all' select all the active data. If a list of surface
                names or a surface name is passed, plot only those.
        """
        if surfaces == 'all':
            geometric_data = data_df
        else:
            geometric_data = pd.concat(
                [data_df.groupby('surface').get_group(group)
                 for group in surfaces])
        return geometric_data

    def remove_actor(self, actor, set_bounds=False):
        """Remove pyvista mesh.

        Args:
            actor: Pyvista mesh
            set_bounds (bool): if True reset the bound
        """
        self.p.remove_actor(actor, reset_camera=False)
        if set_bounds is True:
            self.set_bounds()

    def create_sphere_widgets(self, surface_points: pd.core.frame.DataFrame,
                              test_callback: Union[bool, None] = True, **kwargs) \
            -> List[vtk.vtkInteractionWidgetsPython.vtkSphereWidget]:
        """Create sphere widgets for each surface points with the call back to
        recompute the model.

        Args:
            surface_points (pd.core.frame.DataFrame):
            test_callback (bool):
            **kwargs:

        Returns:
            List[vtkInteractionWidgetsPython.vtkSphereWidget]:
        """
        radius = kwargs.get('radius', None)
        if radius is None:
            _e = self.extent
            _e_dx = _e[1] - _e[0]
            _e_dy = _e[3] - _e[2]
            _e_dz = _e[5] - _e[4]
            _e_d_avrg = (_e_dx + _e_dy + _e_dz) / 3
            radius = _e_d_avrg * .01

            # This is Bane way. It gives me some error with index slicing
            centers = surface_points[['X', 'Y', 'Z']]

            # This is necessary to change the color of the widget if change id
            colors = self._get_color_lot(is_faults=True, is_basement=False)[surface_points['surface']]
            self._color_lot = self._get_color_lot(is_faults=True, is_basement=False, index='id')
            s = self.p.add_sphere_widget(self.call_back_sphere,
                                         center=centers, color=colors.values, pass_widget=True,
                                         test_callback=test_callback,
                                         indices=surface_points.index.values,
                                         radius=radius, **kwargs)
            if type(s) is not list:
                s = [s]
            return s

    def create_orientations_widget(self,
                                   orientations: pd.core.frame.DataFrame)\
            -> List[vtk.vtkInteractionWidgetsPython.vtkPlaneWidget]:
        """Create plane widget for each orientation with interactive recompute
        of the model

        Args:
            orientations (pd.core.frame.DataFrame):

        Returns:
            List[vtkInteractionWidgetsPython.vtkPlaneWidget]:
        """
        colors = self._get_color_lot(is_faults=True, is_basement=False)
        widget_list = []
        # for index, pt, nrm in zip(i, pts, nrms):
        self._color_lot = self._get_color_lot(is_faults=True, is_basement=False, index='id')
        for index, val in orientations.iterrows():
            widget = self.p.add_plane_widget(
                self.call_back_plane,
                normal=val[['G_x', 'G_y', 'G_z']],
                origin=val[['X', 'Y', 'Z']],
                bounds=self.extent,
                factor=0.15,
                implicit=False,
                pass_widget=True,
                test_callback=False,
                color=colors[val['surface']]
            )
            widget.WIDGET_INDEX = index
            widget_list.append(widget)

        return widget_list

    def plot_surface_points(self, surfaces: Union[str, Iterable[str]] = 'all',
                            surface_points: pd.DataFrame = None,
                            clear: bool = True, colors=None, render_points_as_spheres=True,
                            point_size=10, **kwargs):

        # Selecting the surfaces to plot
        """
        Args:
            surfaces:
            surface_points (pd.DataFrame):
            clear (bool):
            colors:
            render_points_as_spheres:
            point_size:
            **kwargs:
        """
        if surface_points is None:
            surface_points = self._select_surfaces_data(self.model._surface_points.df, surfaces)

        if clear is True:
            if self.plotter_type != 'notebook':
                if 'id' not in self.p._scalar_bar_slot_lookup:
                    self.p._scalar_bar_slot_lookup['id'] = None

                self.p.clear_sphere_widgets()
                self.surface_points_widgets = {}
                try:
                    self.p.remove_actor(self.surface_points_actor)
                except KeyError:
                    pass

        if self.live_updating is True:

            sphere_widgets = self.create_sphere_widgets(surface_points, colors, **kwargs)
            self.surface_points_widgets.update(dict(zip(surface_points.index, sphere_widgets)))
            r = self.surface_points_widgets
        else:
            poly = pv.PolyData(surface_points[["X", "Y", "Z"]].values)
            poly['id'] = surface_points['id']
            self.surface_points_mesh = poly
            cmap = mcolors.ListedColormap(list(self._get_color_lot(is_faults=True, is_basement=True)))
            self.surface_points_actor = self.p.add_mesh(poly, cmap=cmap,
                                                        scalars='id',
                                                        render_points_as_spheres=render_points_as_spheres,
                                                        point_size=point_size, show_scalar_bar=False)
            self.set_scalar_bar()

            r = self.surface_points_actor
        self.set_bounds()
        return r

    def plot_orientations(self, surfaces: Union[str, Iterable[str]] = 'all',
                          orientations: pd.DataFrame = None,
                          clear=True, arrow_size=10, **kwargs):
        """

        Args:
            surfaces:
            orientations (pd.DataFrame):
            clear:
            arrow_size:
            **kwargs:
        """
        if orientations is None:
            orientations = self._select_surfaces_data(self.model._orientations.df, surfaces)

        if clear is True:
            self.p.clear_plane_widgets()
            self.orientations_widgets = {}
            try:
                self.p.remove_actor(self.orientations_actor)
            except KeyError:
                pass

        if self.live_updating is True:
            orientations_widgets = self.create_orientations_widget(orientations)
            self.orientations_widgets.update(dict(zip(orientations.index, orientations_widgets)))
            r = self.orientations_widgets
        else:
            poly = pv.PolyData(orientations[["X", "Y", "Z"]].values)
            poly['id'] = orientations['id']
            poly['vectors'] = orientations[['G_x', 'G_y', 'G_z']].values

            min_axes = np.min(np.diff(self.extent)[[0, 2, 4]])

            arrows = poly.glyph(orient='vectors', scale=False,
                                factor=min_axes / (100 / arrow_size))

            cmap = mcolors.ListedColormap(list(self._get_color_lot(is_faults=True, is_basement=True)))
            self.orientations_actor = self.p.add_mesh(arrows, cmap=cmap,
                                                      show_scalar_bar=False)
            self.orientations_mesh = arrows
            r = self.orientations_actor
            self.set_scalar_bar()
        self.set_bounds()
        if self.live_updating is False:
            self.set_scalar_bar()
        return r

    def plot_surfaces(self, surfaces: Union[str, Iterable[str]] = 'all',
                      surfaces_df: pd.DataFrame = None, clear=True,
                      **kwargs):

        # TODO is this necessary for the updates?
        """
        Args:
            surfaces:
            surfaces_df (pd.DataFrame):
            clear:
            **kwargs:
        """
        cmap = mcolors.ListedColormap(list(self._get_color_lot(is_faults=True, is_basement=True)))
        if clear is True and self.plotter_type !='notebook':
            try:
                [self.p.remove_actor(actor) for actor in self.surface_actors.items()]
            except KeyError:
                pass

        if surfaces_df is None:
            surfaces_df = self._select_surfaces_data(self.model._surfaces.df, surfaces)

        select_active = surfaces_df['isActive']
        for idx, val in surfaces_df[select_active][['vertices', 'edges', 'color', 'surface', 'id']].dropna().iterrows():
            surf = pv.PolyData(val['vertices'], np.insert(val['edges'], 0, 3, axis=1).ravel())
            # surf['id'] = val['id']
            self.surface_poly[val['surface']] = surf
            self.surface_actors[val['surface']] = self.p.add_mesh(
                surf, parse_color(val['color']), show_scalar_bar=True,
                cmap=cmap, **kwargs)
        self.set_bounds()

        # In order to set the scalar bar to only surfaces we would need to map
        # every vertex of each layer with the right id. So far I am going to avoid
        # the overhead since usually surfaces will be plotted either with data
        # or the regular grid.
        # self.set_scalar_bar()
        return self.surface_actors

    def update_surfaces(self, recompute=True):
        import time

        if recompute is True:
            try:
                gp.compute_model(self.model, sort_surfaces=False, compute_mesh=True)
            except IndexError:
                t = time.localtime()
                current_time = time.strftime("[%H:%M:%S]", t)
                print(current_time + 'IndexError: Model not computed. Laking data in some surface')
            except AssertionError:
                t = time.localtime()
                current_time = time.strftime("[%H:%M:%S]", t)
                print(current_time + 'AssertionError: Model not computed. Laking data in some surface')

        self.remove_actor(self.regular_grid_actor)
        surfaces = self.model._surfaces
        # TODO add the option of update specific surfaces
        try:
            for idx, val in surfaces.df[['vertices', 'edges', 'surface']].dropna().iterrows():
                self.surface_poly[val['surface']].points = val['vertices']
                self.surface_poly[val['surface']].faces = np.insert(val['edges'], 0, 3, axis=1).ravel()
        except KeyError:
            self.plot_surfaces()

        return True

    def toggle_live_updating(self):

        self.live_updating = self.live_updating ^ True
        self.plot_surface_points()
        self.plot_orientations()

        return self.live_updating

    def plot_topography(
            self,
            topography=None,
            scalars=None,
            contours=True,
            clear=True,
            **kwargs
    ):
        """
        Args:
            topography:
            scalars:
            clear:
            **kwargs:
        """
        rgb = False
        if clear is True and 'topography' in self.surface_actors and self.plotter_type != 'notebook':
                self.p._scalar_bar_slot_lookup['height'] = None
                self.p.remove_actor(self.surface_actors['topography'])
                self.p.remove_actor(self.surface_actors["topography_cont"])

        if not topography:
            try:
                topography = self.model._grid.topography.values
            except AttributeError:
                raise AttributeError("Unable to plot topography: Given geomodel instance "
                                     "does not contain topography grid.")

        polydata = pv.PolyData(topography)

        if scalars is None and self.model.solutions.geological_map is not None:
            scalars = 'geomap'
        elif scalars is None:
            scalars = 'topography'

        if scalars == "geomap":

            colors_hex = self._get_color_lot(is_faults=False, is_basement=True, index='id')
            colors_rgb_ = colors_hex.apply(lambda val: list(mcolors.hex2color(val)))
            colors_rgb = pd.DataFrame(colors_rgb_.to_list(), index=colors_hex.index) * 255

            sel = np.round(self.model.solutions.geological_map[0]).astype(int)[0]

            scalars_val = numpy_to_vtk(colors_rgb.loc[sel], array_type=3)
            cm = mcolors.ListedColormap(list(self._get_color_lot(is_faults=True, is_basement=True)))
            rgb = True

        elif scalars == "topography":
            scalars_val = topography[:, 2]
            cm = 'terrain'

        elif type(scalars) is np.ndarray:
            scalars_val = scalars
            cm = 'terrain'

        else:
            raise AttributeError("Parameter scalars needs to be either \
                      'geomap', 'topography' or a np.ndarray with scalar values")

        polydata.delaunay_2d(inplace=True)
        polydata['id'] = scalars_val
        polydata['height'] = topography[:, 2]
        if scalars != 'geomap':
            show_scalar_bar = True
            scalars = 'height'
        else:
            show_scalar_bar = False
            scalars = 'id'
        sbo = self.scalar_bar_options
        sbo['position_y'] = .35

        topography_actor = self.p.add_mesh(
            polydata,
            scalars=scalars,
            cmap=cm,
            rgb=rgb,
            show_scalar_bar=show_scalar_bar,
            scalar_bar_args=sbo,
            **kwargs
        )

        if scalars == 'geomap':
            self.set_scalar_bar()

        if contours is True:
            contours = polydata.contour(scalars='height')
            contours_actor = self.p.add_mesh(contours, color="white", line_width=3)

            self.surface_poly['topography'] = polydata
            self.surface_poly['topography_cont'] = contours
            self.surface_actors["topography"] = topography_actor
            self.surface_actors["topography_cont"] = contours_actor
        return topography_actor

    def plot_structured_grid(self, scalar_field: str = 'all',
                             data: Union[dict, str] = 'Default',
                             series: str = '',
                             render_topography: bool = True,
                             opacity=.5,
                             clear=True,
                             **kwargs) -> list:
        """Plot a structured grid of the geomodel.

        Args:
            scalar_field (str): Can be either one of the following

                'lith' - Lithology id block. 'scalar' - Scalar field block.
                'values' - Values matrix block.
            data:
            series (str):
            render_topography (bool):
            **kwargs:
        """
        if clear is True:
            try:
                self.p.remove_actor(self.regular_grid_actor)
            except KeyError:
                pass
        regular_grid, cmap = self.create_regular_mesh(scalar_field, data,
                                                      series, render_topography)
        
        return self.add_regular_grid_mesh(regular_grid, cmap, scalar_field, series,
                                          opacity, **kwargs)
    
    def create_regular_mesh(self, scalar_field: str = 'all',
                             data: Union[dict, str] = 'Default',
                             series: str = '',
                             render_topography: bool = True,
                        ):
        
        regular_grid = self.model._grid.regular_grid

        if regular_grid.values is self._grid_values:
            regular_grid_mesh = self.regular_grid_mesh
        else:
            # If the regular grid changes we need to create a new grid. Otherwise we can append it to the
            # previous
            self._grid_values = regular_grid.values

            grid_3d = self._grid_values.reshape(*regular_grid.resolution, 3).T
            regular_grid_mesh = pv.StructuredGrid(*grid_3d)
        
        # Set the scalar field-Activate it-getting cmap?
        regular_grid_mesh, cmap = self.set_scalar_data(regular_grid_mesh,
                                                       data=data, scalar_field=scalar_field,
                                                       series=series)

        if render_topography == True and regular_grid.mask_topo.shape[0] != 0 and True:

            main_scalar = 'id' if scalar_field == 'all' else regular_grid_mesh.array_names[-1]
            regular_grid_mesh[main_scalar][regular_grid.mask_topo.ravel(order='C')] = -100
            regular_grid_mesh = regular_grid_mesh.threshold(-99, scalars=main_scalar)
        
        return regular_grid_mesh, cmap
    
    def add_regular_grid_mesh(self,
                              regular_grid_mesh,
                              cmap,
                              scalar_field: str = 'all',
                              series: str = '',
                              opacity=.5, **kwargs):

        if scalar_field == 'all' or scalar_field == 'lith':
            stitle = 'id'
            show_scalar_bar = False
            main_scalar = 'id'
        else:
            stitle = scalar_field + ': ' + series
            show_scalar_bar = True
            main_scalar_prefix = 'sf_' if scalar_field == 'scalar' else 'values_'
            main_scalar = main_scalar_prefix + series
            if series == '':
                main_scalar = regular_grid_mesh.array_names[-1]

        self.regular_grid_actor = self.p.add_mesh(regular_grid_mesh, cmap=cmap,
                                                  stitle=stitle,
                                                  scalars=main_scalar,
                                                  show_scalar_bar=show_scalar_bar,
                                                  scalar_bar_args=self.scalar_bar_options,
                                                  opacity=opacity,
                                                  **kwargs)

        if scalar_field == 'all' or scalar_field == 'lith':
            self.set_scalar_bar()
        self.regular_grid_mesh = regular_grid_mesh
        return [regular_grid_mesh, cmap]
    
    def set_scalar_data(self, regular_grid, data: Union[dict, gp.Solution, str] = 'Default',
                        scalar_field='all', series='', cmap='viridis'):
        """
        Args:
            regular_grid:
            data: dictionary or solution
            scalar_field: if data is a gp.Solutions object, name of the grid
                that you want to plot.
            series:
            cmap:
        """
        if data == 'Default':
            data = self.model.solutions

        if isinstance(data, gp.Solution):
            if scalar_field == 'lith' or scalar_field == 'all':
                regular_grid['id'] = data.lith_block
                hex_colors = list(self._get_color_lot(is_faults=True, is_basement=True))
                cmap = mcolors.ListedColormap(hex_colors)
            if scalar_field == 'scalar' or scalar_field == 'all' or 'sf_' in scalar_field:
                scalar_field_ = 'sf_'
                for e, series in enumerate(self.model._stack.df.groupby('isActive').groups[True]):
                    regular_grid[scalar_field_ + series] = data.scalar_field_matrix[e]

            if (scalar_field == 'values' or scalar_field == 'all' or 'values_' in scalar_field) and\
                    data.values_matrix.shape[0] \
                    != 0:
                scalar_field_ = 'values_'
                for e, lith_property in enumerate(self.model._surfaces.df.columns[self.model._surfaces._n_properties:]):
                    regular_grid[scalar_field_ + lith_property] = data.values_matrix[e]

        if type(data) == dict:
            for key in data:
                regular_grid[key] = data[key]

        if scalar_field == 'all' or scalar_field == 'lith':
            scalar_field_ = 'lith'
            series = ''
        # else:
        #     scalar_field_ = regular_grid.scalar_names[-1]
        #     series = ''

        self.set_active_scalar_fields(scalar_field_ + series, regular_grid, update_cmap=False)

        return regular_grid, cmap

    def set_active_scalar_fields(self, scalar_field, regular_grid=None, update_cmap=True):
        """
        Args:
            scalar_field:
            regular_grid:
            update_cmap:
        """
        if scalar_field == 'lith':
            scalar_field = 'id'

        if regular_grid is None:
            regular_grid = self.regular_grid_mesh

        # Set the scalar field active
        try:
            regular_grid.set_active_scalars(scalar_field)
        except ValueError:
            raise AttributeError('The scalar field provided does not exist. Please pass '
                                 'a valid field: {}'.format(regular_grid.array_names))

        if update_cmap is True and self.regular_grid_actor is not None:
            cmap = 'lith' if scalar_field == 'lith' else 'viridis'
            self.set_scalar_field_cmap(cmap=cmap)
            arr_ = regular_grid.get_array(scalar_field)
            if scalar_field != 'lith':
                self.p.add_scalar_bar(title='values')
                self.p.update_scalar_bar_range((arr_.min(), arr_.max()))

    def set_scalar_field_cmap(self, cmap: Union[str, dict] = 'viridis',
                              regular_grid_actor = None) -> None:
        """
        Args:
            cmap:
            regular_grid_actor (Union[None, vtkRenderingOpenGL2Python.vtkOpenGLActor):
        """
        if regular_grid_actor is None:
            regular_grid_actor = self.regular_grid_actor

        if type(cmap) is dict:
            self._cmaps = {**self._cmaps, **cmap}
            cmap = cmap.keys()
        elif type(cmap) is str:
            if cmap == 'lith':
                hex_colors = list(self._get_color_lot(is_faults=False))
                n_colors = len(hex_colors)
                cmap_ = mcolors.ListedColormap(hex_colors)
                col = cmap_(np.linspace(0, 1, n_colors)) * 255
                self._cmaps[cmap] = numpy_to_vtk(col, array_type=3)
            if cmap not in self._cmaps.keys():
                col = matplotlib.cm.get_cmap(cmap)(np.linspace(0, 1, 250)) * 255
                nv = numpy_to_vtk(col, array_type=3)
                self._cmaps[cmap] = nv
        else:
            raise AttributeError('cmap must be either a name of a matplotlib string or a dictionary containing the '
                                 'rgb values')
        # Set the scalar field color map
        regular_grid_actor.GetMapper().GetLookupTable().SetTable(self._cmaps[cmap])

    def plot_structured_grid_interactive(
            self,
            scalar_field: str,
            series = None,
            render_topography: bool = False,
            **kwargs,
    ):
        """Plot interactive 3-D geomodel with three cross sections in subplot.

        Args:
            geo_model: Geomodel object with solutions.
            scalar_field (str): Can be either one of the following
                'lith' - Lithology id block.
                'scalar' - Scalar field block.
                'values' - Values matrix block.
            render_topography: Render topography. Defaults to False.
            **kwargs:

        Returns:
            (Vista) GemPy Vista object for plotting.
        """
        # mesh, cmap = self.plot_structured_grid(name=scalar_field, series=series,
        #                                        render_topography=render_topography, **kwargs)

        mesh, cmap = self.create_regular_mesh(scalar_field=scalar_field, series=series,
                                              render_topography=render_topography)

        if scalar_field == 'all' or scalar_field == 'lith':
            main_scalar = 'id'
        else:
            main_scalar_prefix = 'sf_' if scalar_field == 'scalar' else 'values_'
            main_scalar = main_scalar_prefix + series

        # callback functions for subplots
        def xcallback(normal, origin):
            self.p.subplot(1)
            self.p.add_mesh(mesh.slice(normal=normal, origin=origin),
                            scalars=main_scalar, name="xslc", cmap=cmap)
                
        def ycallback(normal, origin):
            self.p.subplot(2)
            self.p.add_mesh(mesh.slice(normal=normal, origin=origin), name="yslc", cmap=cmap)
        
        def zcallback(normal, origin):
            self.p.subplot(3)
            self.p.add_mesh(mesh.slice(normal=normal, origin=origin), name="zslc", cmap=cmap)

        self.add_regular_grid_mesh(mesh, cmap, scalar_field, series, **kwargs)

        # cross section widgets
        self.p.subplot(0)
        self.p.add_plane_widget(xcallback, normal="x")
        self.p.subplot(0)
        self.p.add_plane_widget(ycallback, normal="y")
        self.p.subplot(0)
        self.p.add_plane_widget(zcallback, normal="z")

        # Lock other three views in place
        self.p.subplot(1)
        self.p.view_yz()
        self.p.disable()
        self.p.subplot(2)
        self.p.view_xz()
        self.p.disable()
        self.p.subplot(3)
        self.p.view_xy()
        self.p.disable()

        return self

    def _scale_topology_centroids(
            self,
            centroids: Dict[int, np.ndarray]
    ) -> Dict[int, np.ndarray]:
        """Scale topology centroid coordinates from grid coordinates to
        physical coordinates.

        Args:
            centroids (Dict[int, Array[float, 3]]): Centroid dictionary.

        Returns:
            Dict[int, Array[float, 3]]: Rescaled centroid dictionary.
        """
        res = self.model._grid.regular_grid.resolution
        scaling = np.diff(self.extent)[::2] / res

        scaled_centroids = {}
        for n, pos in centroids.items():
            pos_scaled = pos * scaling
            pos_scaled[0] += np.min(self.extent[:2])
            pos_scaled[1] += np.min(self.extent[2:4])
            pos_scaled[2] += np.min(self.extent[4:])
            scaled_centroids[n] = pos_scaled

        return scaled_centroids

    def plot_topology(
            self,
            edges: Set[Tuple[int, int]],
            centroids: Dict[int, np.ndarray],
            node_kwargs: dict = {},
            edge_kwargs: dict = {}
    ):
        """Plot geomodel topology graph based on given set of topology edges
        and node centroids.

        Args:
            edges (Set[Tuple[int, int]]): Topology edges.
            centroids (Dict[int, Array[float, 3]]): Topology node centroids
            node_kwargs (dict, optional): Node plotting options. Defaults to {}.
            edge_kwargs (dict, optional): Edge plotting options. Defaults to {}.
        """
        lot = gp.assets.topology.get_lot_node_to_lith_id(self.model, centroids)
        centroids_scaled = self._scale_topology_centroids(centroids)
        colors = self._get_color_lot()
        for node, pos in centroids_scaled.items():
            mesh = pv.Sphere(
                center=pos,
                radius=np.average(self.extent) / 15
            )
            # * Requires topo id to lith id lot
            self.p.add_mesh(
                mesh,
                color=colors.iloc[lot[node] - 1],
                **node_kwargs
            )

        ekwargs = dict(
            line_width=3
        )
        ekwargs.update(edge_kwargs)

        for e1, e2 in edges:
            pos1, pos2 = centroids_scaled[e1], centroids_scaled[e2]

            x1, y1, z1 = pos1
            x2, y2, z2 = pos2
            x, y, z = (x1, x2), (y1, y2), (z1, z2)
            pos_mid = (
                min(x) + (max(x) - min(x)) / 2,
                min(y) + (max(y) - min(y)) / 2,
                min(z) + (max(z) - min(z)) / 2
            )
            mesh = pv.Line(
                pointa=pos1,
                pointb=pos_mid,
            )
            self.p.add_mesh(mesh, color=colors.iloc[lot[e1] - 1], **ekwargs)

            mesh = pv.Line(
                pointa=pos_mid,
                pointb=pos2,
            )
            self.p.add_mesh(mesh, color=colors.iloc[lot[e2] - 1], **ekwargs)
