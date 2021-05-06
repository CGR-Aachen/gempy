"""
    This file is part of gempy.

    gempy is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    gempy is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with gempy.  If not, see <http://www.gnu.org/licenses/>.


Module with classes and methods to visualize structural geology data and potential fields of the regional modelling based on
the potential field method.
Tested on Ubuntu 14

Created on 10.04.2019

@author: Miguel de la Varga
"""

import os
import matplotlib.colors as mcolors
import copy
import pandas as pn
import numpy as np
import sys
import gempy as gp
import warnings


warnings.filterwarnings("ignore",
                        message='.*Conversion of the second argument of issubdtype *.',
                        append=True)
try:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk
    VTK_IMPORT = True
except ImportError:
    VTK_IMPORT = False

try:
    import steno3d
    STENO_IMPORT = True
except ImportError:
    STENO_IMPORT = False

try:
    import ipyvolume as ipv
    IPV_IMPORT = True
except ImportError:
    IPV_IMPORT = False

import pyvista as pv

class vtkVisualization(object):
    """
    Class to visualize data and results in 3D. Init will create all the render properties while the method render
    model will lunch the window. Using set_surface_points, set_orientations and set_surfaces in between can be chosen what
    will be displayed.

    Args:
        geo_data(gempy.InputData): All values of a DataManagement object
        ren_name (str): Name of the renderer window
        verbose (int): Verbosity for certain functions

    Attributes:
        renWin(vtk.vtkRenderWindow())
        camera_list (list): list of cameras for the distinct renderers
        ren_list (list): list containing the vtk renderers
    """

    def __init__(self, geo_data, ren_name='GemPy 3D-Editor', verbose=0, real_time=False, bg_color=None, ve=1,
                 **kwargs):
        if VTK_IMPORT is False:
            raise ImportError('vtk is not installed. No vtk capabilities are possible')

        self.ve = ve

        self.real_time = real_time
        self.geo_model = geo_data
        self.interp_data = None
        self.layer_visualization = True
        # self.C_LOT = #dict(zip(self.geo_model.surfaces.df['id'], self.geo_model.surfaces.df['color']))
        #
        # for surf, color in self.C_LOT.items(): #convert hex to rgb
        #     self.C_LOT[surf] = mcolors.hex2color(color)

        self.ren_name = ren_name
        # Number of renders
        self.n_ren = 4
        self.id = geo_data._surface_points.df['id'].unique()
        self.surface_name = geo_data._surface_points.df['surface'].unique()

        # Extents
        self.extent = copy.copy(self.geo_model._grid.regular_grid.extent)
        self.extent[-1] = ve * self.extent[-1]
        self.extent[-2] = ve * self.extent[-2]
        _e = self.extent
        self._e_dx = _e[1] - _e[0]
        self._e_dy = _e[3] - _e[2]
        self._e_dz = _e[5] - _e[4]
        self._e_d_avrg = (self._e_dx + self._e_dy + self._e_dz) / 3

        # Create containers of the vtk objectes
        self.s_rend_1 = pn.DataFrame(columns=['val'])
        self.s_rend_2 = pn.DataFrame(columns=['val'])
        self.s_rend_3 = pn.DataFrame(columns=['val'])
        self.s_rend_4 = pn.DataFrame(columns=['val'])

        self.o_rend_1 = pn.DataFrame(columns=['val'])
        self.o_rend_2 = pn.DataFrame(columns=['val'])
        self.o_rend_3 = pn.DataFrame(columns=['val'])
        self.o_rend_4 = pn.DataFrame(columns=['val'])

        # Resolution
        self.res = self.geo_model._grid.regular_grid.resolution

        # create render window, settings
        self.renwin = vtk.vtkRenderWindow()
        self.renwin.SetWindowName(ren_name)

        # Set 4 renderers. ie 3D, X,Y,Z projections
        self.ren_list = self.create_ren_list()

        # create interactor and set interactor style, assign render window
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.renwin)
        self.interactor.AddObserver("KeyPressEvent", self.key_callbacks)

        # 3d model camera for the 4 renders
        self.camera_list = self._create_cameras(self.extent, verbose=verbose)
        # Setting the camera and the background color to the renders
        self.set_camera_backcolor(color=bg_color)

        # Creating the axis
        for e, r in enumerate(self.ren_list):
            # add axes actor to all renderers
            axe = self._create_axes(self.camera_list[e])

            r.AddActor(axe)
            r.ResetCamera()
        # TODO panel does not deal with text
        self.set_text()

    #

    # def restart(self):
    #     try:
    #         self.close_window()
    #     except AttributeError:
    #         pass
    #
    #     self.__init__(self.geo_model, )

    def render_model(self, **kwargs):
        """
        Method to launch the window

        Args:
            size (tuple): Resolution of the window
            fullscreen (bool): Launch window in full screen or not
        Returns:

        """
        # initialize and start the app
        if 'size' not in kwargs:
            kwargs['size'] = (1920, 1080)

        if 'fullscreen' in kwargs:
            self.renwin.FullScreenOn()

        silent = kwargs.get('silent', False)

        self.renwin.SetSize(kwargs['size'])

        if silent is False:
            self.interactor.Initialize()
            self.interactor.Start()

    def set_text(self):
        txt = vtk.vtkTextActor()
        txt.SetInput("Press L to toggle layers visibility \n"
                     "Press R to toggle real time updates \n"
                     "Press H or P to go back to Python \n"
                     "Press Q to quit")
        txtprop = txt.GetTextProperty()
        txtprop.SetFontFamilyToArial()
        txtprop.SetFontSize(18)
        txtprop.SetColor(1, 1, 1)
        txt.SetDisplayPosition(20, 60)

        # assign actor to the renderer
        self.ren_list[0].AddActor(txt)

    def close_window(self):
        # close_window(interactor)
        render_window = self.interactor.GetRenderWindow()
        render_window.Finalize()
        self.interactor.TerminateApp()
        del self.renwin, self.interactor

    def key_callbacks(self, obj, event):
        key = self.interactor.GetKeySym()

        if key is 'h' or key is 'p':
            print('holding... Use vtk.resume to go back to the interactive window')
            self.interactor.ExitCallback()
            self.interactor.holding = True

        if key is 'l':
            if self.layer_visualization is True:
                for layer in self.surf_rend_1:
                    layer.VisibilityOff()
                self.layer_visualization = False
                self.interactor.Render()
            elif self.layer_visualization is False:
                for layer in self.surf_rend_1:
                    layer.VisibilityOn()
                self.layer_visualization = True
                self.interactor.Render()

        if key is 't':
            if self.topo_visualization is True:
                self.topography_surface.VisibilityOff()
                self.topo_visualization = False
                self.interactor.Render()
            elif self.topo_visualization is False:
                self.topography_surface.VisibilityOn()
                self.topo_visualization = True
                self.interactor.Render()

        if key is 'q':
            print('closing vtk')
            self.close_window()

            # TODO: Decide if this even matter
            # # create render window, settings
            # self.renwin = vtk.vtkRenderWindow()
            # self.renwin.SetWindowName(self.ren_name)
            #
            # # Set 4 renderers. ie 3D, X,Y,Z projections
            # self.ren_list = self.create_ren_list()
            #
            # # create interactor and set interactor style, assign render window
            # self.interactor = vtk.vtkRenderWindowInteractor()
            # self.interactor.SetRenderWindow(self.renwin)
            # self.interactor.AddObserver("KeyPressEvent", self.key_callbacks)
            #
            # # 3d model camera for the 4 renders
            # self.camera_list = self._create_cameras(self.extent)
            # # Setting the camera and the background color to the renders
            # self.set_camera_backcolor()
            #
            # # Creating the axis
            # for e, r in enumerate(self.ren_list):
            #     # add axes actor to all renderer
            #     axe = self._create_axes(self.camera_list[e])
            #
            #     r.AddActor(axe)
            #     r.ResetCamera()

        if key is 'r':
            self.real_time = self.real_time ^ True

    def create_surface_points(self, vertices):
        """
        Method to create the points that form the surfaces
        Args:
            vertices (numpy.array): 2D array (XYZ) with the coordinates of the points

        Returns:
            vtk.vtkPoints: with the coordinates of the points
        """
        Points = vtk.vtkPoints()
        if self.ve != 1:
            vertices[:, 2] = vertices[:, 2] * self.ve
        #     raise NotImplementedError('Vertical exageration for surfaces not implemented yet.')
        # for v in vertices:
        #     v[-1] = self.ve * v[-1]
        #     Points.InsertNextPoint(v)
        Points.SetData(numpy_to_vtk(vertices))

        return Points

    @staticmethod
    def create_surface_triangles(simplices):
        """
        Method to create the Triangles that form the surfaces
        Args:
            simplices (numpy.array): 2D array with the value of the vertices that form every single triangle

        Returns:
            vtk.vtkTriangle
        """

        Triangles = vtk.vtkCellArray()
        Triangle = vtk.vtkTriangle()

        for s in simplices:
            Triangle.GetPointIds().SetId(0, s[0])
            Triangle.GetPointIds().SetId(1, s[1])
            Triangle.GetPointIds().SetId(2, s[2])

            Triangles.InsertNextCell(Triangle)
        return Triangles

    def create_surface(self, vertices, simplices, fn, alpha=.8):
        """
        Method to create the polydata that define the surfaces

        Args:
            vertices (numpy.array): 2D array (XYZ) with the coordinates of the points
            simplices (numpy.array): 2D array with the value of the vertices that form every single triangle
            fn (int): id
            alpha (float): Opacity

        Returns:
            vtk.vtkActor, vtk.vtkPolyDataMapper, vtk.vtkPolyData
        """
        vertices_c = copy.deepcopy(vertices)
        simplices_c = copy.deepcopy(simplices)

        surf_polydata = vtk.vtkPolyData()

        surf_polydata.SetPoints(self.create_surface_points(vertices_c))
        surf_polydata.SetPolys(self.create_surface_triangles(simplices_c))
        surf_polydata.Modified()

        surf_mapper = vtk.vtkPolyDataMapper()
        surf_mapper.SetInputData(surf_polydata)
        surf_mapper.Update()

        surf_actor = vtk.vtkActor()
        surf_actor.SetMapper(surf_mapper)
        surf_actor.GetProperty().SetColor(mcolors.hex2color(self.geo_model._surfaces.df.set_index('id')['color'][fn]))#self.C_LOT[fn])
        surf_actor.GetProperty().SetOpacity(alpha)

        return surf_actor, surf_mapper, surf_polydata

    def create_sphere(self, X, Y, Z, fn, n_sphere=0, n_render=0, n_index=0, r=0.03):
        """
        Method to create the sphere that represent the surface_points points
        Args:
            X: X coord
            Y: Y coord
            Z: Z corrd
            fn (int): id
            n_sphere (int): Number of the sphere
            n_render (int): Number of the render where the sphere belongs
            n_index (int): index value in the PandasDataframe of InupData.surface_points
            r (float): radius of the sphere

        Returns:
            vtk.vtkSphereWidget
        """
        s = vtk.vtkSphereWidget()
        s.SetInteractor(self.interactor)
        s.SetRepresentationToSurface()
        s.SetPriority(2)
        Z = Z * self.ve
        s.r_f = self._e_d_avrg * r
        s.PlaceWidget(X - s.r_f, X + s.r_f, Y - s.r_f, Y + s.r_f, Z - s.r_f, Z + s.r_f)
        s.GetSphereProperty().SetColor(mcolors.hex2color(self.geo_model._surfaces.df.set_index('id')['color'][fn]))#self.C_LOT[fn])

        s.SetCurrentRenderer(self.ren_list[n_render])
        s.n_sphere = n_sphere
        s.n_render = n_render
        s.index = n_index
        s.AddObserver("EndInteractionEvent", self.sphereCallback)  # EndInteractionEvent
        s.AddObserver("InteractionEvent", self.Callback_camera_reset)

        s.On()

        return s

    def create_foliation(self, X, Y, Z, fn,
                         Gx, Gy, Gz,
                         n_plane=0, n_render=0, n_index=0, alpha=0.5):
        """
        Method to create a plane given a foliation

        Args:
            X : X coord
            Y: Y coord
            Z: Z coord
            fn (int): id
            Gx (str): Component of the gradient x
            Gy (str): Component of the gradient y
            Gz (str): Component of the gradient z
            n_plane (int): Number of the plane
            n_render (int): Number of the render where the plane belongs
            n_index (int): index value in the PandasDataframe of InupData.surface_points
            alpha: Opacity of the plane

        Returns:
            vtk.vtkPlaneWidget
        """

        Z = Z * self.ve

        d = vtk.vtkPlaneWidget()
        d.SetInteractor(self.interactor)
        d.SetRepresentationToSurface()

        # Position
        source = vtk.vtkPlaneSource()

        source.SetNormal(Gx, Gy, Gz)
        source.SetCenter(X, Y, Z)
        a, b, c, d_, e, f = self.extent

        source.SetPoint1(X+self._e_dx*.01, Y-self._e_dy*.01, Z)
        source.SetPoint2(X-self._e_dx*.01, Y+self._e_dy*.01, Z)
        source.Update()
        d.SetInputData(source.GetOutput())
        d.SetHandleSize(.05)
        min_extent = np.min([self._e_dx, self._e_dy, self._e_dz])
        d.SetPlaceFactor(0.1)

        d.PlaceWidget(a, b, c, d_, e, f)
        d.SetNormal(Gx, Gy, Gz)
        d.SetCenter(X, Y, Z)
        d.GetPlaneProperty().SetColor(mcolors.hex2color(self.geo_model._surfaces.df.set_index('id')['color'][fn]))#self.C_LOT[fn])
        d.GetHandleProperty().SetColor(mcolors.hex2color(self.geo_model._surfaces.df.set_index('id')['color'][fn]))#self.C_LOT[fn])
        d.GetHandleProperty().SetOpacity(alpha)
        d.SetCurrentRenderer(self.ren_list[n_render])
        d.n_plane = n_plane
        d.n_render = n_render
        d.index = n_index
        d.AddObserver("EndInteractionEvent", self.planesCallback)
        d.AddObserver("InteractionEvent", self.Callback_camera_reset)

        d.On()

        return d

    def set_surfaces_old(self, vertices, simplices, alpha=1):
        """
        Create all the surfaces and set them to the corresponding renders for their posterior visualization with
        render_model

        Args:
            vertices (list): list of 3D numpy arrays containing the points that form each plane
            simplices (list): list of 3D numpy arrays containing the verticies that form every triangle
            surfaces (list): ordered list of strings containing the name of the surfaces to represent
            fns (list): ordered list of surface_numbers (int)
            alpha: Opacity of the plane

        Returns:
            None
        """
        self.surf_rend_1 = []

        surfaces = self.surface_name

        fns = self.geo_model._surface_points.df['id'].unique().squeeze()
        assert type(vertices) is list or type(vertices) is np.ndarray, 'vertices and simpleces have to be a list of' \
                                                                       ' arrays even when only one' \
                                                                       ' surface is passed'
        assert 'DefaultBasement' not in surfaces, 'Remove DefaultBasement from the list of surfaces'
        for v, s, fn in zip(vertices, simplices, np.atleast_1d(fns)):
            act, map, pol = self.create_surface(v, s, fn, alpha)
            self.surf_rend_1.append(act)

            self.ren_list[0].AddActor(act)
            self.ren_list[1].AddActor(act)
            self.ren_list[2].AddActor(act)
            self.ren_list[3].AddActor(act)

    def set_surfaces(self, surfaces, alpha=1):
        self.surf_rend_1 = []
        for idx, val in surfaces.df[['vertices', 'edges', 'id']].dropna().iterrows():
            act, map, pol = self.create_surface(val['vertices'], val['edges'], val['id'], alpha)
            self.surf_rend_1.append(act)

            self.ren_list[0].AddActor(act)
            self.ren_list[1].AddActor(act)
            self.ren_list[2].AddActor(act)
            self.ren_list[3].AddActor(act)

    def set_topography(self):
        # Create points on an XY grid with random Z coordinate
        vertices = copy.copy(self.geo_model._grid.topography.values)

        points = vtk.vtkPoints()
        # for v in vertices:
        #     v[-1] = v[-1]
        #     points.InsertNextPoint(v)
        if self.ve !=1:
            vertices[:, 2]= vertices[:, 2]*self.ve
        points.SetData(numpy_to_vtk(vertices))

        # Add the grid points to a polydata object
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        #
        # glyphFilter = vtk.vtkVertexGlyphFilter()
        # glyphFilter.SetInputData(polydata)
        # glyphFilter.Update()
        #
        # # Create a mapper and actor
        # pointsMapper = vtk.vtkPolyDataMapper()
        # pointsMapper.SetInputConnection(glyphFilter.GetOutputPort())

        #
        # pointsActor = vtk.vtkActor()
        # pointsActor.SetMapper(pointsMapper)
        # pointsActor.GetProperty().SetPointSize(3)
        # pointsActor.GetProperty().SetColor(colors.GetColor3d("Red"))

        # Triangulate the grid points
        delaunay = vtk.vtkDelaunay2D()
        delaunay.SetInputData(polydata)
        delaunay.Update()

        # Create a mapper and actor
        triangulatedMapper = vtk.vtkPolyDataMapper()
        triangulatedMapper.SetInputConnection(delaunay.GetOutputPort())

        triangulatedActor = vtk.vtkActor()
        triangulatedActor.SetMapper(triangulatedMapper)

        self.topography_surface = triangulatedActor
        self._topography_polydata = polydata
        self._topography_delauny = delaunay
        self.ren_list[0].AddActor(triangulatedActor)
        self.ren_list[1].AddActor(triangulatedActor)
        self.ren_list[2].AddActor(triangulatedActor)
        self.ren_list[3].AddActor(triangulatedActor)
        try:
            if self.geo_model.solutions.geological_map is not None:
                self.set_geological_map()
        except AttributeError as ae:
            warnings.warn(str(ae))

    def set_geological_map(self):
        assert self.geo_model.solutions.geological_map is not None, 'Geological map not computed. First' \
                                                                    'set active the topography grid'
        arr_ = np.empty((0, 3), dtype='int')

        # Convert hex colors to rgb
        for idx, val in self.geo_model._surfaces.df['color'].iteritems():
            rgb = (255 * np.array(mcolors.hex2color(val)))
            arr_ = np.vstack((arr_, rgb))

        sel = np.round(self.geo_model.solutions.geological_map[0]).astype(int)[0]
        nv = numpy_to_vtk(arr_[sel - 1], array_type=3)
        self._topography_delauny.GetOutput().GetPointData().SetScalars(nv)

    def set_surface_points(self, indices=None):
        """
        Create all the surface_points points and set them to the corresponding renders for their posterior visualization
         with render_model

        Returns:
            None
        """

        if not indices:

            for e, val in enumerate(self.geo_model._surface_points.df.iterrows()):
                index = val[0]
                row = val[1]
                self.s_rend_1.at[index] = (self.create_sphere(row['X'], row['Y'], row['Z'], row['id'],
                                           n_sphere=e, n_render=0, n_index=index))
                self.s_rend_2.at[index] = (self.create_sphere(row['X'], row['Y'], row['Z'], row['id'],
                                           n_sphere=e, n_render=1, n_index=index))
                self.s_rend_3.at[index] = (self.create_sphere(row['X'], row['Y'], row['Z'], row['id'],
                                           n_sphere=e, n_render=2, n_index=index))
                self.s_rend_4.at[index] = (self.create_sphere(row['X'], row['Y'], row['Z'], row['id'],
                                           n_sphere=e, n_render=3, n_index=index))
        else:
            for e, val in enumerate(self.geo_model._surface_points.df.loc[np.atleast_1d(indices)].iterrows()):
                index = val[0]
                row = val[1]
                self.s_rend_1.at[index] = (self.create_sphere(row['X'], row['Y'], row['Z'], row['id'],
                                           n_sphere=e, n_render=0, n_index=index))
                self.s_rend_2.at[index] = (self.create_sphere(row['X'], row['Y'], row['Z'], row['id'],
                                           n_sphere=e, n_render=1, n_index=index))
                self.s_rend_3.at[index] = (self.create_sphere(row['X'], row['Y'], row['Z'], row['id'],
                                           n_sphere=e, n_render=2, n_index=index))
                self.s_rend_4.at[index] = (self.create_sphere(row['X'], row['Y'], row['Z'], row['id'],
                                           n_sphere=e, n_render=3, n_index=index))

    def set_orientations(self, indices=None):
        """
        Create all the orientations and set them to the corresponding renders for their posterior visualization with
        render_model
        Returns:
            None
        """

        if not indices:
            for e, val in enumerate(self.geo_model._orientations.df.iterrows()):
                index = val[0]
                row = val[1]
                self.o_rend_1.at[index] = (self.create_foliation(row['X'], row['Y'], row['Z'], row['id'],
                                           row['G_x'], row['G_y'], row['G_z'],
                                           n_plane=e, n_render=0, n_index=index))
                self.o_rend_2.at[index] = (self.create_foliation(row['X'], row['Y'], row['Z'], row['id'],
                                           row['G_x'], row['G_y'], row['G_z'],
                                           n_plane=e, n_render=1, n_index=index))
                self.o_rend_3.at[index] = (self.create_foliation(row['X'], row['Y'], row['Z'], row['id'],
                                           row['G_x'], row['G_y'], row['G_z'],
                                           n_plane=e, n_render=2, n_index=index))
                self.o_rend_4.at[index] = (self.create_foliation(row['X'], row['Y'], row['Z'], row['id'],
                                           row['G_x'], row['G_y'], row['G_z'],
                                           n_plane=e, n_render=3, n_index=index))
        else:
            for e, val in enumerate(self.geo_model._orientations.df.loc[np.atleast_1d(indices)].iterrows()):
                index = val[0]
                row = val[1]
                self.o_rend_1.at[index] = (self.create_foliation(row['X'], row['Y'], row['Z'], row['id'],
                                           row['G_x'], row['G_y'], row['G_z'],
                                           n_plane=e, n_render=0, n_index=index))
                self.o_rend_2.at[index] = (self.create_foliation(row['X'], row['Y'], row['Z'], row['id'],
                                           row['G_x'], row['G_y'], row['G_z'],
                                           n_plane=e, n_render=1, n_index=index))
                self.o_rend_3.at[index] = (self.create_foliation(row['X'], row['Y'], row['Z'], row['id'],
                                           row['G_x'], row['G_y'], row['G_z'],
                                           n_plane=e, n_render=2, n_index=index))
                self.o_rend_4.at[index] = (self.create_foliation(row['X'], row['Y'], row['Z'], row['id'],
                                           row['G_x'], row['G_y'], row['G_z'],
                                           n_plane=e, n_render=3, n_index=index))

    def create_slider_rep(self, min=0, max=10, val=0):

        slider_rep = vtk.vtkSliderRepresentation2D()
        slider_rep.SetMinimumValue(min)
        slider_rep.SetMaximumValue(max)
        slider_rep.SetValue(val)
        slider_rep.GetPoint1Coordinate().SetCoordinateSystemToDisplay()
        slider_rep.GetPoint1Coordinate().SetValue(0, 40)
        slider_rep.GetPoint2Coordinate().SetCoordinateSystemToDisplay()
        slider_rep.GetPoint2Coordinate().SetValue(300, 40)

        self.slider_w = vtk.vtkSliderWidget()
        self.slider_w.SetInteractor(self.interactor)
        self.slider_w.SetRepresentation(slider_rep)
        self.slider_w.SetCurrentRenderer(self.ren_list[0])
        self.slider_w.SetAnimationModeToAnimate()
        self.slider_w.On()
        self.slider_w.AddObserver('EndInteractionEvent', self.sliderCallback_traces)

    def create_slider_interactor(self, min=0, max=1, val=1):

        slider_rep = vtk.vtkSliderRepresentation2D()
        slider_rep.SetMinimumValue(min)
        slider_rep.SetMaximumValue(max)
        slider_rep.SetValue(val)
        slider_rep.GetPoint1Coordinate().SetCoordinateSystemToDisplay()
        slider_rep.GetPoint1Coordinate().SetValue(0, 60)
        slider_rep.GetPoint2Coordinate().SetCoordinateSystemToDisplay()
        slider_rep.GetPoint2Coordinate().SetValue(100, 60)
        slider_rep.SetTitleText('Interactor')

        self.slider_w = vtk.vtkSliderWidget()
        self.slider_w.SetInteractor(self.interactor)
        self.slider_w.SetRepresentation(slider_rep)
        self.slider_w.SetCurrentRenderer(self.ren_list[0])
        self.slider_w.SetAnimationModeToJump()

        self.slider_w.On()
        self.slider_w.AddObserver('EndInteractionEvent', self.sliderCallback_interactor)

    def sliderCallback_interactor(self, obj, event):
        if int(obj.GetRepresentation().GetValue()) is 0:
            self.interactor.ExitCallback()

    def sliderCallback_traces(self, obj, event):
        # TODO Check post class
        self.post.change_input_data(self.geo_model, obj.GetRepresentation().GetValue())
        try:
            # for surf in self.surf_rend_1:
            #     self.ren_list[0].RemoveActor(surf)
            #     self.ren_list[1].RemoveActor(surf)
            #     self.ren_list[2].RemoveActor(surf)
            #     self.ren_list[3].RemoveActor(surf)
            self.update_surfaces_real_time()
          #  ver, sim =
          #  self.set_surfaces(ver, sim)
        except AttributeError:
            print('no surf')
            pass
        try:
            for sph in zip(self.s_rend_1['val'], self.s_rend_2['val'], self.s_rend_3['val'],
                           self.s_rend_4['val'], self.geo_model._surface_points.df.iterrows()):

                row_i = sph[4][1]
                sph[0].PlaceWidget(row_i['X'] - sph[0].r_f, row_i['X'] + sph[0].r_f,
                                   row_i['Y'] - sph[0].r_f, row_i['Y'] + sph[0].r_f,
                                   row_i['Z'] - sph[0].r_f, row_i['Z'] + sph[0].r_f)

                sph[1].PlaceWidget(row_i['X'] - sph[1].r_f, row_i['X'] + sph[1].r_f,
                                   row_i['Y'] - sph[1].r_f, row_i['Y'] + sph[1].r_f,
                                   row_i['Z'] - sph[1].r_f, row_i['Z'] + sph[1].r_f)

                sph[2].PlaceWidget(row_i['X'] - sph[2].r_f, row_i['X'] + sph[2].r_f,
                                   row_i['Y'] - sph[2].r_f, row_i['Y'] + sph[2].r_f,
                                   row_i['Z'] - sph[2].r_f, row_i['Z'] + sph[2].r_f)

                sph[3].PlaceWidget(row_i['X'] - sph[3].r_f, row_i['X'] + sph[3].r_f,
                                   row_i['Y'] - sph[3].r_f, row_i['Y'] + sph[3].r_f,
                                   row_i['Z'] - sph[3].r_f, row_i['Z'] + sph[3].r_f)
        except AttributeError:
            pass
        try:
            for fol in (zip(self.f_rend_1, self.f_rend_2, self.f_rend_3, self.f_rend_4, self.geo_model._orientations.iterrows())):
                row_f = fol[4][1]

                fol[0].SetCenter(row_f['X'], row_f['Y'], row_f['Z'])
                fol[0].SetNormal(row_f['G_x'], row_f['G_y'], row_f['G_z'])

        except AttributeError:
            pass

    def sphereCallback(self, obj, event):
        """
        Function that rules what happens when we move a sphere. At the moment we update the other 3 renderers and
        update the pandas data frame.
        """
        #self.interactor.ExitCallback()

       # self.Callback_camera_reset()

        # Get new position of the sphere
        new_center = obj.GetCenter()

        # Get which sphere we are moving
        index = obj.index

        # Check what render we are working with
        render = obj.n_render

        # This must be the radius
        #r_f = obj.r_f

        self.SphereCallback_change_df(index, new_center)
        self.SphereCallbak_move_changes(index)

        if self.real_time:
            try:
                self.update_surfaces_real_time()
                #vertices, simpleces =
                #self.set_surfaces(vertices, simpleces)
            except AssertionError:
                print('Not enough data to compute the model')

    def Callback_camera_reset(self,  obj, event):

        # Resetting the xy camera when a sphere is moving to be able to change only 2D
        fp = self.ren_list[1].GetActiveCamera().GetFocalPoint()
        p = self.ren_list[1].GetActiveCamera().GetPosition()
        dist = np.sqrt((p[0] - fp[0]) ** 2 + (p[1] - fp[1]) ** 2 + (p[2] - fp[2]) ** 2)
        self.ren_list[1].GetActiveCamera().SetPosition(fp[0], fp[1], fp[2] + dist)
        self.ren_list[1].GetActiveCamera().SetViewUp(0.0, 1.0, 0.0)

        # Resetting the yz camera when a sphere is moving to be able to change only 2D
        fp = self.ren_list[2].GetActiveCamera().GetFocalPoint()
        p = self.ren_list[2].GetActiveCamera().GetPosition()
        dist = np.sqrt((p[0] - fp[0]) ** 2 + (p[1] - fp[1]) ** 2 + (p[2] - fp[2]) ** 2)
        self.ren_list[2].GetActiveCamera().SetPosition(fp[0] + dist, fp[1], fp[2])
        self.ren_list[2].GetActiveCamera().SetViewUp(0.0, -1.0, 0.0)
        self.ren_list[2].GetActiveCamera().Roll(90)

        # Resetting the xz camera when a sphere is moving to be able to change only 2D
        fp = self.ren_list[3].GetActiveCamera().GetFocalPoint()
        p = self.ren_list[3].GetActiveCamera().GetPosition()
        dist = np.sqrt((p[0] - fp[0]) ** 2 + (p[1] - fp[1]) ** 2 + (p[2] - fp[2]) ** 2)
        self.ren_list[3].GetActiveCamera().SetPosition(fp[0], fp[1] - dist, fp[2])
        self.ren_list[3].GetActiveCamera().SetViewUp(-1.0, 0.0, 0.0)
        self.ren_list[3].GetActiveCamera().Roll(90)

    def SphereCallback_change_df(self, index, new_center):
        index = np.atleast_1d(index)
        # Modify Pandas DataFrame
        self.geo_model.modify_surface_points(index, X=[new_center[0]], Y=[new_center[1]], Z=[new_center[2]])

    def SphereCallbak_move_changes(self, indices):
       # print(indices)
        df_changes = self.geo_model._surface_points.df.loc[np.atleast_1d(indices)][['X', 'Y', 'Z', 'id']]
        for index, df_row in df_changes.iterrows():
            new_center = df_row[['X', 'Y', 'Z']].values

            # Update renderers
            s1 = self.s_rend_1.loc[index, 'val']

            s1.PlaceWidget(new_center[0] - s1.r_f, new_center[0] + s1.r_f,
                           new_center[1] - s1.r_f, new_center[1] + s1.r_f,
                           new_center[2] - s1.r_f, new_center[2] + s1.r_f)

            s1.GetSphereProperty().SetColor(mcolors.hex2color(
                self.geo_model._surfaces.df.set_index('id')['color'][df_row['id']]))#self.C_LOT[df_row['id']])

            s2 = self.s_rend_2.loc[index, 'val']
            s2.PlaceWidget(new_center[0] - s2.r_f, new_center[0] + s2.r_f,
                           new_center[1] - s2.r_f, new_center[1] + s2.r_f,
                           new_center[2] - s2.r_f, new_center[2] + s2.r_f)

            s2.GetSphereProperty().SetColor(mcolors.hex2color(
                self.geo_model._surfaces.df.set_index('id')['color'][df_row['id']]))

            s3 = self.s_rend_3.loc[index, 'val']
            s3.PlaceWidget(new_center[0] - s3.r_f, new_center[0] + s3.r_f,
                           new_center[1] - s3.r_f, new_center[1] + s3.r_f,
                           new_center[2] - s3.r_f, new_center[2] + s3.r_f)

            s3.GetSphereProperty().SetColor(mcolors.hex2color(
                self.geo_model._surfaces.df.set_index('id')['color'][df_row['id']]))

            s4 = self.s_rend_4.loc[index, 'val']
            s4.PlaceWidget(new_center[0] - s4.r_f, new_center[0] + s4.r_f,
                           new_center[1] - s4.r_f, new_center[1] + s4.r_f,
                           new_center[2] - s4.r_f, new_center[2] + s4.r_f)

            s4.GetSphereProperty().SetColor(mcolors.hex2color(
                self.geo_model._surfaces.df.set_index('id')['color'][df_row['id']]))

    def SphereCallback_delete_point(self, ind_i):
        """
        Warning this deletion system will lead to memory leaks until the vtk object is reseted (hopefully). This is
        mainly a vtk problem to delete objects
        """
        ind_i = np.atleast_1d(ind_i)
        # Deactivating widget
        for i in ind_i:
            self.s_rend_1.loc[i, 'val'].Off()
            self.s_rend_2.loc[i, 'val'].Off()
            self.s_rend_3.loc[i, 'val'].Off()
            self.s_rend_4.loc[i, 'val'].Off()

        self.s_rend_1.drop(ind_i)
        self.s_rend_2.drop(ind_i)
        self.s_rend_3.drop(ind_i)
        self.s_rend_4.drop(ind_i)

    def planesCallback(self, obj, event):
        """
        Function that rules what happens when we move a plane. At the moment we update the other 3 renderers and
        update the pandas data frame
        """

      # self.Callback_camera_reset()

        # Get new position of the plane and GxGyGz
        new_center = obj.GetCenter()
        new_normal = obj.GetNormal()
        # Get which plane we are moving
        index = obj.index

        self.planesCallback_change_df(index, new_center, new_normal)
        self.planesCallback_move_changes(index)

        if self.real_time:
            # try:
            #     if self.real_time:
            #         for surf in self.surf_rend_1:
            #             self.ren_list[0].RemoveActor(surf)
            #             self.ren_list[1].RemoveActor(surf)
            #             self.ren_list[2].RemoveActor(surf)
            #             self.ren_list[3].RemoveActor(surf)
            # except AttributeError:
            #     pass

            try:
                self.update_surfaces_real_time()
              #  vertices, simpleces =
              #  self.set_surfaces(vertices, simpleces)
            except AssertionError:
                print('Not enough data to compute the model')

    def planesCallback_change_df(self, index, new_center, new_normal):

        # Modify Pandas DataFrame
        # update the gradient vector components and its location
        self.geo_model.modify_orientations(index, X=new_center[0], Y=new_center[1], Z=new_center[2],
                                           G_x=new_normal[0], G_y=new_normal[1], G_z=new_normal[2])
        # update the dip and azimuth values according to the new gradient
        # self.geo_model.calculate_orientations()

    def planesCallback_move_changes(self, indices):
        df_changes = self.geo_model._orientations.df.loc[np.atleast_1d(indices)][['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z', 'id']]
        for index, new_values_df in df_changes.iterrows():
            new_center = new_values_df[['X', 'Y', 'Z']].values
            new_normal = new_values_df[['G_x', 'G_y', 'G_z']].values
            new_source = vtk.vtkPlaneSource()
            new_source.SetCenter(new_center)
            new_source.SetNormal(new_normal)
            new_source.Update()

            plane1 = self.o_rend_1.loc[index, 'val']
          #  plane1.SetInputData(new_source.GetOutput())
            plane1.SetNormal(new_normal)
            plane1.SetCenter(new_center[0], new_center[1], new_center[2])
            plane1.GetPlaneProperty().SetColor(mcolors.hex2color(
                self.geo_model._surfaces.df.set_index('id')['color'][new_values_df['id']]))#self.C_LOT[new_values_df['id']])
            plane1.GetHandleProperty().SetColor(mcolors.hex2color(
                self.geo_model._surfaces.df.set_index('id')['color'][new_values_df['id']]))

            plane2 = self.o_rend_2.loc[index, 'val']
            plane2.SetInputData(new_source.GetOutput())
            plane2.SetNormal(new_normal)
            plane2.SetCenter(new_center[0], new_center[1], new_center[2])
            plane2.GetPlaneProperty().SetColor(mcolors.hex2color(
                self.geo_model._surfaces.df.set_index('id')['color'][new_values_df['id']]))
            plane2.GetHandleProperty().SetColor(mcolors.hex2color(
                self.geo_model._surfaces.df.set_index('id')['color'][new_values_df['id']]))

            plane3 = self.o_rend_3.loc[index, 'val']
            plane3.SetInputData(new_source.GetOutput())
            plane3.SetNormal(new_normal)
            plane3.SetCenter(new_center[0], new_center[1], new_center[2])
            plane3.GetPlaneProperty().SetColor(mcolors.hex2color(
                self.geo_model._surfaces.df.set_index('id')['color'][new_values_df['id']]))
            plane3.GetHandleProperty().SetColor(mcolors.hex2color(
                self.geo_model._surfaces.df.set_index('id')['color'][new_values_df['id']]))

            plane4 = self.o_rend_4.loc[index, 'val']
            plane4.SetInputData(new_source.GetOutput())
            plane4.SetNormal(new_normal)
            plane4.SetCenter(new_center[0], new_center[1], new_center[2])
            plane4.GetPlaneProperty().SetColor(mcolors.hex2color(
                self.geo_model._surfaces.df.set_index('id')['color'][new_values_df['id']]))
            plane4.GetHandleProperty().SetColor(mcolors.hex2color(
                self.geo_model._surfaces.df.set_index('id')['color'][new_values_df['id']]))

    def planesCallback_delete_point(self, ind_o):
        """
        Warning this deletion system will lead to memory leaks until the vtk object is reseted (hopefully). This is
        mainly a vtk problem to delete objects
        """
        ind_o = np.atleast_1d(ind_o)
        # Deactivating widget
        for i in ind_o:
            self.o_rend_1.loc[i, 'val'].Off()
            self.o_rend_2.loc[i, 'val'].Off()
            self.o_rend_3.loc[i, 'val'].Off()
            self.o_rend_4.loc[i, 'val'].Off()

        self.s_rend_1.drop(ind_o)
        self.s_rend_2.drop(ind_o)
        self.s_rend_3.drop(ind_o)
        self.s_rend_4.drop(ind_o)

    def create_ren_list(self):
        """
        Create a list of the 4 renderers we use. One general view and 3 cartersian projections
        Returns:
            list: list of renderers
        """

        # viewport dimensions setup
        xmins = [0, 0.6, 0.6, 0.6]
        xmaxs = [0.6, 1, 1, 1]
        ymins = [0, 0, 0.33, 0.66]
        ymaxs = [1, 0.33, 0.66, 1]

        # create list of renderers, set vieport values
        ren_list = []
        for i in range(self.n_ren):
            # append each renderer to list of renderers
            ren_list.append(vtk.vtkRenderer())
            # add each renderer to window
            self.renwin.AddRenderer(ren_list[-1])
            # set viewport for each renderer
            ren_list[-1].SetViewport(xmins[i], ymins[i], xmaxs[i], ymaxs[i])

        return ren_list

    def _create_cameras(self, extent, verbose=0):
        """
        Create the 4 cameras for each renderer
        """
        _e = extent
        _e_dx = _e[1] - _e[0]
        _e_dy = _e[3] - _e[2]
        _e_dz = _e[5] - _e[4]
        _e_d_avrg = (_e_dx + _e_dy + _e_dz) / 3
        _e_max = np.argmax(_e)

        # General camera
        model_cam = vtk.vtkCamera()
        model_cam.SetPosition(_e[_e_max] * 5, _e[_e_max] * 5, _e[_e_max] * 5)
        model_cam.SetFocalPoint(np.min(_e[0:2]) + _e_dx / 2,
                                np.min(_e[2:4]) + _e_dy / 2,
                                np.min(_e[4:]) + _e_dz / 2)

        model_cam.SetViewUp(-0.239, 0.155, 0.958)

        # XY camera RED
        xy_cam = vtk.vtkCamera()

        xy_cam.SetPosition(np.min(_e[0:2]) + _e_dx / 2,
                           np.min(_e[2:4]) + _e_dy / 2,
                           _e[_e_max] * 4)

        xy_cam.SetFocalPoint(np.min(_e[0:2]) + _e_dx / 2,
                             np.min(_e[2:4]) + _e_dy / 2,
                             np.min(_e[4:]) + _e_dz / 2)

        # YZ camera GREEN
        yz_cam = vtk.vtkCamera()
        yz_cam.SetPosition(_e[_e_max] * 4,
                           np.min(_e[2:4]) + _e_dy / 2,
                           np.min(_e[4:]) + _e_dz / 2)

        yz_cam.SetFocalPoint(np.min(_e[0:2]) + _e_dx / 2,
                             np.min(_e[2:4]) + _e_dy / 2,
                             np.min(_e[4:]) + _e_dz / 2)
        yz_cam.SetViewUp(0, -1, 0)
        yz_cam.Roll(90)

        # XZ camera BLUE
        xz_cam = vtk.vtkCamera()
        xz_cam.SetPosition(np.min(_e[0:2]) + _e_dx / 2,
                           - _e[_e_max] * 4,
                           np.min(_e[4:]) + _e_dz / 2)

        xz_cam.SetFocalPoint(np.min(_e[0:2]) + _e_dx / 2,
                             np.min(_e[2:4]) + _e_dy / 2,
                             np.min(_e[4:]) + _e_dz / 2)
        xz_cam.SetViewUp(0, 1, 0)
        xz_cam.Roll(0)

        # camera position debugging
        if verbose == 1:
            print("RED XY:", xy_cam.GetPosition())
            print("RED FP:", xy_cam.GetFocalPoint())
            print("GREEN YZ:", yz_cam.GetPosition())
            print("GREEN FP:", yz_cam.GetFocalPoint())
            print("BLUE XZ:", xz_cam.GetPosition())
            print("BLUE FP:", xz_cam.GetFocalPoint())

        return [model_cam, xy_cam, yz_cam, xz_cam]

    def set_camera_backcolor(self, color=None):
        """
        define background colors of the renderers
        """
        if color is None:
            color = (66 / 250, 66 / 250, 66 / 250)

        ren_color = [color for i in range(self.n_ren)]

        for i in range(self.n_ren):
            # set active camera for each renderer
            self.ren_list[i].SetActiveCamera(self.camera_list[i])
            # set background color for each renderer
            self.ren_list[i].SetBackground(ren_color[i][0], ren_color[i][1], ren_color[i][2])

    def _create_axes(self, camera, verbose=0, tick_vis=False):
        """
        Create the axes boxes
        """
        cube_axes_actor = vtk.vtkCubeAxesActor()
        cube_axes_actor.SetBounds(self.extent)
        cube_axes_actor.SetCamera(camera)
        if verbose == 1:
            print(cube_axes_actor.GetAxisOrigin())

        # set axes and label colors
        cube_axes_actor.GetTitleTextProperty(0).SetColor(1.0, 0.0, 0.0)
        cube_axes_actor.GetLabelTextProperty(0).SetColor(1.0, 0.0, 0.0)

        cube_axes_actor.GetTitleTextProperty(1).SetColor(0.0, 1.0, 0.0)
        cube_axes_actor.GetLabelTextProperty(1).SetColor(0.0, 1.0, 0.0)
        cube_axes_actor.GetTitleTextProperty(2).SetColor(0.0, 0.0, 1.0)
        cube_axes_actor.GetLabelTextProperty(2).SetColor(0.0, 0.0, 1.0)

        cube_axes_actor.DrawXGridlinesOn()
        cube_axes_actor.DrawYGridlinesOn()
        cube_axes_actor.DrawZGridlinesOn()

        if not tick_vis:
            cube_axes_actor.XAxisMinorTickVisibilityOff()
            cube_axes_actor.YAxisMinorTickVisibilityOff()
            cube_axes_actor.ZAxisMinorTickVisibilityOff()

        cube_axes_actor.SetXTitle("X")
        cube_axes_actor.SetYTitle("Y")
        cube_axes_actor.SetZTitle("Z")

        cube_axes_actor.SetXAxisLabelVisibility(1)
        cube_axes_actor.SetYAxisLabelVisibility(1)
        cube_axes_actor.SetZAxisLabelVisibility(1)

        # only plot grid lines furthest from viewpoint
        # ensure platform compatibility for the grid line options
        if sys.platform == "win32":
            cube_axes_actor.SetGridLineLocation(cube_axes_actor.VTK_GRID_LINES_FURTHEST)
        else:  # rather use elif == "linux" ? but what about other platforms
            try:  # apparently this can also go wrong on linux, maybe depends on vtk version?
                cube_axes_actor.SetGridLineLocation(cube_axes_actor.VTK_GRID_LINES_FURTHEST)
            except AttributeError:
                pass

        return cube_axes_actor

    def delete_surfaces(self):
        try:
            for surf in self.surf_rend_1:
                self.ren_list[0].RemoveActor(surf)
                self.ren_list[1].RemoveActor(surf)
                self.ren_list[2].RemoveActor(surf)
                self.ren_list[3].RemoveActor(surf)
        except AttributeError:
            pass

    def update_surfaces_real_time(self, delete=True):

        if delete is True:
            self.delete_surfaces()
        try:
            gp.compute_model(self.geo_model, sort_surfaces=False, compute_mesh=True)
        except IndexError:
            print('IndexError: Model not computed. Laking data in some surface')
        except AssertionError:
            print('AssertionError: Model not computed. Laking data in some surface')
        # try:
        #     v_l, s_l = self.geo_model.surfaces.df['vertices'], self.geo_model.surfaces.df['edges']
        # except IndexError:
        #     try:
        #         v_l, s_l = self.geo_model.surfaces.df['vertices'], self.geo_model.surfaces.df['edges']
        #     except IndexError:
        #         v_l, s_l = self.geo_model.surfaces.df['vertices'], self.geo_model.surfaces.df['edges']

        self.set_surfaces(self.geo_model._surfaces)
        # if self.geo_model.solutions.geological_map is not None:
        #     try:
        #         self.set_geological_map()
        #     except AttributeError:
        #         pass
        return True

    @staticmethod
    def export_vtk_lith_block(geo_data, lith_block=None, path="./default"):
        """
        Export data to a vtk file for posterior visualizations

        Args:
            geo_data(:class:''): All values of a DataManagement object
            block(numpy.array): 3D array containing the lithology block
            path (str): path to the location of the vtk

        Returns:
            None
        """



        # Dimensions

        nx, ny, nz = geo_data._grid.regular_grid.resolution

        lx = geo_data._grid.regular_grid.extent[1] - geo_data._grid.regular_grid.extent[0]
        ly = geo_data._grid.regular_grid.extent[3] - geo_data._grid.regular_grid.extent[2]
        lz = geo_data._grid.regular_grid.extent[5] - geo_data._grid.regular_grid.extent[4]

        dx, dy, dz = lx / nx, ly / ny, lz / nz

        ncells = nx * ny * nz

        npoints = (nx + 1) * (ny + 1) * (nz + 1)

        # Coordinates
        x = np.arange(geo_data._grid.regular_grid.extent[0], geo_data._grid.regular_grid.extent[1] + 0.1, dx, dtype='float64')

        y = np.arange(geo_data._grid.regular_grid.extent[2], geo_data._grid.regular_grid.extent[3] + 0.1, dy, dtype='float64')

        z = np.arange(geo_data._grid.regular_grid.extent[4], geo_data._grid.regular_grid.extent[5] + 0.1, dz, dtype='float64')

        if lith_block is None:
            lith_block = geo_data.solutions.lith_block
        lith = lith_block.reshape((nx, ny, nz))
        out = pv.RectilinearGrid(x,y,z)
        out['Lithology'] = lith
        out.save(path+'_lith_block.vtr')

    @staticmethod
    def export_vtk_surfaces(geo_model, vertices=None, simplices=None, path=None, name='_surfaces', alpha=1):
        """
        Export data to a vtk file for posterior visualizations

        Args:
            geo_model(gempy.InputData): All values of a DataManagement object
            block(numpy.array): 3D array containing the lithology block
            path (str): path to the location of the vtk

        Returns:
            None
        """
        import vtk

        if vertices is None or simplices is None:
            vertices = geo_model.solutions.vertices
            simplices = geo_model.solutions.edges

        s_n = 0
        for e, values in enumerate(vertices):
            # setup points and vertices
            s_n += 1
            Points = vtk.vtkPoints()
            Triangles = vtk.vtkCellArray()
            Triangle = vtk.vtkTriangle()
            for p in values:
                Points.InsertNextPoint(p)

            # Unfortunately in this simple example the following lines are ambiguous.
            # The first 0 is the index of the triangle vertex which is ALWAYS 0-2.
            # The second 0 is the index into the point (geometry) array, so this can range from 0-(NumPoints-1)
            # i.e. a more general statement is triangle->GetPointIds()->SetId(0, PointId);
            for i in simplices[e]:
                Triangle.GetPointIds().SetId(0, i[0])
                Triangle.GetPointIds().SetId(1, i[1])
                Triangle.GetPointIds().SetId(2, i[2])

                Triangles.InsertNextCell(Triangle)

            polydata = vtk.vtkPolyData()
            polydata.SetPoints(Points)
            polydata.SetPolys(Triangles)

            polydata.Modified()
            if vtk.VTK_MAJOR_VERSION <= 5:
                polydata.Update()

            writer = vtk.vtkXMLPolyDataWriter();

            # Add colors
            surf_mapper = vtk.vtkPolyDataMapper()
            surf_mapper.SetInputData(polydata)
            surf_mapper.Update()

            surf_actor = vtk.vtkActor()
            surf_actor.SetMapper(surf_mapper)
            surf_actor.GetProperty().SetColor(mcolors.hex2color(geo_model._surfaces.df.set_index('id')['color'][s_n]))
            surf_actor.GetProperty().SetOpacity(alpha)

            if not path:
                path = "./default_"

            writer.SetFileName(path+'_surfaces_'+str(e)+'.vtp')
            if vtk.VTK_MAJOR_VERSION <= 5:
                writer.SetInput(polydata)
            else:
                writer.SetInputData(polydata)
            writer.Write()


class GemPyvtkInteract(vtkVisualization):

    def resume(self):
        self.interactor.Start()

    def restart(self, render_surfaces=True, **kwargs):
        self.set_surface_points()
        self.set_orientations()
        if render_surfaces is True:
            self.set_surfaces(self.geo_model._surfaces)

        self.render_model(**kwargs)

    def set_real_time_on(self):

        self.real_time = True

    def set_real_time_off(self):
        self.real_time = False

    def render_move_surface_points(self, indices):
        self.SphereCallbak_move_changes(indices)
       # print('vtk-gempy real time is:' + str(self.real_time))
        if self.real_time is True:
            self.update_surfaces_real_time()
        self.interactor.Render()

    def render_add_surface_points(self, indices):
        self.set_surface_points(indices)
       # print('vtk-gempy real time is:' + str(self.real_time))
        if self.real_time is True:
            self.update_surfaces_real_time()
        self.interactor.Render()

    def render_delete_surface_points(self, indices):
        self.SphereCallback_delete_point(indices)
      #  print('vtk-gempy real time is:' + str(self.real_time))
        if self.real_time is True:
            self.update_surfaces_real_time()
        self.interactor.Render()

    def render_move_orientations(self, indices):
        self.planesCallback_move_changes(indices)
      #  print('vtk-gempy real time is:' + str(self.real_time))
        if self.real_time is True:
            self.update_surfaces_real_time()
        self.interactor.Render()

    def render_add_orientations(self, indices):
        self.set_orientations(indices)
     #   print('vtk-gempy real time is:' + str(self.real_time))
        if self.real_time is True:
            self.update_surfaces_real_time()
        self.interactor.Render()

    def render_delete_orientations(self, indices):
        self.planesCallback_delete_point(indices)
     #   print('vtk-gempy real time is:' + str(self.real_time))
        if self.real_time is True:
            self.update_surfaces_real_time()
        self.interactor.Render()

    def render_surfaces(self, alpha=1):
        self.delete_surfaces()
        self.set_surfaces(self.geo_model._surfaces, alpha=alpha)
        self.interactor.Render()

    def render_topography(self):
        try:
            self.ren_list[0].RemoveActor(self.topography_surface)
            self.ren_list[1].RemoveActor(self.topography_surface)
            self.ren_list[2].RemoveActor(self.topography_surface)
            self.ren_list[3].RemoveActor(self.topography_surface)
        except AttributeError:
            pass

      #  print('vtk-gempy real time is:' +str(self.real_time))
        if self.real_time is True:
            self.update_surfaces_real_time()
        self.set_topography()
        self.interactor.Render()

    def update_model(self):
        if self.real_time is True:
            self.update_surfaces_real_time()
            self.interactor.Render()


class steno3D():
    def __init__(self, geo_data, project, **kwargs ):
        if STENO_IMPORT is False:
            raise ImportError( 'Steno 3D package is not installed. No 3D online visualization available.')
        description = kwargs.get('description', 'Nothing')

        self._data = geo_data
        self.surfaces = pn.DataFrame.from_dict(geo_data.get_surface_number(), orient='index')


        steno3d.login()

        self.proj = steno3d.Project(
            title=project,
            description=description,
            public=True,
        )

    def plot3D_steno_grid(self, block, plot=False, **kwargs):


        mesh = steno3d.Mesh3DGrid(h1=np.ones(self._data.resolution[0]) * (self._data.extent[1] - self._data.extent[0]) /
                                                                         (self._data.resolution[0] - 1),
                                  h2=np.ones(self._data.resolution[1]) * (self._data.extent[3] - self._data.extent[2]) /
                                                                         (self._data.resolution[1] - 1),
                                  h3=np.ones(self._data.resolution[2]) * (self._data.extent[5] - self._data.extent[4]) /
                                                                         (self._data.resolution[2] - 1),
                                  O=[self._data.extent[0], self._data.extent[2], self._data.extent[4]])

        data = steno3d.DataArray(
            title='Lithologies_block',
            array=block)

        vol = steno3d.Volume(project=self.proj, mesh=mesh, data=[dict(location='CC', data=data)])
       # vol.upload()

        if plot:
            return vol.plot()

    def plot3D_steno_surface(self, ver, sim):

        for surface in self.surfaces.iterrows():
            if surface[1].values[0] is 0:
                pass

            #for vertices, simpleces in zip(ver[surface[1].values[0]], sim[surface[1].values[0]]):
            surface_mesh = steno3d.Mesh2D(
                vertices=ver[surface[1].values[0]-1],
                triangles=sim[surface[1].values[0]-1],
                opts=dict(wireframe=False)
            )
            surface_obj = steno3d.Surface(
                project=self.proj,
                title='Surface: {}'.format(surface[0]),
                mesh=surface_mesh,
                opts=dict(opacity=1)
            )


def get_fault_ellipse_params(fault_points:np.ndarray):
    """Get the fault ellipse parameters a and b from griven fault points (should
    be the rotated ones.

    Args:
        fault_points (np.ndarray): Fault points

    Returns:
        (tuple) main axis scalars of fault ellipse a,b
    """
    a = (fault_points[:, 0].max() - fault_points[:, 0].min()) / 2
    b = (fault_points[:, 1].max() - fault_points[:, 1].min()) / 2
    return a, b


def get_fault_rotation_objects(geo_model, fault:str):
    """Gets fault rotation objects: rotation matrix U, the rotated fault points,
    rotated centroid, and the ellipse parameters a and b.

    Args:
        geo_model (gempy.core.model.Project): gempy geo_model object
        fault (str): Name of the fault surface.

    Returns:
        U (np.ndarray): Rotation matrix.
        rfpts (np.ndarray): Rotated fault points.
        rctr (np.array): Centroid of the rotated fault points.
        a (float): Horizontal ellipse parameter.
        b (float): Vertical ellipse parameter.
    """
    filter_ = geo_model._surface_points.df.surface == fault
    fpts = geo_model._surface_points.df[filter_][["X", "Y", "Z"]].values.T
    ctr = np.mean(fpts, axis=1)
    x = fpts - ctr.reshape((-1, 1))
    M = np.dot(x, x.T)
    U = np.linalg.svd(M)
    rfpts = np.dot(fpts.T, U[0])
    # rfpts = np.dot(rfpts, U[-1])
    rctr = np.mean(rfpts, axis=0)

    a, b = get_fault_ellipse_params(rfpts)
    return U, rfpts, rctr, a, b


def cut_finite_fault_surfaces(geo_model, ver:dict, sim:dict):
    """Cut vertices and simplices for finite fault surfaces to finite fault ellipse

    Args:
        geo_model (gempy.core.model.Project): gempy geo_model object
        ver (dict): Dictionary with surfaces as keys and vertices ndarray as values.
        sim (dict): Dictionary with surfaces as keys and simplices ndarray as values.

    Returns:
        ver, sim (dict, dict): Updated vertices and simplices with finite fault
            surfaces cut to ellipses.
    """
    from scipy.spatial import Delaunay
    from copy import copy

    finite_ver = copy(ver)
    finite_sim = copy(sim)

    finite_fault_series = list(geo_model._faults.df[geo_model._faults.df["isFinite"] == True].index)
    finite_fault_surfaces = list(
        geo_model._surfaces.df[geo_model._surfaces.df._stack == finite_fault_series].surface.unique())

    for fault in finite_fault_surfaces:
        U, fpoints_rot, fctr_rot, a, b = get_fault_rotation_objects(geo_model, "Fault 1")
        rpoints = np.dot(ver[fault], U[0])
        # rpoints = np.dot(rpoints, U[-1])
        r = (rpoints[:, 0] - fctr_rot[0]) ** 2 / a ** 2 + (rpoints[:, 1] - fctr_rot[1]) ** 2 / b ** 2

        finite_ver[fault] = finite_ver[fault][r < 1]
        delaunay = Delaunay(finite_ver[fault])
        finite_sim[fault] = delaunay.simplices
        # finite_sim[fault] = finite_sim[fault][np.isin(sim[fault], np.argwhere(r<0.33))]

    return finite_ver, finite_sim
