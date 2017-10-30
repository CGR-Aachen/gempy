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
"""

import numpy as np
import theano
import theano.tensor as T
from scipy.constants import G


class GeoPhysicsPreprocessing_pro(object):
    def __init__(self, interp_data, ai_extent, ai_resolution, ai_z=None, range_max=None):

        self.interp_data = interp_data
        self.ai_extent = np.array(ai_extent)
        self.ai_resolution = np.array(ai_resolution)
        self.model_grid = interp_data.geo_data_res.grid.grid

        self.eu = self.compile_eu_f()

        if ai_z is None:
            ai_z = self.model_grid[:, 2].max()

        self.airborne_plane = self.set_airborne_plane(ai_z, self.ai_resolution)

        self.model_resolution = interp_data.geo_data_res.resolution[0] * \
                                interp_data.geo_data_res.resolution[1] * \
                                interp_data.geo_data_res.resolution[2]
        self.vox_size = self.set_vox_size()


        if range_max is None:
            self.range_max = self.default_range()
        else:
            self.range_max = range_max

        # Boolean array that select the voxels that affect each measurement. Size is measurement times resolution
        self.b_all = np.zeros((0, self.model_resolution), dtype=bool)

    def compute_gravity(self, n_chunck_o=25):
        # Init
        i_0 = 0
        n_measurements = self.ai_resolution[0] * self.ai_resolution[1]
        loop_list = np.linspace(0, n_measurements, int(n_measurements/n_chunck_o)+1,
                                      endpoint=True, dtype=int)

        n_chunck_l = loop_list[1:] - loop_list[:-1]

        for e, i_1 in enumerate(loop_list[1:]):

            n_chunck = n_chunck_l[e]
            # print(i_0, i_1)
            # Select the number of measurements to compute in this iteration
            airborne_plane_s = self.airborne_plane[i_0:i_1]
            airborne_plane_s[:, 2] += 0.002
            dist = self.eu(airborne_plane_s, self.model_grid)

            # Boolean selection
            b = dist < self.range_max

            # Release memory
            del dist

            # Save selection
            self.b_all = np.vstack((self.b_all, b))

            # Compute cartesian distances from measurements to each voxel

            model_grid_rep = np.repeat(self.model_grid, n_chunck, axis=1)
            s_gr_x = (
                model_grid_rep[:, :n_chunck].T[b].reshape(n_chunck, -1) -
                airborne_plane_s[:, 0].reshape(n_chunck, -1)).astype('float')
            s_gr_y = (
                model_grid_rep[:, n_chunck:2*n_chunck].T[b].reshape(n_chunck, -1) -
                airborne_plane_s[:, 1].reshape(n_chunck, -1)).astype('float')
            s_gr_z = (
                model_grid_rep[:, 2*n_chunck:].T[b].reshape(n_chunck, -1) -
                airborne_plane_s[:, 2].reshape(n_chunck, -1)).astype('float')

            # getting the coordinates of the corners of the voxel...
            x_cor = np.stack((s_gr_x - self.vox_size[0], s_gr_x + self.vox_size[0]), axis=2)
            y_cor = np.stack((s_gr_y - self.vox_size[1], s_gr_y + self.vox_size[1]), axis=2)
            z_cor = np.stack((s_gr_z - self.vox_size[2], s_gr_z + self.vox_size[2]), axis=2)

            # ...and prepare them for a vectorial op
            x_matrix = np.repeat(x_cor, 4, axis=2)
            y_matrix = np.tile(np.repeat(y_cor, 2, axis=2), (1, 1, 2))
            z_matrix = np.tile(z_cor, (1, 1, 4))

            # Distances to each corner of the voxel
            s_r = np.sqrt(x_matrix ** 2 + y_matrix ** 2 + z_matrix ** 2)

            # This is the vector that determines the sign of the corner of the voxel
            mu = np.array([1, -1, -1, 1, -1, 1, 1, -1])

            # Component z of each voxel
            tz = np.sum(- 1 * mu * (
                x_matrix * np.log(y_matrix + s_r) +
                y_matrix * np.log(x_matrix + s_r) -
                z_matrix * np.arctan(x_matrix * y_matrix / (z_matrix * s_r))),
                        axis=2)

            # Stacking the precomputation
            if i_0 == 0:
                tz_all = tz

            else:
                tz_all = np.vstack((tz_all, tz))

            i_0 = i_1

        return tz_all, np.ravel(self.b_all)

    def default_range(self):
        # Max range to select voxels
        range_ = (self.model_grid[:, 2].max() - self.model_grid[:, 2].min()) * 0.9
        return range_

    @staticmethod
    def compile_eu_f():
        # Compile Theano function
        x_1 = T.matrix()
        x_2 = T.matrix()

        sqd = T.sqrt(T.maximum(
            (x_1 ** 2).sum(1).reshape((x_1.shape[0], 1)) +
            (x_2 ** 2).sum(1).reshape((1, x_2.shape[0])) -
            2 * x_1.dot(x_2.T), 0
        ))
        eu = theano.function([x_1, x_2], sqd, allow_input_downcast=True)
        return eu

    def set_airborne_plane(self, z, ai_resolution):

        # TODO Include all in the loop. At the moment I am tiling all grids and is useless
        # Rescale z
        z_res = (z-self.interp_data.centers[2])/self.interp_data.rescaling_factor + 0.5001
        ai_extent_rescaled = (self.ai_extent - np.repeat(self.interp_data.centers, 2)) / \
                              self.interp_data.rescaling_factor + 0.5001

        # Create xy meshgrid
        xy = np.meshgrid(np.linspace(ai_extent_rescaled.iloc[0], ai_extent_rescaled.iloc[1], self.ai_resolution[0]),
                         np.linspace(ai_extent_rescaled.iloc[2], ai_extent_rescaled.iloc[3], self.ai_resolution[1]))
        z = np.ones(self.ai_resolution[0]*self.ai_resolution[1])*z_res

        # Transformation
        xy_ravel = np.vstack(map(np.ravel, xy))
        airborne_plane = np.vstack((xy_ravel, z)).T.astype(self.interp_data.dtype)

        # Now we need to find what point of the grid are the closest to this grid and choose them. This is important in
        # order to obtain regular matrices when we set a maximum range of effect

        # First we compute the distance between the airborne plane to the grid and choose those closer
        i_0 = 0
        for i_1 in np.arange(25, self.ai_resolution[0] * self.ai_resolution[1] + 1 + 25, 25, dtype=int):

            d = self.eu(self.model_grid.astype('float'), airborne_plane[i_0:i_1])

            if i_0 == 0:
                ab_g = self.model_grid[np.argmin(d, axis=0)]
            else:
                ab_g = np.vstack((ab_g, self.model_grid[np.argmin(d, axis=0)]))

            i_0 = i_1

        return ab_g

    def set_vox_size(self):

        x_extent = self.interp_data.extent_rescaled.iloc[1] - self.interp_data.extent_rescaled.iloc[0]
        y_extent = self.interp_data.extent_rescaled.iloc[3] - self.interp_data.extent_rescaled.iloc[2]
        z_extent = self.interp_data.extent_rescaled.iloc[5] - self.interp_data.extent_rescaled.iloc[4]
        vox_size = np.array([x_extent, y_extent, z_extent]) / self.interp_data.geo_data_res.resolution
        return vox_size



# class GeoPhysicsPreprocessing(object):
#
#     # TODO geophysics grid different that modeling grid
#     def __init__(self, interp_data, z, ai_extent, res_grav=[5, 5], n_cells=1000, grid=None, mode='range'):
#         """
#
#         Args:
#             interp_data: Some model metadata such as rescaling factor or dtype
#             z:
#             res_grav: resolution of the gravity
#             n_cells:
#             grid: This is the model grid
#         """
#         self.interp_data = interp_data
#         self.max_range = (interp_data.extent_rescaled['Z'].max() - interp_data.extent_rescaled['Z'].min())*0.9
#         print(self.max_range)
#         self.mode = mode
#
#         self.res_grav = res_grav
#         self.z = z
#         self.ai_extent = ai_extent
#
#         self.compile_th_fun()
#         self.n_cells = n_cells
#         self.vox_size = self.set_vox_size()
#
#         if not grid:
#             self.grid = interp_data.geo_data_res.grid.grid.astype(self.interp_data.dtype)
#         else:
#             self.grid = grid.astype(self.interp_data.dtype)
#
#         self.airborne_plane = self.set_airborne_plane(z, res_grav)
#         self.n_measurements = self.res_grav[0]*self.res_grav[1]
#         # self.closest_cells_index = self.set_closest_cells()
#         # self.tz = self.z_decomposition()
#
#     def looping_z_decomp(self, chunk_size):
#         n_points_0 = 0
#         final_tz = np.zeros((self.n_cells, 0))
#
#         if self.mode is 'n_closest':
#             self.closest_cells_all = np.zeros((self.n_cells, 0), dtype=np.int)
#         if self.mode is 'range':
#             self.closest_cells_all = False
#
#         n_chunks = int(self.res_grav[0]*self.res_grav[1]/ chunk_size)
#
#         for n_points_1 in np.linspace(chunk_size, self.res_grav[0]*self.res_grav[1], n_chunks,
#                                       endpoint=True, dtype=int):
#             self.n_measurements = n_points_1 - n_points_0
#             self.airborne_plane_op = self.airborne_plane[n_points_0:n_points_1]
#             if not self.closest_cells_all:
#                 self.closest_cells_all = self.set_closest_cells()
#             else:
#                 self.closest_cells_all = np.dstack((self.closest_cells_all, self.set_closest_cells()))
#             tz = self.z_decomposition()
#             final_tz = np.hstack((final_tz, tz))
#             n_points_0 = n_points_1
#
#         return final_tz
#
#     def set_airborne_plane(self, z, res_grav):
#
#         # Rescale z
#         z_res = (z-self.interp_data.centers[2])/self.interp_data.rescaling_factor + 0.5001
#         ai_extent_rescaled = (self.ai_extent - np.repeat(self.interp_data.centers, 2)) / \
#                               self.interp_data.rescaling_factor + 0.5001
#
#         # Create xy meshgrid
#         xy = np.meshgrid(np.linspace(ai_extent_rescaled.iloc[0], ai_extent_rescaled.iloc[1], res_grav[0]),
#                          np.linspace(ai_extent_rescaled.iloc[2], ai_extent_rescaled.iloc[3], res_grav[1]))
#         z = np.ones(res_grav[0]*res_grav[1])*z_res
#
#         # Transformation
#         xy_ravel = np.vstack(map(np.ravel, xy))
#         airborne_plane = np.vstack((xy_ravel, z)).T.astype(self.interp_data.dtype)
#
#         return airborne_plane
#
#     def compile_th_fun(self):
#
#         # Theano function
#         x_1 = T.matrix()
#         x_2 = T.matrix()
#
#         sqd = T.sqrt(T.maximum(
#             (x_1 ** 2).sum(1).reshape((x_1.shape[0], 1)) +
#             (x_2 ** 2).sum(1).reshape((1, x_2.shape[0])) -
#             2 * x_1.dot(x_2.T), 0
#         ))
#         self.eu = theano.function([x_1, x_2], sqd, allow_input_downcast=True)
#
#     def compute_distance(self):
#         # if the resolution is too high is going to consume too much memory
#
#
#         # Distance
#         r = self.eu(self.grid, self.airborne_plane_op)
#
#         return r
#
#     def set_closest_cells(self):
#
#         r = self.compute_distance()
#
#         # This is a integer matrix at least
#         if self.mode =='n_closest':
#             self.closest_cells_index = np.argsort(r, axis=0)[:self.n_cells, :]
#             # DEP?-- I need to make an auxiliary index for axis 1
#             self._axis_1 = np.indices((self.n_cells, self.n_measurements))[1]
#
#             # I think it is better to save it in memory since recompute distance can be too heavy
#             self.selected_dist = r[self.closest_cells_index, self._axis_1]
#
#         if self.mode == 'range':
#             self.closest_cells_index = np.where(r < self.max_range)
#             self.selected_dist = r[self.closest_cells_all]
#
#         return self.closest_cells_index
#
#     def select_grid(self):
#
#         selected_grid_x = np.zeros((0, self.n_cells))
#         selected_grid_y = np.zeros((0, self.n_cells))
#         selected_grid_z = np.zeros((0, self.n_cells))
#
#         n_cells = self.closest_cells_index[0].shape[0]/self.n_measurements
#         i_0 = 0
#         # I am going to loop it in order to keep low memory (first loop in gempy?)
#         for i_1 in np.linspace(n_cells, self.closest_cells_index[0].shape[0], self.n_measurements+1): #range(self.n_measurements):
#             selected_grid_x = np.vstack((selected_grid_x, self.grid[:, 0][self.closest_cells_index[0][i_0:i_1]]))
#             selected_grid_y = np.vstack((selected_grid_y, self.grid[:, 1][self.closest_cells_index[0][i_0:i_1]]))
#             selected_grid_z = np.vstack((selected_grid_z, self.grid[:, 2][self.closest_cells_index[0][i_0:i_1]]))
#
#         return selected_grid_x.T, selected_grid_y.T, selected_grid_z.T
#
#     def set_vox_size(self):
#
#         x_extent = self.interp_data.extent_rescaled.iloc[1] - self.interp_data.extent_rescaled.iloc[0]
#         y_extent = self.interp_data.extent_rescaled.iloc[3] - self.interp_data.extent_rescaled.iloc[2]
#         z_extent = self.interp_data.extent_rescaled.iloc[5] - self.interp_data.extent_rescaled.iloc[4]
#         vox_size = np.array([x_extent, y_extent, z_extent]) / self.interp_data.geo_data_res.resolution
#         return vox_size
#
#     def z_decomposition(self):
#
#         # We get the 100 closest point of our grid!
#         s_gr_x, s_gr_y, s_gr_z = self.select_grid()
#
#         # This is the euclidean distances between the plane (size size chunk) and the selected voxels. Repeated 8 time
#         # for the next eq
#         s_r = np.repeat(np.expand_dims(self.selected_dist, axis=2), 8, axis=2)
#
#         # x_cor = np.expand_dims(np.dstack((s_gr_x - self.vox_size[0], s_gr_x + self.vox_size[0])).T, axis=2)
#         # y_cor = np.expand_dims(np.dstack((s_gr_y - self.vox_size[1], s_gr_y + self.vox_size[1])).T, axis=2)
#         # z_cor = np.expand_dims(np.dstack((s_gr_z - self.vox_size[2], s_gr_z + self.vox_size[2])).T, axis=2)
#
#         # Now we need the coordinates not at the center of the voxel but at the sides
#         x_cor = np.stack((s_gr_x - self.vox_size[0], s_gr_x + self.vox_size[0]), axis=2)
#         y_cor = np.stack((s_gr_y - self.vox_size[1], s_gr_y + self.vox_size[1]), axis=2)
#         z_cor = np.stack((s_gr_z - self.vox_size[2], s_gr_z + self.vox_size[2]), axis=2)
#
#         # Now we expand them in the 8 combinations. Equivalent to 3 nested loops
#         #  see #TODO add paper
#         x_matrix = np.repeat(x_cor, 4, axis=2)
#         y_matrix = np.tile(np.repeat(y_cor, 2, axis=2), (1, 1, 2))
#         z_matrix = np.tile(z_cor, (1, 1, 4))
#
#         mu = np.array([1, -1, -1, 1, -1, 1, 1, -1])
#
#         tz = np.sum(- G/self.interp_data.rescaling_factor * mu * (
#                 x_matrix * np.log(y_matrix + s_r) +
#                 y_matrix * np.log(x_matrix + s_r) -
#                 z_matrix * np.arctan(x_matrix * y_matrix /
#                                     (z_matrix * s_r))), axis=2)
#
#         return tz
#
#     # This has to be also a theano function
#     def compute_gravity(self, block, tz):
#
#         block_matrix = np.tile(block, (1, self.res_grav[0] * self.res_grav[1]))
#         block_matrix_sel = block_matrix[self.closest_cells_all,
#                                         np.indices((self.n_cells, self.res_grav[0] * self.res_grav[1]))[1]]
#         grav = block_matrix_sel
#
#         return grav
#
#
#     # DEP?
#     def set_airbore_plane(self, z, res_grav):
#
#         # Rescale z
#         z_res = (z-self.centers[2])/self.rescaling_factor + 0.5001
#
#         # Create xy meshgrid
#         xy = np.meshgrid(np.linspace(self.extent_rescaled.iloc[0],
#                                      self.extent_rescaled.iloc[1], res_grav[0]),
#                          np.linspace(self.extent_rescaled.iloc[2],
#                                      self.extent_rescaled.iloc[3], res_grav[1]))
#         z = np.ones(res_grav[0]*res_grav[1])*z_res
#
#         # Transformation
#         xy_ravel = np.vstack(map(np.ravel, xy))
#         airborne_plane = np.vstack((xy_ravel, z)).T.astype(self.dtype)
#
#         return airborne_plane