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


DEP-- I need to update this string
Function that generates the symbolic code to perform the interpolation. Calling this function creates
 both the theano functions for the potential field and the block.

Returns:
    theano function for the potential field
    theano function for the block
"""
import theano
import theano.tensor as T
import numpy as np
import sys
from .theano_graph import TheanoGeometry, TheanoOptions

theano.config.openmp_elemwise_minsize = 10000
theano.config.openmp = True

theano.config.optimizer = 'fast_compile'
theano.config.floatX = 'float64'
theano.config.on_opt_error = 'ignore'

theano.config.exception_verbosity = 'high'
theano.config.compute_test_value = 'off'
theano.config.profile_memory = False
theano.config.scan.debug = False
theano.config.profile = False


class TheanoGraphPro(object):
    def __init__(self, output='geology', optimizer='fast_compile', verbose=[0], dtype='float32',
                 is_fault=None, is_lith=None):
        # OPTIONS
        # -------
        if verbose is np.nan:
            self.verbose = [None]
        else:
            self.verbose = verbose
        self.dot_version = False

        theano.config.floatX = dtype

        # CONSTANT PARAMETERS FOR ALL SERIES
        # KRIGING
        # -------
        self.a_T = theano.shared(np.cast[dtype](-1.), "Range")
        self.c_o_T = theano.shared(np.cast[dtype](-1.), 'Covariance at 0')
        self.nugget_effect_grad_T = theano.shared(np.cast[dtype](-1), 'Nugget effect of gradients')
        self.nugget_effect_scalar_T = theano.shared(np.cast[dtype](-1), 'Nugget effect of scalar')
        self.n_universal_eq_T = theano.shared(np.zeros(5, dtype='int32'), "Grade of the universal drift")
        self.n_universal_eq_T_op = theano.shared(3)

        # They weight the contribution of the surface_points against the orientations.
        self.i_reescale = theano.shared(np.cast[dtype](4.))
        self.gi_reescale = theano.shared(np.cast[dtype](2.))

        # Number of dimensions. Now it is not too variable anymore
        self.n_dimensions = 3

        # This is not accumulative
        self.number_of_points_per_surface_T = theano.shared(np.zeros(3, dtype='int32'),
                                                            'Number of points per surface used to split rest-ref')
        self.number_of_points_per_surface_T_op = T.vector('Number of points per surface used to split rest-ref',
                                                          dtype='int32')
        self.npf = T.cumsum(T.concatenate((T.stack([0]), self.number_of_points_per_surface_T[:-1])))
        self.npf_op = self.npf
        self.npf.name = 'Number of points per surfaces after rest-ref. This is used for finding the different' \
                           'surface points withing a layer.'

        # COMPUTE WEIGHTS
        # ---------
        # VARIABLES
        # ---------
        self.dips_position_all = T.matrix("Position of the dips")
        self.dip_angles_all = T.vector("Angle of every dip")
        self.azimuth_all = T.vector("Azimuth")
        self.polarity_all = T.vector("Polarity")

        self.surface_points_all = T.matrix("All the surface_points points at once")

        self.len_points = self.surface_points_all.shape[0] - self.number_of_points_per_surface_T.shape[0]

        # Tiling dips to the 3 spatial coordinations
        self.dips_position = self.dips_position_all
        self.dips_position_tiled = T.tile(self.dips_position, (self.n_dimensions, 1))

        # These are subsets of the data for each series. I initialized them as the whole arrays but then they will take
        # the data of every potential field
        self.dip_angles = self.dip_angles_all
        self.azimuth = self.azimuth_all
        self.polarity = self.polarity_all
       # self.surface_points_op = self.surface_points_all

        self.ref_layer_points_all = self.set_rest_ref_matrix(self.number_of_points_per_surface_T)[0]
        self.rest_layer_points_all = self.set_rest_ref_matrix(self.number_of_points_per_surface_T)[1]

        self.ref_layer_points = self.ref_layer_points_all
        self.rest_layer_points = self.rest_layer_points_all

        # self.ref_layer_points = self.set_rest_ref_matrix()[0]
        # self.rest_layer_points = self.set_rest_ref_matrix()[1]

       # self.fault_drift = T.matrix('Drift matrix due to faults')
        self.fault_matrix = T.matrix('Full block matrix for x_to_interpolate')

        interface_loc = self.fault_matrix.shape[1] - 2 * self.len_points
        self.fault_drift_at_surface_points_rest = self.fault_matrix[
                                                  :, interface_loc: interface_loc + self.len_points]
        self.fault_drift_at_surface_points_ref = self.fault_matrix[
                                                 :,
                                                 interface_loc + self.len_points:]

        self.input_parameters_kriging = [self.dips_position_all, self.dip_angles_all, self.azimuth_all,
                                         self.polarity_all, self.surface_points_all,
                                         self.fault_matrix]

        # COMPUTE SCALAR FIELDS
        # ---------
        # VARIABLES
        # ---------
        self.grid_val_T = T.matrix('Coordinates of the grid points to interpolate')
        #self.fault_matrix = T.matrix('Full block matrix for x_to_interpolate')

        #self.weights = T.vector('kriging weights')

        self.input_parameters_export = [self.dips_position_all, self.surface_points_all,
                                        self.fault_matrix,# self.weights,
                                        self.grid_val_T]

        self.input_parameters_kriging_export = [self.dips_position_all, self.dip_angles_all, self.azimuth_all,
                                                self.polarity_all, self.surface_points_all,
                                                self.fault_matrix, self.grid_val_T,
                                                ]

        # COMPUTE BLOCKS
        # ---------
        # VARIABLES
        # ---------
      #  self.Z_x = T.vector('Scalar')
      #  self.scalar_field_at_surface_points_values = self.Z_x[-2 * self.len_points: -self.len_points][self.npf_op]

        self.values_properties_op = T.matrix('Values that the blocks are taking')

        #self.n_surface = theano.shared(np.arange(1, 5000, dtype='int32'), "ID of the surface")
        self.n_surface = T.arange(1, 5000, dtype='int32')#T.vector('ID of the surface')
        self.n_surface.name = 'ID of surfaces'

        self.input_parameters_block_fault = [self.values_properties_op,

                                             self.surface_points_all, self.grid_val_T]
        self.input_parameters_block_formations = [self.values_properties_op,

                                                  self.surface_points_all]

        # ------
        # Shared
        # ------
        self.is_fault_ctrl = theano.shared(np.zeros(3, dtype='int32'), 'The series (fault) is finite')
        self.is_finite = theano.shared(np.zeros(3, dtype='int32'), 'The series (fault) is finite')
        self.inf_factor = self.is_finite * 10

        # COMPUTE LOOP
        # ------
        # Shared
        # ------
        # Init fault relation matrix
        self.fault_relation = theano.shared(np.array([[0, 1, 0, 1],
                                                      [0, 0, 1, 1],
                                                      [0, 0, 0, 1],
                                                      [0, 0, 0, 0]]), 'fault relation matrix')

        # Results matrix
        self.weights_vector = theano.shared(np.zeros(10000), 'Weights vector')
        self.scalar_fields_matrix = theano.shared(np.zeros((3, 10000)), 'Scalar matrix')
        self.block_matrix = theano.shared(np.zeros((3, 10000)), "block matrix")

        # Structure
        self.n_surfaces_per_series = theano.shared(np.arange(2, dtype='int32'), 'List with the number of surfaces')
        self.len_series_i = theano.shared(np.arange(2, dtype='int32'), 'Length of surface_points in every series')
        self.len_series_f = theano.shared(np.arange(2, dtype='int32'), 'Length of foliations in every series')


        # Control flow
        self.compute_weights_ctrl = T.vector('Vector controlling if weights must be recomputed', dtype='bool')
        self.compute_scalar_ctrl = T.vector('Vector controlling if scalar matrix must be recomputed', dtype='bool')
        self.compute_block_ctrl = T.vector('Vector controlling if block matrix must be recomputed', dtype='bool')
        self.is_fault_ctrl = theano.shared(np.zeros(3, dtype='int32'), 'The series (fault) is finite')


    def compute_weights(self):
     #   self.fault_drift_at_surface_points_ref = T.repeat(fault_drift[:, [0]], self.number_of_points_per_surface_T[0], axis=0)
      #  self.fault_drift_at_surface_points_rest = fault_drift[:, 1:]

        return self.solve_kriging()

    def compute_scalar_field(self, weights, grid, fault_matrix):
        self.fault_matrix = fault_matrix
        grid_val = self.x_to_interpolate(grid)

        return self.scalar_field_at_all(weights, grid_val)

    def compute_formation_block(self, Z_x, scalar_field_at_surface_points, values):
        return self.export_formation_block(Z_x, scalar_field_at_surface_points,  values)

    def compute_fault_block(self, Z_x, scalar_field_at_surface_points, values, n_series, grid):
        grid_val = self.x_to_interpolate(grid)
        finite_faults_sel = self.select_finite_faults(n_series, grid_val)
        return self.export_fault_block(Z_x, scalar_field_at_surface_points, values, finite_faults_sel)

    def compute_model(self):
        # Looping
        series, updates1 = theano.scan(
            fn=self.compute_a_series,
            outputs_info=[
                dict(initial=self.block_matrix), None, None
            ],  # This line may be used for the df network
            sequences=[dict(input=self.len_series_i, taps=[0, 1]),
                       dict(input=self.len_series_f, taps=[0, 1]),
                       dict(input=self.n_surfaces_per_series, taps=[0, 1]),
                       dict(input=self.n_universal_eq_T, taps=[0]),
                       dict(input=self.compute_weights_ctrl, taps=[0]),
                       dict(input=self.compute_scalar_ctrl, taps=[0]),
                       dict(input=self.compute_block_ctrl, taps=[0]),
                       dict(input=self.is_fault_ctrl, taps=[0]),
                       dict(input=T.arange(0, 5000, dtype='int32'), taps=[0])
                       # dict(input=self.weights_vector),
                       # dict(input=self.scalar_fields_matrix)
                       ],
            non_sequences=[self.weights_vector, self.scalar_fields_matrix],
            name='Looping',
            return_list=True,
            profile=False
        )

        return series

    # region Geometry
    def set_rest_ref_matrix(self, number_of_points_per_surface):
        ref_positions = T.cumsum(T.concatenate((T.stack([0]), number_of_points_per_surface[:-1] + 1)))
        ref_points = T.repeat(self.surface_points_all[ref_positions], number_of_points_per_surface, axis=0)

        rest_mask = T.ones(T.stack([self.surface_points_all.shape[0]]), dtype='int16')
        rest_mask = T.set_subtensor(rest_mask[ref_positions], 0)
        rest_points = self.surface_points_all[T.nonzero(rest_mask)[0]]
        return [ref_points, rest_points, rest_mask, T.nonzero(rest_mask)[0]]

    @staticmethod
    def squared_euclidean_distances(x_1, x_2):
        """
        Compute the euclidian distances in 3D between all the points in x_1 and x_2

        Args:
            x_1 (theano.tensor.matrix): shape n_points x number dimension
            x_2 (theano.tensor.matrix): shape n_points x number dimension

        Returns:
            theano.tensor.matrix: Distancse matrix. shape n_points x n_points
        """

        # T.maximum avoid negative numbers increasing stability
        sqd = T.sqrt(T.maximum(
            (x_1 ** 2).sum(1).reshape((x_1.shape[0], 1)) +
            (x_2 ** 2).sum(1).reshape((1, x_2.shape[0])) -
            2 * x_1.dot(x_2.T), 1e-12
        ))


        #    sqd = theano.printing.Print('sed')(sqd)

        return sqd

    def matrices_shapes(self):
        """
        Get all the lengths of the matrices that form the covariance matrix

        Returns:
             length_of_CG, length_of_CGI, length_of_U_I, length_of_faults, length_of_C
        """

        # Calculating the dimensions of the
        length_of_CG = self.dips_position_tiled.shape[0]
        length_of_CGI = self.rest_layer_points.shape[0]
        length_of_U_I = self.n_universal_eq_T_op

        # Self fault matrix contains the block and the potential field (I am not able to split them). Therefore we need
        # to divide it by 2
        length_of_faults = T.cast(self.fault_matrix.shape[0], 'int32')
        length_of_C = length_of_CG + length_of_CGI + length_of_U_I + length_of_faults

        if 'matrices_shapes' in self.verbose:
            length_of_CG = theano.printing.Print("length_of_CG")(length_of_CG)
            length_of_CGI = theano.printing.Print("length_of_CGI")(length_of_CGI)
            length_of_U_I = theano.printing.Print("length_of_U_I")(length_of_U_I)
            length_of_faults = theano.printing.Print("length_of_faults")(length_of_faults)
            length_of_C = theano.printing.Print("length_of_C")(length_of_C)

        return length_of_CG, length_of_CGI, length_of_U_I, length_of_faults, length_of_C
    # endregion

    # region Kriging
    def cov_surface_points(self):
        """
        Create covariance function for the surface_points

        Returns:
            theano.tensor.matrix: covariance of the surface_points. Shape number of points in rest x number of
            points in rest

        """

        # Compute euclidian distances
        sed_rest_rest = self.squared_euclidean_distances(self.rest_layer_points, self.rest_layer_points)
        sed_ref_rest = self.squared_euclidean_distances(self.ref_layer_points, self.rest_layer_points)
        sed_rest_ref = self.squared_euclidean_distances(self.rest_layer_points, self.ref_layer_points)
        sed_ref_ref = self.squared_euclidean_distances(self.ref_layer_points, self.ref_layer_points)

        # Covariance matrix for surface_points
        C_I = (self.c_o_T * self.i_reescale * (
                (sed_rest_rest < self.a_T) *  # Rest - Rest Covariances Matrix
                (1 - 7 * (sed_rest_rest / self.a_T) ** 2 +
                 35 / 4 * (sed_rest_rest / self.a_T) ** 3 -
                 7 / 2 * (sed_rest_rest / self.a_T) ** 5 +
                 3 / 4 * (sed_rest_rest / self.a_T) ** 7) -
                ((sed_ref_rest < self.a_T) *  # Reference - Rest
                 (1 - 7 * (sed_ref_rest / self.a_T) ** 2 +
                  35 / 4 * (sed_ref_rest / self.a_T) ** 3 -
                  7 / 2 * (sed_ref_rest / self.a_T) ** 5 +
                  3 / 4 * (sed_ref_rest / self.a_T) ** 7)) -
                ((sed_rest_ref < self.a_T) *  # Rest - Reference
                 (1 - 7 * (sed_rest_ref / self.a_T) ** 2 +
                  35 / 4 * (sed_rest_ref / self.a_T) ** 3 -
                  7 / 2 * (sed_rest_ref / self.a_T) ** 5 +
                  3 / 4 * (sed_rest_ref / self.a_T) ** 7)) +
                ((sed_ref_ref < self.a_T) *  # Reference - References
                 (1 - 7 * (sed_ref_ref / self.a_T) ** 2 +
                  35 / 4 * (sed_ref_ref / self.a_T) ** 3 -
                  7 / 2 * (sed_ref_ref / self.a_T) ** 5 +
                  3 / 4 * (sed_ref_ref / self.a_T) ** 7))))

        C_I += T.eye(C_I.shape[0]) * 2 * self.nugget_effect_scalar_T
        # Add name to the theano node
        C_I.name = 'Covariance SurfacePoints'

        if str(sys._getframe().f_code.co_name) in self.verbose:
            C_I = theano.printing.Print('Cov surface_points')(C_I)

        return C_I

    def cov_gradients(self, verbose=0):
        """
         Create covariance function for the gradients

         Returns:
             theano.tensor.matrix: covariance of the gradients. Shape number of points in dip_pos x number of
             points in dip_pos

         """

        # Euclidean distances
        sed_dips_dips = self.squared_euclidean_distances(self.dips_position_tiled, self.dips_position_tiled)

        if 'sed_dips_dips' in self.verbose:
            sed_dips_dips = theano.printing.Print('sed_dips_dips')(sed_dips_dips)

        # Cartesian distances between dips positions
        h_u = T.vertical_stack(
            T.tile(self.dips_position[:, 0] - self.dips_position[:, 0].reshape((self.dips_position[:, 0].shape[0], 1)),
                   self.n_dimensions),
            T.tile(self.dips_position[:, 1] - self.dips_position[:, 1].reshape((self.dips_position[:, 1].shape[0], 1)),
                   self.n_dimensions),
            T.tile(self.dips_position[:, 2] - self.dips_position[:, 2].reshape((self.dips_position[:, 2].shape[0], 1)),
                   self.n_dimensions))

        # Transpose
        h_v = h_u.T

        # Perpendicularity matrix. Boolean matrix to separate cross-covariance and
        # every gradient direction covariance (block diagonal)
        perpendicularity_matrix = T.zeros_like(sed_dips_dips)

        # Cross-covariances of x
        perpendicularity_matrix = T.set_subtensor(
            perpendicularity_matrix[0:self.dips_position.shape[0], 0:self.dips_position.shape[0]], 1)

        # Cross-covariances of y
        perpendicularity_matrix = T.set_subtensor(
            perpendicularity_matrix[self.dips_position.shape[0]:self.dips_position.shape[0] * 2,
            self.dips_position.shape[0]:self.dips_position.shape[0] * 2], 1)

        # Cross-covariances of z
        perpendicularity_matrix = T.set_subtensor(
            perpendicularity_matrix[self.dips_position.shape[0] * 2:self.dips_position.shape[0] * 3,
            self.dips_position.shape[0] * 2:self.dips_position.shape[0] * 3], 1)

        # Covariance matrix for gradients at every xyz direction and their cross-covariances
        C_G = T.switch(
            T.eq(sed_dips_dips, 0),  # This is the condition
            0,  # If true it is equal to 0. This is how a direction affect another
            (  # else, following Chiles book
                    (h_u * h_v / sed_dips_dips ** 2) *
                    ((
                             (sed_dips_dips < self.a_T) *  # first derivative
                             (-self.c_o_T * ((-14 / self.a_T ** 2) + 105 / 4 * sed_dips_dips / self.a_T ** 3 -
                                             35 / 2 * sed_dips_dips ** 3 / self.a_T ** 5 +
                                             21 / 4 * sed_dips_dips ** 5 / self.a_T ** 7))) +
                     (sed_dips_dips < self.a_T) *  # Second derivative
                     self.c_o_T * 7 * (9 * sed_dips_dips ** 5 - 20 * self.a_T ** 2 * sed_dips_dips ** 3 +
                                       15 * self.a_T ** 4 * sed_dips_dips - 4 * self.a_T ** 5) / (2 * self.a_T ** 7)) -
                    (perpendicularity_matrix *
                     (sed_dips_dips < self.a_T) *  # first derivative
                     self.c_o_T * ((-14 / self.a_T ** 2) + 105 / 4 * sed_dips_dips / self.a_T ** 3 -
                                   35 / 2 * sed_dips_dips ** 3 / self.a_T ** 5 +
                                   21 / 4 * sed_dips_dips ** 5 / self.a_T ** 7)))
        )

        # Setting nugget effect of the gradients
        # TODO: This function can be substitued by simply adding the nugget effect to the diag if I remove the condition
        C_G += T.eye(C_G.shape[0]) * self.nugget_effect_grad_T

        # Add name to the theano node
        C_G.name = 'Covariance Gradient'

        if verbose > 1:
            theano.printing.pydotprint(C_G, outfile="graphs/" + sys._getframe().f_code.co_name + ".png",
                                       var_with_name_simple=True)

        if str(sys._getframe().f_code.co_name) in self.verbose:
            C_G = theano.printing.Print('Cov Gradients')(C_G)

        return C_G

    def cov_interface_gradients(self):
        """
        Create covariance function for the gradiens
        Returns:
            theano.tensor.matrix: covariance of the gradients. Shape number of points in rest x number of
              points in dip_pos
        """

        # Euclidian distances
        sed_dips_rest = self.squared_euclidean_distances(self.dips_position_tiled, self.rest_layer_points)
        sed_dips_ref = self.squared_euclidean_distances(self.dips_position_tiled, self.ref_layer_points)

        # Cartesian distances between dips and interface points
        # Rest
        hu_rest = T.vertical_stack(
            (self.dips_position[:, 0] - self.rest_layer_points[:, 0].reshape(
                (self.rest_layer_points[:, 0].shape[0], 1))).T,
            (self.dips_position[:, 1] - self.rest_layer_points[:, 1].reshape(
                (self.rest_layer_points[:, 1].shape[0], 1))).T,
            (self.dips_position[:, 2] - self.rest_layer_points[:, 2].reshape(
                (self.rest_layer_points[:, 2].shape[0], 1))).T
        )

        # Reference point
        hu_ref = T.vertical_stack(
            (self.dips_position[:, 0] - self.ref_layer_points[:, 0].reshape(
                (self.ref_layer_points[:, 0].shape[0], 1))).T,
            (self.dips_position[:, 1] - self.ref_layer_points[:, 1].reshape(
                (self.ref_layer_points[:, 1].shape[0], 1))).T,
            (self.dips_position[:, 2] - self.ref_layer_points[:, 2].reshape(
                (self.ref_layer_points[:, 2].shape[0], 1))).T
        )

        # Cross-Covariance gradients-surface_points
        C_GI = self.gi_reescale * (
                (hu_rest *
                 (sed_dips_rest < self.a_T) *  # first derivative
                 (- self.c_o_T * ((-14 / self.a_T ** 2) + 105 / 4 * sed_dips_rest / self.a_T ** 3 -
                                  35 / 2 * sed_dips_rest ** 3 / self.a_T ** 5 +
                                  21 / 4 * sed_dips_rest ** 5 / self.a_T ** 7))) -
                (hu_ref *
                 (sed_dips_ref < self.a_T) *  # first derivative
                 (- self.c_o_T * ((-14 / self.a_T ** 2) + 105 / 4 * sed_dips_ref / self.a_T ** 3 -
                                  35 / 2 * sed_dips_ref ** 3 / self.a_T ** 5 +
                                  21 / 4 * sed_dips_ref ** 5 / self.a_T ** 7)))
        ).T

        # Add name to the theano node
        C_GI.name = 'Covariance gradient interface'

        if str(sys._getframe().f_code.co_name) + '_g' in self.verbose:
            theano.printing.pydotprint(C_GI, outfile="graphs/" + sys._getframe().f_code.co_name + ".png",
                                       var_with_name_simple=True)
        return C_GI

    def universal_matrix(self):
        """
        Create the drift matrices for the potential field and its gradient

        Returns:
            theano.tensor.matrix: Drift matrix for the surface_points. Shape number of points in rest x 3**degree drift
            (except degree 0 that is 0)

            theano.tensor.matrix: Drift matrix for the gradients. Shape number of points in dips x 3**degree drift
            (except degree 0 that is 0)
        """

        # Condition of universality 2 degree
        # Gradients

        n = self.dips_position.shape[0]
        U_G = T.zeros((n * self.n_dimensions, 3 * self.n_dimensions))
        # x
        U_G = T.set_subtensor(U_G[:n, 0], 1)
        # y
        U_G = T.set_subtensor(U_G[n * 1:n * 2, 1], 1)
        # z
        U_G = T.set_subtensor(U_G[n * 2: n * 3, 2], 1)
        # x**2
        U_G = T.set_subtensor(U_G[:n, 3], 2 * self.gi_reescale * self.dips_position[:, 0])
        # y**2
        U_G = T.set_subtensor(U_G[n * 1:n * 2, 4], 2 * self.gi_reescale * self.dips_position[:, 1])
        # z**2
        U_G = T.set_subtensor(U_G[n * 2: n * 3, 5], 2 * self.gi_reescale * self.dips_position[:, 2])
        # xy
        U_G = T.set_subtensor(U_G[:n, 6], self.gi_reescale * self.dips_position[:, 1])  # This is y
        U_G = T.set_subtensor(U_G[n * 1:n * 2, 6], self.gi_reescale * self.dips_position[:, 0])  # This is x
        # xz
        U_G = T.set_subtensor(U_G[:n, 7], self.gi_reescale * self.dips_position[:, 2])  # This is z
        U_G = T.set_subtensor(U_G[n * 2: n * 3, 7], self.gi_reescale * self.dips_position[:, 0])  # This is x
        # yz
        U_G = T.set_subtensor(U_G[n * 1:n * 2, 8], self.gi_reescale * self.dips_position[:, 2])  # This is z
        U_G = T.set_subtensor(U_G[n * 2:n * 3, 8], self.gi_reescale * self.dips_position[:, 1])  # This is y

        # Interface
        U_I = - T.stack(
            (self.gi_reescale * (self.rest_layer_points[:, 0] - self.ref_layer_points[:, 0]),
             self.gi_reescale * (self.rest_layer_points[:, 1] - self.ref_layer_points[:, 1]),
             self.gi_reescale * (self.rest_layer_points[:, 2] - self.ref_layer_points[:, 2]),
             self.gi_reescale ** 2 * (self.rest_layer_points[:, 0] ** 2 - self.ref_layer_points[:, 0] ** 2),
             self.gi_reescale ** 2 * (self.rest_layer_points[:, 1] ** 2 - self.ref_layer_points[:, 1] ** 2),
             self.gi_reescale ** 2 * (self.rest_layer_points[:, 2] ** 2 - self.ref_layer_points[:, 2] ** 2),
             self.gi_reescale ** 2 * (
                     self.rest_layer_points[:, 0] * self.rest_layer_points[:, 1] - self.ref_layer_points[:, 0] *
                     self.ref_layer_points[:, 1]),
             self.gi_reescale ** 2 * (
                     self.rest_layer_points[:, 0] * self.rest_layer_points[:, 2] - self.ref_layer_points[:, 0] *
                     self.ref_layer_points[:, 2]),
             self.gi_reescale ** 2 * (
                     self.rest_layer_points[:, 1] * self.rest_layer_points[:, 2] - self.ref_layer_points[:, 1] *
                     self.ref_layer_points[:, 2]),
             )).T

        if 'U_I' in self.verbose:
            U_I = theano.printing.Print('U_I')(U_I)

        if 'U_G' in self.verbose:
            U_G = theano.printing.Print('U_G')(U_G)

        if str(sys._getframe().f_code.co_name) + '_g' in self.verbose:
            theano.printing.pydotprint(U_I, outfile="graphs/" + sys._getframe().f_code.co_name + "_i.png",
                                       var_with_name_simple=True)

            theano.printing.pydotprint(U_G, outfile="graphs/" + sys._getframe().f_code.co_name + "_g.png",
                                       var_with_name_simple=True)

        # Add name to the theano node
        if U_I:
            U_I.name = 'Drift surface_points'
            U_G.name = 'Drift foliations'

        return U_I[:, :self.n_universal_eq_T_op], U_G[:, :self.n_universal_eq_T_op]

    def faults_matrix(self):
        """
        This function creates the part of the graph that generates the df function creating a "block model" at the
        references and the rest of the points. Then this vector has to be appended to the covariance function

        Returns:

            list:

            - theano.tensor.matrix: Drift matrix for the surface_points. Shape number of points in rest x n df. This drif
              is a simple addition of an arbitrary number

            - theano.tensor.matrix: Drift matrix for the gradients. Shape number of points in dips x n df. For
              discrete values this matrix will be null since the derivative of a constant is 0
        """

        length_of_CG, length_of_CGI, length_of_U_I, length_of_faults = self.matrices_shapes()[:4]

        # self.fault_drift contains the df volume of the grid and the rest and ref points. For the drift we need
        # to make it relative to the reference point
        if 'fault matrix' in self.verbose:
            self.fault_drift = theano.printing.Print('self.fault_drift')(self.fault_drift)
        # interface_loc = self.fault_drift.shape[1] - 2 * self.len_points
        #
        # fault_drift_at_surface_points_rest = self.fault_drift
        # fault_drift_at_surface_points_ref = self.fault_drift

        F_I = (self.fault_drift_at_surface_points_ref - self.fault_drift_at_surface_points_rest) + 0.0001

        # As long as the drift is a constant F_G is null
        F_G = T.zeros((length_of_faults, length_of_CG)) + 0.0001

        if str(sys._getframe().f_code.co_name) in self.verbose:
            F_I = theano.printing.Print('Faults surface_points matrix')(F_I)
            F_G = theano.printing.Print('Faults gradients matrix')(F_G)

        return F_I, F_G

    def covariance_matrix(self):
        """
        Set all the previous covariances together in the universal cokriging matrix

        Returns:
            theano.tensor.matrix: Multivariate covariance
        """

        # Lengths
        length_of_CG, length_of_CGI, length_of_U_I, length_of_faults, length_of_C = self.matrices_shapes()

        # Individual matrices
        C_G = self.cov_gradients()
        C_I = self.cov_surface_points()
        C_GI = self.cov_interface_gradients()
        U_I, U_G = self.universal_matrix()
        F_I, F_G = self.faults_matrix()

        # =================================
        # Creation of the Covariance Matrix
        # =================================
        C_matrix = T.zeros((length_of_C, length_of_C))

        # First row of matrices
        # Set C_G
        C_matrix = T.set_subtensor(C_matrix[0:length_of_CG, 0:length_of_CG], C_G)
        # Set CGI
        C_matrix = T.set_subtensor(C_matrix[0:length_of_CG, length_of_CG:length_of_CG + length_of_CGI], C_GI.T)
        # Set UG
        C_matrix = T.set_subtensor(C_matrix[0:length_of_CG,
                                   length_of_CG + length_of_CGI:length_of_CG + length_of_CGI + length_of_U_I], U_G)
        # Set FG. I cannot use -index because when is -0 is equivalent to 0
        C_matrix = T.set_subtensor(C_matrix[0:length_of_CG, length_of_CG + length_of_CGI + length_of_U_I:], F_G.T)
        # Second row of matrices
        # Set C_IG
        C_matrix = T.set_subtensor(C_matrix[length_of_CG:length_of_CG + length_of_CGI, 0:length_of_CG], C_GI)
        # Set C_I
        C_matrix = T.set_subtensor(C_matrix[length_of_CG:length_of_CG + length_of_CGI,
                                   length_of_CG:length_of_CG + length_of_CGI], C_I)
        # Set U_I
        # if not self.u_grade_T.get_value() == 0:
        C_matrix = T.set_subtensor(C_matrix[length_of_CG:length_of_CG + length_of_CGI,
                                   length_of_CG + length_of_CGI:length_of_CG + length_of_CGI + length_of_U_I], U_I)
        # Set F_I
        C_matrix = T.set_subtensor(
            C_matrix[length_of_CG:length_of_CG + length_of_CGI, length_of_CG + length_of_CGI + length_of_U_I:], F_I.T)
        # Third row of matrices
        # Set U_G
        C_matrix = T.set_subtensor(
            C_matrix[length_of_CG + length_of_CGI:length_of_CG + length_of_CGI + length_of_U_I, 0:length_of_CG], U_G.T)
        # Set U_I
        C_matrix = T.set_subtensor(C_matrix[length_of_CG + length_of_CGI:length_of_CG + length_of_CGI + length_of_U_I,
                                   length_of_CG:length_of_CG + length_of_CGI], U_I.T)
        # Fourth row of matrices
        # Set F_G
        C_matrix = T.set_subtensor(C_matrix[length_of_CG + length_of_CGI + length_of_U_I:, 0:length_of_CG], F_G)
        # Set F_I
        C_matrix = T.set_subtensor(
            C_matrix[length_of_CG + length_of_CGI + length_of_U_I:, length_of_CG:length_of_CG + length_of_CGI], F_I)
        # Add name to the theano node
        C_matrix.name = 'Block Covariance Matrix'
        if str(sys._getframe().f_code.co_name) in self.verbose:
            C_matrix = theano.printing.Print('cov_function')(C_matrix)

        return C_matrix

    def b_vector(self):
        """
        Creation of the independent vector b to solve the kriging system

        Args:
            verbose: -deprecated-

        Returns:
            theano.tensor.vector: independent vector
        """

        length_of_C = self.matrices_shapes()[-1]
        # =====================
        # Creation of the gradients G vector
        # Calculation of the cartesian components of the dips assuming the unit module
        G_x = T.sin(T.deg2rad(self.dip_angles)) * T.sin(T.deg2rad(self.azimuth)) * self.polarity
        G_y = T.sin(T.deg2rad(self.dip_angles)) * T.cos(T.deg2rad(self.azimuth)) * self.polarity
        G_z = T.cos(T.deg2rad(self.dip_angles)) * self.polarity

        G = T.concatenate((G_x, G_y, G_z))

        # Creation of the Dual Kriging vector
        b = T.zeros((length_of_C,))
        b = T.set_subtensor(b[0:G.shape[0]], G)

        if str(sys._getframe().f_code.co_name) in self.verbose:
            b = theano.printing.Print('b vector')(b)

        # Add name to the theano node
        b.name = 'b vector'
        return b

    def solve_kriging(self):
        """
        Solve the kriging system. This has to get substituted by a more efficient and stable method QR
        decomposition in all likelihood

        Returns:
            theano.tensor.vector: Dual kriging parameters

        """
        C_matrix = self.covariance_matrix()
        b = self.b_vector()
        # Solving the kriging system
        import theano.tensor.slinalg
        b2 = T.tile(b, (1, 1)).T
        DK_parameters = theano.tensor.slinalg.solve(C_matrix, b2)
        DK_parameters = DK_parameters.reshape((DK_parameters.shape[0],))

        # Add name to the theano node
        DK_parameters.name = 'Dual Kriging parameters'

        if str(sys._getframe().f_code.co_name) in self.verbose:
            DK_parameters = theano.printing.Print(DK_parameters.name)(DK_parameters)
        return DK_parameters
    # endregion

    # region Evaluate
    def x_to_interpolate(self, grid, verbose=0):
        """
        here I add to the grid points also the references points(to check the value of the potential field at the
        surface_points). Also here I will check what parts of the grid have been already computed in a previous series
        to avoid to recompute.

        Returns:
            theano.tensor.matrix: The 3D points of the given grid plus the reference and rest points
        """

        grid_val = T.concatenate([grid, self.rest_layer_points_all,
                                  self.ref_layer_points_all])

        if verbose > 1:
            theano.printing.pydotprint(grid_val, outfile="graphs/" + sys._getframe().f_code.co_name + ".png",
                                       var_with_name_simple=True)

        if 'grid_val' in self.verbose:
            grid_val = theano.printing.Print('Points to interpolate')(grid_val)

        return grid_val

    def extend_dual_kriging(self, weights, grid_shape):
        # TODO Think what object is worth to save to speed up computation
        """
        Tile the dual kriging vector to cover all the points to interpolate.So far I just make a matrix with the
        dimensions len(DK)x(grid) but in the future maybe I have to try to loop all this part so consume less memory

        Returns:
            theano.tensor.matrix: Matrix with the Dk parameters repeated for all the points to interpolate
        """

      #  grid_val = self.x_to_interpolate()
        # if self.weights.get_value() is None:
        #     DK_parameters = self.solve_kriging()
        # else:
        #     DK_parameters = self.weights
        DK_parameters = weights
        # Creation of a matrix of dimensions equal to the grid with the weights for every point (big 4D matrix in
        # ravel form)
        # TODO IMP: Change the tile by a simple dot op -> The DOT version in gpu is slower
        DK_weights = T.tile(DK_parameters, (grid_shape, 1)).T

        if self.dot_version:
            DK_weights = DK_parameters

        return DK_weights
    # endregion

    # region Evaluate Geology
    def contribution_gradient_interface(self, grid_val=None, weights=None):
        """
        Computation of the contribution of the foliations at every point to interpolate

        Returns:
            theano.tensor.vector: Contribution of all foliations (input) at every point to interpolate
        """
        if weights is None:
            weights = self.extend_dual_kriging()
        if grid_val is None:
            grid_val = self.x_to_interpolate()

        length_of_CG = self.matrices_shapes()[0]

        # Cartesian distances between the point to simulate and the dips
        hu_SimPoint = T.vertical_stack(
            (self.dips_position[:, 0] - grid_val[:, 0].reshape((grid_val[:, 0].shape[0], 1))).T,
            (self.dips_position[:, 1] - grid_val[:, 1].reshape((grid_val[:, 1].shape[0], 1))).T,
            (self.dips_position[:, 2] - grid_val[:, 2].reshape((grid_val[:, 2].shape[0], 1))).T
        )

        # Euclidian distances
        sed_dips_SimPoint = self.squared_euclidean_distances(self.dips_position_tiled, grid_val)
        # Gradient contribution
        sigma_0_grad = T.sum(
            (weights[:length_of_CG] *
             self.gi_reescale *
             (-hu_SimPoint *
              (sed_dips_SimPoint < self.a_T) *  # first derivative
              (- self.c_o_T * ((-14 / self.a_T ** 2) + 105 / 4 * sed_dips_SimPoint / self.a_T ** 3 -
                               35 / 2 * sed_dips_SimPoint ** 3 / self.a_T ** 5 +
                               21 / 4 * sed_dips_SimPoint ** 5 / self.a_T ** 7)))),
            axis=0)

        if self.dot_version:
            sigma_0_grad = T.dot(
                weights[:length_of_CG],
                self.gi_reescale *
                (-hu_SimPoint *
                 (sed_dips_SimPoint < self.a_T) *  # first derivative
                 (- self.c_o_T * ((-14 / self.a_T ** 2) + 105 / 4 * sed_dips_SimPoint / self.a_T ** 3 -
                                  35 / 2 * sed_dips_SimPoint ** 3 / self.a_T ** 5 +
                                  21 / 4 * sed_dips_SimPoint ** 5 / self.a_T ** 7))))

        # Add name to the theano node
        sigma_0_grad.name = 'Contribution of the foliations to the potential field at every point of the grid'

        if str(sys._getframe().f_code.co_name) in self.verbose:
            sigma_0_grad = theano.printing.Print('interface_gradient_contribution')(sigma_0_grad)

        return sigma_0_grad

    def contribution_interface(self, grid_val=None, weights=None):
        """
          Computation of the contribution of the surface_points at every point to interpolate

          Returns:
              theano.tensor.vector: Contribution of all surface_points (input) at every point to interpolate
          """

        if weights is None:
            weights = self.extend_dual_kriging()
        if grid_val is None:
            grid_val = self.x_to_interpolate()
        length_of_CG, length_of_CGI = self.matrices_shapes()[:2]

        # Euclidian distances
        sed_rest_SimPoint = self.squared_euclidean_distances(self.rest_layer_points, grid_val)
        sed_ref_SimPoint = self.squared_euclidean_distances(self.ref_layer_points, grid_val)

        # Interface contribution
        sigma_0_interf = (T.sum(
            -weights[length_of_CG:length_of_CG + length_of_CGI, :] *
            (self.c_o_T * self.i_reescale * (
                    (sed_rest_SimPoint < self.a_T) *  # SimPoint - Rest Covariances Matrix
                    (1 - 7 * (sed_rest_SimPoint / self.a_T) ** 2 +
                     35 / 4 * (sed_rest_SimPoint / self.a_T) ** 3 -
                     7 / 2 * (sed_rest_SimPoint / self.a_T) ** 5 +
                     3 / 4 * (sed_rest_SimPoint / self.a_T) ** 7) -
                    ((sed_ref_SimPoint < self.a_T) *  # SimPoint- Ref
                     (1 - 7 * (sed_ref_SimPoint / self.a_T) ** 2 +
                      35 / 4 * (sed_ref_SimPoint / self.a_T) ** 3 -
                      7 / 2 * (sed_ref_SimPoint / self.a_T) ** 5 +
                      3 / 4 * (sed_ref_SimPoint / self.a_T) ** 7)))), axis=0))

        if self.dot_version:
            sigma_0_interf = (
                T.dot(-weights[length_of_CG:length_of_CG + length_of_CGI],
                      (self.c_o_T * self.i_reescale * (
                              (sed_rest_SimPoint < self.a_T) *  # SimPoint - Rest Covariances Matrix
                              (1 - 7 * (sed_rest_SimPoint / self.a_T) ** 2 +
                               35 / 4 * (sed_rest_SimPoint / self.a_T) ** 3 -
                               7 / 2 * (sed_rest_SimPoint / self.a_T) ** 5 +
                               3 / 4 * (sed_rest_SimPoint / self.a_T) ** 7) -
                              ((sed_ref_SimPoint < self.a_T) *  # SimPoint- Ref
                               (1 - 7 * (sed_ref_SimPoint / self.a_T) ** 2 +
                                35 / 4 * (sed_ref_SimPoint / self.a_T) ** 3 -
                                7 / 2 * (sed_ref_SimPoint / self.a_T) ** 5 +
                                3 / 4 * (sed_ref_SimPoint / self.a_T) ** 7))))))

        # Add name to the theano node
        sigma_0_interf.name = 'Contribution of the surface_points to the potential field at every point of the grid'

        return sigma_0_interf

    def contribution_universal_drift(self, grid_val=None, weights=None, a=0, b=100000000):
        """
        Computation of the contribution of the universal drift at every point to interpolate

        Returns:
            theano.tensor.vector: Contribution of the universal drift (input) at every point to interpolate
        """
        if weights is None:
            weights = self.extend_dual_kriging()
        if grid_val is None:
            grid_val = self.x_to_interpolate()

        length_of_CG, length_of_CGI, length_of_U_I, length_of_faults, length_of_C = self.matrices_shapes()

        universal_grid_surface_points_matrix = T.horizontal_stack(
            grid_val,
            (grid_val ** 2),
            T.stack((grid_val[:, 0] * grid_val[:, 1],
                     grid_val[:, 0] * grid_val[:, 2],
                     grid_val[:, 1] * grid_val[:, 2]), axis=1)).T

        # These are the magic terms to get the same as geomodeller
        i_rescale_aux = T.repeat(self.gi_reescale, 9)
        i_rescale_aux = T.set_subtensor(i_rescale_aux[:3], 1)
        _aux_magic_term = T.tile(i_rescale_aux[:self.n_universal_eq_T_op], (grid_val.shape[0], 1)).T

        # Drif contribution
        f_0 = (T.sum(
            weights[
            length_of_CG + length_of_CGI:length_of_CG + length_of_CGI + length_of_U_I] * self.gi_reescale * _aux_magic_term *
            universal_grid_surface_points_matrix[:self.n_universal_eq_T_op]
            , axis=0))

        if self.dot_version:
            f_0 = T.dot(
                weights[length_of_CG + length_of_CGI:length_of_CG + length_of_CGI + length_of_U_I],
                self.gi_reescale * _aux_magic_term *
                universal_grid_surface_points_matrix[:self.n_universal_eq_T_op])

        if not type(f_0) == int:
            f_0.name = 'Contribution of the universal drift to the potential field at every point of the grid'

        if str(sys._getframe().f_code.co_name) in self.verbose:
            f_0 = theano.printing.Print('Universal terms contribution')(f_0)

        return f_0

    def contribution_faults(self, weights=None, a=0, b=100000000):
        """
        Computation of the contribution of the df drift at every point to interpolate. To get these we need to
        compute a whole block model with the df data

        Returns:
            theano.tensor.vector: Contribution of the df drift (input) at every point to interpolate
        """
        if weights is None:
            weights = self.extend_dual_kriging()
        length_of_CG, length_of_CGI, length_of_U_I, length_of_faults, length_of_C = self.matrices_shapes()

        fault_matrix_selection_non_zero = self.fault_matrix[a:b]  # * self.fault_mask[a:b] + 1

        f_1 = T.sum(
            weights[length_of_CG + length_of_CGI + length_of_U_I:, :] * fault_matrix_selection_non_zero, axis=0)

        if self.dot_version:
            f_1 = T.dot(
                weights[length_of_CG + length_of_CGI + length_of_U_I:], fault_matrix_selection_non_zero)

        # Add name to the theano node
        f_1.name = 'Faults contribution'

        if str(sys._getframe().f_code.co_name) in self.verbose:
            f_1 = theano.printing.Print('Faults contribution')(f_1)

        return f_1

    def scalar_field_loop(self, a, b, Z_x, grid_val, weights):

        sigma_0_grad = self.contribution_gradient_interface(grid_val[a:b], weights[:, a:b])
        sigma_0_interf = self.contribution_interface(grid_val[a:b], weights[:, a:b])
        f_0 = self.contribution_universal_drift(grid_val[a:b], weights[:, a:b], a, b)
        f_1 = self.contribution_faults(weights[:, a:b], a, b)

        # Add an arbitrary number at the potential field to get unique values for each of them
        partial_Z_x = (sigma_0_grad + sigma_0_interf + f_0 + f_1)
        Z_x = T.set_subtensor(Z_x[a:b], partial_Z_x)

        return Z_x

    def scalar_field_at_all(self, weights, grid_val):
        """
        Compute the potential field at all the interpolation points, i.e. grid plus rest plus ref
        Returns:
            theano.tensor.vector: Potential fields at all points

        """
        tiled_weights = self.extend_dual_kriging(weights, grid_val.shape[0])

        grid_shape = T.stack([grid_val.shape[0]], axis=0)
        Z_x_init = T.zeros(grid_shape, dtype='float32')
        if 'grid_shape' in self.verbose:
            grid_shape = theano.printing.Print('grid_shape')(grid_shape)

        steps = 1e13 / self.matrices_shapes()[-1] / grid_shape
        slices = T.concatenate((T.arange(0, grid_shape[0], steps[0], dtype='int64'), grid_shape))

        if 'slices' in self.verbose:
            slices = theano.printing.Print('slices')(slices)

        Z_x_loop, updates3 = theano.scan(
            fn=self.scalar_field_loop,
            outputs_info=[Z_x_init],
            sequences=[dict(input=slices, taps=[0, 1])],
            non_sequences=[grid_val, tiled_weights],
            profile=False,
            name='Looping grid',
            return_list=True)

        Z_x = Z_x_loop[-1][-1]
        Z_x.name = 'Value of the potential field at every point'

        if str(sys._getframe().f_code.co_name) in self.verbose:
            Z_x = theano.printing.Print('Potential field at all points')(Z_x)
        scalar_field_at_surface_points_values = Z_x[-2 * self.len_points: -self.len_points][self.npf_op]
        return Z_x, scalar_field_at_surface_points_values
    # endregion

    # region Block export
    def select_finite_faults(self, n_series, grid):
        fault_points = T.vertical_stack(T.stack([self.ref_layer_points[0]], axis=0), self.rest_layer_points).T
        ctr = T.mean(fault_points, axis=1)
        x = fault_points - ctr.reshape((-1, 1))
        M = T.dot(x, x.T)
        U = T.nlinalg.svd(M)[2]
        rotated_x = T.dot(grid, U)
        rotated_fault_points = T.dot(fault_points.T, U)
        rotated_ctr = T.mean(rotated_fault_points, axis=0)
        a_radio = (rotated_fault_points[:, 0].max() - rotated_fault_points[:, 0].min()) / 2 + self.inf_factor[
            n_series - 1]
        b_radio = (rotated_fault_points[:, 1].max() - rotated_fault_points[:, 1].min()) / 2 + self.inf_factor[
            n_series - 1]
        sel = T.lt((rotated_x[:, 0] - rotated_ctr[0]) ** 2 / a_radio ** 2 + (
                rotated_x[:, 1] - rotated_ctr[1]) ** 2 / b_radio ** 2,
                   1)

        if "select_finite_faults" in self.verbose:
            sel = theano.printing.Print("scalar_field_iter")(sel)

        return sel

    def compare(self, a, b, slice_init, Z_x, l, n_surface, drift):
        """
        Treshold of the points to interpolate given 2 potential field values. TODO: This function is the one we
        need to change for a sigmoid function

        Args:
            a (scalar): Upper limit of the potential field
            b (scalar): Lower limit of the potential field
            n_surface (scalar): Value given to the segmentation, i.e. lithology number
            Zx (vector): Potential field values at all the interpolated points

        Returns:
            theano.tensor.vector: segmented values
        """

        slice_init = slice_init
        n_surface_0 = n_surface[:, slice_init:slice_init + 1]
        n_surface_1 = n_surface[:, slice_init + 1:slice_init + 2]
        drift = drift[:, slice_init:slice_init + 1]

        if 'compare' in self.verbose:
            a = theano.printing.Print("a")(a)
            b = theano.printing.Print("b")(b)
            # l = 200/ (a - b)
            slice_init = theano.printing.Print("slice_init")(slice_init)
            n_surface_0 = theano.printing.Print("n_surface_0")(n_surface_0)
            n_surface_1 = theano.printing.Print("n_surface_1")(n_surface_1)
            drift = theano.printing.Print("drift[slice_init:slice_init+1][0]")(drift)

        # drift = T.switch(slice_init == 0, n_surface_1, n_surface_0)
        #    drift = T.set_subtensor(n_surface[0], n_surface[1])

        # The 5 rules the slope of the function
        sigm = (-n_surface_0.reshape((-1, 1)) / (1 + T.exp(-l * (Z_x - a)))) - \
               (n_surface_1.reshape((-1, 1)) / (1 + T.exp(l * (Z_x - b)))) + drift.reshape((-1, 1))
        if 'sigma' in self.verbose:
            sigm = theano.printing.Print("middle point")(sigm)
        #      n_surface = theano.printing.Print("n_surface")(n_surface)
        return sigm

    def export_fault_block(self, Z_x, scalar_field_at_surface_points, values_properties_op, finite_faults_sel, slope=50, offset_slope=5000):
        """
        Compute the part of the block model of a given series (dictated by the bool array yet to be computed)

        Returns:
            theano.tensor.vector: Value of lithology at every interpolated point
        """

        # if Z_x is None:
        #     Z_x = self.Z_x

        # Max and min values of the potential field.
        # max_pot = T.max(Z_x) + 1
        # min_pot = T.min(Z_x) - 1
        # max_pot += max_pot * 0.1
        # min_pot -= min_pot * 0.1

        # Value of the potential field at the surface_points of the computed series
        # TODO timeit
        max_pot = T.max(Z_x)
        # max_pot = theano.printing.Print("max_pot")(max_pot)

        min_pot = T.min(Z_x)
        #     min_pot = theano.printing.Print("min_pot")(min_pot)

        # max_pot_sigm = 2 * max_pot - self.scalar_field_at_surface_points_values[0]
        # min_pot_sigm = 2 * min_pot - self.scalar_field_at_surface_points_values[-1]

        boundary_pad = (max_pot - min_pot) * 0.01
        # l = slope / (max_pot - min_pot)  # (max_pot - min_pot)

        # This is the different line with respect layers
        l = T.switch(finite_faults_sel, offset_slope / (max_pot - min_pot), slope / (max_pot - min_pot))
        #  l = theano.printing.Print("l")(l)

        # A tensor with the values to segment
        scalar_field_iter = T.concatenate((
            T.stack([max_pot + boundary_pad], axis=0),
            scalar_field_at_surface_points,
            T.stack([min_pot - boundary_pad], axis=0)
        ))

        if "scalar_field_iter" in self.verbose:
            scalar_field_iter = theano.printing.Print("scalar_field_iter")(scalar_field_iter)

        # Here we just take the first element of values properties because at least so far we do not find a reason
        # to populate fault blocks with anything else

        n_surface_op_float_sigmoid = T.repeat(values_properties_op[[0], :], 2, axis=1)

        # TODO: instead -1 at the border look for the average distance of the input!
        # TODO I think should be -> n_surface_op_float_sigmoid[:, 2] - n_surface_op_float_sigmoid[:, 1]
        n_surface_op_float_sigmoid = T.set_subtensor(n_surface_op_float_sigmoid[:, 1], -1)
        # - T.sqrt(T.square(n_surface_op_float_sigmoid[0] - n_surface_op_float_sigmoid[2])))

        n_surface_op_float_sigmoid = T.set_subtensor(n_surface_op_float_sigmoid[:, -1], -1)
        # - T.sqrt(T.square(n_surface_op_float_sigmoid[3] - n_surface_op_float_sigmoid[-1])))

        drift = T.set_subtensor(n_surface_op_float_sigmoid[:, 0], n_surface_op_float_sigmoid[:, 1])

        if 'n_surface_op_float_sigmoid' in self.verbose:
            n_surface_op_float_sigmoid = theano.printing.Print("n_surface_op_float_sigmoid") \
                (n_surface_op_float_sigmoid)

        fault_block, updates2 = theano.scan(
            fn=self.compare,
            outputs_info=None,
            sequences=[dict(input=scalar_field_iter, taps=[0, 1]),
                       T.arange(0, n_surface_op_float_sigmoid.shape[1], 2, dtype='int64')],
            non_sequences=[Z_x, l, n_surface_op_float_sigmoid, drift],
            name='Looping compare',
            profile=False,
            return_list=False)

        # For every surface we get a vector so we need to sum compress them to one dimension
        fault_block = fault_block.sum(axis=0)

        # Add name to the theano node
        fault_block.name = 'The chunk of block model of a specific series'
        if str(sys._getframe().f_code.co_name) in self.verbose:
            fault_block = theano.printing.Print(fault_block.name)(fault_block)

        return fault_block

    def export_formation_block(self, Z_x, scalar_field_at_surface_points, values_properties_op, slope=5000):
        """
        Compute the part of the block model of a given series (dictated by the bool array yet to be computed)

        Returns:
            theano.tensor.vector: Value of lithology at every interpolated point
        """
        # TODO: IMP set soft max in the borders

        # if Z_x is None:
        #     Z_x = self.Z_x

        max_pot = T.max(Z_x)
        # max_pot = theano.printing.Print("max_pot")(max_pot)

        min_pot = T.min(Z_x)
        #     min_pot = theano.printing.Print("min_pot")(min_pot)

     #   max_pot_sigm = 2 * max_pot - self.scalar_field_at_surface_points_values[0]
     #   min_pot_sigm = 2 * min_pot - self.scalar_field_at_surface_points_values[-1]

        boundary_pad = (max_pot - min_pot) * 0.01
        l = slope / (max_pot - min_pot)

        # A tensor with the values to segment
        scalar_field_iter = T.concatenate((
            T.stack([max_pot + boundary_pad], axis=0),
            scalar_field_at_surface_points,
            T.stack([min_pot - boundary_pad], axis=0)
        ))

        if "scalar_field_iter" in self.verbose:
            scalar_field_iter = theano.printing.Print("scalar_field_iter")(scalar_field_iter)

        # Loop to segment the distinct lithologies

        n_surface_op_float_sigmoid = T.repeat(values_properties_op, 2, axis=1)

        # TODO: instead -1 at the border look for the average distance of the input!
        n_surface_op_float_sigmoid = T.set_subtensor(n_surface_op_float_sigmoid[:, 0], -1)
        # - T.sqrt(T.square(n_surface_op_float_sigmoid[0] - n_surface_op_float_sigmoid[2])))

        n_surface_op_float_sigmoid = T.set_subtensor(n_surface_op_float_sigmoid[:, -1], -1)
        # - T.sqrt(T.square(n_surface_op_float_sigmoid[3] - n_surface_op_float_sigmoid[-1])))

        drift = T.set_subtensor(n_surface_op_float_sigmoid[:, 0], n_surface_op_float_sigmoid[:, 1])

        if 'n_surface_op_float_sigmoid' in self.verbose:
            n_surface_op_float_sigmoid = theano.printing.Print("n_surface_op_float_sigmoid") \
                (n_surface_op_float_sigmoid)

        formations_block, updates2 = theano.scan(
            fn=self.compare,
            outputs_info=None,
            sequences=[dict(input=scalar_field_iter, taps=[0, 1]), T.arange(0, n_surface_op_float_sigmoid.shape[1],
                                                                            2, dtype='int64')],
            non_sequences=[Z_x, l, n_surface_op_float_sigmoid, drift],
            name='Looping compare',
            profile=False,
            return_list=False)

        # For every surface we get a vector so we need to sum compress them to one dimension
        formations_block = formations_block.sum(axis=0)

        # Add name to the theano node
        formations_block.name = 'The chunk of block model of a specific series'
        if str(sys._getframe().f_code.co_name) in self.verbose:
            formations_block = theano.printing.Print(formations_block.name)(formations_block)

        return formations_block
    # endregion

    # region Compute model
    def compute_a_series(self,
                         len_i_0, len_i_1,
                         len_f_0, len_f_1,
                         n_form_per_serie_0, n_form_per_serie_1,
                         u_grade_iter,
                         compute_weight, compute_scalar, compute_block, is_fault, n_series,
                         block_matrix, weights_vector, scalar_field_matrix,

                         ):
        """
        Function that loops each fault, generating a potential field for each on them with the respective block model

        Args:
            len_i_0: Lenght of rest of previous series
            len_i_1: Lenght of rest for the computed series
            len_f_0: Lenght of dips of previous series
            len_f_1: Length of dips of the computed series
            n_form_per_serie_0: Number of surfaces of previous series
            n_form_per_serie_1: Number of surfaces of the computed series

        Returns:
            theano.tensor.matrix: block model derived from the df that afterwards is used as a drift for the "real"
            data
        """

        self.number_of_points_per_surface_T_op = self.number_of_points_per_surface_T[
                                                 n_form_per_serie_0: n_form_per_serie_1]
        self.npf_op = self.npf[n_form_per_serie_0: n_form_per_serie_1]
        n_surface_op = self.n_surface[n_form_per_serie_0: n_form_per_serie_1]

        self.dips_position = self.dips_position_all[len_f_0: len_f_1, :]
        self.dips_position_tiled = T.tile(self.dips_position, (self.n_dimensions, 1))

        self.dip_angles = self.dip_angles_all[len_f_0: len_f_1]

        self.azimuth = self.azimuth_all[len_f_0: len_f_1]
        self.polarity = self.polarity_all[len_f_0: len_f_1]

     #   self.surface_points_op = self.surface_points_all[len_i_0: len_i_1, :]

        self.ref_layer_points = self.ref_layer_points_all[len_i_0: len_i_1, :]
        self.rest_layer_points = self.rest_layer_points_all[len_i_0: len_i_1, :]

        self.n_universal_eq_T_op = u_grade_iter

        # Extracting faults matrices
        faults_relation_op = self.fault_relation[:, T.cast(n_surface_op[0]-1, 'int8')]
        fault_matrix = block_matrix[T.nonzero(T.cast(faults_relation_op, "int8"))[0], :]

        # TODO this is wrong

        interface_loc = self.fault_matrix.shape[1] - 2 * self.len_points

        self.fault_drift_at_surface_points_rest = fault_matrix[
                                                  :, interface_loc + len_i_0: interface_loc + len_i_1]
        self.fault_drift_at_surface_points_ref = fault_matrix[
                                                 :, interface_loc + self.len_points + len_i_0: interface_loc + self.len_points + len_i_1]

#        fault_drift = fault_matrix[:-2 * self.len_points][len_i_0:, len_i_1]

        # weights = theano.ifelse.ifelse(compute_weight, self.compute_weights(), weights_vector[n_series])
        weights = self.compute_weights()
        weights_vector = T.set_subtensor(weights_vector[self.matrices_shapes()], weights)
        #
        # Z_x = theano.ifelse.ifelse(compute_scalar,
        #                            self.compute_scalar_field(weights, self.grid_val_T, fault_matrix),
        #                            scalar_field_vector)
        #
        # block = theano.ifelse.ifelse(
        #     compute_block,
        #     theano.iflese.ifelse(is_fault,
        #                          self.compute_fault_block(
        #                              Z_x,
        #                              self.values_properties_op[:, n_form_per_serie_0: n_form_per_serie_1 + 1],
        #                              n_series
        #                          ),
        #                          self.compute_formation_block(
        #                              Z_x,
        #                              self.values_properties_op[:, n_form_per_serie_0: n_form_per_serie_1 + 1])),
        #     block_vector
        # )
        #
        # #aux_ind = T.max(n_surface_op, 0)
        #
        # block_matrix = T.set_subtensor(block_matrix[n_series, :], block)
        return block_matrix, weights_vector, scalar_field_matrix  # Z_x, block_matrix
