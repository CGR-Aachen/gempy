import tensorflow as tf  # tensorflow_nightly
import numpy as np
import sys


class TFGraph(tf.Module):

    def __init__(self, dip_angles, azimuth, polarity,
                 fault_drift, grid, values_properties,
                 number_of_points_per_surface, Range, C_o, nugget_effect_scalar,
                 nugget_effect_grad, rescalefactor, output=None, **kwargs):
        super(TFGraph, self).__init__()
        self.dtype = kwargs.get('dtype', tf.float32)
        self.lengh_of_faults = tf.constant(0, dtype=tf.int32)

        # OPETIONS
        # -------
        self.gradient = kwargs.get('gradient', True)
        self.max_speed = kwargs.get('max_speed', 1)
        self.gravity = kwargs.get('gravity', False)

        # CONSTANT PARAMETERS FOR ALL SERIES
        # KRIGING
        # -------
        self.a_T = tf.cast(tf.divide(Range, rescalefactor), self.dtype)
        self.a_T_surface = self.a_T
        self.c_o_T = tf.divide(C_o, rescalefactor)

        self.n_universal_eq_T = tf.ones(
            5, dtype=tf.int32)
        self.n_universal_eq_T_op = tf.constant(3)  # 9 for 2nd order drift

        # They weight the contribution of the surface_points against the orientations.
        self.i_rescale = tf.constant(4., dtype=self.dtype)
        self.gi_rescale = tf.constant(2., dtype=self.dtype)

        # Number of dimensions. Now it is not too variable anymore
        self.n_dimensions = 3

        self.number_of_points_per_surface = tf.cast(
            number_of_points_per_surface, dtype=tf.int32)
        self.npf = tf.concat([[0], self.number_of_points_per_surface[:-1]], -1)

        self.nugget_effect_grad = nugget_effect_grad
        self.nugget_effect_scalar = nugget_effect_scalar

        # COMPUTE WEIGHTS
        # ---------
        # VARIABLES DEFINITION
        # ---------
        if self.gradient:
            self.sig_slope = tf.constant(50, dtype=self.dtype)
        else:
            self.sig_slope = tf.constant(
                50000, dtype=self.dtype, name='Sigmoid slope')
            self.not_l = tf.constant(
                50, dtype=self.dtype, name='Sigmoid Outside')
            self.ellipse_factor_exponent = tf.constant(
                2, dtype=self.dtype, name='Attenuation factor')

        self.dip_angles_all = dip_angles
        self.azimuth_all = azimuth
        self.polarity_all = polarity

        self.dip_angles = self.dip_angles_all
        self.azimuth = self.azimuth_all
        self.polarity = self.polarity_all



        self.values_properties_op = values_properties

        self.grid_val = grid

        self.fault_matrix = fault_drift


        if output is None:
            output = ['geology']
        self.output = output

        self.compute_type = output

        self.is_erosion = tf.constant([1., 0.], dtype=self.dtype)
        self.is_onlap = tf.constant([0, 1], dtype=self.dtype)

        self.offset = tf.constant(10., dtype=self.dtype)
        self.shift = 0

        if 'gravity' in self.compute_type:
            self.lg0 = tf.constant(np.array(0, dtype='int32'))
            self.lg1 = tf.constant(np.array(1, dtype='int32'))

            self.tz = tf.constant(np.empty(0, dtype=self.dtype))
            self.pos_density = tf.constant(np.array(1, dtype='int32'))

        self.n_surface = range(1, 5000)

    
    def set_rest_ref_matrix(self, number_of_points_per_surface, surface_points_all, nugget_effect_scalar):
        # reference point: every first point of each layer
        ref_positions = tf.cumsum(
            tf.concat([[0], number_of_points_per_surface[:-1] + 1], axis=0))

        partitions = tf.reduce_sum(tf.one_hot(ref_positions,tf.reduce_sum(number_of_points_per_surface+1),dtype = tf.int32),axis=0)
        # reference:1 rest: 0

        ## selecting surface points as rest set and reference set
        rest_points = tf.dynamic_partition(surface_points_all, partitions, 2)[0]
        ref_points = tf.dynamic_partition(surface_points_all, partitions, 2)[1]
        
        rest_nugget = tf.dynamic_partition(nugget_effect_scalar, partitions, 2)[0]
        ref_nugget = tf.dynamic_partition(nugget_effect_scalar, partitions, 2)[1]
        
        # repeat the reference points (the number of persurface -1)  times
        ref_points_repeated = tf.repeat(
            ref_points, number_of_points_per_surface, 0)
        ref_nugget_repeated = tf.repeat(
            ref_nugget, number_of_points_per_surface, 0)

        return ref_points_repeated, rest_points, ref_nugget_repeated, rest_nugget

    
    def squared_euclidean_distance(self, x_1, x_2):
        """
        Compute the euclidian distances in 3D between all the points in x_1 and x_2

        Arguments:
            x_1 {[Tensor]} -- shape n_points x number dimension
            x_2 {[Tensor]} -- shape n_points x number dimension

        Returns:
            [Tensor] -- Distancse matrix. shape n_points x n_points
        """
        # tf.maximum avoid negative numbers increasing stability

        sqd = tf.sqrt(tf.maximum(tf.reshape(tf.reduce_sum(x_1**2, 1), shape=(tf.shape(x_1)[0], 1)) +
                                 tf.reshape(tf.reduce_sum(x_2**2, 1), shape=(1, tf.shape(x_2)[0])) -
                                 2 * tf.tensordot(x_1, tf.transpose(x_2), 1), tf.constant(1e-12, dtype=self.dtype)))

        return sqd

    
    def matrices_shapes(self):
        """
        Get all the lengths of the matrices that form the covariance matrix

        Returns:
             length_of_CG, length_of_CGI, length_of_U_I, length_of_faults, length_of_C
        """
        length_of_CG = tf.constant(self.dips_position_all_tiled.shape[0])
        length_of_CGI = tf.shape(self.ref_layer_points)[0]
        length_of_U_I = self.n_universal_eq_T_op
        length_of_faults = self.lengh_of_faults

        length_of_C = length_of_CG + length_of_CGI + length_of_U_I + length_of_faults

        return length_of_CG, length_of_CGI, length_of_U_I, length_of_faults, length_of_C

    
    def cov_surface_points(self, ref_layer_points, rest_layer_points):

        sed_rest_rest = self.squared_euclidean_distance(
            rest_layer_points, rest_layer_points)
        sed_ref_rest = self.squared_euclidean_distance(
            ref_layer_points, rest_layer_points)
        sed_rest_ref = self.squared_euclidean_distance(
            rest_layer_points, ref_layer_points)
        sed_ref_ref = self.squared_euclidean_distance(
            ref_layer_points, ref_layer_points)

        C_I = self.c_o_T * self.i_rescale * (
            tf.where(sed_rest_rest < self.a_T, x=(1 - 7 * (sed_rest_rest / self.a_T) ** 2 +
                                                  35 / 4 * (sed_rest_rest / self.a_T) ** 3 -
                                                  7 / 2 * (sed_rest_rest / self.a_T) ** 5 +
                                                  3 / 4 * (sed_rest_rest / self.a_T) ** 7), y=0) -
            tf.where(sed_ref_rest < self.a_T, x=(1 - 7 * (sed_ref_rest / self.a_T) ** 2 +
                                                 35 / 4 * (sed_ref_rest / self.a_T) ** 3 -
                                                 7 / 2 * (sed_ref_rest / self.a_T) ** 5 +
                                                 3 / 4 * (sed_ref_rest / self.a_T) ** 7), y=0) -
            tf.where(sed_rest_ref < self.a_T, x=(1 - 7 * (sed_rest_ref / self.a_T) ** 2 +
                                                 35 / 4 * (sed_rest_ref / self.a_T) ** 3 -
                                                 7 / 2 * (sed_rest_ref / self.a_T) ** 5 +
                                                 3 / 4 * (sed_rest_ref / self.a_T) ** 7), y=0) +
            tf.where(sed_ref_ref < self.a_T, x=(1 - 7 * (sed_ref_ref / self.a_T) ** 2 +
                                                35 / 4 * (sed_ref_ref / self.a_T) ** 3 -
                                                7 / 2 * (sed_ref_ref / self.a_T) ** 5 +
                                                3 / 4 * (sed_ref_ref / self.a_T) ** 7), y=0))

        C_I = C_I + tf.eye(tf.shape(C_I)[0], dtype=self.dtype) * \
            self.nugget_effect_scalar_ref_rest
        return C_I

    
    def cov_gradients(self, dips_position_all):
        dips_position_all_tiled = tf.tile(
            dips_position_all, [self.n_dimensions, 1])

        sed_dips_dips = self.squared_euclidean_distance(
            dips_position_all_tiled, dips_position_all_tiled)

        h_u = tf.concat([
            tf.tile(dips_position_all[:, 0] - tf.reshape(
                dips_position_all[:, 0], [dips_position_all.shape[0], 1]), [1, 3]),
            tf.tile(dips_position_all[:, 1] - tf.reshape(
                dips_position_all[:, 1], [dips_position_all.shape[0], 1]), [1, 3]),
            tf.tile(dips_position_all[:, 2] - tf.reshape(dips_position_all[:, 2], [dips_position_all.shape[0], 1]), [1, 3])], axis=0)

        h_v = tf.transpose(h_u)

        sub_x = tf.concat([tf.ones([dips_position_all.shape[0], dips_position_all.shape[0]]), tf.zeros(
            [dips_position_all.shape[0], 2 * dips_position_all.shape[0]])], axis=1)

        sub_y = tf.concat([tf.concat([tf.zeros([dips_position_all.shape[0], dips_position_all.shape[0]]), tf.ones(
            [dips_position_all.shape[0], 1 * dips_position_all.shape[0]])], axis=1), tf.zeros([dips_position_all.shape[0], dips_position_all.shape[0]])], 1)
        sub_z = tf.concat([tf.zeros([dips_position_all.shape[0], 2 * dips_position_all.shape[0]]),
                           tf.ones([dips_position_all.shape[0], dips_position_all.shape[0]])], axis=1)

        perpendicularity_matrix = tf.cast(
            tf.concat([sub_x, sub_y, sub_z], axis=0), dtype=self.dtype)

        condistion_fail = tf.math.divide_no_nan(h_u * h_v, sed_dips_dips ** 2) * (
            tf.where(sed_dips_dips < self.a_T,
                     x=(((-self.c_o_T * ((-14. / self.a_T ** 2.) + 105. / 4. * sed_dips_dips / self.a_T ** 3. -
                                         35. / 2. * sed_dips_dips ** 3. / self.a_T ** 5. +
                                         21. / 4. * sed_dips_dips ** 5. / self.a_T ** 7.))) +
                        self.c_o_T * 7. * (9. * sed_dips_dips ** 5. - 20. * self.a_T ** 2. * sed_dips_dips ** 3. +
                                           15. * self.a_T ** 4. * sed_dips_dips - 4. * self.a_T ** 5.) / (2. * self.a_T ** 7.)), y=0.)) -\
            perpendicularity_matrix * tf.where(sed_dips_dips < self.a_T, x=self.c_o_T * ((-14. / self.a_T ** 2.) + 105. / 4. * sed_dips_dips / self.a_T ** 3. -
                                                                                         35. / 2. * sed_dips_dips ** 3. / self.a_T ** 5. +
                                                                                         21. / 4. * sed_dips_dips ** 5. / self.a_T ** 7.), y=0.)

        C_G = tf.where(sed_dips_dips == 0, x=tf.constant(
            0., dtype=self.dtype), y=condistion_fail)
        C_G = C_G + tf.eye(C_G.shape[0],
                           dtype=self.dtype) * self.nugget_effect_grad

        return C_G

    
    def cartesian_dist(self, x_1, x_2):
        return tf.concat([
            tf.transpose(
                (x_1[:, 0] - tf.expand_dims(x_2[:, 0], axis=1))),
            tf.transpose(
                (x_1[:, 1] - tf.expand_dims(x_2[:, 1], axis=1))),
            tf.transpose(
                (x_1[:, 2] - tf.expand_dims(x_2[:, 2], axis=1)))], axis=0)

    
    def cov_interface_gradients(self, dips_position_all, rest_layer_points, ref_layer_points):
        dips_position_all_tiled = tf.tile(
            dips_position_all, [self.n_dimensions, 1])

        sed_dips_rest = self.squared_euclidean_distance(
            dips_position_all_tiled, rest_layer_points)
        sed_dips_ref = self.squared_euclidean_distance(
            dips_position_all_tiled, ref_layer_points)

        hu_rest = self.cartesian_dist(
            dips_position_all, rest_layer_points)
        hu_ref = self.cartesian_dist(
            dips_position_all, ref_layer_points)

        C_GI = self.gi_rescale * tf.transpose(hu_rest *
                                               tf.where(sed_dips_rest < self.a_T_surface, x=(- self.c_o_T * ((-14 / self.a_T_surface ** 2) + 105 / 4 * sed_dips_rest / self.a_T_surface ** 3 -
                                                                                                             35 / 2 * sed_dips_rest ** 3 / self.a_T_surface ** 5 +
                                                                                                             21 / 4 * sed_dips_rest ** 5 / self.a_T_surface ** 7)), y=tf.constant(0., dtype=self.dtype)) -
                                               (hu_ref * tf.where(sed_dips_ref < self.a_T_surface, x=- self.c_o_T * ((-14 / self.a_T_surface ** 2) + 105 / 4 * sed_dips_ref / self.a_T_surface ** 3 -
                                                                                                                     35 / 2 * sed_dips_ref ** 3 / self.a_T_surface ** 5 +
                                                                                                                     21 / 4 * sed_dips_ref ** 5 / self.a_T_surface ** 7), y=tf.constant(0., dtype=self.dtype))))

        return C_GI

    
    def universal_matrix(self, dips_position_all, ref_layer_points, rest_layer_points):

        n = dips_position_all.shape[0]

        sub_x = tf.tile(tf.constant([[1., 0., 0.]], self.dtype), [n, 1])
        sub_y = tf.tile(tf.constant([[0., 1., 0.]], self.dtype), [n, 1])
        sub_z = tf.tile(tf.constant([[0., 0., 1.]], self.dtype), [n, 1])
        sub_block1 = tf.concat([sub_x, sub_y, sub_z], 0)

        sub_x_2 = tf.reshape(2 * self.gi_rescale *
                             dips_position_all[:, 0], [n, 1])
        sub_y_2 = tf.reshape(2 * self.gi_rescale *
                             dips_position_all[:, 1], [n, 1])
        sub_z_2 = tf.reshape(2 * self.gi_rescale *
                             dips_position_all[:, 2], [n, 1])

        sub_x_2 = tf.pad(sub_x_2, [[0, 0], [0, 2]])
        sub_y_2 = tf.pad(sub_y_2, [[0, 0], [1, 1]])
        sub_z_2 = tf.pad(sub_z_2, [[0, 0], [2, 0]])
        sub_block2 = tf.concat([sub_x_2, sub_y_2, sub_z_2], 0)

        sub_xy = tf.reshape(tf.concat([self.gi_rescale * dips_position_all[:, 1],
                                       self.gi_rescale * dips_position_all[:, 0]], 0), [2 * n, 1])
        sub_xy = tf.pad(sub_xy, [[0, n], [0, 0]])
        sub_xz = tf.concat([tf.pad(tf.reshape(self.gi_rescale * dips_position_all[:, 2], [n, 1]), [
            [0, n], [0, 0]]), tf.reshape(self.gi_rescale * dips_position_all[:, 0], [n, 1])], 0)
        sub_yz = tf.reshape(tf.concat([self.gi_rescale * dips_position_all[:, 2],
                                       self.gi_rescale * dips_position_all[:, 1]], 0), [2 * n, 1])
        sub_yz = tf.pad(sub_yz, [[n, 0], [0, 0]])

        sub_block3 = tf.concat([sub_xy, sub_xz, sub_yz], 1)

        U_G = tf.concat([sub_block1, sub_block2, sub_block3], 1)

        U_I = -tf.stack([self.gi_rescale * (rest_layer_points[:, 0] - ref_layer_points[:, 0]), self.gi_rescale *
                         (rest_layer_points[:, 1] -
                          ref_layer_points[:, 1]),
                         self.gi_rescale *
                         (rest_layer_points[:, 2] -
                          ref_layer_points[:, 2]),
                         self.gi_rescale ** 2 *
                         (rest_layer_points[:, 0] ** 2 -
                          ref_layer_points[:, 0] ** 2),
                         self.gi_rescale ** 2 *
                         (rest_layer_points[:, 1] ** 2 -
                          ref_layer_points[:, 1] ** 2),
                         self.gi_rescale ** 2 *
                         (rest_layer_points[:, 2] ** 2 -
                          ref_layer_points[:, 2] ** 2),
                         self.gi_rescale ** 2 * (
            rest_layer_points[:, 0] * rest_layer_points[:, 1] - ref_layer_points[:, 0] *
            ref_layer_points[:, 1]),
            self.gi_rescale ** 2 * (
            rest_layer_points[:, 0] * rest_layer_points[:, 2] - ref_layer_points[:, 0] *
            ref_layer_points[:, 2]),
            self.gi_rescale ** 2 * (
            rest_layer_points[:, 1] * rest_layer_points[:, 2] - ref_layer_points[:, 1] *
            ref_layer_points[:, 2])], 1)

        return U_G, U_I

    
    def faults_matrix(self, f_ref=None, f_res=None):
        length_of_CG, _, _, length_of_faults = self.matrices_shapes()[
            :4]

        # self.fault_drift_at_surface_points_rest = self.fault_matrix
        # self.fault_drift_at_surface_points_ref = self.fault_matrix

        F_I = (self.fault_drift_at_surface_points_ref -
               self.fault_drift_at_surface_points_rest) + 0.0001

        F_G = tf.zeros((length_of_faults, length_of_CG),
                       dtype=self.dtype) + 0.0001

        return F_I, F_G

    
    def covariance_matrix(self, dips_position_all, ref_layer_points, rest_layer_points):

        length_of_CG, length_of_CGI, length_of_U_I, length_of_faults, length_of_C = self.matrices_shapes()

        C_G = self.cov_gradients(dips_position_all)
        C_I = self.cov_surface_points(ref_layer_points, rest_layer_points)
        C_GI = self.cov_interface_gradients(
            dips_position_all, rest_layer_points, ref_layer_points)
        U_G, U_I = self.universal_matrix(
            dips_position_all, ref_layer_points, rest_layer_points)
        U_G = U_G[:length_of_CG, :3]
        U_I = U_I[:length_of_CGI, :3]
        F_I, F_G = self.faults_matrix()

        A = tf.concat([tf.concat([C_G, tf.transpose(C_GI)], -1),
                       tf.concat([C_GI, C_I], -1)], 0)



        B = tf.concat([U_G, U_I], 0)

        AB = tf.concat([A, B], -1)

        B_T = tf.transpose(B)

        paddings = tf.constant([[0, 0], [0, 3]])
        C = tf.pad(B_T, paddings)

        C_matrix = tf.concat([AB, C], 0)

        C_matrix = tf.where(tf.logical_and(tf.abs(C_matrix) > 0, tf.abs(
            C_matrix) < 1e-9), tf.constant(0, dtype=self.dtype), y=C_matrix)

        return C_matrix

    
    def deg2rad(self, degree_matrix):
        return degree_matrix * tf.constant(0.0174533, dtype=self.dtype)

    
    def b_vector(self, dip_angles_=None, azimuth_=None, polarity_=None):

        length_of_C = self.matrices_shapes()[-1]
        if dip_angles_ is None:
            dip_angles_ = self.dip_angles
        if azimuth_ is None:
            azimuth_ = self.azimuth
        if polarity_ is None:
            polarity_ = self.polarity

        G_x = tf.sin(self.deg2rad(dip_angles_)) * \
            tf.sin(self.deg2rad(azimuth_)) * polarity_
        G_y = tf.sin(self.deg2rad(dip_angles_)) * \
            tf.cos(self.deg2rad(azimuth_)) * polarity_
        G_z = tf.cos(self.deg2rad(dip_angles_)) * polarity_

        G = tf.concat([G_x, G_y, G_z], -1)

        G = tf.expand_dims(G, axis=1)
        b_vector = tf.pad(G, [[0, length_of_C - G.shape[0]], [0, 0]])

        return b_vector

    
    def solve_kriging(self, dips_position_all, ref_layer_points, rest_layer_points, b=None):

        C_matrix = self.covariance_matrix(
            dips_position_all, ref_layer_points, rest_layer_points)

        b_vector = self.b_vector()



        DK = tf.linalg.solve(C_matrix, b_vector)

        return DK

    
    def x_to_interpolate(self, grid, ref_layer_points, rest_layer_points):
        grid_val = tf.concat([grid, rest_layer_points], 0)
        grid_val = tf.concat([grid_val, ref_layer_points], 0)

        return grid_val

    
    def extend_dual_kriging(self, weights, grid_shape):

        DK_parameters = weights

        DK_parameters = tf.tile(DK_parameters, [1, grid_shape])

        return DK_parameters

    
    def contribution_gradient_interface(self, dips_position_all, grid_val=None, weights=None):
        dips_position_all_tiled = tf.tile(
            dips_position_all, [self.n_dimensions, 1])
        length_of_CG = self.matrices_shapes()[0]

        hu_SimPoint = self.cartesian_dist(dips_position_all, grid_val)

        sed_dips_SimPoint = self.squared_euclidean_distance(
            dips_position_all_tiled, grid_val)

        sigma_0_grad = tf.reduce_sum((weights[:length_of_CG] *
                                      self.gi_rescale * (tf.negative(hu_SimPoint) *\
                                                          # first derivative
                                                          tf.where((sed_dips_SimPoint < self.a_T_surface),
                                                                   x=(- self.c_o_T * ((-14 / self.a_T_surface ** 2) + 105 / 4 * sed_dips_SimPoint / self.a_T_surface ** 3 -
                                                                                      35 / 2 * sed_dips_SimPoint ** 3 / self.a_T_surface ** 5 +
                                                                                      21 / 4 * sed_dips_SimPoint ** 5 / self.a_T_surface ** 7)),
                                                                   y=tf.constant(0, dtype=self.dtype)))), 0)


        
        return sigma_0_grad

    
    def contribution_interface(self, ref_layer_points, rest_layer_points, grid_val, weights=None):

        length_of_CG, length_of_CGI = self.matrices_shapes()[:2]

        # Euclidian distances
        sed_rest_SimPoint = self.squared_euclidean_distance(
            rest_layer_points, grid_val)
        sed_ref_SimPoint = self.squared_euclidean_distance(
            ref_layer_points, grid_val)

        sigma_0_interf = tf.reduce_sum(-weights[length_of_CG:length_of_CG + length_of_CGI, :] *
                                       (self.c_o_T * self.i_rescale * (
                                           tf.where(sed_rest_SimPoint < self.a_T_surface,  # SimPoint - Rest Covariances Matrix
                                                    x=(1 - 7 * (sed_rest_SimPoint / self.a_T_surface) ** 2 +
                                                        35 / 4 * (sed_rest_SimPoint / self.a_T_surface) ** 3 -
                                                        7 / 2 * (sed_rest_SimPoint / self.a_T_surface) ** 5 +
                                                        3 / 4 * (sed_rest_SimPoint / self.a_T_surface) ** 7),
                                                    y=tf.constant(0., dtype=self.dtype)) -
                                        (tf.where(sed_ref_SimPoint < self.a_T_surface,  # SimPoint- Ref
                                                  x=(1 - 7 * (sed_ref_SimPoint / self.a_T_surface) ** 2 +
                                                     35 / 4 * (sed_ref_SimPoint / self.a_T_surface) ** 3 -
                                                      7 / 2 * (sed_ref_SimPoint / self.a_T_surface) ** 5 +
                                                      3 / 4 * (sed_ref_SimPoint / self.a_T_surface) ** 7),
                                                  y=tf.constant(0., dtype=self.dtype))))), 0)


        return sigma_0_interf

    
    def contribution_universal_drift(self, grid_val, weights=None):

        length_of_CG, length_of_CGI, length_of_U_I, length_of_faults, length_of_C = self.matrices_shapes()

        _submatrix1 = tf.transpose(tf.concat([grid_val, grid_val**2], 1))

        _submatrix2 = tf.stack([grid_val[:, 0] * grid_val[:, 1], grid_val[:, 0]
                                * grid_val[:, 2], grid_val[:, 1] * grid_val[:, 2]])

        universal_grid_surface_points_matrix = tf.concat(
            [_submatrix1, _submatrix2], 0)

        i_rescale_aux = tf.concat([tf.ones([3], dtype=self.dtype), tf.tile(
            tf.expand_dims(self.gi_rescale, 0), [6])], -1)

        _aux_magic_term = tf.tile(tf.expand_dims(
            i_rescale_aux[:self.n_universal_eq_T_op], 0), [tf.shape(grid_val)[0], 1])

        f_0 = tf.reduce_sum(weights[length_of_CG + length_of_CGI:length_of_CG + length_of_CGI + length_of_U_I] * self.gi_rescale *
                            tf.transpose(_aux_magic_term) *
                            universal_grid_surface_points_matrix[:self.n_universal_eq_T_op], 0)

        return f_0
    
    
    def scalar_field(self, surface_point,dips_position):

        self.dips_position_all_tiled = tf.tile(
            dips_position, [self.n_dimensions, 1])

        self.ref_layer_points, self.rest_layer_points, self.ref_nugget, self.rest_nugget = self.set_rest_ref_matrix(
            self.number_of_points_per_surface, surface_point, self.nugget_effect_scalar)

        self.nugget_effect_scalar_ref_rest = tf.expand_dims(
            self.ref_nugget + self.rest_nugget, 1)

        self.len_points = surface_point.shape[0] - \
            self.number_of_points_per_surface.shape[0]

        interface_loc = self.grid_val.shape[0]
        self.fault_drift_at_surface_points_rest = self.fault_matrix[
            :, interface_loc: interface_loc + self.len_points]
        self.fault_drift_at_surface_points_ref = self.fault_matrix[
            :, interface_loc + self.len_points:]

        grid_val = self.x_to_interpolate(
            self.grid_val, self.ref_layer_points, self.rest_layer_points)
        weights = self.solve_kriging(
            dips_position, self.ref_layer_points, self.rest_layer_points)


        tiled_weights = self.extend_dual_kriging(
            weights, tf.shape(grid_val)[0])
        sigma_0_grad = self.contribution_gradient_interface(dips_position,
                                                            grid_val, tiled_weights)
        sigma_0_interf = self.contribution_interface(
            self.ref_layer_points, self.rest_layer_points, grid_val, tiled_weights)
        f_0 = self.contribution_universal_drift(grid_val, weights)

        scalar_field_results = sigma_0_grad + sigma_0_interf + f_0

        return scalar_field_results


    
    def get_scalar_field_at_surface_points(self, Z_x):
        partitions = tf.reduce_sum(tf.one_hot(self.npf, self.len_points, dtype='int32'), axis=0)
        #
        scalar_field_at_surface_points_values = tf.dynamic_partition(Z_x[-2 * self.len_points: -self.len_points],partitions,2)[1]
        # scalar_field_at_surface_points_values = tf.gather_nd(
        #     Z_x[-2 * self.len_points: -self.len_points], tf.expand_dims(self.npf, 1))

        return scalar_field_at_surface_points_values
    
    
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
            Tensor: segmented values
        """
        n_surface_0 = n_surface[:, slice_init:slice_init + 1]
        n_surface_1 = n_surface[:, slice_init + 1:slice_init + 2]
        drift = drift[:, slice_init:slice_init + 1]

        # The 5 rules the slope of the function
        sigm = (-tf.reshape(n_surface_0, (-1, 1)) / (1 + tf.exp(-l * (Z_x - a)))) - \
            (tf.reshape(n_surface_1, (-1, 1)) /
             (1 + tf.exp(l * (Z_x - b)))) + tf.reshape(drift, (-1, 1))
        sigm.set_shape([None, None])
        return sigm

    
    def export_formation_block(self, Z_x, scalar_field_at_surface_points, values_properties):

        

        slope = self.sig_slope
        scalar_field_iter = tf.pad(tf.expand_dims(
            scalar_field_at_surface_points, 0), [[0, 0], [1, 1]])[0]

        n_surface_op_float_sigmoid_mask = tf.repeat(
            values_properties, 2, axis=1)
        n_surface_op_float_sigmoid = tf.pad(
            n_surface_op_float_sigmoid_mask[:, 1:-1], [[0, 0], [1, 1]])
        drift = tf.pad(
            n_surface_op_float_sigmoid_mask[:, 0:-1], [[0, 0], [0, 1]])
        formations_block = tf.zeros([1, tf.shape(Z_x)[0]], dtype=self.dtype)

        # need to check if Hessian works for this, otherwise vectorize
        # code for vectorization
        # tf.concat([tf.expand_dims(tf.range(scalar_field_iter.shape[1]-1),1),
        # tf.expand_dims(tf.range(scalar_field_iter.shape[1]-1)+1,1)],-1)

        for i in tf.range(tf.shape(scalar_field_iter)[0] - 1):
        # for i in range(3):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[(formations_block, tf.TensorShape([None, Z_x.shape[0]]))])

            formations_block = formations_block + \
                self.compare(scalar_field_iter[i], scalar_field_iter[i + 1],
                             2 * i, Z_x, slope, n_surface_op_float_sigmoid, drift)

        if self.gradient is True:
            ReLU_up = - 0.01 * tf.nn.relu(Z_x - scalar_field_iter[1])
            ReLU_down = 0.01 * tf.nn.relu(Z_x - scalar_field_iter[-2])

            formations_block += ReLU_down + ReLU_up

        return formations_block

    
    def compute_forward_gravity(self, tz, lg0, lg1, densities=None):
        n_devices = tf.math.floordiv((tf.shape(densities)[0]), tf.shape(tz)[0])
        tz_rep = tf.tile(tz, [n_devices])
        grav = tf.reduce_sum(tf.reshape(densities * tz_rep, [n_devices, -1]), axis=1)
        return grav


