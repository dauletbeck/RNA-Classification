from math import atan2, pi, sqrt, cos, sin
import numpy as np
from scipy.optimize import leastsq
import scipy.optimize as opt
import os
import warnings
from utils.plot_functions import build_fancy_chain_plot
from utils.constants import BACKBONE_ATOMS_VALIDATION


def dihedral(point_list, verbose=False, rna_distances=True, long=False):
    """
    This function gets a list of four points in |R^3 and returns the dihedral angle.
    :param point_list: A np.array of shape (4, 3).
    :param verbose: If False no documentation.
    :param rna_distances: If this parameter is wrong then every angle will be calculated.
    :param long: Tolerance parameter for the distances of the atoms.
    :return: The dihedral angle.
    """
    b = [__diff(point_list[i], point_list[i + 1]) for i in range(3)]
    for bi in b:
        bi = __dot(bi, bi)
        # here a change:
        if bi > (20 if long else 3) and rna_distances:
            if verbose:
                print('Atoms too far apart:', bi)
            return None
    c = [__x(b[i], b[i + 1]) for i in range(2)]
    tmp = atan2(__dot(__x(c[0], c[1]), __n(b[1])), __dot(c[0], c[1]))
    return (360 - tmp * (180 / pi)) % 360


def __diff(p, q):
    """
    Calculates p-q for two lists or arrarys p and q. This is a help function for dihedral.
    :param p: An array/list with 3 values.
    :param q: An array/list with 3 values.
    :return: An array/list with 3 values.
    """
    return [p[i] - q[i] for i in range(3)]


def __x(p, q):
    """
    The cross product of p and q. This is a help function for dihedral.
    :param p: An array/list with 3 values.
    :param q: An array/list with 3 values.
    :return: An array/list with 3 values.
    """
    return [p[1] * q[2] - p[2] * q[1], p[2] * q[0] - p[0] * q[2], p[0] * q[1] - p[1] * q[0]]


def __dot(p, q):
    """
    The dot product of two lists/arrays p and q.
    :param p: An array/list with 3 values.
    :param q: An array/list with 3 values.
    :return: A scalar
    """
    return p[0] * q[0] + p[1] * q[1] + p[2] * q[2]


def __n(v):
    """
    This function gets a list or an array and returns the same list/array divided by its norm. This is a help function
    for dihedral.
    :param v: An array/list with 3 values.
    :return: An array/list with 3 values.
    """
    tmp = sqrt(__dot(v, v))
    return [x / tmp for x in v]


def calculate_angle_3_points(point_list):
    """
    This function calculates and returns the angle between 3 points in |R^3.
    :param point_list: A list of three points.
    :return: The angle between the three points.
    """
    v_1 = point_list[0] - point_list[1]
    v_2 = point_list[2] - point_list[1]
    angle = np.arccos(np.dot(v_1, v_2) / (np.linalg.norm(v_1) * np.linalg.norm(v_2))) * 180 / np.pi
    return angle


def procrustes_algorithm_short(input_data, plot_string=None, shape=False, origin_index=None, mean_shape=None):
    # step 1 of the procrustes algorithm: shift and center the data.
    shift_array = np.zeros((input_data.shape[0], input_data.shape[2]))
    scale_array = np.zeros(input_data.shape[0])
    rotation_matrices = np.array([np.eye(3)] * input_data.shape[0])

    procrustes_data = input_data.copy()
    for i in range(input_data.shape[0]):
        if origin_index is None:
            shift_array[i] = np.mean(procrustes_data[i], axis=0)
        else:
            shift_array[i] = procrustes_data[i][origin_index]

        procrustes_data[i] = procrustes_data[i] - shift_array[i]
        if shape:
            # changed for shape space (instead of size and shape)
            scale_array[i] = np.linalg.norm(procrustes_data[i])
            procrustes_data[i] = procrustes_data[i] / scale_array[i]

    # Start shape:
    if mean_shape is None:
        mean_shape = procrustes_data[0].copy()
    abs_distance = 1
    counter = 1
    while abs_distance > 1e-06:
        old_procrustes = procrustes_data.copy()
        for i in range(input_data.shape[0]):
            x = procrustes_data[i]
            y = mean_shape
            procrustes_data[i], rotation_matrix = rotate_y_optimal_to_x(y, x, True)
            rotation_matrices[i] = np.dot(rotation_matrix, rotation_matrices[i])
        mean_shape = np.mean(procrustes_data, axis=0)
        if origin_index is None:
            mean_shape = mean_shape - np.mean(mean_shape, axis=0)
        if shape:
            mean_shape = mean_shape / np.linalg.norm(mean_shape)
        abs_distance = np.linalg.norm(mean_shape - y)
        counter = counter + 1
        print(abs_distance)
        print('procrustes distance:', np.linalg.norm(procrustes_data - old_procrustes))

    # print(f'---------- mean shape: {mean_shape}')
    if plot_string is not None:
        if not os.path.exists(plot_string):
            os.makedirs(plot_string)
        color = ['black'] * old_procrustes.shape[0] + ['red'] + ['blue']
        different_lw = [0.1] * (len(color) - 2) + [2.5] + [1]
        alpha_line = [0.3] * (len(color) - 2) + [1]
        build_fancy_chain_plot(np.vstack((procrustes_data,
                                          mean_shape.reshape((1, input_data.shape[1], 3)))), colors=color,
                               atom_size_vector=different_lw, lw_vec=different_lw,
                               atom_alpha=0.3,
                               alpha_line=0.3, without_legend=True, alpha_line_vec=alpha_line,
                               atom_alpha_vector=alpha_line, atom_color_vector=color,
                               filename=plot_string + '/' + str(counter))  # , plot_atoms=True)
    return procrustes_data, shift_array, scale_array, rotation_matrices


def calculate_rotation_matrix(alpha):
    """
    This function calculates the rotation matrix for a list with three angles.
    :param alpha: A list with three angles.
    :return:
    """
    rotation_matrix = np.dot(np.dot(np.transpose(rotation_matrix_x_axis(alpha[0][0])),
                                    np.transpose(rotation_matrix_y_axis(alpha[0][1]))),
                             np.transpose(rotation_matrix_z_axis(alpha[0][2])))
    return rotation_matrix


def rotation_function(alpha, x, y):
    """
    This function calculate y-R(alpha)x for the arrays x amd y and the the rotation matrix R(alpha). The array alpha
    contains three angles.
    :param alpha:
    :param x:
    :param y:
    :return:
    """
    x_rot = np.dot(x, np.transpose(rotation_matrix_x_axis(alpha[0])))
    x_y_rot = np.dot(x_rot, np.transpose(rotation_matrix_y_axis(alpha[1])))
    x_y_z_rot = np.dot(x_y_rot, np.transpose(rotation_matrix_z_axis(alpha[2])))
    return np.ndarray.flatten(y - x_y_z_rot)


def rotation_function_first_and_last_landmark(alpha, x, y):
    """
    This function calculate y-R(alpha)x for the arrays x amd y and the the rotation matrix R(alpha). The array alpha
    contains three angles.
    :param alpha:
    :param x:
    :param y:
    :return:
    """
    x_rot = np.dot(x, np.transpose(rotation_matrix_x_axis(alpha[0])))
    x_y_rot = np.dot(x_rot, np.transpose(rotation_matrix_y_axis(alpha[1])))
    x_y_z_rot = np.dot(x_y_rot, np.transpose(rotation_matrix_z_axis(alpha[2])))
    return np.ndarray.flatten(y[[0, x_y_z_rot.shape[0] - 1]] - x_y_z_rot[[0, x_y_z_rot.shape[0] - 1]])


def rotation_matrix_axis_angle(v, alpha):
    R = np.array([[v[0] ** 2 * (1 - np.cos(alpha)) + np.cos(alpha),
                   v[0] * v[1] * (1 - np.cos(alpha)) - v[2] * np.sin(alpha),
                   v[0] * v[2] * (1 - np.cos(alpha)) + v[1] * np.sin(alpha)],
                  [v[0] * v[1] * (1 - np.cos(alpha)) + v[2] * np.sin(alpha),
                   v[1] ** 2 * (1 - np.cos(alpha)) + np.cos(alpha),
                   v[1] * v[2] * (1 - np.cos(alpha)) - v[0] * np.sin(alpha)],
                  [v[0] * v[2] * (1 - np.cos(alpha)) - v[1] * np.sin(alpha),
                   v[1] * v[2] * (1 - np.cos(alpha)) + v[0] * np.sin(alpha),
                   v[2] ** 2 * (1 - np.cos(alpha)) + np.cos(alpha)]])
    return R


def distance_rotation(alpha, v, x, y):
    x_rot = np.dot(x, rotation_matrix_axis_angle(v, alpha))
    return np.ndarray.flatten(y.reshape((6, 3)) - x_rot.reshape((6, 3)))


def rotation_matrix_x_axis(angle):
    """
    This function returns the rotation matrix around the x-axis.
    :param angle: The input angle.
    :return: A rotation matrix with the dimension 3 x 3.
    """
    return np.array([[1, 0, 0],
                     [0, cos(angle), -sin(angle)],
                     [0, sin(angle), cos(angle)]])


def rotation_matrix_y_axis(angle):
    """
    This function returns the rotation matrix around the y-axis.
    :param angle: The input angle.
    :return: A rotation matrix with the dimension 3 x 3.
    """
    return np.array([[cos(angle), 0, -sin(angle)],
                     [0, 1, 0],
                     [sin(angle), 0, cos(angle)]])


def rotation_matrix_z_axis(angle):
    """
    This function returns the rotation matrix around the z-axis.
    :param angle: The input angle.
    :return: A rotation matrix with the dimension 3 x 3.
    """
    return np.array([[cos(angle), sin(angle), 0],
                     [-sin(angle), cos(angle), 0],
                     [0, 0, 1]])


def procrustes_algorithm(input_data, starting_int=9, number_points=6, number_extra_points=0, backbone=False,
                         plot_sub=None, important_plot=False, return_procrustes_steps=False):
    # step 1 of the procrustes algorithm: shift and center the data.
    reshape_data = input_data[:, starting_int:starting_int + number_points * 3].reshape(
        (input_data.shape[0], number_points, 3))
    if number_extra_points > 0:
        reshape_data_extra = input_data[:,
                             starting_int:(starting_int + number_points + number_extra_points) * 3].reshape(
            (input_data.shape[0],
             number_points + number_extra_points, 3)).copy()
    for i in range(reshape_data.shape[0]):
        mean = np.mean(reshape_data[i], axis=0)
        reshape_data[i] = reshape_data[i] - mean
        norm = np.linalg.norm(reshape_data[i])
        reshape_data[i] = reshape_data[i] / norm
        if number_extra_points > 0:
            reshape_data_extra[i] = reshape_data_extra[i] - mean
            reshape_data_extra[i] = reshape_data_extra[i] / norm

    # Start shape:
    procrustes_data = reshape_data.copy()
    if number_extra_points > 0:
        procrustes_data_extra = reshape_data_extra.copy()
    mean_shape = reshape_data[0]
    abs_distance = 1
    counter = 1
    while abs_distance > 1e-06:
        old_procrustes = procrustes_data.copy()
        for i in range(reshape_data.shape[0]):
            x = procrustes_data[i]
            y = mean_shape
            alpha = leastsq(rotation_function, x0=np.array([0.0, 0.0, 0.0]), args=(x, y))
            x_rot = np.dot(x, np.transpose(rotation_matrix_x_axis(alpha[0][0])))
            x_y_rot = np.dot(x_rot, np.transpose(rotation_matrix_y_axis(alpha[0][1])))
            procrustes_data[i] = np.dot(x_y_rot, np.transpose(rotation_matrix_z_axis(alpha[0][2])))
            if number_extra_points > 0:
                x_rot_extra = np.dot(procrustes_data_extra[i], np.transpose(rotation_matrix_x_axis(alpha[0][0])))
                x_y_rot_extra = np.dot(x_rot_extra, np.transpose(rotation_matrix_y_axis(alpha[0][1])))
                procrustes_data_extra[i] = np.dot(x_y_rot_extra, np.transpose(rotation_matrix_z_axis(alpha[0][2])))
        mean_shape = np.mean(procrustes_data, axis=0)
        mean_shape = mean_shape - np.mean(mean_shape)
        mean_shape = mean_shape / np.linalg.norm(mean_shape)
        abs_distance = np.linalg.norm(mean_shape - y)

        name = './out/procrustes/' + str(backbone)
        if plot_sub is not None:
            name = name + '_subplot_nr_' + str(plot_sub)
        if not os.path.exists(name):
            os.makedirs(name)
        color = ['black'] * old_procrustes.shape[0] + ['red'] + ['blue']
        if important_plot:
            different_lw = [0.1] * (len(color) - 2) + [1.5] + [1]
            build_fancy_chain_plot(np.vstack((np.vstack((procrustes_data, y.reshape((1, number_points, 3)))),
                                              mean_shape.reshape((1, number_points, 3)))), colors=color,
                                   atom_color_vector=color, plot_atoms=True, without_legend=True,
                                   filename=name + '/' + str(counter), lw_vec=different_lw,
                                   atom_size_vector=different_lw,
                                   atom_alpha=0.3, alpha_line=0.3)
        else:
            build_fancy_chain_plot(np.vstack((np.vstack((procrustes_data, y.reshape((1, number_points, 3)))),
                                              mean_shape.reshape((1, number_points, 3)))), colors=color,
                                   filename=name + '/' + str(counter))
        counter = counter + 1
        print(abs_distance)
        print('procrustes distance:', np.linalg.norm(procrustes_data - old_procrustes))
    if number_extra_points > 0:
        return procrustes_data_extra
    return procrustes_data


def mean_on_sphere(points):
    d = points.shape[1]
    def f(x):
        return np.arccos(np.einsum('ij,j->i', points, x / np.linalg.norm(x)).clip(-1, 1))
    mean = __fit(f, d)
    variance = np.mean(f(mean) ** 2)
    return mean / np.linalg.norm(mean), variance


def mean_on_sphere_init(points):
    d = points.shape[1]
    def f(x):
        return np.arccos(np.einsum('ij,j->i', points, x / np.linalg.norm(x)).clip(-1, 1))
    init = np.mean(points, axis=0)/ np.linalg.norm(np.mean(points, axis=0))
    mean = __fit_initial(f, d, init)
    variance = np.mean(f(mean) ** 2)
    return mean / np.linalg.norm(mean), variance

def __fit(f, d):
    #warnings.filterwarnings('error')
    tol = 1e-8
    initial = np.random.rand(d)
    exit_code = 6
    fails = -1
    counter = 0
    EPS = 1e-8
    while exit_code > 1 and np.linalg.norm(f(initial)) > EPS:
        fails += 1
        if fails > 3 or exit_code == 6:
            initial = np.random.rand(d)
            fails = 0
        try:
            out = opt.least_squares(f, initial, ftol=tol, xtol=tol, gtol=tol)
            initial = out.x
            exit_code = out.status
        except:
            exit_code = 6
            counter += 1
        if counter > 20:
            return None
        tol *= 2
    return initial

def __fit_initial(f, d, init):
    #warnings.filterwarnings('error')
    tol = 1e-8
    initial = init
    out = opt.least_squares(f, initial, ftol=tol, xtol=tol, gtol=tol)
    initial = out.x
    return initial


def rotation(v_from, v_to):
    """
    This function returns a rotation matrix. Therefore v_from and v_to must have the property norm(v_from) and
    norm(v_to)=1.
    :param v_from: The array which should be rotated.
    :param v_to: The vector to be rotated on.
    :return: The rotation matrix.
    """
    prod = float(np.einsum('i,i->', v_from, v_to))
    v_aux = v_from - prod * v_to
    if np.linalg.norm(v_aux) == 0:
        return np.eye(len(v_from))
    v_aux /= np.linalg.norm(v_aux)
    m1 = np.einsum('i,j->ij', v_aux, v_to)
    m1 = m1.T - m1
    m2 = np.einsum('i,j->ij', v_aux, v_aux) + np.einsum('i,j->ij', v_to, v_to)
    return np.eye(len(v_from)) + np.sqrt(1 - prod ** 2) * m1 + (prod - 1) * m2


def return_index_of_smallest_k_elements(list, k):
    index_set = [np.argmin(list)]
    for i in range(k - 1):
        sub_list = [list[j] if j not in index_set else np.inf for j in range(len(list))]
        index_set = index_set + [np.argmin(sub_list)]
    return index_set


def return_atom_number(clash_atom, suite_number, return_list=False):
    if clash_atom in BACKBONE_ATOMS_VALIDATION:
        atom_backbone_list_double = [["C5'", "H5'", "H5''"], ["C4'", "H4'"], ["C3'", "H3'"], ["O3'"]]
        hydrogen_atoms = ["H5'", "H5''", "H4'", "H3'"]
        oxygen_atoms = ["OP1", "OP2"]
        atom_backbone_list_single = [['P'], ["O5'"]]
        if suite_number == 0:
            hydrogen_number = 14
        else:
            hydrogen_number = 10
        for j in range(4):
            if clash_atom == hydrogen_atoms[j]:
                hydrogen_number = hydrogen_number + j
        for j in range(2):
            if clash_atom == oxygen_atoms[j]:
                oxgen_number = 32 + j
        for j in range(4):
            if clash_atom in atom_backbone_list_double[j]:
                if suite_number == 0:
                    if return_list is False:
                        return j + 6
                    else:
                        if clash_atom == atom_backbone_list_double[j][0]:
                            return [j + 6]
                        else:
                            return [j + 6, hydrogen_number]

                else:
                    if return_list is False:
                        return j
                    else:
                        if clash_atom == atom_backbone_list_double[j][0]:
                            return [j]
                        else:
                            return [j, hydrogen_number]

        for j in range(2):
            if clash_atom in atom_backbone_list_single[j]:
                if return_list is False:
                    return j + 4
                else:
                    return [j + 4]
            if clash_atom in oxygen_atoms:
                return [4, oxgen_number]

    else:
        atom_ring_list_double = [["O4'"], ["C1'", "H1'"], ["C2'", "H2'"], ["O2'", "HO2'"]]
        ring_hydrogen = ["H1'", "H2'", "HO2'"]
        if suite_number == 0:
            int_value = 22
            int_value_hydrogen = 28
        else:
            int_value = 18
            int_value_hydrogen = 25
        for j in range(len(atom_ring_list_double)):
            if clash_atom in atom_ring_list_double[j]:
                if clash_atom in ring_hydrogen:
                    # print(clash_atom, suite_number, [int_value+j, int_value_hydrogen+j])
                    return [int_value + j, int_value_hydrogen + j]
                else:
                    # print(clash_atom, suite_number, [int_value+j])
                    return [int_value + j]


def rotate_y_optimal_to_x(x, y, return_rotation=False):
    u, d, v = np.linalg.svd(x.T @ y)
    if np.linalg.det(u) * np.linalg.det(v) > 0:
        s = np.eye(3)
    else:
        s = np.diag([1, 1, -1])
    rotation_matrix = u @ s @ v
    a = np.transpose(rotation_matrix @ y.T)
    if not return_rotation:
        return a
    else:
        return a, rotation_matrix
