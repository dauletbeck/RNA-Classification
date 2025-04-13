import os

import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

ATOM_COLORS = ['darkred', 'midnightblue', 'magenta']

def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


def help_chain_plot_legend_and_axis(chains, diag, **kwargs):
    """
    This function is a help function for the 'build_fancy_chain_plot'. It is responsible for the legend and for the
    scale of the axes.
    :param diag: plot parameter.
    :param maximum: float value for the axes.
    :param minimum: float value for the axes.
    :param kwargs: strings: 'xlim', 'ylim', 'zlim', 'x_label', 'y_label', 'z_label'
                   list with two integers: 'xlim', 'ylim', 'zlim'
    :return:
    """
    # set the x,y and z lims: If 'xlim', 'ylim' and 'zlim' not in kwars then the scale of all axis is equal.

    maximum = np.max(np.max(chains, axis=0), axis=0)
    minimum = np.min(np.min(chains, axis=0), axis=0)
    #max_range = np.absolute(0.5 * np.max(maximum - minimum))
    max_range_vector = np.absolute(0.5 * (maximum - minimum))
    x_scale = max_range_vector[0]
    y_scale = max_range_vector[1]
    z_scale = max_range_vector[2]

    constant = 2.5
    if x_scale/np.max([x_scale, y_scale, z_scale]) < 1/constant:
        x_scale = np.max([x_scale, y_scale, z_scale])/constant
    if y_scale/np.max([x_scale, y_scale, z_scale]) < 1/constant:
        y_scale = np.max([x_scale, y_scale, z_scale])/constant
    if z_scale/np.max([x_scale, y_scale, z_scale]) < 1/constant:
        z_scale = np.max([x_scale, y_scale, z_scale])/constant

    scale = np.diag([x_scale, y_scale, z_scale, 1.0])
    scale = scale*(1.0/scale.max())
    scale[3, 3] = 1.0
    if 'not_scale' not in kwargs:
        def short_proj():
            return np.dot(Axes3D.get_proj(diag), scale)
        diag.get_proj=short_proj
    means = 0.5 * (maximum + minimum)
    if 'xlim' in kwargs:
        diag.set_xlim(*kwargs['xlim'])
    else:
        #diag.set_xlim(means[0] - max_range, means[0] + max_range)
        diag.set_xlim(means[0] - x_scale, means[0] + x_scale)
    if 'ylim' in kwargs:
        diag.set_ylim(*kwargs['ylim'])
    else:
        #diag.set_ylim(means[1] - max_range, means[1] + max_range)
        diag.set_ylim(means[1] - y_scale, means[1] + y_scale)
    if 'zlim' in kwargs:
        diag.set_zlim(*kwargs['zlim'])
    else:
        #diag.set_zlim(means[2] - max_range, means[2] + max_range)
        diag.set_zlim(means[2] - z_scale, means[2] + z_scale)


    # scale = np.diag([x_scale, y_scale, z_scale, 1.0])
    # scale = scale*(1.0/scale.max())
    # scale[3, 3] = 1.0
    # if 'not_scale' not in kwargs:
    #     def short_proj():
    #         return np.dot(Axes3D.get_proj(diag), scale)
    #
    #     diag.get_proj=short_proj
    if 'specific_axis_label_size' not in kwargs:
        diag.tick_params(direction='out', length=6, width=2, labelsize=4.5)
    else:
        diag.tick_params(direction='out', length=6, width=2, labelsize=kwargs['specific_axis_label_size'])
    #axisEqual3D(diag)
    if 'without_legend' not in kwargs:
        leg = diag.legend(loc='upper center', fontsize='small')
        # set the linewidth of each legend object
        for legobj in leg.legendHandles:
            legobj.set_linewidth(2.0)
            legobj.set_alpha(1)

    # Set the labels of the x,y and z-axes.
    #diag.set_xlabel(kwargs['x_label']) if 'x_label' in kwargs else diag.set_xlabel('x')
    #diag.set_xlabel(kwargs['y_label']) if 'y_label' in kwargs else diag.set_ylabel('z')
    #diag.set_xlabel(kwargs['z_label']) if 'z_label' in kwargs else diag.set_zlabel('y')


def help_chain_plot(chains, diag, **kwargs):
    """
    This function is a help function for the 'build_fancy_chain_plot'. It is responsible for 'chain_plot'. One can plot
    all chains in the same color (if colors is not in kwargs), different colors (colors is in matrix) or each connection
    line in a different color (if color_matrix is in kwargs and has dimension (number of chains) x (number of points).
    The same holds for alpha_line_matrix and the lw_matrix.
    :param chains: (nr elements) x (nr points) x (3)
    :param create_label: If True a label is created.
    :param kwargs: 'colors', 'color_matrix', 'lw', 'lw_vev', 'lw_matrix', 'alpha_line', 'alpha_line_vec',
                   'alpha_line_matrix', 'chain_length'
    :return:
    """
    color = kwargs['color'] if 'color' in kwargs else 'black'
    colors = kwargs['colors'] if 'colors' in kwargs else len(chains) * [color]
    lw = kwargs['lw'] if 'lw' in kwargs else 0.1
    lw_vec = kwargs['lw_vec'] if 'lw_vec' in kwargs else len(chains) * [lw]
    alpha_line = kwargs['alpha_line'] if 'alpha_line' in kwargs else 1
    alpha_line_vec = kwargs['alpha_line_vec'] if 'alpha_line_vec' in kwargs else len(chains) * [alpha_line]
    chain_length = kwargs['chain_length'] if 'chain_length' in kwargs else chains.shape[1]
    for i, chain in enumerate(chains[:, :chain_length, :]):
        # one chain has one color, one alpha and one lw value:
        if (not 'alpha_line_matrix' in kwargs) and (not 'lw_matrix' in kwargs) and (not 'color_matrix' in kwargs):
            diag.plot(chain[:, 0], chain[:, 1], chain[:, 2], c=colors[i], linewidth=lw_vec[i], alpha=alpha_line_vec[i])

        # one chain has one or more color, alpha and lw value:
        else:
            for j in range(chain.shape[0] - 1):
                diag.plot(chain[j:j + 2, 0], chain[j:j + 2, 1], chain[j:j + 2, 2],
                          c=(kwargs['color_matrix'][i][j] if 'color_matrix' in kwargs else colors[i]),
                          linewidth=(kwargs['lw_matrix'][i][j] if 'lw_matrix' in kwargs else lw_vec[i]),
                          alpha=(kwargs['alpha_line_matrix'][i][j] if 'alpha_line_matrix' in kwargs
                                 else alpha_line_vec[i]))
    # Create the legend of the chains:
    if (kwargs['create_label'] if 'create_label' in kwargs else True):
        create_label_of_legend(colors, diag, kwargs)


def create_specific_legend(chains, diag, **kwargs):
    specific_legend_strings = kwargs['specific_legend_strings']
    specific_legend_colors = kwargs['specific_legend_colors']
    for i in range(len(specific_legend_colors)):
        diag.plot([], [], [], c=specific_legend_colors[i], label=specific_legend_strings[i])


def create_label_of_legend(colors, diag, kwargs):
    chain_string = kwargs['chain_legend_string'] if 'chain_legend_string' in kwargs else ' elements'
    chain_string_first = kwargs['chain_legend_string_first'] if 'chain_legend_string_first' in kwargs else ''
    if len(set(colors)) == 1:
        diag.plot([], [], [], c=colors[0], label=chain_string_first + str(len(colors)) + chain_string)
    else:
        counter_cluster = 1
        for i in range(len(colors)):
            if 'first_chain_legend_string' in kwargs:
                first_chain_string = kwargs['first_chain_legend_string']
            else:
                first_chain_string = 'Cluster ' + str(counter_cluster) + ' with ' + str(sum([colors[j] == colors[i] for j in range(len(colors))]))
            if i == 0:
                diag.plot([], [], [], c=colors[i], label='Cluster ' + str(counter_cluster) + ' with ' +
                                                         str(sum([colors[j] == colors[i] for j in range(len(colors))]))
                                                         + chain_string)
                counter_cluster = counter_cluster + 1
                #  diag.plot([], [], [], c=colors[i], label='Repaired suites  ' +
                #                                          str(sum([colors[j] == colors[i] for j in range(len(colors))])))
            else:
                if colors[i - 1] != colors[i]:
                    counter_cluster = counter_cluster + 1
                    diag.plot([], [], [], c=colors[i],
                              label=first_chain_string + chain_string)
                    # diag.plot([], [], [], c=colors[i], label='Added RNA suites ' +
                    #                                          str(sum([colors[j] == colors[i] for j in range(len(colors))])))


def plot_biggest_backbone_cluster(diag, **kwargs):
    """
    This function is a help function for the 'build_fancy_chain_plot'. It plots the two largest backbone cluster with a
    very small lw and a small alpha value.
    :param diag:
    :param kwargs:
    :return:
    """
    background_bb_color = kwargs['background_bb_color'] if 'background_bb_color' in kwargs else 'orange'
    background_bb_lw = kwargs['background_bb_lw'] if 'background_bb_lw' in kwargs else 0.03
    alpha_bb_first = kwargs['alpha_bb_first'] if 'alpha_bb_first' in kwargs else 0.03
    alpha_bb_second = kwargs['alpha_bb_second'] if 'alpha_bb_second' in kwargs else 0.6
    # plot the first cluster:
    if os.path.isfile('./out/procrustes_backbone_first.csv'):
        biggest_cluster_chains = pd.read_csv('./out/procrustes_backbone_first.csv', header=None).to_numpy()
        procrustes_data_backbone = biggest_cluster_chains.reshape((biggest_cluster_chains.shape[0], 10, 3))
        # plot the first cluster:
        help_chain_plot(procrustes_data_backbone, diag, create_label=False, lw=background_bb_lw,
                        alpha_line=alpha_bb_first, color=background_bb_color)
    else:
        print('Cant plot the first backbone cluster')
    # plot the second cluster:
    if os.path.isfile('./out/procrustes_backbone_second.csv'):
        biggest_cluster_chains = pd.read_csv('./out/procrustes_backbone_second.csv', header=None).to_numpy()
        procrustes_data_backbone = biggest_cluster_chains.reshape((biggest_cluster_chains.shape[0], 10, 3))
        help_chain_plot(procrustes_data_backbone, diag, create_label=False, lw=background_bb_lw,
                        alpha_line=alpha_bb_second, color=background_bb_color)
        diag.plot([], [], [], c=background_bb_color, linewidth=background_bb_lw, alpha=alpha_bb_second,
                  label='The largest and second largest \nsuite-MAKP cluster')

    else:
        print('Cant plot the second backbone cluster')


def plot_biggest_mesoscopic_cluster(diag, **kwargs):
    background_color = kwargs['background_color'] if 'background_color' in kwargs else 'orange'
    background_lw = kwargs['background_lw'] if 'background_lw' in kwargs else 0.01
    alpha_largest = kwargs['alpha_first'] if 'alpha_first' in kwargs else 0.2
    # plot the largest cluster:
    if os.path.isfile('./out/procrustes_shape_biggest_cluster.csv'):
        biggest_cluster_chains = pd.read_csv('./out/procrustes_shape_biggest_cluster.csv', header=None).to_numpy()
        biggest_data = biggest_cluster_chains.reshape((biggest_cluster_chains.shape[0], 6, 3))
        help_chain_plot(biggest_data, diag, create_label=False, lw=background_lw, alpha_line=alpha_largest,
                        color=background_color)
        diag.plot([], [], [], c=background_color, label='The largest mesoscopic-MAKP cluster')

    else:
        print('Cant plot the largest mesoscopic cluster')


def plot_backbone_atoms(chains, diag, **kwargs):
    """
    This function is a help function for the 'build_fancy_chain_plot'. It plots the carbon, oxygen and phosphat atoms.
    :param diag: A plot parameter.
    :param chains: (nr elements) x (10) x (3)
    :param kwargs: 'alpha_backbone_atoms', 'atom_size'
    :return:
    """
    alpha_atoms = kwargs['alpha_backbone_atoms'] if 'alpha_backbone_atoms' in kwargs else 0.9
    atom_size = kwargs['atom_size'] if 'atom_size' in kwargs else 0.5
    atom_size_vector = kwargs['atom_size_vector'] if 'atom_size_vector' in kwargs else len(chains)*[atom_size]
    atom_colors = 3 * [ATOM_COLORS[0]] + [ATOM_COLORS[1]] + [ATOM_COLORS[2]] + [ATOM_COLORS[1]] + \
                  3 * [ATOM_COLORS[0]] + [ATOM_COLORS[1]]
    help_point_chains(chains[:, :10, :], diag, atom_color_matrix=len(chains)*[atom_colors],
                      atom_size_vector=atom_size_vector, atom_alpha=alpha_atoms)
    diag.scatter([], [], [], marker="o", c=atom_colors[0], label='Carbon Atom')
    diag.scatter([], [], [], marker="o", c=atom_colors[3], label='Oxygen Atom')
    diag.scatter([], [], [], marker="o", c=atom_colors[4], label='Phosphorus Atom')


def plot_backbone_backbone_one_suite_clashes(chains, diag, **kwargs):
    """
    This function is a help function for the 'build_fancy_chain_plot'. It plots the backbone-backbone-one-suite clashes.
    :param chains: (nr elements) x (10) x (3)
    :param diag:
    :param kwargs:
    :return:
    """
    if 'backbone_ring_one_suite_clash' in kwargs:
        clash_info = kwargs['backbone_ring_one_suite_clash']
    if 'backbone_backbone_one_suite_clash' in kwargs:
        clash_info = kwargs['backbone_backbone_one_suite_clash']
    size_clash_atom = kwargs['size_clash_atom'] if 'size_clash_atom' in kwargs else 0.5
    alpha_connection = kwargs['alpha_connection'] if 'alpha_connection' in kwargs else 0.5
    lw = kwargs['lw'] if 'lw' in kwargs else 0.1
    for i in range(len(clash_info)):
        clash_list = clash_info[i]['clash_list']
        for k in range(len(clash_list)):
            sub_list = clash_list[k]
            first_list = sub_list[0]
            second_list = sub_list[1]
            first_atom_number = first_list[0]
            second_atom_number = first_list[len(first_list) - 1]
            third_atom_number = second_list[0]
            fourth_atom_number = second_list[len(second_list) - 1]
            dummy_list = [second_atom_number, fourth_atom_number]
            diag.plot(chains[i][first_list, 0], chains[i][first_list, 1],
                      chains[i][first_list, 2], c='green', linewidth=lw / 2,
                      alpha=alpha_connection)  # , alpha=0.5)
            diag.plot(chains[i][second_list, 0], chains[i][second_list, 1],
                      chains[i][second_list, 2], c='green', linewidth=lw / 2,
                      alpha=alpha_connection)
            diag.plot(chains[i][dummy_list, 0], chains[i][dummy_list, 1],
                      chains[i][dummy_list, 2], '--', c='darkred', linewidth=lw, alpha=alpha_connection, )
            # diag.scatter(chains[i][first_atom_number, 0], chains[i][first_atom_number, 1],
            #              chains[i][first_atom_number, 2],
            #              marker="o", s=size_clash_atom, c='green')
            # diag.scatter(chains[i][third_atom_number, 0], chains[i][third_atom_number, 1],
            #              chains[i][third_atom_number, 2],
            #              marker="o", s=size_clash_atom, c='green')
            diag.scatter(chains[i][second_atom_number, 0], chains[i][second_atom_number, 1],
                         chains[i][second_atom_number, 2],
                         marker="o", s=size_clash_atom, c='darkred')
            diag.scatter(chains[i][fourth_atom_number, 0], chains[i][fourth_atom_number, 1],
                         chains[i][fourth_atom_number, 2],
                         marker="o", s=size_clash_atom, c='darkred')

    diag.plot([], [], [], '--', c='darkred', linewidth=lw, label='Clash line', alpha=alpha_connection)
    diag.plot([], [], [], c='green', linewidth=lw / 2, label='Connection to clash atom', alpha=alpha_connection)
    #diag.scatter([], [], [], marker="o", c='green', label='Atom connected to clash atom')
    diag.scatter([], [], [], marker="o", c='darkred', label='Clash atom')


def plot_ring(chains, diag, **kwargs):
    lw = kwargs['lw'] if 'lw' in kwargs else 0.1
    ring_lw = kwargs['ring_lw'] if 'ring_lw' in kwargs else lw
    ring_color = kwargs['ring_color'] if 'ring_color' in kwargs else 'gray'
    ring_colors = kwargs['ring_colors'] if 'ring_colors' in kwargs else len(chains) * [ring_color]
    ring_alpha_line = kwargs['ring_alpha_line'] if 'ring_alpha_line' in kwargs else 0.5
    ring_alphas_line = kwargs['ring_alphas_line'] if 'ring_alphas_line' in kwargs else len(chains) * [ring_alpha_line]

    dummy_colors = []
    # list for ring plots
    first_list = [1, 18, 19, 20, 2]
    first_o_list = [20, 21]
    second_list = [7, 22, 23, 24, 8]
    second_o_list = [24, 25]

    help_chain_plot(chains[:, first_list, :], diag, colors=ring_colors, alpha_line_vec=ring_alphas_line, lw=ring_lw,
                    create_label=False)
    help_chain_plot(chains[:, first_o_list, :], diag, colors=ring_colors, alpha_line_vec=ring_alphas_line, lw=ring_lw,
                    create_label=False)
    help_chain_plot(chains[:, second_list, :], diag, colors=ring_colors, alpha_line_vec=ring_alphas_line, lw=ring_lw,
                    create_label=False)
    help_chain_plot(chains[:, second_o_list, :], diag, colors=ring_colors, alpha_line_vec=ring_alphas_line, lw=ring_lw,
                    create_label=False)
    help_point_chains(chains[:, 18:26, :], diag, atom_color='gray', **kwargs)

    diag.scatter([], [], [], marker="o", c=ring_color, label='Ring Atom')


def help_point_chains(chains, diag, **kwargs):
    atom_size = kwargs['atom_size'] if 'atom_size' in kwargs else 0.5
    atom_size_vector = kwargs['atom_size_vector'] if 'atom_size_vector' in kwargs else len(chains)*[atom_size]
    if 'atom_size_matrix' in kwargs:
        atom_size_matrix = kwargs['atom_size_matrix']
    else:
        atom_size_matrix = np.array(chains.shape[1]*[atom_size_vector]).T.tolist()
    atom_alpha = kwargs['atom_alpha'] if 'atom_alpha' in kwargs else 0.5
    atom_alpha_vector = kwargs['atom_alpha_vector'] if 'atom_alpha_vector' in kwargs else len(chains)*[atom_alpha]
    if 'atom_alpha_matrix' in kwargs:
        atom_alpha_matrix = kwargs['atom_alpha_matrix']
    else:
        atom_alpha_matrix = np.array(chains.shape[1]*[atom_alpha_vector]).T.tolist()
    atom_color = kwargs['atom_color'] if 'atom_color' in kwargs else 'black'
    atom_color_vector = kwargs['atom_color_vector'] if 'atom_color_vector' in kwargs else len(chains)*[atom_color]
    if 'atom_color_matrix' in kwargs:
        atom_color_matrix = kwargs['atom_color_matrix']
    else:
        atom_color_matrix = np.array(chains.shape[1]*[atom_color_vector]).T.tolist()

    for i in range(chains.shape[0]):
        for j in range(chains.shape[1]):
            diag.scatter(chains[i][j][0], chains[i][j][1], chains[i][j][2], marker="o",
                         s=atom_size_matrix[i][j],
                         c=atom_color_matrix[i][j],
                         alpha=atom_alpha_matrix[i][j])

    if 'label_atoms' in kwargs:
        diag.scatter([], [], [], marker='o', c=atom_color, label=kwargs['label_atoms'])


def plot_clash_atoms(chains, diag, **kwargs):
    if 'backbone_clash_atoms' in kwargs:
        clash_info = kwargs['backbone_clash_atoms']
    size_value = kwargs['size_value']
    for i in range(chains.shape[0]):
        clash_list = clash_info[i]['clash_list']
        for j in range(len(clash_list)):
            if size_value == 2:
                if clash_list[j][size_value] < 2:
                    color = 'limegreen'
                elif clash_list[j][size_value] < 4:
                    color = 'yellow'
                else:
                    color = 'darkred'
            else:
                if clash_list[j][size_value] < 0.5:
                    color = 'limegreen'
                elif clash_list[j][size_value] < 0.8:
                    color = 'yellow'
                else:
                    color = 'darkred'
            diag.scatter(chains[i][clash_list[j][0], 0], chains[i][clash_list[j][0], 1],
                         chains[i][clash_list[j][0], 2], marker="o", s=0.6, c=color)
    diag.scatter([], [], [], marker="o", c='limegreen', label='Weak clash')
    diag.scatter([], [], [], marker="o", c='yellow', label='Medium clash')
    diag.scatter([], [], [], marker="o", c='darkred', label='Hard clash')


def bootstrap_help(chains, diag, **kwargs):
    atom_number = kwargs['atom_number']
    atom_numbers = kwargs['atom_numbers']
    atom_alpha = kwargs['atom_alpha'] if 'atom_alpha' in kwargs else 0.9
    atom_colors = 3 * [ATOM_COLORS[0]] + [ATOM_COLORS[1]] + [ATOM_COLORS[2]] + [ATOM_COLORS[1]] + \
                  3 * [ATOM_COLORS[0]] + [ATOM_COLORS[1]]
    atom_size = kwargs['atom_size'] if 'atom_size' in kwargs else 0.5
    atom_size_vector = kwargs['atom_size_vector'] if 'atom_size_vector' in kwargs else len(chains)*[atom_size]
    not_atom_list = [atom_number + h for h in range(atom_numbers + 1)]
    atom_list = [i for i in range(10) if i not in not_atom_list]
    for k in range(10 - (atom_numbers + 1)):
        help_point_chains(chains[:, k, :].reshape((chains.shape[0], 1, chains.shape[2])), diag,
                          atom_color=atom_colors[atom_list[k]], atom_alpha=atom_alpha,
                          atom_size_vector=atom_size_vector)
    diag.scatter([], [], [], marker="o", c=atom_colors[0], label='Carbon Atom')
    diag.scatter([], [], [], marker="o", c=atom_colors[9], label='Oxygen Atom')
    diag.scatter([], [], [], marker="o", c=atom_colors[4], label='Phosphorus Atom')