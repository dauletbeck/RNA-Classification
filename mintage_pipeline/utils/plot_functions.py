from auxiliary_plot_functions import *
import matplotlib.pyplot as plot
from constants import COLORS, MARKERS, COLORS_SCATTER


def build_fancy_chain_plot(chains, filename=None, **kwargs):
    """
    This function plots chains[0] different chains in a 3D plot.
    :param chains: Needs the shape: [nr_chains, nr_points_in_chain, 3]
    :param colors: The colors of each chain.
    :param filename: The filepath of the saved plot.
    :param number_of_elements: If you want to plot more then one data set.
    """
    fig = plot.figure()
    # diag = fig.gca(projection='3d')
    diag = fig.add_subplot(111, projection='3d')
    help_chain_plot(chains, diag, **kwargs)
    # For suites: Add the largest cluster to the plot:
    if (kwargs['backbone_plot_biggest'] if 'backbone_plot_biggest' in kwargs else False):
        plot_biggest_backbone_cluster(diag)
    if (kwargs['mesoscopic_plot_biggest'] if 'mesoscopic_plot_biggest' in kwargs else False):
        plot_biggest_mesoscopic_cluster(diag, **kwargs)
    # For suites: Add the carbon, oxygen and phosphat atoms:
    if (kwargs['plot_backbone_atoms'] if 'plot_backbone_atoms' in kwargs else False):
        plot_backbone_atoms(chains, diag, **kwargs)
    # For suites: Add the backbone-backbone-one-clashes:
    if 'backbone_backbone_one_suite_clash' in kwargs:
        plot_backbone_backbone_one_suite_clashes(chains, diag, **kwargs)
    if 'backbone_ring_one_suite_clash' in kwargs:
        plot_ring(chains, diag, **kwargs)
        plot_backbone_backbone_one_suite_clashes(chains, diag, **kwargs)
    if 'backbone_clash_atoms' in kwargs and 'size_value' in kwargs:
        plot_clash_atoms(chains, diag, **kwargs)
    if (kwargs['plot_atoms'] if 'plot_atoms' in kwargs else False):
        help_point_chains(chains, diag, **kwargs)
    if ('atom_number' in kwargs and 'atom_numbers' in kwargs):
        bootstrap_help(chains, diag, **kwargs)
    if ('specific_legend_strings' in kwargs and 'specific_legend_colors' in kwargs):
        create_specific_legend(chains, diag, **kwargs)

    help_chain_plot_legend_and_axis(chains, diag, **kwargs)
    if ('elev' in kwargs and 'azim' in kwargs):
        diag.view_init(elev=kwargs['elev'], azim=kwargs['azim'])
    if 'hide_axes' in kwargs:
        diag.set_axis_off()
    if False:
        diag.grid(False)
        # Hide axes ticks
        diag.set_axis_off()
    if not (filename is None):
        plot.savefig(filename + '.png', dpi=300, bbox_inches='tight', pad_inches=0)
        # Image.open(filename + '.png').save(filename + '.jpg', 'JPEG')
    else:
        plot.show()
    plot.close()
    arr = plot.imread(filename + '.png')
    min_1 = np.min([i for i in range(arr.shape[0]) if not np.all(arr[i, :, :] == 1)])
    max_1 = np.max([i for i in range(arr.shape[0]) if not np.all(arr[i, :, :] == 1)])
    min_2 = np.min([i for i in range(arr.shape[1]) if not np.all(arr[:, i, :] == 1)])
    max_2 = np.max([i for i in range(arr.shape[1]) if not np.all(arr[:, i, :] == 1)])
    arr_new = arr[min_1:max_1 + 1, min_2:max_2 + 1, :]
    plot.imsave(filename + '.png', arr_new)


def scatter_plots(input_data, filename=None, axis_min=0, axis_max=360, set_title=None, number_of_elements=None,
                  suite_titles=None, alpha_first=1, s=5, all_titles=False, fontsize=40, legend=True,
                  dummy_legend=False, list_ranges=None, legend_names=None, color_and_marker_list = None,
                  fontsize_axis=None, fontsize_legend=30, input_data2=None, input_data3=None, plot_line=False):
    """
    This function gets input data and creates scatter plots.
    :param input_data: A matrix of dimension (number of data_points) x (number of dimensions)
    :param filename: Should be a string indicating where the plot should be stored and what the plot should be named.
    :param axis_min: The minimum of the range.
    :param axis_max: The maximum of the range.
    :param set_title:
    :param legend_: Boolean: If False: without legend.
    :param number_of_elements: If you have more than one group of data.
    """
    fig = plot.figure()
    n = input_data.shape[1]
    size = fig.get_size_inches()
    # To avoid overlapping of the plots:
    if n > 3:
        fig.set_size_inches((1.2 * size[0] * (n - 1), 1.2 * size[1] * (n - 1)))
        # fig.subplots_adjust(left=-0.2, bottom=0.05, right=1.2, top=0.95, wspace=-0.8, hspace=0.4)
    else:
        fig.set_size_inches((1.2 * size[0] * (n), 1.2 * size[1] * (n)))
        # fig.subplots_adjust(left=-0.8, bottom=0.1, right=1.7, top=0.95, wspace=-0.8, hspace=0.4)
    for y in range(n):
        for x in range(y):
            diag = fig.add_subplot(n - 1, n - 1, 1 + x + (n - 1 - y) * (n - 1))
            if fontsize_axis is not None:
                diag.tick_params(axis='both', labelsize=fontsize_axis)
            if suite_titles is None:
                if set_title is None:
                    diag.set_title(r'$x = \alpha_' + str(x + 1) +
                                   r'$, $y = \alpha_' + str(y + 1) + '$', fontsize=20)
                else:
                    diag.set_title(set_title + str(x + 1) + ', ' +
                                   set_title + str(y + 1), fontsize=20)
            else:
                # diag.set_title('x-axis is ' + suite_titles[x] + ', ' + 'y axis is' + suite_titles[y], fontsize=20)
                if y == n - 1 or all_titles:
                    if not all_titles:
                        diag.set_title(suite_titles[x], fontsize=fontsize)
                    else:
                        diag.set_xlabel(suite_titles[x], fontsize=fontsize)
                if x == 0 or all_titles:
                    diag.set_ylabel(suite_titles[y], fontsize=fontsize)
            if list_ranges is None:
                if axis_min is not None:
                    diag.set_aspect('equal')
                    diag.set_xlim(axis_min, axis_max)
                    diag.set_ylim(axis_min, axis_max)
                else:
                    diffs = [np.abs(np.min(input_data[:, z]) - np.max(input_data[:, z])) for z in
                             range(input_data.shape[1])]
                    diag.set_xlim((np.min(input_data[:, x]) + np.max(input_data[:, x])) / 2 - np.max(diffs) / 2 - 1,
                                  (np.min(input_data[:, x]) + np.max(input_data[:, x])) / 2 + np.max(diffs) / 2 + 1)
                    diag.set_ylim((np.min(input_data[:, y]) + np.max(input_data[:, y])) / 2 - np.max(diffs) / 2 - 1,
                                  (np.min(input_data[:, y]) + np.max(input_data[:, y])) / 2 + np.max(diffs) / 2 + 1)
                    # diag.set_xlim(np.min(input_data[:, x]), np.max(input_data[:, x]))
                    # diag.set_ylim(np.min(input_data[:, y]), np.max(input_data[:, y]))
                #diag.set_aspect('equal')
            # only one data set:
            else:
                #diag.set_aspect('equal')
                diag.set_xlim(np.min(list_ranges[x]), np.max(list_ranges[x]))
                diag.set_ylim(np.min(list_ranges[y]), np.max(list_ranges[y]))
            if number_of_elements is None:
                if color_and_marker_list is None:
                    diag.scatter(input_data[:, x], input_data[:, y], marker="D", linewidth=0.1, s=s, c='black')
                else:
                    color_list = color_and_marker_list[0]
                    marker_list = color_and_marker_list[1]
                    for i in range(len(color_list)):
                        diag.scatter(input_data[i, x], input_data[i, y], color=color_list[i], linewidth=0.1, s=s,
                                     marker=marker_list[i])
            else:
                for number_element in range(len(number_of_elements)):
                    if number_element == 0:
                        diag.scatter(input_data[:number_of_elements[number_element], x],
                                     input_data[:number_of_elements[number_element], y],
                                     c=COLORS_SCATTER[number_element], linewidth=0.1, s=s,
                                     alpha=alpha_first, marker=MARKERS[number_element])
                    else:
                        diag.scatter(input_data[sum(number_of_elements[:number_element]):
                                                sum(number_of_elements[:number_element + 1]), x],
                                     input_data[sum(number_of_elements[:number_element]):
                                                sum(number_of_elements[:number_element + 1]), y],
                                     c=COLORS_SCATTER[number_element], linewidth=0.1, s=s,
                                     marker=MARKERS[number_element])

    if number_of_elements is not None and legend:
        x = n - 3
        y = n - 4

        diag = fig.add_subplot(n - 1, n - 1, 1 + x + (n - 1 - y) * (n - 1))
        for i in range(len(number_of_elements)):
            if i == 0:
                if legend_names is None:
                    plot.scatter([], [], c=COLORS_SCATTER[i], s=s, marker=MARKERS[i],
                                 label='Class ' + str(i + 1), alpha=alpha_first)
                else:
                    plot.scatter([], [], c=COLORS_SCATTER[i], s=s, marker=MARKERS[i],
                                 label=legend_names[i], alpha=alpha_first)
            else:
                if legend_names is None:
                    plot.scatter([], [], c=COLORS_SCATTER[i], s=s, marker=MARKERS[i],
                                 label='Class ' + str(i + 1))
                else:
                    plot.scatter([], [], c=COLORS_SCATTER[i], s=s, marker=MARKERS[i],
                                 label=legend_names[i])
        legend = diag.legend(loc='center', markerscale=3, prop={"size": fontsize_legend})
        legend.get_frame().set_edgecolor('white')
        diag.set_xticks([])
        diag.set_yticks([])
        diag.axis("off")

    if dummy_legend:
        x = n - 3
        y = n - 4

        diag = fig.add_subplot(n - 1, n - 1, 1 + x + (n - 1 - y) * (n - 1))

        plot.scatter([], [], c=COLORS_SCATTER[0], s=s, marker=MARKERS[0],
                     label='MINT-AGE Class 6')

        plot.scatter([], [], c=COLORS_SCATTER[1], s=s, marker=MARKERS[1], label='Mean of Richardson cluster 7a')
        plot.scatter([], [], c=COLORS_SCATTER[2], s=s, marker=MARKERS[2], label='Mean of Richardson cluster 3a')
        plot.scatter([], [], c=COLORS_SCATTER[4], s=s, marker=MARKERS[4], label='Mean of Richardson cluster 9a')
        diag.legend(loc='center', markerscale=1.3, prop={"size": 30})
        diag.set_xticks([])
        diag.set_yticks([])
        diag.axis("off")
    if not (filename is None):
        plot.savefig(filename + '.png', dpi=150)
    else:
        plot.show()
    plot.close()
    arr = plot.imread(filename + '.png')
    min_1 = np.min([i for i in range(arr.shape[0]) if not np.all(arr[i, :, :] == 1)])
    max_1 = np.max([i for i in range(arr.shape[0]) if not np.all(arr[i, :, :] == 1)])
    min_2 = np.min([i for i in range(arr.shape[1]) if not np.all(arr[:, i, :] == 1)])
    max_2 = np.max([i for i in range(arr.shape[1]) if not np.all(arr[:, i, :] == 1)])
    arr_new = arr[min_1:max_1 + 1, min_2:max_2 + 1, :]
    # transparent_region_1 = [i for i in range(arr_new.shape[0]) if np.all(arr_new[i, :, :] == 1)]
    transparent_region_2 = [i for i in range(arr_new.shape[1]) if np.all(arr_new[:, i, :] == 1)]
    list_2 = []
    for i in transparent_region_2:
        added = False
        for sub_list in list_2:
            if (i - 1) in sub_list:
                sub_list.append(i)
                added = True
        if not added:
            list_2.append([i])
    remove_list = []
    for list in list_2:
        if len(list) > 75:
            for element in list[74:]:
                remove_list.append(element)
    not_remove_list = [i for i in range(arr_new.shape[1]) if i not in remove_list]
    arr_new = arr_new[:, not_remove_list, :]

    # "ValueError: ndarray is not C-contiguous" else
    arr_new = arr_new.copy(order='C')
    plot.imsave(filename + '.png', arr_new)


def my_scatter_plots(input_data, filename=None, axis_min=0, axis_max=360, set_title=None, number_of_elements=None,
                     suite_titles=None, alpha_first=1, s=5, all_titles=False, fontsize=40, legend=True,
                     legend_with_clustersize=False, color_numbers=None, legend_titles=None, markerscale=5):
    """
    This function gets input data and creates scatter plots.
    :param legend_titles: needs to be list of len number_of_elements
    :param color_numbers: needs to be list of len number_of_elements
    :param input_data: A matrix of dimension (number of data_points) x (number of dimensions)
    :param filename: Should be a string indicating where the plot should be stored and what the plot should be named.
    :param axis_min: The minimum of the range.
    :param axis_max: The maximum of the range.
    :param set_title:
    :param legend: Boolean: If False: without legend.
    :param number_of_elements: If you have more than one group of data.
    """
    fig = plot.figure()
    n = input_data.shape[1]
    size = fig.get_size_inches()
    if suite_titles is None:
        suite_titles = [r'$\delta_{1}$', r'$\epsilon$', r'$\zeta$', r'$\alpha$', r'$\beta$', r'$\gamma$',
                        r'$\delta_{2}$']

    if color_numbers is None and number_of_elements is not None:
        color_numbers = range(len(number_of_elements))
    else:
        color_numbers = [n - 1 for n in color_numbers]

    # To avoid overlapping of the plots:
    if n > 3:
        fig.set_size_inches((1.2 * size[0] * (n - 1), 1.2 * size[1] * (n - 1)))
        # fig.subplots_adjust(left=-0.2, bottom=0.05, right=1.2, top=0.95, wspace=-0.8, hspace=0.4)
    else:
        fig.set_size_inches((1.2 * size[0] * n, 1.2 * size[1] * n))
        # fig.subplots_adjust(left=-0.8, bottom=0.1, right=1.7, top=0.95, wspace=-0.8, hspace=0.4)
    for y in range(n):
        for x in range(y):
            diag = fig.add_subplot(n - 1, n - 1, 1 + x + (n - 1 - y) * (n - 1))
            if suite_titles is None:
                if set_title is None:
                    diag.set_title(r'$x = \alpha_' + str(x + 1) +
                                   r'$, $y = \alpha_' + str(y + 1) + '$', fontsize=20)
                else:
                    diag.set_title(set_title + str(x + 1) + ', ' +
                                   set_title + str(y + 1), fontsize=20)
            else:
                # diag.set_title('x-axis is ' + suite_titles[x] + ', ' + 'y axis is' + suite_titles[y], fontsize=20)
                if y == n - 1 or all_titles:
                    if not all_titles:
                        diag.set_title(suite_titles[x], fontsize=fontsize)
                    else:
                        diag.set_xlabel(suite_titles[x], fontsize=fontsize)
                if x == 0 or all_titles:
                    diag.set_ylabel(suite_titles[y], fontsize=fontsize)

            if axis_min is not None:
                diag.set_aspect('equal')
                diag.set_xlim(axis_min, axis_max)
                diag.set_ylim(axis_min, axis_max)
            else:
                diffs = [np.abs(np.min(input_data[:, z]) - np.max(input_data[:, z])) for z in
                         range(input_data.shape[1])]
                diag.set_xlim((np.min(input_data[:, x]) + np.max(input_data[:, x])) / 2 - np.max(diffs) / 2 - 1,
                              (np.min(input_data[:, x]) + np.max(input_data[:, x])) / 2 + np.max(diffs) / 2 + 1)
                diag.set_ylim((np.min(input_data[:, y]) + np.max(input_data[:, y])) / 2 - np.max(diffs) / 2 - 1,
                              (np.min(input_data[:, y]) + np.max(input_data[:, y])) / 2 + np.max(diffs) / 2 + 1)
                # diag.set_xlim(np.min(input_data[:, x]), np.max(input_data[:, x]))
                # diag.set_ylim(np.min(input_data[:, y]), np.max(input_data[:, y]))
            # only one data set:
            if number_of_elements is None:
                diag.scatter(input_data[:, x], input_data[:, y], marker="D", linewidth=0.1, s=s, c='black')
            else:

                for number_element, number_color in zip(range(len(number_of_elements)), color_numbers):
                    if number_element == 0:
                        diag.scatter(input_data[:number_of_elements[number_element], x],
                                     input_data[:number_of_elements[number_element], y],
                                     c=COLORS_SCATTER[number_color], linewidth=0.1, s=s,
                                     alpha=alpha_first, marker=MARKERS[number_color])
                    else:
                        diag.scatter(input_data[sum(number_of_elements[:number_element]):
                                                sum(number_of_elements[:number_element + 1]), x],
                                     input_data[sum(number_of_elements[:number_element]):
                                                sum(number_of_elements[:number_element + 1]), y],
                                     c=COLORS_SCATTER[number_color], linewidth=0.1, s=s,
                                     marker=MARKERS[number_color])

    if number_of_elements is not None and legend:
        x = n - 3
        y = n - 4

        if legend_titles is None:
            legend_titles = range(1, len(number_of_elements)+1)

        if legend_with_clustersize:
            # temp = [f'size: {l}' for l in number_of_elements]
            # legend_titles = zip(legend_titles, temp)
            legend_titles = [f"{c}, size: {l}" for c, l in zip(legend_titles, number_of_elements)]

        diag = fig.add_subplot(n - 1, n - 1, 1 + x + (n - 1 - y) * (n - 1))
        for i, j in zip(legend_titles, color_numbers):
            if not isinstance(i, str):
                i = str(i)
            if i == 0:
                plot.scatter([], [], c=COLORS_SCATTER[j], s=s, marker=MARKERS[j],
                             label='Class ' + i, alpha=alpha_first)
            else:
                plot.scatter([], [], c=COLORS_SCATTER[j], s=s, marker=MARKERS[j],
                             label='Class ' + i)
        diag.legend(loc='center', markerscale=markerscale, prop={"size": 30})
        diag.set_xticks([])
        diag.set_yticks([])
        diag.axis("off")

    if not (filename is None):
        plot.savefig(filename + '.png', dpi=150)
    else:
        plot.show()
    plot.close()
    arr = plot.imread(filename + '.png')
    min_1 = np.min([i for i in range(arr.shape[0]) if not np.all(arr[i, :, :] == 1)])
    max_1 = np.max([i for i in range(arr.shape[0]) if not np.all(arr[i, :, :] == 1)])
    min_2 = np.min([i for i in range(arr.shape[1]) if not np.all(arr[:, i, :] == 1)])
    max_2 = np.max([i for i in range(arr.shape[1]) if not np.all(arr[:, i, :] == 1)])
    arr_new = arr[min_1:max_1 + 1, min_2:max_2 + 1, :]
    # transparent_region_1 = [i for i in range(arr_new.shape[0]) if np.all(arr_new[i, :, :] == 1)]
    transparent_region_2 = [i for i in range(arr_new.shape[1]) if np.all(arr_new[:, i, :] == 1)]
    list_2 = []
    for i in transparent_region_2:
        added = False
        for sub_list in list_2:
            if (i - 1) in sub_list:
                sub_list.append(i)
                added = True
        if not added:
            list_2.append([i])
    remove_list = []
    for list in list_2:
        if len(list) > 75:
            for element in list[74:]:
                remove_list.append(element)
    not_remove_list = [i for i in range(arr_new.shape[1]) if i not in remove_list]
    arr_new = arr_new[:, not_remove_list, :]

    # "ValueError: ndarray is not C-contiguous" else
    arr_new = arr_new.copy(order='C')
    plot.imsave(filename + '.png', arr_new)


def scatter_plots_two(input_data1, input_data2=None, input_data3=None, input_data4=None, filename=None, axis_min=0,
                      axis_max=360, set_title=None, number_of_elements=None,
                      legend_=False, suite_titles=None, alpha_first=1, s=10, plot_line=False, all_titles=False,
                      without_axis=True):
    """
    This function gets input data and creates scatter plots.
    :param input_data3:
    :param input_data: A matrix of dimension (number of data_points) x (number of dimensions)
    :param filename: Should be a string indicating where the plot should be stored and what the plot should be named.
    :param axis_min: The minimum of the range.
    :param axis_max: The maximum of the range.
    :param set_title:
    :param legend_: Boolean: If False: without legend.
    :param number_of_elements: If you have more than one group of data.
    """
    fig = plot.figure()
    n = input_data1.shape[1]
    size = fig.get_size_inches()
    # To avoid overlapping of the plots:
    if n > 3:
        fig.set_size_inches((1.2 * size[0] * (n - 1), 1.2 * size[1] * (n - 1)))
        # fig.subplots_adjust(left=-0.2, bottom=0.05, right=1.2, top=0.95, wspace=-0.8, hspace=0.4)
    else:
        fig.set_size_inches((1.2 * size[0] * (n), 1.2 * size[1] * (n)))
        # fig.subplots_adjust(left=-0.8, bottom=0.1, right=1.7, top=0.95, wspace=-0.8, hspace=0.4)
    for y in range(n):
        for x in range(y):
            diag = fig.add_subplot(n - 1, n - 1, 1 + x + (n - 1 - y) * (n - 1))
            if suite_titles is None:
                if set_title is None:
                    diag.set_title(r'$x = \alpha_' + str(x + 1) +
                                   r'$, $y = \alpha_' + str(y + 1) + '$', fontsize=20)
                else:
                    diag.set_title(set_title[x] + ', ' +
                                   set_title[y], fontsize=20)
            else:
                if y == n - 1 or all_titles:
                    if not all_titles:
                        diag.set_title(suite_titles[x], fontsize=20)
                    else:
                        diag.set_xlabel(suite_titles[x], fontsize=20)
                if x == 0 or all_titles:
                    diag.set_ylabel(suite_titles[y], fontsize=20)
            diag.set_aspect('equal')
            diag.set_xlim(axis_min, axis_max)
            diag.set_ylim(axis_min, axis_max)
            if without_axis:
                diag.set_xticks([])
                diag.set_yticks([])
            # only one data set:

            diag.scatter(input_data1[:, x], input_data1[:, y], marker="D", linewidth=0.1, s=s, c='black', alpha=0.7)
            if input_data2 is not None:
                diag.scatter(input_data2[:, x], input_data2[:, y], marker="o", linewidth=0.1, s=s * 1.5,
                             c='darkmagenta', alpha=0.7)
                if plot_line == True:
                    plot_lines_scatter(diag, input_data1, input_data2, x, y, 'darkmagenta')
            if input_data3 is not None:
                diag.scatter(input_data3[:, x], input_data3[:, y], marker="o", linewidth=0.1, s=s * 1.5, c='darkgreen',
                             alpha=0.7)
                if plot_line == True:
                    plot_lines_scatter(diag, input_data1, input_data3, x, y, 'darkgreen')
            if input_data4 is not None:
                for k in range(len(input_data4)):
                    if input_data4[k] is not None:
                        diag.scatter(input_data4[k][x], input_data4[k][y], marker='X', linewidth=0.1, s=s * 1.5,
                                     c='darkblue', alpha=0.7)
                        plot_lines_scatter(diag, input_data1[k].reshape((1, n)),
                                           np.array(input_data4[k]).reshape((1, n)), x, y, 'darkblue')

    if legend_:
        x = n - 3
        y = n - 4
        diag = fig.add_subplot(n - 1, n - 1, 1 + x + (n - 1 - y) * (n - 1))
        plot.scatter([], [], c='black', s=40, marker='D', label='Clash suites ')
        if input_data2 is not None:
            plot.scatter([], [], c='darkmagenta', s=40, marker='o', label="ERRASER correction")
        if input_data3 is not None:
            plot.scatter([], [], c='darkgreen', s=40, marker='o', label="CLEAN, ABSOLUTE")
        if input_data4 is not None:
            plot.scatter([], [], c='darkblue', s=40, marker='X', label="CLEAN, RELATIVE")

        diag.legend(loc='center', markerscale=5, prop={"size": 30})
        diag.set_xticks([])
        diag.set_yticks([])
        diag.axis("off")

    if not (filename is None):
        plot.savefig(filename + '.png', dpi=150)
    else:
        plot.show()
    plot.close()
    arr = plot.imread(filename + '.png')
    min_1 = np.min([i for i in range(arr.shape[0]) if not np.all(arr[i, :, :] == 1)])
    max_1 = np.max([i for i in range(arr.shape[0]) if not np.all(arr[i, :, :] == 1)])
    min_2 = np.min([i for i in range(arr.shape[1]) if not np.all(arr[:, i, :] == 1)])
    max_2 = np.max([i for i in range(arr.shape[1]) if not np.all(arr[:, i, :] == 1)])
    arr_new = arr[min_1:max_1 + 1, min_2:max_2 + 1, :]
    transparent_region_2 = [i for i in range(arr_new.shape[1]) if np.all(arr_new[:, i, :] == 1)]
    list_2 = []
    for i in transparent_region_2:
        added = False
        for sub_list in list_2:
            if (i - 1) in sub_list:
                sub_list.append(i)
                added = True
        if not added:
            list_2.append([i])
    remove_list = []
    for list in list_2:
        if len(list) > 75:
            for element in list[74:]:
                remove_list.append(element)
    not_remove_list = [i for i in range(arr_new.shape[1]) if i not in remove_list]
    arr_new = arr_new[:, not_remove_list, :]
    plot.imsave(filename + '.png', arr_new)


def plot_lines_scatter(diag, input_data1, input_data2, x, y, col):
    for i in range(input_data2.shape[0]):
        if np.abs(input_data2[i, x] - input_data1[i, x]) < 180 and np.abs(input_data2[i, y] - input_data1[i, y]) < 180:
            diag.plot([input_data2[i, x], input_data1[i, x]], [input_data2[i, y], input_data1[i, y]],
                      c=col, linewidth=1, alpha=0.5)
        elif np.abs(input_data2[i, x] - input_data1[i, x]) > 180 and np.abs(
                input_data2[i, y] - input_data1[i, y]) < 180:
            help_x_axis_torus_plot(col, diag, i, input_data1, input_data2, x, y)
        elif np.abs(input_data2[i, x] - input_data1[i, x]) < 180 and np.abs(
                input_data2[i, y] - input_data1[i, y]) > 180:
            help_y_axis_torus_plot(col, diag, i, input_data1, input_data2, x, y)
        else:
            list_y = [input_data2[i, y], input_data1[i, y]]
            argmin_y = np.argmin(list_y)
            argmax_y = np.argmax(list_y)
            lower_element_y = [input_data2[i, y], input_data1[i, y]][argmin_y]
            upper_element_y = [input_data2[i, y], input_data1[i, y]][argmax_y]
            lower_element_x = [input_data2[i, x], input_data1[i, x]][argmin_y]
            upper_element_x = [input_data2[i, x], input_data1[i, x]][argmax_y]
            c = (360 - np.abs(input_data2[i, y] - input_data1[i, y])) / (
                    360 - np.abs(input_data2[i, x] - input_data1[i, x]))

            if np.min((lower_element_x, 360 - lower_element_x)) * c < lower_element_y:
                # print('plot durch y achse')
                closer_value = 360 if lower_element_x > 180 else 0
                other_value = 0 if lower_element_x > 180 else 360
                diag.plot([closer_value, lower_element_x],
                          [lower_element_y - c * np.min((lower_element_x, 360 - lower_element_x)), lower_element_y],
                          c=col, linewidth=1, alpha=0.5)

                y_value_1 = lower_element_y - c * np.min((lower_element_x, 360 - lower_element_x))
                x_value_1 = other_value

                y_value_2 = 0
                x_value_2 = 360 - (1 / c) * np.min((360 - y_value_1, y_value_1)) if other_value == 360 else (
                                                                                                                        1 / c) * np.min(
                    (360 - y_value_1, y_value_1))

                diag.plot([x_value_1, x_value_2], [y_value_1, y_value_2], c=col, linewidth=1, alpha=0.5)
                diag.plot([upper_element_x, x_value_2], [upper_element_y, 360], c=col, linewidth=1, alpha=0.5)
            else:
                list_y = [input_data2[i, y], input_data1[i, y]]
                argmin_y = np.argmin(list_y)
                argmax_y = np.argmax(list_y)
                lower_element_y = [input_data2[i, y], input_data1[i, y]][argmin_y]
                upper_element_y = [input_data2[i, y], input_data1[i, y]][argmax_y]
                lower_element_x = [input_data2[i, x], input_data1[i, x]][argmin_y]
                upper_element_x = [input_data2[i, x], input_data1[i, x]][argmax_y]

                closer_value = 360 if upper_element_x > 180 else 0
                other_value = 0 if upper_element_x > 180 else 360

                diag.plot([closer_value, upper_element_x],
                          [upper_element_y + c * np.min((upper_element_x, 360 - upper_element_x)), upper_element_y],
                          c=col, linewidth=1, alpha=0.5)

                y_value_1 = upper_element_y + c * np.min((upper_element_x, 360 - upper_element_x))
                x_value_1 = other_value

                y_value_2 = 360
                x_value_2 = 360 - (1 / c) * np.min((360 - y_value_1, y_value_1)) if other_value == 360 else (
                                                                                                                        1 / c) * np.min(
                    (360 - y_value_1, y_value_1))  # (1 / c) * np.min((360 - y_value_1, y_value_1))
                diag.plot([x_value_1, x_value_2], [y_value_1, y_value_2], c=col, linewidth=1, alpha=0.5)
                diag.plot([lower_element_x, x_value_2], [lower_element_y, 0], c=col, linewidth=1, alpha=0.5)

            # print('test')


def help_y_axis_torus_plot(col, diag, i, input_data1, input_data2, x, y):
    list_y = [input_data2[i, y], input_data1[i, y]]
    argmin_y = np.argmin(list_y)
    argmax_y = np.argmax(list_y)
    lower_element_y = [input_data2[i, y], input_data1[i, y]][argmin_y]
    upper_element_y = [input_data2[i, y], input_data1[i, y]][argmax_y]
    lower_element_x = [input_data2[i, x], input_data1[i, x]][argmin_y]
    upper_element_x = [input_data2[i, x], input_data1[i, x]][argmax_y]
    diag.plot([lower_element_x - (lower_element_x - upper_element_x) * (lower_element_y) / (
            360 - upper_element_y + lower_element_y), lower_element_x], [0, lower_element_y],
              c=col, linewidth=1, alpha=0.5)
    diag.plot([lower_element_x - (lower_element_x - upper_element_x) * (lower_element_y) / (
            360 - upper_element_y + lower_element_y), upper_element_x], [360, upper_element_y],
              c=col, linewidth=1, alpha=0.5)


def help_x_axis_torus_plot(col, diag, i, input_data1, input_data2, x, y):
    list_x = [input_data2[i, x], input_data1[i, x]]
    argmin_x = np.argmin(list_x)
    argmax_x = np.argmax(list_x)
    left_element_y = [input_data2[i, y], input_data1[i, y]][argmin_x]
    right_element_y = [input_data2[i, y], input_data1[i, y]][argmax_x]
    left_element_x = [input_data2[i, x], input_data1[i, x]][argmin_x]
    right_element_x = [input_data2[i, x], input_data1[i, x]][argmax_x]
    c = (right_element_y - left_element_y) / (360 - np.abs(input_data2[i, x] - input_data1[i, x]))
    diag.plot([0, left_element_x], [left_element_y + c * left_element_x, left_element_y],
              c=col, linewidth=1, alpha=0.5)
    diag.plot([360, right_element_x], [left_element_y + c * left_element_x, right_element_y],
              c=col, linewidth=1, alpha=0.5)


def hist_own_plot(data, x_label, y_label, filename, bins=50, calculate_xlim=False, density=True, y_ticks=None,
                  x_ticks_ticks=None, x_ticks_ticks_label=None, fontsize=15):
    n, bins, patches = plot.hist(data, bins, density=density, color='darkgreen', edgecolor='black')
    plot.xlabel(x_label, fontdict={'size': fontsize})
    plot.ylabel(y_label, fontdict={'size': fontsize})
    plot.tight_layout()
    if y_ticks is not None:
        plot.yticks(y_ticks)
    if (x_ticks_ticks is not None) and (x_ticks_ticks_label is not None):
        plot.xticks(ticks=x_ticks_ticks, labels=x_ticks_ticks_label)
    # plot.xlim(np.min(data), np.max(data))
    plot.grid(False)
    if not (filename is None):
        plot.savefig(filename + '.png')
    plot.close()
