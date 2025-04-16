# -*- coding: utf-8 -*-
"""
Copyright (C) 2014-2016 Benjamin Eltzner

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 3
of the License, or (at your option) any later version.

This software is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.

You should have received a copy of the GNU General Public
License along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA
or see <http://www.gnu.org/licenses/>.
"""
# Standard library imports
import math
import os
import sys

# Third-party imports
import matplotlib
try:
    matplotlib.use('Agg')
except:
    print('Rerunning')
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la

# Local/custom imports
from clustering.cluster_improving import large_cluster_separation
from clustering.gaussian_modehunting import modehunting_gaussian, mode_test_gaussian
from multiscale_analysis.Multiscale_modes import get_quantile, get_modes
from pnds.PNDS_geometry import RESHify_1D, unRESHify_1D, torus_distances, euclideanize
from pnds.PNDS_io import find_files, import_csv, import_lists, export_csv
from pnds.PNDS_plot import (
    scatter_plots,
    var_plot,
    inv_var_plot,
    residual_plots,
    sphere_views,
    make_circle,
    make_gauss_flex,
    colored_scatter_plots,
    one_scatter_plot,
    abs_var_plot,
    residues_plot,
    rainbow,
    one_two_plot,
    scatter_plot,
    circle_shade_plot,
    one_sphere_view,
    linear_1d_plot,
    plot_thread
)
from pnds.PNDS_PNS import pns_loop, fold_points, unfold_points, as_matrix 

################################################################################
################################   Constants   #################################
################################################################################

DEG = math.degrees(1)
OUTPUT_FOLDER = "./out/Torus_PCA/"


################################################################################
############################   Auxiliary function   ############################
################################################################################

def histogram_plot(data, bin_size=1):
    hi = max(data)
    lo = bin_size * math.floor(min(data / bin_size))
    bins = math.ceil((hi - lo) / bin_size)
    x = np.arange(lo, hi, bin_size)
    y = np.histogram(data, bins=bins, range=(lo, hi))[0]
    fig = matplotlib.pyplot.figure()
    diag = fig.add_subplot(111)
    diag.bar(x, y, width=bin_size, linewidth=0)
    matplotlib.pyplot.show()
    matplotlib.pyplot.close()


################################################################################
#############################   Control function   #############################
################################################################################

""" Clustering """


def new_multi_slink(scale, data=None, cluster_list=None, outlier_list=None, min_cluster_size=2):
    if data is None:
        points = import_csv(find_files('RNA_data_richardson.csv')[0])['PDB-Data']
    else:
        points = data
    # Import single-linkage clusters. this will have to be changed!
    if cluster_list is None:
        clusters = import_lists(find_files('RNA_data_richardson*multiSLINK_result.csv')[0])
    else:
        clusters = cluster_list
    noise = [np.array(outlier_list)]  # [np.array(sorted(clusters[-1]))]

    # Sort clusters by size.
    clusters = list(reversed(sorted([np.array(sorted(x)) for x in clusters], key=len)))

    # Lists for plottable data.
    scree_data = []
    scree_data_euclid = []
    std_data = []
    scree_labels = []

    # Perform further clustering.
    clusters = __slink_pns(clusters, points, scree_data, scree_data_euclid, std_data, scree_labels, 'filtered', scale,
                           min_cluster=min_cluster_size)
    return clusters, noise


"""
The main data processing function.
"""


def __slink_pns(new_clusters, points, scree_data, scree_data_euclid, std_data,
                scree_labels, type_name, scale, old_modehunting=False, min_cluster=2):
    folder = "./out/Gaussian_mode_hunting/"
    if not os.path.exists(folder):
        os.makedirs(folder)
    # c = clusters[i] indices of the cluster points
    # clusters: array of clusters. each cluster consists of the indices
    # final_clusters: append only final clusters
    # new_clusters: append every newfound cluster after split
    # p: dihedral_angles[clusters[i]] = points[c]
    # points = dihedral angles of each POINT, not sorted by cluster
    cluster_separation = True  # TODO

    split = True
    count = 0
    final_clusters = []
    inv_modes = [[False, 'gap'], [True, 'gap'], [False, 'mean'], [True, 'mean']]
    while split:
        split = False
        count += 1
        clusters = list(reversed(sorted(new_clusters, key=len)))
        if type_name == 'filtered':
            export_csv({('cluster%02d' % i): c for i, c in enumerate(clusters)},
                       'slink_rich_clusters_1d.csv', mode='Int')
        elif type_name == 'noise':
            export_csv({('cluster%02d' % i): c for i, c in enumerate(clusters)},
                       'slink_rich_noise_1d.csv', mode='Int')
        new_clusters = []
        for i, c in enumerate(clusters):
            # Ignore clusters which are final.
            if any(np.array_equal(c, f) for f in final_clusters):
                new_clusters.append(c)
                continue

            # Run PNS and collect 1D projections.
            name = OUTPUT_FOLDER + 'slink_rich_' + type_name + '_run%d_cluster%02d_%d_points_' % (count, i, len(c))
            name_gaussplot = folder + type_name + '_run%d_cluster%02d_%d_points_' % (count, i, len(c))
            name_separation_plot = type_name + '_run%d_cluster%02d_%d_points_' % (count, i, len(c))
            relative_residual_variances = []  # not needed atm
            p = points[c]
            print(p.shape, flush=True)
            dist_list = []
            point_list = []
            for [invert, mode] in inv_modes:
                print('SO' if invert else 'SI', mode, flush=True)
                suffix = ('SO_' if invert else 'SI_') + mode + '_'
                sphere_points, means, half = RESHify_1D(p[:, :7], invert, mode)
                spheres, projected_points, distances = pns_loop(sphere_points, 10, 10, degenerate=False, verbose=False,
                                                                mode='torus')
                center_point = unRESHify_1D(unfold_points(as_matrix(projected_points[-1]), spheres[:-1]), means, half)
                unfolded_1d = unRESHify_1D(unfold_points(projected_points[-2], spheres[:-1]), means, half)
                relative_residual_variance = (np.mean(torus_distances(unfolded_1d, p[:, :7]) ** 2) / np.mean(
                    torus_distances(center_point, p[:, :7]) ** 2))

                # for plot of the circles
                point_list.append([unfolded_1d, suffix])
                # relative_residual_variances.append(relative_residual_variance)

                sys.stdout.flush()
                mode_hunting = True
                unimodal_test = True
                if unimodal_test:
                    order = 1
                    while relative_residual_variance > 0.25:
                        print('WARNING: High residual variance in', 'SO' if invert else 'SI', mode, ':',
                              relative_residual_variance)
                        order += 1
                        # safety stop:
                        if order >= 7:
                            dist_list.append(None)
                            mode_hunting = False
                            break

                        # distances[-order]   Gaussian modes
                        modes = mode_test_gaussian(distances[-order], 0.05)
                        if modes == 2:
                            dist_list.append(None)
                            mode_hunting = False
                            break
                        unfolded = unRESHify_1D(unfold_points(projected_points[-2], spheres[:-1]), means, half)
                        relative_residual_variance = (np.mean(torus_distances(unfolded, p[:, :7]) ** 2) / np.mean(
                            torus_distances(center_point, p[:, :7]) ** 2))

                    if mode_hunting:
                        dist_list.append(distances[-1])
                    else:
                        pass
                        # c = clusters[i] indices of the cluster points
                        # clusters: array of clusters. each cluster consists of the indices
                        # final_clusters: append only final clusters
                        # new_clusters: append every newfound cluster after split
                        # p: dihedral_angles[clusters[i]] = points[c]
                        # points = dihedral angles of each POINT, not sorted by cluster
                        # large_c, leftover_c = large_cluster_separation(p, c, plot_names=name_separation_plot)
                        # new_clusters.append(large_c)  # c
                        # new_clusters.append(leftover_c)
                        # final_clusters.append(large_c)

                else:  # if not unimodal_test:
                    if relative_residual_variance > 0.25:  # default: 0.25
                        print('WARNING: High residual variance in', 'SO' if invert else 'SI', mode, ':',
                              relative_residual_variance)
                        dist_list.append(None)
                        mode_hunting = False
                    else:
                        dist_list.append(distances[-1])  # for mode hunting

            ##### We need distances[-1] for mode hunting

            if old_modehunting:
                # First check the highest alpha for quick stop.
                this_split = True
                mode_list = [np.array(range(len(c)))]
                mins = np.zeros(1)
                #### Change
                quantile = get_quantile(len(p), 0.05)
                for i, dists in enumerate(dist_list):
                    if dists is None:
                        continue
                    mode_list, mins = get_modes(dists, 360., quantile)
                    if len(mins) > 1:
                        this_split = False
                        print(len(mins), 'modes expected')
                        break

                # Perform mode hunting.
                alpha = 0
                while (not this_split) and (alpha < 5):
                    alpha += 1
                    quantile = get_quantile(len(p), 0.01 * alpha)
                    for i, dists in enumerate(dist_list):
                        if dists is None:
                            continue
                        mode_list, mins = get_modes(dists, 360., quantile)
                        [invert, mode] = inv_modes[i]
                        if len(mins) > 1:
                            print('SO' if invert else 'SI', mode, mins)
                            print(len(mins), 'modes found at alpha = %.2f' % (0.01 * alpha), flush=True)
                            suffix = ('SO_' if invert else 'SI_') + mode + '_'
                            sphere_points, means, half = RESHify_1D(p[:, :7], invert, mode)
                            spheres, projected_points, distances = pns_loop(sphere_points, 10, 10, degenerate=False,
                                                                            verbose=False, mode='torus')
                            __slink_plotting(name, suffix, p, distances, projected_points,
                                             spheres, means, half, c, count,
                                             scree_data, scree_data_euclid,
                                             std_data, scree_labels, False, scale, relative_residual_variances)
                            new_clusters += [c[x] for x in mode_list]
                            this_split = True
                            split = True
                            break
                if len(mins) <= 1:
                    print('No modes found!')
                    suffix = ('SO_' if invert else 'SI_') + mode + '_'
                    __slink_plotting(name, suffix, p, distances, projected_points,
                                     spheres, means, half, c, count, scree_data,
                                     scree_data_euclid, std_data, scree_labels,
                                     True, scale, relative_residual_variances)
                    new_clusters += [c[x] for x in mode_list]
                    final_clusters.append(c)
            else:
                print("gaussian modehunting")
                __slink_plotting(name_gaussplot, "", p, distances, projected_points,
                                 spheres, means, half, c, count,
                                 scree_data, scree_data_euclid,
                                 std_data, scree_labels, False, scale, relative_residual_variances)

                point_plot = False
                if point_plot:
                    for j in range(len(point_list)):
                        plot_folder = "./out/point_list/" + type_name + \
                                      '_run%d_cluster%02d_%d_points_' % (count, i, len(c)) + str(j)
                        _plot_circles(plot_folder + point_list[j][1], point_list[j][0][:, :7], s=25)

                dist_list_sorted = [dist for dist in dist_list if dist is not None]
                if len(dist_list_sorted) > 0:
                    # this_split = True
                    # alpha = 0
                    # while (not this_split) and (alpha < 5):
                    # goes through whole dist_list 5 times
                    # alpha += 1
                    gesplitted = False
                    gauss_means = []
                    for i, dists in enumerate(dist_list_sorted):
                        # if dists is None: continue  # - therefore take dist_list_sorted
                        indexlist, gesplitted, gauss_means_for_plot = modehunting_gaussian(dists, 0.05,
                                                                                           min_cluster_size=min_cluster)  # dist_list[i]
                        # -> bekommt array mit Clusterindizes und einen boolean mit ob gesplittet werden soll zurück
                        gauss_means.append(gauss_means_for_plot)

                        if gesplitted:
                            # sort the new clusters and add to new_clusters
                            if np.sum(indexlist) > 0.5 * len(c):
                                indexlist = 1 - indexlist
                            new_clusters += [c[indexlist == 0], c[indexlist == 1]]
                            split = True
                            # this_split = True
                            _plot_gaussian_mode(name_gaussplot, dist_list_sorted, p, indexlist, split=True,
                                                means_plot=gauss_means)
                            # [points[c[indexlist == 0]], points[c[indexlist == 1]]]
                            break

                    if not gesplitted:
                        # keep old clustering
                        new_clusters.append(c)
                        final_clusters.append(c)
                        _plot_gaussian_mode(name_gaussplot, dist_list_sorted, p, None, split=False, stop=False,
                                            means_plot=gauss_means)
                        continue

                else:
                    # if no elements in dist_list
                    _plot_gaussian_mode(name_gaussplot, dist_list_sorted, p, None, split=False, stop=False)
                    if cluster_separation and len(c) > min_cluster:
                        # large cluster sep
                        large_c, leftover_c = large_cluster_separation(p, c, plot_names=name_separation_plot)
                        new_clusters.append(large_c)  # c
                        if len(leftover_c) > 0:
                            new_clusters.append(leftover_c)
                        final_clusters.append(large_c)
                    else:
                        new_clusters.append(c)
                        final_clusters.append(c)

    print('Clustering done.', len(clusters), 'clusters found.', flush=True)
    return clusters


def _plot_circles(folder, p, s=25):
    scatter_plots(p[:, :7], folder + '7d', s=25)


def _plot_gaussian_mode(folder, distances, p, cluster, split=False, stop=False, means_plot=None):
    if stop:
        folder = folder.replace('_run', '_stop_run')

    scatter_plots(p[:, :7], folder + '7d', s=25, color=cluster)
    # residual_plots(distances, 1, 4, 'blue', folder + 'residuals')
    gauss_plot_helper(folder, distances, split=split, means_plot=means_plot)


def gauss_plot_helper(folder, distances, split, means_plot):
    if distances is None:
        return
    if len(distances) == 0:
        return
    if means_plot is not None:
        while len(distances) > len(means_plot):
            means_plot.append(None)
        # means_plot = np.append(means_plot, None)
    means_plot = np.array(means_plot, dtype=object)
    sigma = 10
    for i, dist in enumerate(distances):
        bin_size = (np.max(dist) - np.min(dist)) / 20
        rounded = np.round(dist / bin_size).astype(int)
        min_bin = np.min(rounded)
        max_bin = np.max(rounded)
        d = max_bin - min_bin + 1

        plt.subplot(len(distances), 2, 1 + i * 2)
        plt.hist(dist, bins=d, edgecolor='black')
        plt.xlim(-180, 180)
        # plt.tight_layout(top=0.75)

        plt.subplot(len(distances), 2, 2 + i * 2)
        plt.plot(make_gauss_flex(np.array(dist), sigma, 180)[0],
                 make_gauss_flex(np.array(dist), sigma, 180)[1])
        plt.xlim(-180, 180)

        if means_plot[i] is None:
            continue
        plt.plot(means_plot[i][0], 0, marker='X', color='r', markersize=5)
        if split:
            if len(means_plot[i]) < 2:
                continue
            plt.plot(means_plot[i][1], 0, marker='X', color='r', markersize=5)
            plt.axvline(x=means_plot[i][2], color='g')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle(f"cluster with {len(dist)} points - split: {split}")
    plt.savefig(folder + "gauss")
    plt.close()


def __slink_plotting(name, suffix, p, distances, projected_points, spheres,
                     means, half, c, count, scree_data, scree_data_euclid,
                     std_data, scree_labels, stop, scale, rel_residual_variance):
    if stop:
        name = name.replace('_run', '_stop_run')

    def worker():
        scatter_plots(p[:, :7], name + '7d')
        #        scatter_plots(p[:,7:9], name + '2d')
        residual_plots(distances, 1, min(4, len(distances)), 'blue', rel_residual_variance,
                       filename=name + suffix + 'residuals')

    #        angles_1d = unRESHify_1D(unfold_points(projected_points[-2], spheres[:-1]), means, half)
    #        scatter_plots(angles_1d, name + suffix + 'projection_1d')
    # plot_thread(worker)
    worker()
    # loop = asyncio.get_event_loop()
    # loop.run_in_executor(None, worker)

    if count == 1:
        center_point = unRESHify_1D(unfold_points(as_matrix(projected_points[-1]), spheres[:-1]), means, half)
        unfolded_angles = [unRESHify_1D(unfold_points(pts, spheres[:i + 1]), means, half) for i, pts in
                           enumerate(projected_points[:-1])]
        var_data = []
        for ang in unfolded_angles:
            var_data.append(np.mean(torus_distances(ang, p[:, :7]) ** 2))
        var_data.append(np.mean(torus_distances(center_point, p[:, :7]) ** 2))
        scree_data.append(np.array([[i, 100 * (1 - v / scale)] for i, v in enumerate(reversed(var_data))]))
        std_data.append(np.array([[i, math.sqrt(v)] for i, v in enumerate(reversed(var_data))]))
        scree_labels.append(('noise ' if 'noise' in name else '') + str(len(p)) +
                            ' ' + suffix[:-1].replace('_', ' '))
        val, vec = la.eig(np.cov(euclideanize(p[:, :7].copy()).T))
        val = np.hstack((np.zeros(1), np.cumsum(np.sort(val)[::-1])))
        val = val[-1] - val
        scree_data_euclid.append(np.array([[i, 100 * (1 - v / scale)] for i, v in enumerate(val[:-1])]))


if __name__ == '__main__':
    new_multi_slink(13742.1871427)
