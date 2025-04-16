# -*- coding: utf-8 -*-
"""
Copyright (C) 2014 Benjamin Eltzner

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

import numpy as np
import numpy.linalg as la
import math, subprocess
import matplotlib.pyplot as plot
import multiprocessing as mp
from random import sample
from mpl_toolkits.mplot3d import Axes3D  # Necessary for 3D plots to work!
from matplotlib.colors import LinearSegmentedColormap
from math import sqrt, sin, cos
from pnds.PNDS_geometry import apply_matrix, rotation

################################################################################
################################   Constants   #################################
################################################################################

PI = np.pi
DEG = math.degrees(1)
RAD = math.radians(1)
TURN_SPHERE = 0.05
LABEL_SIZE = 18.0
AXES_LABEL_SIZE = 20.0
EPS = 1e-10
OUTPUT_FOLDER = "out/"


def heat_map():
    cdict = {'red': ((0.0, 0.0, 0.0),
                     (1. / 2., 1.0, 1.0),
                     (1.0, 1.0, 1.0)),
             'green': ((0.0, 0.9, 0.9),
                       (1. / 2., 0.9, 0.9),
                       (1.0, 0.0, 0.0)),
             'blue': ((0.0, 0.1, 0.1),
                      (1. / 2., 0.0, 0.0),
                      (1.0, 0.0, 0.0))}
    return LinearSegmentedColormap('my_colormap', cdict, 256)


HEAT = heat_map()


################################################################################
#################################   Plotting   #################################
################################################################################

def make_binning(data, is2d=False, step=1):
    if is2d:
        data = np.arctan2(data[:, 1], data[:, 0])
    binned = np.zeros(int(360 // step + 1))
    for i in range(len(data)):
        binned[int(((180 + round(data[i] * DEG)) % 360) // step)] += 1
    binned = (np.vstack([np.array(list(range(-180, 181, step))), binned])).T
    return binned


def make_bar_chart(data, mean=None, filename=None):
    if mean != None:
        lines = [mean[0] - sqrt(mean[1]), mean[0] + sqrt(mean[1]),
                 mean[0] - sqrt(mean[1]) + 2 * PI, mean[0] + sqrt(mean[1]) - 2 * PI]
    fig = plot.figure()
    diag = fig.add_subplot(111)
    xdata = data[:, 0] - (180. / float(len(data) - 1))
    ydata = data[:, 1]
    binWidth = 360 / float(len(data) - 1)
    plot.bar(xdata, ydata, width=binWidth, linewidth=0, color='gray')
    if mean != None:
        plot.axvline(x=mean[0] * DEG, linewidth=0.3, color='black')
        for line in lines:
            plot.axvline(x=line * DEG, linewidth=0.3, dashes=(4, 1), color='black')
    diag.set_xlim(-180, 180)
    if filename != None:
        plot.savefig(OUTPUT_FOLDER + filename + '.png', dpi=300)
    else:
        plot.show()
    plot.close()


def plot_distances_test(x, y, filename=None):
    # Diagram setup
    fig = plot.figure()
    diag = fig.add_subplot(111)
    diag.set_ylim(-0.01, 0.2)
    # Data to plot
    for i in range(y.shape[1]):
        color = rainbow(i, y.shape[1])
        diag.plot(x, y[:, i], linewidth=1, color=color, label='')
    if (not filename == None):
        #        filename = OUTPUT_FOLDER + filename + '.pdf'
        #        plot.savefig(filename)
        #        subprocess.call(['sed', '-i', 's/\/Group.*R//g', filename])
        plot.savefig(OUTPUT_FOLDER + filename + '.png', dpi=300)
    else:
        plot.show()
    plot.close()


def make_diagram(title=None):
    fig = plot.figure()
    if title:
        fig.suptitle(title, fontsize=20)
    diag = fig.add_subplot(111, projection='3d')
    diag.autoscale(enable=False, axis='both', tight=None)
    diag.set_aspect('equal')
    diag.set_xlim(-1.1, 1.1)
    diag.set_ylim(-1.1, 1.1)
    diag.set_zlim(-1.1, 1.1)
    return fig, diag


def plot_sphere(diag):
    # Make mesh for the spheres
    theta = np.arange(0, PI + 0.01, PI / 18.)
    phi = np.arange(TURN_SPHERE, 2 * PI + TURN_SPHERE + 0.01, PI / 18.)
    theta, phi = np.meshgrid(theta, phi)
    sphere = [np.sin(theta) * np.cos(phi),
              np.sin(theta) * np.sin(phi),
              np.cos(theta)]
    # Plot sphere to given diagram
    diag.plot_surface(sphere[0], sphere[1], sphere[2], rstride=1, cstride=1,
                      linewidths=0.5, edgecolors='#000066', color='#0000FF',
                      alpha=0.08)


def plot_points(diag, colors, points, number, marker, size):
    if number:
        random_sample = sample(range(len(colors)), number)
        colors = np.array(colors)[random_sample]
        points = points[random_sample]
    points = np.array(points).astype(float)
    diag.scatter(points[:, 0], points[:, 1], points[:, 2], marker=marker,
                 linewidth=0.2, c=colors, s=size)


def plot_3d(colors, points, number=None, title=None, filename=None):
    fig, diag = make_diagram(title)
    plot_sphere(diag)
    if points != None:
        plot_points(diag, colors, points, number, 'o', 15)
    if filename != None:
        plot.savefig(OUTPUT_FOLDER + filename + '.png', dpi=300)
    #        filename = OUTPUT_FOLDER + filename + '.pdf'
    #        plot.savefig(filename)
    #        subprocess.call(['sed', '-i', 's/\/Group.*R//g', filename])
    else:
        plot.show()
    plot.close()


def scatter_plots(data, filename=None, s=5, color=None):
    fig = plot.figure()
    n = data.shape[1]
    size = fig.get_size_inches()
    fig.set_size_inches((size[0] * (n - 1), size[1] * (n - 1)))
    fig.subplots_adjust(left=-0.2, bottom=0.05, right=1.2, top=0.95,
                        wspace=-0.8, hspace=0.4)
    if color is not None:
        col = np.array(["blue", "red"])
        color = col[color]
    for y in range(n):
        for x in range(y):
            diag = fig.add_subplot(n - 1, n - 1, 1 + x + (n - 1 - y) * (n - 1))
            diag.set_title(r'$x = \alpha_' + str(x + 1) +
                           r'$, $y = \alpha_' + str(y + 1) + '$', fontsize=20)
            diag.set_aspect('equal')
            diag.set_xlim(0, 360)
            diag.set_ylim(0, 360)
            diag.scatter(data[:, x], data[:, y], marker="D", linewidth=0.1, s=s, c=color)
    if (not filename == None):
        plot.savefig(filename + '.png', dpi=150)
    else:
        plot.show()
    plot.close()


def scatter_path_plots(data, filename=None):
    fig = plot.figure()
    n = data.shape[1]
    size = fig.get_size_inches()
    fig.set_size_inches((size[0] * (n - 1), size[1] * (n - 1)))
    fig.subplots_adjust(left=-0.2, bottom=0.05, right=1.2, top=0.95,
                        wspace=-0.8, hspace=0.4)
    for y in range(n):
        for x in range(y):
            diag = fig.add_subplot(n - 1, n - 1, 1 + x + (n - 1 - y) * (n - 1))
            diag.set_title(r'$x = \alpha_' + str(x + 1) +
                           r'$, $y = \alpha_' + str(y + 1) + '$', fontsize=20)
            diag.set_aspect('equal')
            diag.set_xlim(0, 360)
            diag.set_ylim(0, 360)
            diag.plot(data[:, x], data[:, y], linewidth=1, c='blue')
            diag.scatter(data[:, x], data[:, y], marker="D", linewidth=0.1, s=5)
    if (not filename == None):
        plot.savefig(OUTPUT_FOLDER + filename + '.png', dpi=150)
    else:
        plot.show()
    plot.close()


def one_scatter_plot(data, x, y, labels, filename=None, means=None):
    fig = plot.figure()
    diag = fig.add_subplot(111)
    diag.set_aspect('equal')
    diag.set_xlim(0, 360)
    diag.set_ylim(0, 360)
    diag.set_xlabel(labels[0], fontsize=AXES_LABEL_SIZE)
    diag.set_ylabel(labels[1], fontsize=AXES_LABEL_SIZE)
    for [p, c, m, s] in data:
        lw = s / (40 if ((m == '+') or (m == 'x')) else 100)
        diag.scatter(p[:, x], p[:, y], marker=m, linewidth=lw, s=s, c=c,
                     edgecolor='black')
    if not means is None:
        diag.plot([means[x]] * 2, [0, 360], c='black', lw=0.5, ls='--')
        diag.plot([0, 360], [means[y]] * 2, c='black', lw=0.5, ls='--')
    if (not filename == None):
        plot.savefig(OUTPUT_FOLDER + filename + '.png', dpi=300)
    else:
        plot.show()
    plot.close()


def scatter_plot(x, y, colors, filename=None):
    fig = plot.figure()
    size = 0.5 * min(100, 400 / sqrt(len(x)))
    diag = fig.add_subplot(111)
    # square_subplot(diag)
    diag.set_xlim(10 * math.floor(0.1 * np.min(x)), 10 * math.ceil(0.1 * np.max(x)))
    diag.set_ylim(10 * math.floor(0.1 * np.min(y)), 10 * math.ceil(0.1 * np.max(y)))
    diag.set_aspect('equal')
    diag.scatter(x, y, c=colors, marker="D", linewidth=0.1, s=size)
    if colors[-1] == 'red':
        a = (np.array(colors) == 'red')
        diag.scatter(x[a], y[a], c='red', marker="*", linewidth=0.1, s=10 * size)
    if (not filename == None):
        plot.savefig(OUTPUT_FOLDER + filename + '.png', dpi=150)
    else:
        plot.show()
    plot.close()


def scatter_plot_3d(data, colors, filename=None):
    fig = plot.figure()
    diag = fig.add_subplot(111, projection='3d')
    diag.autoscale(enable=False, axis='both', tight=None)
    size = 0.5 * min(100, 100 / sqrt(len(data)))
    diag.set_aspect('equal')
    diag.set_xlim(0, 360)
    diag.set_ylim(0, 360)
    diag.set_zlim(0, 360)
    diag.set_aspect('equal')
    diag.scatter(data[:, 0], data[:, 1], data[:, 2], c=colors, marker="o",
                 linewidth=0, s=size)
    if (not filename == None):
        plot.savefig(OUTPUT_FOLDER + filename + '.png', dpi=150)
    else:
        plot.show()
    plot.close()


def scatter_plot_plus_3d(data, labels, filename=None):
    fig = plot.figure()
    diag = fig.add_subplot(111, projection='3d')
    diag.autoscale(enable=False, axis='both', tight=None)
    diag.set_aspect('equal')
    diag.set_xlim(0, 360)
    diag.set_ylim(0, 360)
    diag.set_zlim(0, 360)
    diag.set_xlabel(labels[0], fontsize=AXES_LABEL_SIZE)
    diag.set_ylabel(labels[1], fontsize=AXES_LABEL_SIZE)
    diag.set_zlabel(labels[2], fontsize=AXES_LABEL_SIZE)
    plot.tick_params(labelsize=LABEL_SIZE - 6)
    plot.tight_layout()
    for [p, c, m, s] in data:
        lw = s / (40 if ((m == '+') or (m == 'x')) else 100)
        tmp = diag.scatter(p[:, 0], p[:, 1], p[:, 2], c=c, marker=m, linewidth=lw,
                           edgecolor='black', s=s)
        tmp.set_facecolors = lambda *args: None
        tmp.set_edgecolors = lambda *args: None
    if (not filename == None):
        plot.savefig(OUTPUT_FOLDER + filename + '.png', dpi=150)
    else:
        plot.show()
    plot.close()


def colored_scatter_plots(data, n0, n1, filename=None):
    fig = plot.figure()
    n = n1 - n0
    size = fig.get_size_inches()
    fig.set_size_inches((size[0] * (n - 1), size[1] * (n - 1)))
    fig.subplots_adjust(left=-0.2, bottom=0.05, right=1.2, top=0.95,
                        wspace=-0.8, hspace=0.4)
    for y in range(n0, n1):
        for x in range(n0, y):
            diag = fig.add_subplot(n - 1, n - 1, 1 + x - n0 + (n - 1 - y + n0) * (n - 1))
            diag.set_title(r'$x = \alpha_' + str(x + 1) +
                           r'$, $y = \alpha_' + str(y + 1) + '$', fontsize=20)
            diag.set_aspect('equal')
            diag.set_xlim(0, 360)
            diag.set_ylim(0, 360)
            for [c, d] in data:
                diag.scatter(d[:, x], d[:, y], marker="D", linewidth=0, s=1, c=c,
                             facecolors=c, edgecolor=None)
    if (not filename == None):
        plot.savefig(OUTPUT_FOLDER + filename + '.png', dpi=150)
    else:
        plot.show()
    plot.close()


def colored_scatter_plots_flex_range(data, n0, n1, filename=None):
    fig = plot.figure()
    n = n1 - n0
    size = fig.get_size_inches()
    fig.set_size_inches((size[0] * (n - 1), size[1] * (n - 1)))
    fig.subplots_adjust(left=-0.2, bottom=0.05, right=1.2, top=0.95,
                        wspace=-0.8, hspace=0.4)
    lo = min([np.min(d) for [c, d] in data])
    hi = max([np.max(d) for [c, d] in data])
    diff = 0.05 * (hi - lo);
    lo -= diff;
    hi += diff
    for y in range(n0, n1):
        for x in range(n0, y):
            diag = fig.add_subplot(n - 1, n - 1, 1 + x - n0 + (n - 1 - y + n0) * (n - 1))
            diag.set_title(r'$x = \alpha_{' + str(x + 1) +
                           r'}$, $y = \alpha_{' + str(y + 1) + '}$', fontsize=20)
            diag.set_aspect('equal')
            diag.set_xlim(lo, hi)
            diag.set_ylim(lo, hi)
            for [c, d] in data:
                diag.scatter(d[:, x], d[:, y], marker="D", linewidth=0, s=1, c=c)
    if (not filename == None):
        plot.savefig(OUTPUT_FOLDER + filename + '.png', dpi=150)
    else:
        plot.show()
    plot.close()


def rainbow_scatter_plots(data, filename=None):
    fig = plot.figure()
    n = data[0].shape[1]
    size = fig.get_size_inches()
    fig.set_size_inches((size[0] * (n - 1), size[1] * (n - 1)))
    fig.subplots_adjust(left=-0.2, bottom=0.05, right=1.2, top=0.95,
                        wspace=-0.8, hspace=0.4)
    for y in range(n):
        for x in range(y):
            diag = fig.add_subplot(n - 1, n - 1, 1 + x + (n - 1 - y) * (n - 1))
            diag.set_title(r'$x = \alpha_' + str(x + 1) +
                           r'$, $y = \alpha_' + str(y + 1) + '$', fontsize=20)
            diag.set_aspect('equal')
            diag.set_xlim(0, 360)
            diag.set_ylim(0, 360)
            for i, d in enumerate(data):
                diag.scatter(d[:, x], d[:, y], marker="D", linewidth=0.1, s=5,
                             c=rainbow(i, len(data) + 1))
    if (not filename is None):
        plot.savefig(OUTPUT_FOLDER + filename + '.png', dpi=150)
    else:
        plot.show()
    plot.close()


def scatter_2D(data, colors, labels=['', ''], filename=None, size=20, ran=[-180, 180]):
    fig = plot.figure()
    diag = fig.add_subplot(111)
    diag.set_xlim(*ran)
    diag.set_ylim(*ran)
    diag.set_aspect('equal')
    diag.set_xlabel(labels[0], fontsize=AXES_LABEL_SIZE)
    diag.set_ylabel(labels[1], fontsize=AXES_LABEL_SIZE)
    plot.tick_params(labelsize=LABEL_SIZE)
    plot.tight_layout()
    diag.scatter(data[:, 0], data[:, 1], marker="D", linewidth=0.1, c=colors, s=size)
    if (not filename is None):
        plot.savefig(OUTPUT_FOLDER + filename + '.png', dpi=300)
    else:
        plot.show()
    plot.close()


def residual_plots(data, scale, n, colors, rel_residual_variances, filename=None):
    fig = plot.figure()
    size = fig.get_size_inches()
    fig.set_size_inches((size[0] * n, size[1] * n))
    if n > 3:
        fig.subplots_adjust(left=-0.2, bottom=0.05, right=1.2, top=0.95,
                            wspace=-0.8, hspace=0.4)
    else:
        fig.subplots_adjust(left=-0.2, bottom=0.05, right=1.2, top=0.95,
                            wspace=-0.7, hspace=0.4)
    variances = [d.var() for d in data]
    variances = [100 * v / sum(variances) for v in variances][-n:]
    data = data[-n:]
    if scale is None:
        scale = 1.1 * np.max(np.abs(data))
    else:
        scale *= 180
    size = min(100, 400 / sqrt(len(data[0])))  # (60 if len(data[0]) < 50 else (35 if len(data[0]) < 120 else 10))
    for x in range(n):
        for y in range(n):
            diag = fig.add_subplot(n, n, (n - x) + n * (n - y - 1))
            # square_subplot(diag)
            if x == y:
                tmp = make_gauss(data[x], scale)
                diag.set_xlim(-scale, scale)
                # diag.set_title('component ' + str(n-x) + '\n' + r'variance = %.2f%%' %
                #                variances[x], fontsize=20)
                diag.set_title('component ' + str(n - x) + '\n' + r'variance = %.2f%%' %
                               variances[x], fontsize=20)
                diag.plot(tmp[0], tmp[1])
                x0, x1 = diag.get_xlim()
                y0, y1 = diag.get_ylim()
                diag.set_aspect((x1 - x0) / (y1 - y0))
            else:
                diag.set_xlim(-scale, scale)
                diag.set_ylim(-scale, scale)
                diag.set_aspect('equal')
                diag.set_title('components ' + str(n - x) + ' and ' + str(n - y), fontsize=20)
                if x > y:
                    diag.scatter(data[x], data[y],
                                 marker="D", linewidth=0.1, c=colors, s=size)
                if x < y:
                    diag.scatter(data[x], data[y],
                                 marker="D", linewidth=0.1, c=colors, s=size)
    if (not filename is None):
        #        filename = OUTPUT_FOLDER + filename + '_tex.pdf'
        #        plot.savefig(filename)
        #        subprocess.call(['sed', '-i', 's/\/Group.*R//g', filename])
        plot.savefig(filename + '.png', dpi=(300 if n < 4 else 150))
    else:
        plot.show()
    plot.close()


def one_two_plot(data, colors='blue', filename=None):
    fig = plot.figure()
    size = 0.25 * min(100, 400 / sqrt(len(data[0])))
    diag = fig.add_subplot(111)
    # square_subplot(diag)
    diag.set_xlim(min(-180, np.min(data[-1])), max(180, np.max(data[-1])))
    diag.set_ylim(-180, 180)
    diag.set_aspect('equal')
    diag.scatter(data[-1], data[-2], marker="D", c=colors, linewidth=0.2, s=size)
    if (not filename == None):
        plot.savefig(OUTPUT_FOLDER + filename + '.png', dpi=150)
    else:
        plot.show()
    plot.close()


def stratified_plots(data, filename=None):
    fig = plot.figure()
    n = data.shape[2]
    colors = [rainbow(i, n) for i in range(n)]
    size = fig.get_size_inches()
    fig.set_size_inches((size[0] * (n - 1), size[1] * (n - 1)))
    fig.subplots_adjust(left=-0.2, bottom=0.05, right=1.2, top=0.95,
                        wspace=-0.8, hspace=0.4)
    for y in range(n):
        for x in range(y):
            diag = fig.add_subplot(n - 1, n - 1, 1 + x + (n - 1 - y) * (n - 1))
            diag.set_title(r'$x = \alpha_' + str(x + 1) +
                           r'$, $y = \alpha_' + str(y + 1) + '$', fontsize=20)
            diag.set_aspect('equal')
            diag.set_xlim(0, 360)
            diag.set_ylim(0, 360)
            for i in range(data.shape[0]):
                x_data, y_data = data[i, :, x], data[i, :, y]
                splits = np.where((np.abs(x_data[1:] - x_data[:-1]) +
                                   np.abs(y_data[1:] - y_data[:-1])) > 300)
                if len(splits[0]) == 0:
                    diag.plot(x_data, y_data, linewidth=1, color=colors[i])
                else:
                    splits = [0] + (np.array(splits) + 1).flatten().tolist()
                    for j in range(len(splits) - 1):
                        diag.plot(x_data[splits[j]:splits[j + 1]],
                                  y_data[splits[j]:splits[j + 1]],
                                  linewidth=1, color=colors[i])
                    diag.plot(x_data[splits[-1]:], y_data[splits[-1]:],
                              linewidth=1, color=colors[i])
    if (not filename == None):
        plot.savefig(OUTPUT_FOLDER + filename + '.png', dpi=300)
    else:
        plot.show()
    plot.close()


def circle_plot(data, colors, circles=True, filename=None):
    data = np.array(data)
    fig = plot.figure()
    diag = fig.add_subplot(111)
    diag.set_aspect('equal')
    diag.set_xlim(-1.2, 1.2)
    diag.set_ylim(-1.2, 1.2)
    if circles:
        angles = np.arange(0, 2 * PI + 0.01, PI / float(18))
        circle = np.array([[cos(phi), sin(phi)] for phi in angles])
        diag.plot(circle[:, 0], circle[:, 1], linewidth=0.5, color='black', label='')
    diag.scatter(data[:, 0], data[:, 1], marker="o", linewidth=0.5, color='black',
                 facecolor=colors, s=20)
    if (not filename is None):
        plot.savefig(OUTPUT_FOLDER + filename + '.png', dpi=300)
    else:
        plot.show()
    plot.close()


def line_plot(data, labels, filename=None):
    data = np.array(data)
    fig = plot.figure()
    diag = fig.add_subplot(111)
    diag.plot(data[:, 0], data[:, 1])
    diag.scatter(data[:, 0], data[:, 1], marker='*', s=100)
    diag.set_xlim(0, len(data) + 0.5)
    diag.set_xlabel(labels[0], fontsize=AXES_LABEL_SIZE)
    diag.set_ylim(bottom=0)
    diag.set_ylabel(labels[1], fontsize=AXES_LABEL_SIZE)
    plot.tick_params(labelsize=LABEL_SIZE)
    plot.tight_layout()
    if (not filename == None):
        plot.savefig(OUTPUT_FOLDER + filename + '.png', dpi=300)
    else:
        plot.show()
    plot.close()


def lines_plot(data, filename=None):
    data = np.array(data)
    fig = plot.figure()
    diag = fig.add_subplot(111)
    for d in data:
        diag.plot(range(len(d)), d)
    diag.legend([1, 2, 3], loc='best')
    plot.tight_layout()
    if (not filename == None):
        plot.savefig(OUTPUT_FOLDER + filename + '.png', dpi=300)
    else:
        plot.show()
    plot.close()


def torus_plot(data, max_jump, lims, labels, filename=None):
    data = np.array(data)
    fig = plot.figure()
    diag = fig.add_subplot(111)
    for [d, c] in data:
        jumps = np.where(np.max(np.abs(d[1:] - d[:-1]), axis=1) > max_jump)[0]
        jumps = np.hstack(([0], jumps + 1, [len(d)]))
        for n, j in enumerate(jumps[:-1]):
            diag.plot(d[j:jumps[n + 1], 0], d[j:jumps[n + 1], 1], c)
    diag.set_xlim(lims[0][0], lims[0][1])
    diag.set_ylim(lims[1][0], lims[1][1])
    if not (labels is None):
        diag.set_xlabel(labels[0], fontsize=AXES_LABEL_SIZE)
        diag.set_ylabel(labels[1], fontsize=AXES_LABEL_SIZE)
    plot.tick_params(labelsize=LABEL_SIZE)
    plot.tight_layout()
    if (not filename == None):
        plot.savefig(OUTPUT_FOLDER + filename + '.png', dpi=300)
    else:
        plot.show()
    plot.close()


def circle_shade_plot(data, sigma, star, filename=None):
    fig = plot.figure()
    A = 250
    data = (data + A) * RAD
    diag = fig.add_subplot(111, projection='3d')
    diag.set_xlim(-0.7, 0.7)
    diag.set_ylim(-0.7, 0.7)
    phi = np.linspace(0, 2 * PI, 1000)
    circle = np.array([np.cos(phi), np.sin(phi), np.zeros(len(phi))])
    curve = np.vstack((circle[:-1], gaussianize(data, sigma) ** (2)))
    phi_grid = np.vstack((phi, phi)).T
    z_grid = np.vstack((np.zeros(len(phi)), curve[-1])).T
    tube = [np.cos(phi_grid), np.sin(phi_grid), z_grid]
    # diag.plot(circle[0], circle[1], circle[2], color='black')
    diag.plot(curve[0], curve[1], curve[2], color='blue')
    diag.scatter(np.cos(data), np.sin(data), np.zeros(len(data)), c='blue',
                 linewidth=0, s=5)
    diag.plot_surface(tube[0], tube[1], tube[2], rstride=1, cstride=1,
                      linewidths=0.5, edgecolors='#000066', color='#6666FF',
                      alpha=0.08)
    if star != None:
        star = np.array([cos(A * RAD) * star[0] - sin(A * RAD) * star[1],
                         cos(A * RAD) * star[1] + sin(A * RAD) * star[0]])
        print(star)
        diag.scatter(star[0], star[1], 0, c='red', marker="*",
                     linewidth=0.1, s=200)
    diag.set_axis_off()
    if (not filename == None):
        plot.savefig(OUTPUT_FOLDER + filename + '.png', dpi=300)
    else:
        plot.show()
    plot.close()


def linear_1d_plot(data, sigma, colors, filename=None):
    fig = plot.figure()
    diag = fig.add_subplot(111)
    diag.plot(*gaussianize_linear(data, sigma), color='black')
    colors = np.array(colors)
    for c in set(colors):
        diag.plot(*gaussianize_linear(data[colors == c], sigma), color=c)
    if (not filename == None):
        plot.savefig(OUTPUT_FOLDER + filename + '.png', dpi=300)
    else:
        plot.show()
    plot.close()


def var_plot(data, threshold, names, do_sum, filename=None):
    fig = plot.figure()
    diag = fig.add_subplot(111)
    x_range = max([len(d) for d in data]) - 0.5
    colors = [rainbow(i, 4 * len(data) // 3) for i in range(len(data))]
    for i, d in enumerate(data):
        limit = 1
        if do_sum:
            while (limit < len(d)) and (d[:limit, 1].sum() < threshold): limit += 1
            d[:, 1] = [sum(d[:i + 1, 1]) for i in range(len(d))]
            d = np.vstack((np.zeros(2), d))
        else:
            while (limit < len(d)) and (d[limit, 1] < threshold): limit += 1
        limit += 1
        diag.plot(d[:, 0], d[:, 1], c=colors[i])
        diag.scatter(d[1:limit, 0], d[1:limit, 1], marker='*', s=100, c=colors[i], linewidth=0.5)
        diag.scatter(d[limit:, 0], d[limit:, 1], marker='.', s=50, c=colors[i], linewidth=0.5)
    diag.set_xlim(0, x_range)
    diag.set_xlabel('Dimension', fontsize=AXES_LABEL_SIZE)
    # diag.set_yscale('log')
    diag.set_ylim(0, 100)
    diag.set_ylabel(r'Variances [$\%$]', fontsize=AXES_LABEL_SIZE)
    plot.tick_params(labelsize=LABEL_SIZE)
    if names == None:
        names = ('PNDS', 'CPNS')
    diag.legend(names, loc='best', fontsize=min(18, 150 // len(names)), labelspacing=0.5)
    plot.tight_layout()
    if (not filename == None):
        #        filename = OUTPUT_FOLDER + filename + '_tex.pdf'
        #        plot.savefig(filename)
        #        subprocess.call(['sed', '-i', 's/\/Group.*R//g', filename])
        plot.savefig(OUTPUT_FOLDER + filename + '.png', dpi=300)
    else:
        plot.show()
    plot.close()


def inv_var_plot(data, names, colors=None, y_axis_label=None, filename=None):
    fig = plot.figure()
    diag = fig.add_subplot(111)
    x_range = max([len(d) for d in data]) - 0.5
    if colors is None:
        colors = [rainbow(i, 4 * len(data) // 3) for i in range(len(data))]
    y_max = 0
    for i, d in enumerate(data):
        y_max = max(y_max, np.max(d[:, 1]))
        diag.plot(d[:, 0], d[:, 1], c=colors[i], lw=3)
        diag.scatter(d[:, 0], d[:, 1], marker='.', s=200, c=colors[i], linewidth=0.5)
    diag.set_xlim(0, x_range)
    diag.set_ylim(bottom=0)  # , ((y_max // 50) + 1) * 50)
    if y_axis_label == None:
        y_axis_label = r'Residual Variance [$\%$]'
    diag.set_xlabel('Dimension', fontsize=AXES_LABEL_SIZE)
    diag.set_ylabel(y_axis_label, fontsize=AXES_LABEL_SIZE)
    plot.tick_params(labelsize=LABEL_SIZE)
    diag.legend(names, loc='best', fontsize=min(18, 150 // len(names)), labelspacing=0.5)
    plot.tight_layout()
    if (not filename == None):
        plot.savefig(OUTPUT_FOLDER + filename + '.png', dpi=300)
    else:
        plot.show()
    plot.close()


def abs_var_plot(data, threshold, names, y_label, filename=None):
    fig = plot.figure()
    diag = fig.add_subplot(111)
    x_range = max([len(d) for d in data]) - 0.5
    colors = [rainbow(i, 4 * len(data) // 3) for i in range(len(data))]
    for i, d in enumerate(data):
        limit = 1
        while (limit < len(d)) and (d[limit, 1] > threshold): limit += 1
        limit += 1
        diag.plot(d[:, 0], d[:, 1], c=colors[i])
        diag.scatter(d[1:limit, 0], d[1:limit, 1], marker='*', s=100, c=colors[i], linewidth=0.5)
        diag.scatter(d[limit:, 0], d[limit:, 1], marker='.', s=50, c=colors[i], linewidth=0.5)
    diag.set_xlim(0, x_range)
    diag.set_xlabel('Dimension', fontsize=AXES_LABEL_SIZE)
    diag.set_ylim(0, 100)
    diag.set_ylabel(y_label, fontsize=AXES_LABEL_SIZE)
    plot.tick_params(labelsize=LABEL_SIZE)
    if names == None:
        names = ('PNDS', 'CPNS')
    diag.legend(names, loc='best', fontsize=LABEL_SIZE - 6.0, labelspacing=0.5)
    plot.tight_layout()
    if (not filename == None):
        plot.savefig(OUTPUT_FOLDER + filename + '.png', dpi=300)
    else:
        plot.show()
    plot.close()


def residues_plot(residues, colors, offset, filename=None):
    fig = plot.figure()
    diag = fig.add_subplot(111, projection='3d')
    maxs = np.zeros(3)
    mins = np.zeros(3)
    for i, res in enumerate(residues):
        chain = np.array([[-1 / sqrt(5), 0, -2 / sqrt(5)], [0, 0, 0], [1, 0, 0]])
        for angle in res[offset:]:
            vec1 = chain[-3] - chain[-2]
            vec2 = chain[-1] - chain[-2]
            vec3 = vec1 - 2 * np.einsum('i,i->', vec1, vec2) * vec2
            vec3 = (cos(angle * RAD) * vec3 + sin(angle * RAD) * __x(vec2, vec3) +
                    (1 - cos(angle * RAD)) * np.einsum('i,i->', vec2, vec3) * vec2)
            chain = np.vstack((chain, chain[-1] + vec3 / la.norm(vec3)))
        for angle in reversed(res[:offset]):
            angle = - angle
            vec1 = chain[2] - chain[1]
            vec2 = chain[0] - chain[1]
            vec3 = vec1 - 2 * np.einsum('i,i->', vec1, vec2) * vec2
            vec3 = (cos(angle * RAD) * vec3 + sin(angle * RAD) * __x(vec2, vec3) +
                    (1 - cos(angle * RAD)) * np.einsum('i,i->', vec2, vec3) * vec2)
            chain = np.vstack((chain[0] + vec3 / la.norm(vec3), chain))
        lw = 0.1  # 0.5 + 31.5 / len(chain)
        s = 1  # 1 + 891 / len(chain)
        diag.plot(chain[:, 0], chain[:, 1], chain[:, 2], c=colors[i], linewidth=lw)
        s = diag.scatter(chain[:, 0], chain[:, 1], chain[:, 2], marker="o",
                         c=colors[i], linewidth=0.2, s=s)
        # s.set_facecolors = lambda *args:None
        maxs = np.maximum(maxs, np.max(chain, axis=0))
        mins = np.minimum(mins, np.min(chain, axis=0))
    max_range = 0.5 * np.max(maxs - mins) - 1
    means = 0.5 * (maxs + mins)
    diag.set_xlim(means[0] - max_range, means[0] + max_range)
    diag.set_ylim(means[1] - max_range, means[1] + max_range)
    diag.set_zlim(means[2] - max_range, means[2] + max_range)
    # diag.set_axis_off()
    if (not filename == None):
        plot.savefig(OUTPUT_FOLDER + filename + '.png', dpi=300)
    else:
        plot.show()
    plot.close()


def __x(p, q):
    return np.array([p[1] * q[2] - p[2] * q[1], p[2] * q[0] - p[0] * q[2],
                     p[0] * q[1] - p[1] * q[0]])


def sphere_plot(data, circles=None, name=None):
    data = np.array(data)
    # Make mesh for the sphere
    theta = np.arange(0, PI + 0.01, PI / 18.)
    phi = np.arange(TURN_SPHERE, 2 * PI + TURN_SPHERE + 0.01, PI / 18.)
    theta, phi = np.meshgrid(theta, phi)
    sphere = [np.sin(theta) * np.cos(phi),
              np.sin(theta) * np.sin(phi),
              np.cos(theta)]
    fig = plot.figure()
    diag = fig.add_subplot(111, projection='3d')
    diag.autoscale(enable=False, axis='both', tight=None)
    diag.set_aspect('equal')
    diag.set_xlim(-0.7, 0.7)
    diag.set_ylim(-0.7, 0.7)
    diag.set_zlim(-0.7, 0.7)
    diag.plot_surface(sphere[0], sphere[1], sphere[2], rstride=1, cstride=1,
                      linewidths=0.09, edgecolors='#000066', color='#6666FF',
                      alpha=0.08)
    if (not circles is None):
        for [circ, col] in circles:
            diag.plot(circ[:, 0], circ[:, 1], circ[:, 2], c=col, lw=1)
    colormap = True
    if not data is None:
        for [p, c, m, s] in data:
            lw = s / (40 if ((m == '+') or (m == 'x')) else 100)
            sc = diag.scatter(p[:, 0], p[:, 1], p[:, 2], marker=m, linewidth=lw,
                              edgecolor='black', c=c, cmap=HEAT, s=s)
            colormap = colormap and (c is None)
    if colormap:
        fig.colorbar(sc)
    diag.set_axis_off()
    if (not name is None):
        #        name = OUTPUT_FOLDER + name + '_tex.pdf'
        #        plot.savefig(name)
        #        subprocess.call(['sed', '-i', 's/\/Group.*R//g', name])
        plot.savefig(OUTPUT_FOLDER + name + '.png', dpi=300)
    else:
        plot.show()
    plot.close()


def sphere_views(data, center, circles=None, name=None):
    def rot(x, v): return apply_matrix(x, rotation(center, np.array(v)))

    def view(v, n):
        d = [[rot(p, v), c, m, s] for [p, c, m, s] in data]
        ci = None if circles is None else [[rot(circ, v), col] for [circ, col] in circles]
        sphere_plot(d, ci, None if name == None else name + '_view' + str(n))

    view([sqrt(0.94), 0, sqrt(0.06)], 1)
    view([sqrt(0.45), -sqrt(0.45), sqrt(0.1)], 2)
    view([0, -sqrt(0.9), sqrt(0.1)], 3)


def one_sphere_view(data, center, v, circles=None, name=None):
    def rot(x, v): return apply_matrix(x, rotation(center, np.array(v)))

    def view(v, n):
        d = [[rot(p, v), c, m, s] for [p, c, m, s] in data]
        sphere_plot(d, [[rot(circ, v), col] for [circ, col] in circles],
                    None if name == None else name + '_view' + str(n))

    view(v, '_custom')


def make_S2_plots(colors, data, folder):
    plot_3d(colors, data, title='Full data set', filename=folder + 'data_2d')
    for i in [100, 250, 500, 1000]:
        plot_3d(colors, data, number=i, title='Random sample (' + str(i) + ')',
                filename=folder + 'data_2d_sample_' + str(i))


def make_histogram(data, filename=None):
    fig = plot.figure()
    diag = fig.add_subplot(111)
    # Bin data
    bin_size = (np.max(data) - np.min(data)) / 50
    rounded = np.round(data / bin_size).astype(int)
    min_bin = np.min(rounded)
    max_bin = np.max(rounded)
    d = max_bin - min_bin + 1
    x = np.arange(bin_size * min_bin, bin_size * (max_bin + 0.1), bin_size) + 0.5 * bin_size
    y = np.histogram(data, bins=d, range=(bin_size * min_bin, bin_size * max_bin))[0]
    diag.bar(x, y, width=bin_size, linewidth=0)
    if (not filename == None):
        #        filename = filename + '.pdf'
        #        plot.savefig(filename)
        #        subprocess.call(['sed', '-i', 's/\/Group.*R//g', filename])
        plot.savefig(OUTPUT_FOLDER + filename + '.png', dpi=300)
    else:
        plot.show()
    plot.close()


def custom_histogram(data, xrange, bin_size, filename=None):
    fig = plot.figure()
    diag = fig.add_subplot(111)
    # Bin data
    d = round((xrange[1] - xrange[0]) / bin_size)
    y, x = np.histogram(data, bins=d, range=xrange)
    diag.bar(x[:-1], y, width=bin_size, linewidth=0)
    diag.set_xlim(xrange)
    if (not filename == None):
        #        filename = filename + '.pdf'
        #        plot.savefig(filename)
        #        subprocess.call(['sed', '-i', 's/\/Group.*R//g', filename])
        plot.savefig(OUTPUT_FOLDER + filename + '.png', dpi=300)
    else:
        plot.show()
    plot.close()


# Worker thread to encapsulate plotting and avoid pyplot memory leak.
def plot_thread(worker):
    proc = mp.Process(target=worker)
    proc.daemon = True
    proc.start()
    proc.join()


"""#############################################################################
Plot supporting function
#############################################################################"""


def rainbow(number, total):
    # Normalize to a total of 6 * 255
    number = number % total
    number += (total if number < 0 else 0)
    number = int(1530. * number / float(total))
    # Define color values.
    r = (255 if number < 511 else
         (765 - number if number < 766 else
          (0 if number < 1021 else
           (number - 1020 if number < 1276 else
            255))))
    g = (int(number / 2) if number < 511 else
         (255 if number < 766 else
          (1020 - number if number < 1021 else
           0)))
    b = (0 if number < 766 else
         (number - 765 if number < 1021 else
          (255 if number < 1276 else
           1530 - number)));
    return '#%02X%02X%02X' % (r, g, b)


def make_circle(normal, height):
    n = 200
    phi = np.arange(0, 2 * PI + EPS, 2 * PI / n)
    width = sqrt(1 - height ** 2)
    circle = np.vstack((width * np.cos(phi), width * np.sin(phi), height * np.ones(n + 1))).T
    return apply_matrix(circle, rotation(np.array([0, 0, 1]), normal))


def make_gauss(values, scale):
    sigma = 3 + values.std() / 10.
    x_values = np.arange(-1.1 * scale, 1.1 * scale + .001, 0.1)
    y_values = np.einsum('i,j->ij', x_values, np.ones(len(values))) - values
    y_values = np.einsum('ij->i', (np.exp(-y_values ** 2 / (2.0 * sigma ** 2)) /
                                   (sqrt(2 * PI) * sigma)))
    return np.vstack((x_values, y_values))


def make_gauss_flex(values, sigma, scale):
    x_values = np.arange(-1.1 * scale, 1.1 * scale + .001, 0.1)
    y_values = x_values[:, np.newaxis] - values[np.newaxis, :]
    y_values = np.einsum('ij->i', (np.exp(-y_values ** 2 / (2.0 * sigma ** 2)) /
                                   (sqrt(2 * PI) * sigma)))
    return np.vstack((x_values, y_values))


def gaussianize(data, sigma):
    data = (data + 2 * PI) % (2 * PI)
    x_values = np.linspace(0., 2 * PI, 1000)
    y_values = np.abs(np.einsum('i,j->ij', x_values, np.ones(len(data))) - data)
    y_values[y_values > PI] -= 2 * PI
    y_values = np.einsum('ij->i', (np.exp(-y_values ** 2 / (2.0 * sigma ** 2)) /
                                   (sqrt(2 * PI) * sigma)))
    return y_values


def gaussianize_linear(data, sigma):
    lo = np.min(data);
    hi = np.max(data)
    x_values = np.linspace(lo - 4 * sigma, hi + 4 * sigma, 1000)
    y_values = np.abs(np.einsum('i,j->ij', x_values, np.ones(len(data))) - data)
    y_values = np.einsum('ij->i', (np.exp(-y_values ** 2 / (2.0 * sigma ** 2)) /
                                   (sqrt(2 * PI) * sigma)))
    return x_values, y_values


def unique_list(l):
    unique = set();
    unique_add = unique.add
    return [x for x in l if not (x in unique or unique_add(x))]


def square_subplot(diag):
    pos = diag.get_position().get_points()
    x0 = 0.5 * (pos[0, 0] + pos[1, 0] + pos[0, 1] - pos[1, 1])
    diag.set_position([x0, pos[0, 1], 2 * (pos[1, 1] - pos[0, 1]) / 3.0, (pos[1, 1] - pos[0, 1])])
