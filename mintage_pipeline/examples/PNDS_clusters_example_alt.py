# -*- coding: utf-8 -*-
"""
Copyright (C) 2014-2018 Benjamin Eltzner

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


import matplotlib
#try: matplotlib.use('Agg')
#except: print('Rerunning')
import numpy as np
import numpy.linalg as la
import numpy.random as arandom
import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D        # Necessary for 3D plots to work!
import math
from PNDS_io import find_files, import_csv, import_lists, export_csv
from PNDS_plot import (scatter_plots, var_plot, inv_var_plot, residual_plots,
                       sphere_views, make_circle, colored_scatter_plots,
                       one_scatter_plot, abs_var_plot, residues_plot, rainbow,
                       one_two_plot, scatter_plot, custom_histogram,
                       one_sphere_view, linear_1d_plot, scatter_plot_3d,
                       scatter_plot_plus_3d, rainbow_scatter_plots, plot_thread)
from PNDS_geometry import RESHify_1D, unRESHify_1D, torus_distances, euclideanize
from PNDS_PNS  import (pns_loop, fold_points, unfold_points, as_matrix)

"""
AUTHOR'S DISCLAIMER:
This software was written in view of a specific data set on a specific system
and is provided for the sake of verifiability of its results. It is not
recommended to reuse this software in any other scenario.

The functions in this program do not check for well-formed input which may lead
to unexpected or even harmful behavior. Output functions do not check for
pre-existing data. Running the program CAN RESULT IN LOSS OF DATA by unchecked
overwriting and CAN POTENTIALLY DAMAGE YOUR SYSTEM.

This disclaimer need not be reproduced in copies or derived works of this
software. However, the author would be grateful if it was kept intact at least
in unmodified copies.
"""

################################################################################
################################   Constants   #################################
################################################################################

DEG = math.degrees(1)
PI = np.pi
LABEL_SIZE = 18.0
AXES_LABEL_SIZE = 20.0
OUTPUT_FOLDER = "out/"

################################################################################
############################   Auxiliary functions   ###########################
################################################################################

def scatter_2D (data, circle, labels, ran=[-180,180], filename = None):
    fig = plot.figure()
    diag = fig.add_subplot(111)
    diag.set_xlim(*ran)
    diag.set_ylim(*ran)
    diag.set_aspect('equal')
    diag.set_xlabel(labels[0], fontsize=AXES_LABEL_SIZE)
    diag.set_ylabel(labels[1], fontsize=AXES_LABEL_SIZE)
    plot.tick_params(labelsize=LABEL_SIZE)
    plot.tight_layout()
    for [p, c, m, s] in data:
        lw = s/(40 if ((m == '+') or (m == 'x')) else 100)
        tmp = diag.scatter(p[:,0], p[:,1], c=c, marker=m, linewidth=lw,
                           edgecolor='black', s=s)
        tmp.set_facecolors = lambda *args:None
        tmp.set_edgecolors = lambda *args:None
    if not circle is None:
        jumps = [0] + list(np.where(np.abs(circle[1:] - circle[:-1]) > 300)[0]+1) + [len(circle)]
        for i in range(len(jumps[:-1])):
            c = circle[jumps[i]:jumps[i+1]]
            diag.plot(c[:,0], c[:,1], c='black')
    if (not filename == None):
        plot.savefig(OUTPUT_FOLDER + filename + '.png', dpi=300)
    else:
        plot.show()
    plot.close()

def scatter_3D (data, circle, man_2d, labels, ran=[0, 360], filename = None):
    fig = plot.figure()
    diag = fig.add_subplot(111, projection='3d')
    diag.autoscale(enable=False, axis='both', tight=None)
    diag.set_aspect('equal')
    diag.set_xlim(*ran)
    diag.set_ylim(*ran)
    diag.set_zlim(*ran)
    diag.set_xlabel(labels[0], fontsize=AXES_LABEL_SIZE)
    diag.set_ylabel(labels[1], fontsize=AXES_LABEL_SIZE)
    diag.set_zlabel(labels[2], fontsize=AXES_LABEL_SIZE)
    plot.tick_params(labelsize=LABEL_SIZE-6)
    plot.tight_layout()
    for [p, c, m, s] in data:
        lw = s/(40 if ((m == '+') or (m == 'x')) else 100)
        tmp = diag.scatter(p[:,0], p[:,1], p[:,2], c=c, marker=m, linewidth=lw,
                           edgecolor='black', s=s)
        tmp.set_facecolors = lambda *args:None
        tmp.set_edgecolors = lambda *args:None
    if not circle is None:
        jumps = [0] + list(np.where(np.abs(circle[1:] - circle[:-1]) > 300)[0]+1) + [len(circle)]
        for i in range(len(jumps[:-1])):
            c = circle[jumps[i]:jumps[i+1]]
            diag.plot(c[:,0], c[:,1], c[:,2], c='black')
    if not man_2d is None:
        diag.scatter(man_2d[:,0], man_2d[:,1], man_2d[:,2], c='#444444', marker='o', linewidth=0, s=1)
    if (not filename == None):
        plot.savefig(OUTPUT_FOLDER + filename + '.png', dpi=150)
    else:
        plot.show()
    #plot.close()

################################################################################
#############################   Control functions   ############################
################################################################################

""" Full data processing """
def explore ():
    files = find_files('cluster*.txt', 'data')
    colors = ['red'] * 500 + ['#00dd00'] * 500 + ['#00dddd'] * 500 + ['#6600ff'] *500
    for f in sorted(files):
        print(f)
        data = np.array(import_lists(f, as_type='String')).astype(np.float)*DEG
        print(f, data.shape, np.min(data), np.max(data))
        sphere_points, means, half = RESHify_1D(data, False)
#        sphere_views([[sphere_points, colors, 'o', 5]], [0,0,1], name='protein_' + f[-5] + '_spheres')
        spheres, projected_points, distances = pns_loop(sphere_points, False, False, mode='torus', half=half)
        residual_plots(distances, 1, data.shape[-1], colors, f[-15:-4] + '_residuals')
        print(distances[0].shape, distances[1].shape)
        scatter_2D([[np.array([distances[-1], distances[-2]]).T, colors, 'o', 5]], None,
                     [r'PC 1',r'PC 2'], [-180,180], f[-15:-4] + '_PCs')
        phi = np.linspace(0,2*PI,1001)
        phi = np.vstack((np.sin(phi), np.cos(phi))).T
        circle = unRESHify_1D(unfold_points(phi, spheres[:-1]), means, half)
        circle[circle>180] -= 360
        if data.shape[-1] < 3:
            scatter_2D([[data, colors, 'o', 5]], circle, [r'$\phi$',r'$\theta$'], [-180,180], f[-15:-4])
        else:
            man_2d = arandom.randn(10000, 3)
            man_2d /= la.norm(man_2d, axis=1)[:,np.newaxis]
            man_2d = unRESHify_1D(unfold_points(man_2d, spheres[:-2]), means, half)
            man_2d[man_2d>180] -= 360
            scatter_3D([[data, colors, 'o', 5]], circle, man_2d, [r'$\phi$',r'$\theta$', r'$\psi$'], [-180,180])#, f[-15:-4])



if __name__ == '__main__':
    explore()
