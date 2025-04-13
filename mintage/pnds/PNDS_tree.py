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
import platform

import numpy as np
import sys
from math import sqrt

if platform.system() == "Linux":
    import resource

    resource.setrlimit(resource.RLIMIT_STACK, (2 ** 29, -1))

sys.setrecursionlimit(10**5)

################################################################################
##############################   Tree functions   ##############################
################################################################################

def get_points (node):
    if 'points' in node:
        return node['points']
    return get_points(node[0]) + get_points(node[1])

def get_significant_clusters (root):
    clusters = []
    while root.size > 100:
        min_size = sqrt(root.size)
        node = root
        final = node
        candidates = []
        while node.size > min_size:
            tmp = node.split()
            max_size = 0
            for c in tmp:
                if c.size > max_size:
                    max_size = c.size
                    node = c
            tmp = [c for c in tmp if c.size > min_size]
            if len(tmp) > 1:
                tmp.remove(node)
                final = node
                candidates += tmp
        candidates.append(final)
        for c in candidates:
            if c.size > max_size:
                max_size = c.size
                node = c
        if node == root:
            clusters.append(node.get_points())
            return clusters
        clusters.append(node.trim())
        root.count()
    clusters.append(root.get_points())
    return clusters
    

def get_significant_clusters_old (root, tree, min_points, min_size):
    thresholds = sorted([node.level for node in tree], reverse=True)
    clusters = []
    noise = []
    node = root
    i = 0
    cut = get_cut(root, thresholds, min_points)
    threshold = cut + 1
    while threshold > cut:
        tmp = node.split()
        max_size = 0
        for c in tmp:
            if c.size > max_size:
                max_size = c.size
                node = c
        threshold = node.level
        tmp.remove(node)
        for c in tmp:
            if c.size < min_size:
                noise.append(c)
            else:
                clusters.append(c)
        i += 1
        sys.stdout.flush()
        if i > len(thresholds) - 1:
            break
    clusters.append(node)
    return [c.get_points() for c in clusters] + [np.vstack([c.get_points() for c in noise])]

def get_cut (root, thresholds, min_points):
    node = root
    i = 0
    cut = None
    while node.size > min_points:
        tmp = node.split()
        max_size = 0
        for c in tmp:
            if c.size > max_size:
                max_size = c.size
                node = c
        tmp = [c for c in tmp if c.size > min_points]
        if len(tmp) > 1:
            cut = node.level
        i += 1
    return cut

################################################################################
################################   Node Class   ################################
################################################################################

class Node (object):
    def __init__ (self, point = None, child1 = None, child2 = None, level = 0,
                  size = 0):
        self.parent = None
        self.point = point
        self.child1 = child1
        self.child2 = child2
        if self.child1:
            self.child1.__set_parent(self)
        if self.child2:
            self.child2.__set_parent(self)
        self.level = level
        self.size = size

    def __set_parent (self, parent):
        self.parent = parent

    def __disappear (self, caller):
        new = (self.child1 if caller == self.child2 else
               (self.child2 if caller == self.child1 else None))
        if new == None:
            print('Invalid caller!')
            return
        if self == self.parent.child1:
            self.parent.child1 = new
        elif self == self.parent.child2:
            self.parent.child2 = new
        else:
            print('Invalid parent!')
            return
        new.parent = self.parent

    def is_leaf (self):
        return self.point != None

    def higher_child (self):
        if self.child1.size > self.child2.size:
            return self.child1
        return self.child2

    def is_root (self):
        return self.parent == None

    def get_points (self):
        if self.is_leaf():
            return self.point
        return np.vstack((self.child1.get_points(), self.child2.get_points()))

    def split (self):
        if self.is_leaf():
            return []
        return [self.child1, self.child2]

    def trim (self):
        self.parent.__disappear(self)
        return self.get_points()

    def count (self):
        if self.is_leaf():
            self.size = 1
        else:
            self.size = self.child1.count() + self.child2.count()
        return self.size

    def branch_cut (self, threshold):
        if self.level <= threshold:
            return [self]
        return self.child1.branch_cut(threshold) + self.child2.branch_cut(threshold)