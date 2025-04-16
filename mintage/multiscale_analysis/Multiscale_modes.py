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

import sys
import numpy as np
import numpy.random as arandom
import scipy.linalg as la
import itertools as it
from math import ceil, sqrt, log2
from pnds.PNDS_io import export_csv, import_csv

SIZES = [100,200,500,1000,2000]
MC_COUNT = 10000
PI = np.pi
EPS = 1e-9

def get_quantile (n, alpha=0.05):
    alpha = 1 - alpha
    if n < SIZES[0]:
        return simulate_density(n, MC_COUNT)[int(MC_COUNT * alpha) - 1]
    d = import_csv('out/quantile_table.csv')
    for i in range(len(SIZES)-1):
        if n <= SIZES[i+1] and n > SIZES[i]:
            # Logarithmic extrapolation.
            table_lo = d[str(SIZES[i]) + 'points']
            table_hi = d[str(SIZES[i+1]) + 'points']
            quantile_lo = table_lo[int(len(table_lo) * alpha) - 1]
            quantile_hi = table_hi[int(len(table_hi) * alpha) - 1]
            scale = (n - SIZES[i]) / float(SIZES[i+1] - SIZES[i])
            return quantile_lo**(1-scale) * quantile_hi**scale
    table_lo = d[str(SIZES[-2]) + 'points']
    table_hi = d[str(SIZES[-1]) + 'points']
    quantile_lo = table_lo[int(len(table_lo) * alpha) - 1]
    quantile_hi = table_hi[int(len(table_hi) * alpha) - 1]
    return quantile_hi + log2(n/float(SIZES[-1])) * (quantile_hi - quantile_lo)

def arange (begin, end):
    return np.array(list(range(begin,end)))

def __where(array):
    return np.squeeze(np.array(np.where(array)))

def get_modes (data, max_dist, quantile):
    data = (data + max_dist) % max_dist
    maxima, minima, extrema = find_extrema_regions(data, max_dist, quantile)
    n = count_modes(maxima, minima, extrema)
    n[-1] = (n[-1] + 1) // 2
    n = max(n)
    if n < 2:
        return [np.array(range(len(data)))], [-1]
    tmp = find_kde_extrema(data, max_dist, n, maxima, minima, extrema)
    if tmp == None:
        return [np.array(range(len(data)))], [-1]
    mins = sorted(tmp[0])
    mode_list = []
    for i in range(len(mins)-1):
        mode_list.append(__where((data > mins[i]) & (data <= mins[i+1])))
    mode_list.append(__where((data > mins[-1]) | (data <= mins[0])))
    return mode_list, mins

def find_extrema_regions (data, max_dist, quantile):
    n = len(data)
    data = np.sort((data + max_dist) % max_dist)
    k1 = np.sqrt(arange(1,n-1) / 3).reshape((n-2,1))
    k2 = np.sqrt(2*(1 + np.log((n+1) / arange(2,n)))).reshape((n-2,1))
    # dists[i,j] = data[j+i] - data[j]
    dists = la.circulant(data[::-1])[::-1,:] - data
    dists[dists<0] += max_dist
    dists[dists==0] = EPS
    test_stat = (2 * np.cumsum(dists, axis=0)[2:,:] / dists[2:,:] -
                 arange(3,n+1).reshape((n-2,1)))
    test_stat[dists[2:,:] > 0.5 * max_dist] = 0
    def slope_intervals (test, up):
        test = test > (quantile + k2) * k1
        slope = np.vstack((np.argmax(test, axis=0), np.array(range(n))))
        slope = slope[:,slope[0,:] > 0 | test[0,:]]
        slope = slope[:,np.argsort(slope[0])]
        tmp = []
        for [size,left] in slope.T:
            if len(tmp) > 1000:
                tmp = cleanup(tmp)
            interval = [data[left], data[(left+size+2)%n]]
            if up:
                a = data[left]
                b = data[(left+1)%n]
                if a > b:
                    b += max_dist
                interval[0] = a/3 + 2*b/3
            else:
                a = data[(left+size+1)%n]
                b = data[(left+size+2)%n]
                if a > b:
                    b += max_dist
                interval[1] = 2*a/3 + b/3
            if interval[1] < interval[0]:
                interval[1] += max_dist
            tmp.append(interval)
        return cleanup(tmp)
    increase = slope_intervals(test_stat, True)
    decrease = slope_intervals(-test_stat, False)
    return process_slopes(increase, decrease)

def cleanup (raw_list):
    tmp = []
    for i, interval in enumerate(raw_list):
        add = True
        for j, other in enumerate(raw_list):
            if other[0] >= interval[0] and other[1] <= interval[1] and i != j:
                #print('Interval', interval, 'contains interval', other)
                add = False
                break
        if add:
            tmp.append(interval)
    return tmp

def process_slopes (increase, decrease):
    if len(increase) <= 0 or len(decrease) <= 0:
        return [], [], []
    maxima = []
    minima = []
    extrema = []
    for i in increase:
        if len(maxima) > 1000:
            maxima = cleanup(maxima)
        if len(minima) > 1000:
            minima = cleanup(minima)
        if len(extrema) > 1000:
            extrema = cleanup(extrema)
        best_max = None
        best_min = None
        for d in decrease:
            if (best_max == None or d[1] < best_max) and i[1] <= d[0]:
                best_max = d[1]
            if (best_min == None or d[0] > best_min) and d[1] <= i[0]:
                best_min = d[0]
            if i[0] < d[1] and d[0] < i[1]:
                extrema.append([min(i[0], d[0]), max(i[1], d[1])])
            if i[0] + 360 < d[1] and d[0] < i[1] + 360:
                extrema.append([min(i[0] + 360, d[0]), max(i[1] + 360, d[1])])
            if i[0] < d[1] + 360 and d[0] + 360 < i[1]:
                extrema.append([min(i[0], d[0] + 360), max(i[1], d[1] + 360)])
        if best_max != None:
            maxima.append([i[0],best_max])
        if best_min != None:
            minima.append([best_min,i[1]])
    maxima = cleanup(maxima)
    minima = cleanup(minima)
    extrema = cleanup(extrema + minima + maxima)
    return maxima, minima, extrema

def count_modes (maxima, minima, extrema):
    counters = []
    for intervals in [maxima, minima, extrema]:
        if len(intervals) < 2:
            counters.append(len(intervals))
            continue
        for n in range(2,len(intervals)+1):
            disjoint = False
            for c in it.combinations(intervals,n):
                disjoint = True
                for [a,b] in it.combinations(c,2):
                    if ((a[0] < b[1] and b[0] < a[1]) or
                        (a[0] + 360 < b[1] and b[0] < a[1] + 360) or
                        (a[0] < b[1] + 360 and b[0] + 360 < a[1])):
                        disjoint = False
                        break
                if disjoint:
                    break
            if not disjoint:
                counters.append(n - 1)
                break
            if disjoint and n == len(intervals):
                counters.append(n)
    return counters

def find_kde_extrema (data, max_dist, n_modes, maxima, minima, extrema):
    factor = max_dist / 360.
    data = np.sort((data + max_dist) % max_dist) / factor
    sigma = 0.0
    increment = 1.0
    out_min = []
    out_max = []
    while True:
        sigma += increment
        tmp = __make_circular_gauss(data, sigma)
        mins, maxs = __min_max(tmp)
        if len(mins) != len(maxs):
            print('WARNING: Different number of minima and maxima:', len(mins), len(maxs))
        if __any_empty(mins, minima) or __any_empty(maxs, maxima):
            sigma -= increment
            increment *= 0.5
            continue
        out_min += __all_unique(mins, minima, [])
        out_max += __all_unique(maxs, maxima, [])
        minima = __full_filtered(minima, out_min)
        maxima = __full_filtered(maxima, out_max)
        if (len(out_min) == n_modes and len(minima) < 1 and
            not (len(out_max) == n_modes and len(maxima) < 1)):
            out_max = __fill_gaps(sorted(out_min), data, False)
            maxima = []
        if (len(out_max) == n_modes and len(maxima) < 1 and
            not (len(out_min) == n_modes and len(minima) < 1)):
            out_min = __fill_gaps(sorted(out_max), data, True)
            minima = []
        extrema = __full_filtered(extrema, out_min + out_max)
        if ((len(out_min) >= n_modes) and (len(out_max) >= n_modes)):
            if len(out_min) > n_modes:
                print('WARNING: Too many minima.')
            if len(out_max) > n_modes:
                print('WARNING: Too many maxima.')
            if len(minima) >= 1:
                print('WARNING: Remaining minima intervals.')
            if len(maxima) >= 1:
                print('WARNING: Remaining maxima intervals.')
            if len(extrema) >= 1:
                print('WARNING: Remaining extrema intervals.')
            return out_min, out_max
        if sigma > 360:
            print('WARNING: Huge sigma. Something is probably wrong.')
            return None

def __min_max (data):
    mins = []
    maxs = []
    n = len(data)
    for i in range(n):
        if data[i] < data[i-1] and data[i] <= data[(i+1) % n]:
            mins.append(i)
        elif data[i] > data[i-1] and data[i] >= data[(i+1) % n]:
            maxs.append(i)
    return mins, maxs

def __make_circular_gauss (data, sigma):
    bins = np.array(range(360))
    gauss = np.abs(np.einsum('i,j->ij', bins, np.ones(len(data))) - data)
    gauss[gauss > 180] -= 360
    gauss = np.einsum('ij->i', (np.exp(-gauss**2 / (2.0 * sigma**2)) /
                               (sqrt(2 * PI) * sigma)))
    return gauss

def __all_unique (values, intervals, result_list):
    for i in intervals:
        u = __find_unique(values, i)
        if u != None:
            result_list.append(u)
            return __all_unique ([x for x in values if x != u],
                                 __filtered(intervals, u), result_list)
    return result_list

def __find_unique (values, interval):
    [a,b] = interval
    out = None
    for v in values:
        if (v >= a and v <= b) or (v + 360 >= a and v + 360 <= b):
            if out != None:
                return None
            out = v
    return out

def __any_empty (values, intervals):
    for [a,b] in intervals:
        empty = True
        for v in values:
            if (v >= a and v <= b) or (v + 360 >= a and v + 360 <= b):
                empty = False
                break
        if empty:
            return True
    return False

def __fill_gaps (points, data, get_mins):
    sigma = 0.0
    increment = 1.0
    out = []
    intervals = [[points[i],points[i+1]] for i in range(len(points)-1)]
    intervals.append([points[-1], 360 + points[0]])
    while True:
        sigma += increment
        tmp = __make_circular_gauss(data, sigma)
        mins, maxs = __min_max(tmp)
        candidates = (mins if get_mins else maxs)
        if __any_empty(candidates, intervals):
            sigma -= increment
            increment *= 0.5
            continue
        out += __all_unique(candidates, intervals, [])
        intervals = __full_filtered(intervals, out)
        if (len(out) == len(points)):
            return out
        if sigma > 360:
            print('WARNING: Huge sigma. Something is probably wrong.')
            return None

def __filtered (intervals, v):
    filtered = []
    for [a,b] in intervals:
        if (v >= a and v <= b) or (v + 360 >= a and v + 360 <= b):
            continue
        filtered.append([a,b])
    return filtered

def __full_filtered (intervals, values):
    filtered = []
    for [a,b] in intervals:
        match = False
        for v in values:
            if (v >= a and v <= b) or (v + 360 >= a and v + 360 <= b):
                match = True
                break
        if match:
            continue
        filtered.append([a,b])
    return filtered

def __all_in (values, intervals):
    for [a,b] in intervals:
        found = False
        for v in values:
            if (v >= a and v <= b) or (v + 360 >= a and v + 360 <= b):
                found = True
                break
        if not found:
            return False
    return True

def simulate_density (n, runs, l=1.):
    runs = max(19, runs)
    dmax = ceil(l * (n+1))
    k1 = np.sqrt(3 / arange(1,n+1))
    k2 = np.sqrt(2*(1 + np.log((n+1) / arange(2,n+2))))
    runs_at_once = min(1e7 / n, runs)
    a = 0
    dist_list = []
    while a < runs:
#        print('n =', n, '\ta =', a, flush=True)
        dist = - sqrt(2) * np.ones(runs_at_once)
        uni = np.sort(arandom.rand(runs_at_once,n), axis=1)
        uni = np.hstack((np.zeros((runs_at_once,1)), uni, np.ones((runs_at_once,1))))
        for j in range(n):
            tmp = uni[:,j+1:] - uni[:,j].reshape((uni.shape[0],1))
            tmp = np.abs(2 * np.cumsum(tmp[:,:-1], axis=1) / tmp[:,1:] -
                         arange(1,n-j+1)) * k1[:n-j] - k2[:n-j]
            dist = np.maximum(dist, tmp[:,:min(n-j,dmax)].max(axis=1))
        dist_list.append(dist)
        a += runs_at_once
    return np.sort(np.hstack(dist_list)[:runs])

def make_table():
    d = {}
    for n in SIZES:
        d[str(n) + 'points'] = simulate_density(n, MC_COUNT)
    export_csv(d, 'quantile_table.csv')
    q95 = []; q99 = []
    for n in SIZES:
        q95.append(d[str(n) + 'points'][9500])
        q99.append(d[str(n) + 'points'][9900])
    print('5% quantiles: ' + str(q95))
    print('1% quantiles: ' + str(q99))


if __name__ == '__main__':
    #make_table()
    find_extrema_regions(np.hstack((arandom.rand(100),0.25+0.05*arandom.randn(1900))),1., 2.3)