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

import time
import re
import numpy as np
import numpy.random as arandom
import scipy.linalg as la
import itertools as it
import matplotlib.pyplot as plot
from math import ceil, sqrt, log2, log10

SIZES = [100,200,500,1000,2000,5000,10000]
MC_COUNT = 10000
PI = np.pi
EPS = 1e-9
AX = np.newaxis

"""#############################################################################
################################   Data and I/O  ###############################
#############################################################################"""

def import_csv (filename):
    data = __get_raw_data(filename, data_type='float')
    for key in data:
        tmp = np.array(__discard_empty_lists(data[key]))
        data[key] = np.squeeze(tmp)
    if len(data) == 1 and 'NO_LABEL' in data:
        return data['NO_LABEL']
    return data

def import_lists (filename, data_type='int'):
    data = __get_raw_data(filename, data_type=data_type)
    for key in data:
        data[key] = __discard_empty_lists(data[key])
        if len(data[key]) == 1:
            data[key] = data[key][0]
    if len(data) == 1 and 'NO_LABEL' in data:
        return data['NO_LABEL']
    return data

def __get_raw_data (filename, data_type='int'):
    data = {}
    label = None
    depth = 0
    with open(filename) as datafile:
        for line in datafile:
            if line[0] == '%':
                label = line[1:].strip()
                data[label] = []
                depth = 0
            elif line[0] != '#' and line[0] != ',':
                if label == None:
                    label = 'NO_LABEL'
                    data[label] = []
                level = data[label]
                for i in range(depth):
                    level = level[-1]
                line = line.replace("NA", "nan")
                tmp = filter(None, re.split(',| |\t', line.strip()))
                if data_type=='int':
                    level.append([int(i) for i in tmp])
                elif data_type=='float':
                    level.append([float(i) for i in tmp])
                else:
                    level.append(list(tmp))
            elif line[0] == ',' and label != None:
                tmp = 0
                while ','*(tmp+2) == line[:tmp+2]:
                    tmp += 1
                if tmp == 0:
                    continue
                for i in range(tmp - depth):
                    data[label] = [data[label]]
                    depth += 1
                level = data[label]
                for i in range(depth - tmp):
                    level = level[-1]
                for i in range(tmp):
                    level.append([])
                    level = level[-1]
    return data

def __discard_empty_lists (some_list):
    some_list = [x for x in some_list if x]
    for sublist in some_list:
        if isinstance(sublist, list):
            __discard_empty_lists (sublist)
    some_list = [x for x in some_list if x]
    return some_list

def export_csv (data, filename):
    with open(filename, 'w') as output:
        for key in data.keys():
            output.write('%%%s\n' % key)
            output.write(array2csv(data[key], '%.6f'))
        output.close()

def array2csv (array, form):
    if len(array.shape) == 1:
        return ','.join(form % item for item in array) + '\n'
    if len(array.shape) == 2:
        return '\n'.join(','.join(form % i for i in k) for k in array) + '\n'
    if len(array.shape) > 2:
        marker = ',' * (len(array.shape) - 1) + '\n'
        return marker.join([array2csv(m, form) for m in array])
    return None

def get_quantile (n, alpha=0.05):
    alpha = 1 - alpha
    if n < SIZES[0]:
        return simulate_density(n, MC_COUNT)[int(MC_COUNT * alpha) - 1]
    d = import_csv('quantile_table.csv')
    for i in range(len(SIZES)-1):
        if n < SIZES[i+1] and n >= SIZES[i]:
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

def __where(array):
    return np.squeeze(np.array(np.where(array)))

def get_modes (data, quantile):
    maxima, minima, extrema = find_extrema_regions(data, quantile)
    n = count_modes(maxima, minima, extrema)[1]
#    n[-1] = (n[-1] + 1) // 2
#    n = max(n)
    if n < 1:
        return [np.array(range(len(data)))], []
    tmp = find_kde_extrema(data, n, maxima, minima, extrema)
    if tmp == None:
        return None
    mins = sorted(tmp[0])
    if len(mins) < 1:
        return [np.array(range(len(data)))], []
    mode_list = [__where(data < mins[0])]
    for i in range(1,len(mins)):
        mode_list.append(__where((data < mins[i]) & (data >= mins[i-1])))
    mode_list.append(__where(data > mins[-1]))
    return mode_list, mins

def find_extrema_regions (data, quantile):
    increase, decrease = find_slopes(data, quantile)
    return process_slopes(increase, decrease)

def find_slopes (data, quantile):
    n = len(data)
    data = np.sort(data)
    k1 = np.sqrt(np.arange(1,n-1,1) / 3).reshape((n-2,1))
    k2 = np.sqrt(2*(1 + np.log((n+1) / np.arange(2,n,1)))).reshape((n-2,1))
    dists = la.circulant(data[::-1])[::-1,:] - data
    dists[dists<=0] = 0
    tot_dists = np.cumsum(dists, axis=0)
    dists[dists==0] = -EPS
    test_stat = (2 * tot_dists / dists - np.arange(1,n+1,1)[:,AX])[2:,:]
    test_stat[dists[2:,:] <= 0] = 0
    def slope_intervals (test):
        test = test > (quantile + k2) * k1
        slope = np.vstack((np.argmax(test, axis=0), np.array(range(n))))
        slope = slope[:,slope[0,:] > 0 | test[0,:]]
        slope = slope[:,np.argsort(slope[0])]
        tmp = []
        for [size,left] in slope.T:
            if len(tmp) > 1000:
                tmp = list(cleanup(tmp))
            interval = [data[left+1], data[left+size+1]]
            tmp.append(interval)
        return cleanup(tmp)
    increase = slope_intervals(test_stat)
    decrease = slope_intervals(-test_stat)
    return increase, decrease

def filter_first_increase (data, quantile):
    increase, decrease = find_slopes(data, quantile)
    first_decrease = decrease[0]
    for dec in decrease[1:]:
        if dec[0] > first_decrease[1]:
            break
        first_decrease[1] = dec[1]
    for inc in increase:
        if inc[0] > first_decrease[1]:
            return inc[0]
    return np.inf

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
    tmp = np.array(tmp)
    if len(tmp.shape) == 2:
        tmp = tmp[np.argsort(tmp[:,0])]
    return tmp

def process_slopes (increase, decrease):
    if len(increase) <= 0 or len(decrease) <= 0:
        return [], [], []
    maxima = []
    minima = []
    extrema = []
    for i in increase:
        if len(maxima) > 1000:
            maxima = list(cleanup(maxima))
        if len(minima) > 1000:
            minima = list(cleanup(minima))
        if len(extrema) > 1000:
            extrema = list(cleanup(extrema))
        best_max = None
        best_min = None
        for d in decrease:
            if (best_max == None or d[1] < best_max) and i[1] <= d[0]:
                best_max = d[1]
            if (best_min == None or d[0] > best_min) and d[1] <= i[0]:
                best_min = d[0]
            if i[0] < d[1] and d[0] < i[1]:
                extrema.append([min(i[0], d[0]), max(i[1], d[1])])
        if best_max != None:
            maxima.append([i[0],best_max])
        if best_min != None:
            minima.append([best_min,i[1]])
    maxima = cleanup(maxima)
    minima = cleanup(minima)
    extrema = cleanup(list(extrema) + list(minima) + list(maxima))
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
                    if (a[0] < b[1] and b[0] < a[1]):
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

""" #TODO: Use non-wrapped Gaussians """
def find_kde_extrema (data, n_modes, maxima, minima, extrema):
    data = np.sort(data)
    sigma = 0.0
    increment = 10**(int(round(log10(data[-1] - data[0])))-3)
    out_min = []
    out_max = []
    while True:
        sigma += increment
        x, tmp = __gaussian_kde(data, sigma)
        mins, maxs = __min_max(tmp, x)
#        if mins[-1] > maxs[-1]:
#            mins = mins[:-1]
#        if len(mins)+1 != len(maxs):
#            print('WARNING: Incompatible number of minima and maxima:', len(mins), len(maxs))
        if __any_empty(mins, minima) or __any_empty(maxs, maxima):
            sigma -= increment
            increment *= 0.5
            continue
        out_min += __all_unique(mins, minima, [])
        out_max += __all_unique(maxs, maxima, [])
        minima = __full_filtered(minima, out_min)
        maxima = __full_filtered(maxima, out_max)
#        if (len(out_min) >= n_modes-1 and len(minima) < 1 and
#            not (len(out_max) == n_modes and len(maxima) < 1)):
#            out_max = __fill_gaps(sorted(out_min), data, False)
#            maxima = []
#        if (len(out_max) == n_modes and len(maxima) < 1 and
#            not (len(out_min) >= n_modes-1 and len(minima) < 1)):
#            out_min = __fill_gaps(sorted(out_max), data, True)
#            minima = []
        extrema = __full_filtered(extrema, out_min + out_max)
        if ((len(minima) < 1) and (len(maxima) < 1)): #((len(out_min) >= n_modes-1) and (len(out_max) >= n_modes)):
            if len(out_min) > n_modes:
                print('WARNING: Too many minima.')
#            if len(out_max) > n_modes:
#                print('WARNING: Too many maxima.')
            if len(minima) >= 1:
                print('WARNING: Remaining minima intervals.')
            if len(maxima) >= 1:
                print('WARNING: Remaining maxima intervals.')
            if len(extrema) >= 1:
                print('WARNING: Remaining extrema intervals.')
            return out_min, out_max
        if sigma > 1000*increment:
            print('WARNING: Huge sigma. Something is probably wrong.')
            return None

def __gaussian_kde (data, sigma):
    lo = np.min(data)
    hi = np.max(data)
    length = hi-lo
    lo -= 0.1*length
    hi += 0.1*length
    x = np.linspace(lo, hi, 1001)
    return x, np.sum(np.exp(-(data[:,AX] - x[AX,:])**2/(2*sigma**2)) / (sqrt(2*PI)*sigma), axis=0)

def __min_max (data, x):
    mins = []
    maxs = []
    n = len(data)
    for i in range(1,n-1):
        if data[i] < data[i-1] and data[i] <= data[i+1]:
            mins.append(x[i])
        elif data[i] > data[i-1] and data[i] >= data[i+1]:
            maxs.append(x[i])
    return mins, maxs

def __all_unique (values, intervals, result_list):
    for i in intervals:
        u = __find_unique(values, i)
        if u != None:
            result_list.append(u)
            return __all_unique([x for x in values if x != u],
                                 __filtered(intervals, u), result_list)
    return result_list

def __find_unique (values, interval):
    [a,b] = interval
    out = None
    for v in values:
        if (v >= a and v <= b):
            if out != None:
                return None
            out = v
    return out

def __any_empty (values, intervals):
    for [a,b] in intervals:
        empty = True
        for v in values:
            if (v >= a and v <= b):
                empty = False
                break
        if empty:
            return True
    return False

""" #TODO: Use non-wrapped Gaussians """
def __fill_gaps (points, data, get_mins):
    sigma = 0.0
    increment = 10**(int(round(log10(data[-1] - data[0])))-3)
    out = []
    intervals = [[points[i],points[i+1]] for i in range(len(points)-1)]
    while True:
        if len(intervals) < 1:
            print('WARNING: No gap to fill. Skipping')
            return []
        sigma += increment
        x, tmp = __gaussian_kde(data, sigma)
        mins, maxs = __min_max(tmp, x)
        if mins[-1] > maxs[-1]:
            mins = mins[:-1]
        candidates = (mins if get_mins else maxs)
        if __any_empty(candidates, intervals):
            sigma -= increment
            increment *= 0.5
            continue
        out += __all_unique(candidates, intervals, [])
        intervals = __full_filtered(intervals, out)
        if (len(out) == len(points)):
            return out
        if sigma > 1000*increment:
            print('WARNING: Huge sigma. Something is probably wrong.')
            return None

def __filtered (intervals, v):
    filtered = []
    for [a,b] in intervals:
        if (v >= a and v <= b):
            continue
        filtered.append([a,b])
    return filtered

def __full_filtered (intervals, values):
    filtered = []
    for [a,b] in intervals:
        match = False
        for v in values:
            if (v >= a and v <= b):
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
            if (v >= a and v <= b):
                found = True
                break
        if not found:
            return False
    return True

def simulate_density (n, runs, l=1.):
    runs = max(19, runs)
    dmax = ceil(l * (n+1))
    k1 = np.sqrt(3 / np.arange(1,n+1,1))
    k2 = np.sqrt(2*(1 + np.log((n+1) / np.arange(2,n+2,1))))
    runs_at_once = max(int(min(1e5 / n, runs)), 1)
#    runs_at_once = 10
    n_range = np.arange(1,n+1,1)
    a = 0
    dist_list = []
    while a < runs:
#        if a % 1000 == 0:
#            print('n =', n, '\ta =', a, flush=True)
        dist = - sqrt(2) * np.ones(runs_at_once)
        uni = np.sort(arandom.rand(runs_at_once,n), axis=1)
        uni = np.hstack((np.zeros((runs_at_once,1)), uni, np.ones((runs_at_once,1))))
        for j in range(n):
            tmp = uni[:,j+1:] - (uni[:,j])[:,AX]
            tmp = np.abs(2 * np.cumsum(tmp[:,:-1], axis=1) / tmp[:,1:] -
                         n_range[:n-j]) * k1[:n-j] - k2[:n-j]
            dist = np.maximum(dist, np.max(tmp[:,:min(n-j,dmax)], axis=1))
        dist_list.append(dist)
        a += runs_at_once
    return np.sort(np.hstack(dist_list)[:runs])

def make_table():
    d = {}
    t1 = time.time()
    for n in SIZES:
        d[str(n) + 'points'] = simulate_density(n, MC_COUNT)
        t2 = time.time()
        print(n, ':', t2-t1, ' --- ', d[str(n) + 'points'][9500], flush=True)
        export_csv(d, 'quantile_table.csv')
    q95 = []; q99 = []
    for n in SIZES:
        q95.append(d[str(n) + 'points'][9500])
        q99.append(d[str(n) + 'points'][9900])
    print('5% quantiles: ' + str(q95))
    print('1% quantiles: ' + str(q99))

def test_simulations():
    n = 1000
    t1 = time.time()
    q95 = []
    for a in range(10):
        x = simulate_density(n, MC_COUNT)
        t2 = time.time()
        print(n, ':', t2-t1, ' --- ', x[9500], flush=True)
        q95.append(x[9500])
    q95 = np.array(q95)
    print(np. mean(q95), np.std(q95))

def test_run ():
    n = 1000
    data = np.random.randn(n) + np.random.choice([0,8,16], n, p=[0.5,0.2,0.3])
    quantile = get_quantile(n, alpha=0.05)
    up, down = find_slopes(data, quantile)
    maxr, minr, extr = find_extrema_regions(data, quantile)
    mlist, mins = get_modes(data, quantile)
    print([len(l) for l in mlist], mins)

def test_file (name):
    data = import_csv(name)
    quantile = get_quantile(len(data), alpha=0.05)
    up, down = find_slopes(data, quantile)
    maxr, minr, extr = find_extrema_regions(data, quantile)
    mlist, mins = get_modes(data, quantile)
    print([len(l) for l in mlist], mins)

if __name__ == '__main__':
#    make_table()
#    test_simulations()
#    find_extrema_regions(np.hstack((arandom.rand(100),0.25+0.05*arandom.randn(1900))),1., 2.3)
#    test_run()
    test_file('test_cluster_2.csv')
    