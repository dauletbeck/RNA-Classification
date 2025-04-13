# -*- coding: utf-8 -*-
"""
Copyright (c) 2021, Benjamin Eltzer

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software. 

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER
TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF
THIS SOFTWARE.
"""

import numpy as np

AX = np.newaxis


def em_normal(data, k):
    n = len(data)
    scale = 0.01 * np.std(data)
    centers = data[np.random.choice(n, k, replace=False)] + scale * np.random.randn(k)
    v0 = np.var(data)
    p0 = 1 / v0
    var_invs = np.ones(k) / np.var(data)
    weights = np.ones(k) / k
    counter = 0
    while True:
        scores = expectation(data, centers, var_invs, weights, p0)
        weights = np.sum(scores, axis=1) / np.sum(scores)
        # Reset if weights too low
        while ((np.min(weights) < 1e-5) or np.isnan(np.sum(weights))):
            for j in range(k):
                if ((weights[j] < 1e-5) or np.isnan(weights[j])):
                    centers[j] = data[np.random.choice(n, 1)] + scale * np.random.randn(1)
                    var_invs[j] = p0
            scores = expectation(data, centers, var_invs, np.ones(k) / k, p0)
            weights = np.sum(scores, axis=1) / np.sum(scores)
        new_centers, new_sigma_invs = maximization(data, scores, v0)
        #        print(counter, np.array([la.det(s) for s in new_sigma_invs]))
        change = np.sum((new_centers - centers) ** 2) + np.sum((new_sigma_invs - var_invs) ** 2)
        order = np.argsort(new_centers)
        centers = new_centers[order]
        sigma_invs = new_sigma_invs[order]
        counter += 1
        if ((change < 1e-12) or (counter > 100)):
            break
    #    scatter_plot(data, centers, np.argmax(scores,axis=0))
    #    print(counter, change, likelihood(data, centers, sigma_invs))
    return centers, sigma_invs, expectation(data, centers, sigma_invs, weights, None)


def expectation(data, mus, var_invs, weights, d0):
    diffs = data[AX, :] - mus[:, AX]
    if np.min(var_invs) <= 0:
        if d0 is None:
            return None
        var_invs[var_invs <= 0] = d0
    dets = np.log(var_invs)
    scores = dets[:, AX] - np.einsum('kn,kn->kn', diffs * var_invs[:, AX], diffs)
    scores = np.exp(scores - np.max(scores, axis=0)[AX, :]) * weights[:, AX]
    scores = scores / np.sum(scores, axis=0)[AX, :]
    return scores


def maximization(data, scores, s0):
    mus = np.sum(data[AX, :] * scores, axis=1) / np.sum(scores, axis=1)
    x = data[AX, :] - mus[:, AX]
    sigmas = np.einsum('kn,kn->k', x * scores, x) / np.sum(scores, axis=1)
    invalid = (sigmas <= 0)
    if np.max(invalid):
        sigmas[invalid] = s0
        mus[invalid] *= 0.1
        mus[invalid] += 0.9 * data[np.random.choice(len(data), 1)]
    return mus, 1 / sigmas


def likelihood(data, mus, sigma_invs, scores):
    if scores is None:
        return -np.inf
    diffs = data[AX, :] - mus[:, AX]
    dets = np.log(sigma_invs)
    return 0.5 * np.sum(dets * np.sum(scores, axis=1) -
                        np.einsum('kn,kn->k', diffs * scores * sigma_invs[:, AX], diffs))


def em_wrapper(data, k, repetitions, counter=None):
    out = []
    scores = []
    for j in range(repetitions):
        out.append(em_normal(data, k))
        scores.append(likelihood(data, out[-1][0], out[-1][1], out[-1][2]))
    #    print(counter, flush=True)
    return out[np.nanargmax(np.array(scores))]


def em_wrapper_loglikelihood(data, k, repetitions, counter=None):
    out = []
    scores = []
    for j in range(repetitions):
        out.append(em_normal(data, k))
        scores.append(likelihood(data, out[-1][0], out[-1][1], out[-1][2]))

    #    print(counter, flush=True)
    return np.log(np.nanmax(np.array(scores)))


def test_sim():
    data = np.random.randn(50)
    data[:10] += 10
    print(em_wrapper(data, 2, 1))

# if __name__ == '__main__':
#     test_sim()
