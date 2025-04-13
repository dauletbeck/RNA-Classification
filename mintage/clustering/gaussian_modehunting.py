import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from mintage_pipeline.clustering import gaussian_mixture_model_1d as gm
from sklearn.mixture import GaussianMixture
from scipy.stats import chi2


def modehunting_gaussian(distances, alpha, min_cluster_size):
    # 1D mode hunting for small data size
    # 1. Fit a mixture of two Gaussians to the data via EM-algorithm, compute distance d of the two modes
    # 2. fit single Gaussian to data
    # 3. simulate N times from that single Gaussian and fit a mixture of two Gaussians as above,
    # compute the distances d1,...,dN of the two modes abd the empirical quantile d^_{1-alpha} such that 1-alpha
    # of the distances are smaller than d^_{1-alpha}
    # reject the null hypothesis (data comes from a single Gaussian) at level alpha if the d > d_{1-alpha}
    # 3'. Likelihood-Ratio-Test:
    # chi^2 dist
    # 95% quantil

    if distances is None:
        return np.zeros(len(distances)), False

    # split distances at furthest distance
    distances_sorted = sorted(enumerate(distances), key=lambda x: x[1])
    distances_sort = np.array(distances_sorted)[:, 1]
    ranges = [distances_sort[i + 1] - distances_sort[i] for i in range(len(distances_sort) - 1)]
    # append range from last to first point : 180 degree - last + first
    ranges.append(360 - distances_sort[-1] + distances_sort[0])

    max_range_index = np.argmax(ranges)  # 0. index = 0 and 1 in distances

    split = True
    # 1. ------------
    # centers, sigma_invs, expectations = gm.em_normal(distances, 2)
    # d = expectations[0] - expectations[1]
    # loglikelihood
    # print(distances)
    # two_gaus = gm.em_wrapper_loglikelihood(distances, 2, 1)

    # 2. ------------
    # n = 2
    gaussian_fit1 = GaussianMixture(n_components=1, n_init=10).fit(distances.reshape(-1, 1))
    gaussian_likelihood1 = gaussian_fit1.score(distances.reshape(-1, 1))
    gaussian_fit2 = GaussianMixture(n_components=2, n_init=10).fit(distances.reshape(-1, 1))
    gaussian_likelihood2 = gaussian_fit2.score(distances.reshape(-1, 1))
    # GaussianMixture(n_components=2).score(distances)
    # Compute the per-sample average log-likelihood of the given data distances

    cluster_indizes2 = gaussian_fit2.fit_predict(distances.reshape(-1, 1))
    cluster_indizes1 = gaussian_fit1.fit_predict(distances.reshape(-1, 1))

    means_for_plot = gaussian_fit1.means_

    # minimal cluster size
    check_sum = np.sum(cluster_indizes2)
    if (check_sum > len(cluster_indizes2) - min_cluster_size) or (check_sum < min_cluster_size):
        print("modehunting: don't split data")
        split = False
        return cluster_indizes1, split, means_for_plot

    # 3' ------------
    # the log-likelihoods
    ll_0 = gaussian_likelihood1  # np.mean(two_gaus)
    ll_1 = gaussian_likelihood2

    delta = -2 * (ll_0 - ll_1) * len(distances)  # likelihood ratio
    df = 2  # given the difference in dof - var and mean
    # pvalue = 1 - chi2(df).cdf(delta)  # since Λ follows χ2
    pvalue = chi2.sf(delta, df)
    if pvalue < alpha:
        print("modehunting: split data")
        # reject null hypothesis (data comes from a single Gaussian)
        # divide and get 2 modes
        means_for_plot = gaussian_fit2.means_

        if max(distances[cluster_indizes2 == 1]) >= max(distances[cluster_indizes2 == 0]):
            dividing_point = max(distances[cluster_indizes2 == 0]) \
                             + np.abs(min(distances[cluster_indizes2 == 1]) - max(distances[cluster_indizes2 == 0])) / 2
        else:
            dividing_point = max(distances[cluster_indizes2 == 1]) \
                             + np.abs(min(distances[cluster_indizes2 == 0]) - max(distances[cluster_indizes2 == 1])) / 2
        means_for_plot = np.append(means_for_plot, dividing_point)  # gaussian_fit2.lower_bound_)
        split = True
        return cluster_indizes2, split, means_for_plot

    # from em: got where to divide:
    # divide clusters and do modehunting again
    # otherwise leave clusters as they are
    print("modehunting: don't split data")
    split = False
    return cluster_indizes1, split, means_for_plot


def mode_test_gaussian(distances, alpha):
    # 1D mode hunting for small data size
    # 1. Fit a mixture of two Gaussians to the data via EM-algorithm, compute distance d of the two modes
    # 2. fit single Gaussian to data
    # 3'. Likelihood-Ratio-Test:
    # chi^2 dist
    # 95% quantil

    if distances is None:
        return 0

    split = True
    # n = 2
    gaussian_fit1 = GaussianMixture(n_components=1, n_init=10).fit(distances.reshape(-1, 1))
    gaussian_likelihood1 = gaussian_fit1.score(distances.reshape(-1, 1))
    gaussian_fit2 = GaussianMixture(n_components=2, n_init=10).fit(distances.reshape(-1, 1))
    gaussian_likelihood2 = gaussian_fit2.score(distances.reshape(-1, 1))
    # Compute the per-sample average log-likelihood of the given data distances

    # 3' ------------
    # the log-likelihoods
    ll_0 = gaussian_likelihood1  # np.mean(two_gaus)
    ll_1 = gaussian_likelihood2

    delta = -2 * (ll_0 - ll_1) * len(distances)  # likelihood ratio
    df = 2  # given the difference in dof - var and mean
    # pvalue = 1 - chi2(df).cdf(delta)  # since Λ follows χ2
    pvalue = chi2.sf(delta, df)
    if pvalue < alpha:
        print("modes: 2")
        # reject null hypothesis (data comes from a single Gaussian)
        # divide and get 2 modes
        split = True
        return 2

    print("modes: 1")
    split = False
    return 1
