import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import chi2

def fit_gaussian_models(distances):
    """Fit one-Gaussian and two-Gaussian models, return likelihoods and cluster indices."""
    gaussian_fit1 = GaussianMixture(n_components=1, n_init=10).fit(distances.reshape(-1, 1))
    gaussian_fit2 = GaussianMixture(n_components=2, n_init=10).fit(distances.reshape(-1, 1))
    
    gaussian_likelihood1 = gaussian_fit1.score(distances.reshape(-1, 1))
    gaussian_likelihood2 = gaussian_fit2.score(distances.reshape(-1, 1))
    
    return gaussian_fit1, gaussian_fit2, gaussian_likelihood1, gaussian_likelihood2

def likelihood_ratio_test(ll_0, ll_1, num_samples, alpha, df=2):
    """Perform likelihood ratio test and return p-value and decision."""
    delta = -2 * (ll_0 - ll_1) * num_samples
    pvalue = chi2.sf(delta, df)
    return pvalue < alpha

def mode_hunting(distances, alpha, min_cluster_size=None):
    if distances is None:
        return 0 if min_cluster_size is None else (np.zeros(len(distances)), False)

    # fit Gaussians and get likelihoods
    gaussian_fit1, gaussian_fit2, ll_0, ll_1 = fit_gaussian_models(distances)
    
    # likelihood ratio test
    if likelihood_ratio_test(ll_0, ll_1, len(distances), alpha):
        if min_cluster_size is not None:
            # check minimal cluster size and return cluster indices if needed
            cluster_indizes2 = gaussian_fit2.fit_predict(distances.reshape(-1, 1))
            check_sum = np.sum(cluster_indizes2)
            if check_sum > len(cluster_indizes2) - min_cluster_size or check_sum < min_cluster_size:
                return gaussian_fit1.fit_predict(distances.reshape(-1, 1)), False, gaussian_fit1.means_
            return cluster_indizes2, True, gaussian_fit2.means_
        return 2  # 2 modes detected
    return 1  # 1 mode detected if min_cluster_size is None

def modehunting_gaussian(distances, alpha, min_cluster_size):
    """
    Perform mode hunting with Gaussian Mixture Models (GMMs) and Likelihood-Ratio Test.
    This tests whether the data comes from one or two Gaussian distributions.
    
    distances: 1D array of data points (distances)
    alpha: significance level for rejecting the null hypothesis (single Gaussian model)
    min_cluster_size: minimum size for valid clusters in two-mode case
    
    Returns:
    cluster_indices: array of cluster labels for each point
    split: Boolean indicating if data was split into two modes
    means_for_plot: Means of the Gaussian components for plotting
    """
    return mode_hunting(distances, alpha, min_cluster_size)

def mode_test_gaussian(distances, alpha):
    """
    Perform simple mode test without needing minimal cluster size.
    Returns:
    1 if data comes from a single Gaussian, 2 if from two Gaussians.
    """
    return mode_hunting(distances, alpha)
