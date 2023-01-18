from random import random

import numpy as np
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt
from ex6_utils import (plot_ims, load_MNIST, outlier_data, gmm_data, plot_2D_gmm, load_dogs_vs_frogs,
                       BayesianLinearRegression, poly_kernel, cluster_purity)
from scipy.special import logsumexp
from typing import Tuple


def outlier_regression(model: BayesianLinearRegression, X: np.ndarray, y: np.ndarray, p_out: float, T: int,
                       mu_o: float = 0, sig_o: float = 10) -> Tuple[BayesianLinearRegression, np.ndarray]:
    """
    Gibbs sampling algorithm for robust regression (i.e. regression assuming there are outliers in the data)
    :param model: the Bayesian linear regression that will be used to fit the data
    :param X: the training data, as a numpy array of shape [N, d] where N is the number of points and d is the dimension
    :param y: the regression targets, as a numpy array of shape [N,]
    :param p_out: the assumed probability for outliers in the data
    :param T: number of Gibbs sampling iterations to use in order to fit the model
    :param mu_o: the assumed mean of the outlier points
    :param sig_o: the assumed variance of the outlier points
    :return: the fitted model assuming outliers, as a BayesianLinearRegression model, as well as a numpy array of the
             indices of points which were considered as outliers
    """
    p_outliers, p_inliers = p_out, 1 - p_out

    p_k_arr = np.full(X.shape, p_inliers)
    outlier_indices = np.random.binomial(1, p_outliers, size=len(y))
    model.fit(X, y, True)

    for i_T in range(T):
        outlier_indices = []
        likelihood = np.exp(model.log_likelihood(X, y))
        for i, x in enumerate(X):
            y_i = y[i]
            likelihood_y = likelihood[i]
            p_k = p_outlier_per_x_T(p_outliers, mu_o, sig_o, likelihood_y, y_i)
            p_k_arr[i] = p_k  # outlier prob'
            is_outlier = np.random.binomial(1, p_k)
            if is_outlier > 0:
                outlier_indices.append(i)
        X_T, y_T = get_new_D(X, y, outlier_indices)
        model.fit(X_T, y_T, True)
    return model, np.array(outlier_indices)


def get_new_D(X, y, outlier_indices):
    X_T, y_T = [], []
    for i in range(len(y)):
        if i not in outlier_indices:
            X_T.append(X[i])
            y_T.append(y[i])
    return np.array(X_T), np.array(y_T)


def p_outlier_per_x_T(p_outlier, mu_o, sig_o, likelihood, y_i):
    # model after fit to theta_T new
    normal_o = multivariate_normal.pdf(y_i, mean=mu_o, cov=sig_o)
    nominator = p_outlier * normal_o
    denominator = nominator + ((1 - p_outlier) * likelihood)
    return nominator / denominator


class BayesianGMM:
    def __init__(self, k: int, alpha: float, mu_0: np.ndarray, sig_0: float, nu: float, beta: float,
                 learn_cov: bool = True):
        """
        Initialize a Bayesian GMM model
        :param k: the number of clusters to use
        :param alpha: the value of alpha to use for the Dirichlet prior over the mixture probabilities
        :param mu_0: the mean of the prior over the means of each Gaussian in the GMM
        :param sig_0: the variance of the prior over the means of each Gaussian in the GMM
        :param nu: the nu parameter of the inverse-Wishart distribution used as a prior for the Gaussian covariances
        :param beta: the variance of the inverse-Wishart distribution used as a prior for the covariances
        :param learn_cov: a boolean indicating whether the cluster covariances should be learned or not
        """
        self._k = k
        self._alpha = alpha
        self._mu_prior = mu_0
        self._sig_prior = sig_0
        self._nu = nu
        self._beta = beta
        self._learn_cov = learn_cov

        # for better calculations
        self._d = mu_0.size
        self._empty_k = np.zeros(self._k)
        self._I_d = np.identity(self._d)
        self._inv_beta = 1 / self._beta
        self._inv_sig = 1 / self._sig_prior

        # for gibbs sampling
        self._N_k = np.zeros(shape=self._k)
        self._pi = np.random.dirichlet(alpha=np.array([self._alpha for k in range(self._k)]))
        self._mu_ks = np.random.multivariate_normal(mean=self._mu_prior,
                                                    cov=self._I_d * self._sig_prior,
                                                    size=self._k)
        self._sig_ks = np.array([self._I_d * self._beta for i_k in range(self._k)])

    def log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates the log-likelihood of each data point under each Gaussian in the GMM
        :param X: the data points whose log-likelihood should be calculated, as a numpy array of shape [N, d]
        :return: the log-likelihood of each point under each Gaussian in the GMM
        """

        # ----
        dist = np.sum(X*X, axis=1)[None, :] - 2*self._mu_ks @ X.T + np.sum(self._mu_ks*self._mu_ks, axis=1)[:, None]
        log_likelihood_1 = -.5 * (self._d * np.log(2*np.pi*self._beta) + (self._inv_beta) * dist)
        log_likelihood = np.log(self._pi)[:, None] + log_likelihood_1
        # ----
        # if self._learn_cov: log_likelihood = np.array( [multivariate_normal.logpdf(X, mean=self._mu_ks[v_k],
        # cov=self._sig_ks[v_k]) for v_k in range(self._k)]) else: common_value = self._d * np.log(2 * np.pi *
        # self._beta) x_k = np.array([X - mu_k for mu_k in self._mu_ks]) inner = np.array([np.linalg.norm(x,
        # axis=1) ** 2 for x in x_k]) log_likelihood = -.5 * ( common_value + self._inv_beta * inner) # L[j,
        # i] : log likelihood of cluster j of point i
        return log_likelihood

    def cluster(self, X: np.ndarray) -> np.ndarray:
        """
        Clusters the data according to the learned GMM
        :param X: the data points to be clustered, as a numpy array of shape [N, d]
        :return: a numpy array containing the indices of the Gaussians the data points are most likely to belong to,
                 as a numpy array with shape [N,]
        """
        log_likelihood = self.log_likelihood(X)
        clusters = np.argmax(log_likelihood, axis=0)
        return clusters

    # def get_mu_ks(self, cov_k):
    #     if self._learn_cov:
    #         inv_sig = np.array([np.linalg.inv(cov_k[v_k]) for v_k in range(self._k)])
    #         cov = np.array([self._N_k[v_k] * inv_sig[v_k] + self._sig_prior * self._I_d for v_k in range(self._k)])
    #         cov = np.array([np.linalg.inv(m) for m in cov])
    #         mu = np.array(
    #             [cov[v_k] @ ((inv_sig[v_k] @ self._sum_X_per_k[v_k]) + (self._inv_sig * self._mu_prior)) for v_k in
    #              range(self._k)])
    #     else:
    #         denominator_arr = np.array([1 / (n_k * self._inv_beta + self._inv_sig) for n_k in self._N_k])
    #         cov = np.array([self._I_d * denom for denom in denominator_arr])
    #         mu = self._inv_beta * self._sum_X_per_k + (self._inv_sig * self._mu_prior)
    #         mu = np.array([mu[i] / denominator_arr[i] for i in range(self._k)])
    #     return np.array([np.random.multivariate_normal(mu[v_k], cov[v_k]) for v_k in range(self._k)])

    def get_dist(self,X, mu, indices):
        x_indices = X[indices]
        if np.size(x_indices) == 0:
            return np.zeros(shape=self._I_d.shape)
        else:
            sub = x_indices - mu[None, :]
            s = sub.T @ sub
            return s
    def gibbs_fit(self, X: np.ndarray, T: int) -> 'BayesianGMM':
        """
        Fits the Bayesian GMM model using a Gibbs sampling algorithm
        :param X: the training data, as a numpy array of shape [N, d] where N is the number of points
        :param T: the number of sampling iterations to run the algorithm
        :return: the fitted model
        """
        self._N = X.shape[0]
        for t in range(T):
            self._L = self.log_likelihood(X)
            self._q = np.exp(self._L - logsumexp(self._L, axis=0)[None, :]).T
            self._z_t = np.random.default_rng().multinomial(1, self._q, size=self._N).argmax(axis=-1)
            self._N_k = np.bincount(self._z_t, minlength=self._k)
            self._pi = np.random.dirichlet(alpha=self._alpha + self._N_k)
            for v_k in range(self._k):
                indices_k = np.where(self._z_t == v_k)[0]
                x_k = np.sum(X[indices_k], axis=0)
                if self._learn_cov:
                    dist = self.get_dist(X, self._mu_ks[v_k], indices_k)
                    self._sig_ks[v_k] = ((self._nu * self._beta * self._I_d) + dist) / (self._nu + self._N_k[v_k])
                inv_cov = np.linalg.inv(self._sig_ks[v_k])
                mu_cov = np.linalg.inv(self._N_k[v_k] * inv_cov + self._inv_sig * self._I_d)
                m_k = mu_cov @ (inv_cov @ x_k + self._inv_sig * self._mu_prior)
                self._mu_ks[v_k] = np.random.multivariate_normal(m_k, mu_cov)
        return self
        #
        # for t in range(T):
        #     self._L = self.log_likelihood(X)
        #     self._q = np.exp(self._L - logsumexp(self._L, axis=0)[None, :])
        #     self._z_t = np.random.default_rng().multinomial(1, self._q.T, size=self._N).argmax(axis=-1)
        #     indices_per_k = [np.where(self._z_t == v_k) for v_k in range(self._k)]
        #     self._sum_X_per_k = np.array([np.sum(X[indices], axis=0) for indices in indices_per_k])
        #     self._N_k = np.array([np.count_nonzero(self._z_t == v_k) for v_k in range(self._k)])
        #     self._pi = np.random.dirichlet(alpha=self._alpha + self._N_k)
        #     # self._sig_ks = (self._nu * self._I_d * self._beta + sub_inner @ sub_inner.T) / self._nu + self._N_k
        #     if self._learn_cov:
        #         sig_common = self._nu + self._I_d * self._beta
        #         sig_array_denominator = self._nu + self._N_k
        #         inner_array = [X[indices] for indices in indices_per_k]
        #         inner_array_minus_mu_ks = [inner_array[v_k] - self._mu_ks[v_k] for v_k in range(self._k)]
        #         inner_value = [np.zeros(self._d) if i_v_k.shape[0] == 0 else i_v_k.T @ i_v_k for i_v_k in
        #                        inner_array_minus_mu_ks]
        #         self._sig_ks = np.array(
        #             [(sig_common + inner_value[v_k]) / sig_array_denominator[v_k] for v_k
        #              in range(self._k)])
        #     self._mu_ks = self.get_mu_ks(self._sig_ks)


if __name__ == '__main__':
    # # ------------------------------------------------------ section 2 - Robust Regression
    # # ---------------------- question 2
    # # load the outlier data
    # x, y = outlier_data(50)
    # # init BLR model that will be used to fit the data
    # mdl = BayesianLinearRegression(theta_mean=np.zeros(2), theta_cov=np.eye(2), sample_noise=0.15)
    #
    # # sample using the Gibbs sampling algorithm and plot the results
    # plt.figure()
    # plt.scatter(x, y, 15, 'k', alpha=.75)
    # xx = np.linspace(-0.2, 5.2, 100)
    # for t in [0, 1, 5, 10, 25]:
    #     samp, outliers = outlier_regression(mdl, x, y, T=t, p_out=0.1, mu_o=4, sig_o=2)
    #     plt.plot(xx, samp.predict(xx), lw=2, label=f'T={t}')
    # plt.xlim([np.min(xx), np.max(xx)])
    # plt.legend()
    # plt.show()
    #
    # # ---------------------- question 3
    # # load the images to use for classification
    # N = 1000
    # ims, labs = load_dogs_vs_frogs(N)
    # # define BLR model that should be used to fit the data
    # mdl = BayesianLinearRegression(sample_noise=0.001, kernel_function=poly_kernel(2))
    # # use Gibbs sampling to sample model and outliers
    # samp, outliers = outlier_regression(mdl, ims, labs, p_out=0.01, T=50, mu_o=0, sig_o=.5)
    # # plot the outliers
    # plot_ims(ims[outliers], title='outliers')

    # ------------------------------------------------------ section 3 - Bayesian GMM
    # ---------------------- question 5
    # load 2D GMM data
    k, N = 5, 1000
    X = gmm_data(N, k)

    for i in range(5):
        gmm = BayesianGMM(k=50, alpha=.01, mu_0=np.zeros(2), sig_0=.5, nu=5, beta=.5)
        gmm.gibbs_fit(X, T=100)

        # plot a histogram of the mixture probabilities (in descending order)
        pi = gmm._pi  # mixture probabilities from the fitted GMM
        plt.figure()
        plt.bar(np.arange(len(pi)), np.sort(pi)[::-1])
        plt.ylabel(r'$\pi_k$')
        plt.xlabel('cluster number')
        plt.title(f'histogram #{i}')
        plt.show()

        # plot the fitted 2D GMM
        plot_2D_gmm(X, gmm._mu_ks, gmm._sig_ks, gmm.cluster(X))  # the second input are the means and the third are the covariances

    # # ---------------------- questions 6-7
    # # load image data
    # MNIST, labs = load_MNIST()
    # # flatten the images
    # ims = MNIST.copy().reshape(MNIST.shape[0], -1)
    # gmm = BayesianGMM(k=500, alpha=1, mu_0=0.5 * np.ones(ims.shape[1]), sig_0=.1, nu=1, beta=.25, learn_cov=False)
    # gmm.gibbs_fit(ims, 100)
    #
    # # plot a histogram of the mixture probabilities (in descending order)
    # pi = None  # mixture probabilities from the fitted GMM
    # plt.figure()
    # plt.bar(np.arange(len(pi)), np.sort(pi)[::-1])
    # plt.ylabel(r'$\pi_k$')
    # plt.xlabel('cluster number')
    # plt.show()
    #
    # # find the clustering of the images to different Gaussians
    # cl = gmm.cluster(ims)
    # clusters = np.unique(cl)
    # print(f'{len(clusters)} clusters used')
    # # calculate the purity of each of the clusters
    # purities = np.array([cluster_purity(labs[cl == k]) for k in clusters])
    # purity_inds = np.argsort(purities)
    #
    # # plot 25 images from each of the clusters with the top 5 purities
    # for ind in purity_inds[-5:]:
    #     clust = clusters[ind]
    #     plot_ims(MNIST[cl == clust][:25].astype(float), f'cluster {clust}: purity={purities[ind]:.2f}')
    #
    # # plot 25 images from each of the clusters with the bottom 5 purities
    # for ind in purity_inds[:5]:
    #     clust = clusters[ind]
    #     plot_ims(MNIST[cl == clust][:25].astype(float), f'cluster {clust}: purity={purities[ind]:.2f}')
