from __future__ import division
import numpy as np
import pandas as pd
from .stats import autocorrelation
from numpy.random import dirichlet as Dir
from numpy.random import poisson as Poisson

def covariance_matrix(n_variables, sigma, mu_r, sigma_r):
    '''Simulate Covariance matrix'''
    cov = np.random.randn(n_variables, n_variables) * sigma_r + mu_r
    cov[np.diag_indices(n_variables)] = sigma
    return cov


def covariance_matrix_ar(n_variables, sigma, mu_r, sigma_r, alpha, cov):
    '''Calcuate multivariate covariance matrix with autocorrelation'''
    if len(alpha) != len(cov):
        raise ValueError('alpha must have same number of elements as cov.')

    cached_cov = np.zeros((n_variables, n_variables))
    for lag, ar in enumerate(alpha):
        cached_cov += ar*cov[-1*(lag+1)]
    cached_cov = cached_cov + np.random.randn(n_variables, n_variables) * sigma_r + mu_r
    cached_cov[np.diag_indices(n_variables)] = sigma
    return cached_cov

def covariance_matrix_ar1(n_variables, sigma, mu_r, sigma_r, alpha, cov):
    '''Simulate Covariance matrix with AR1'''
    cov_t1 = deepcopy(cov)
    cov_t = alpha*cov_t1 + np.random.randn(n_variables, n_variables) * sigma_r + mu_r
    cov_t[np.diag_indices(n_variables)] = sigma
    return cov_t

def simulate_time_series(n_tr=1000, n_variables=2, mu_r=0.2, sigma_r=0.1,
                         alpha_time=0.8, alpha_cov=0.5, mu=0, sigma = 1):
    ''' Simulate a time by feature timeseries with dynamic fluctuating connectivity
        and AR1 on dynamic connectivity and timeseries intensity. Covariance is drawn from normal distribution N(mu_r, sigma_r)

        Based on:
        Thompson WH, Richter CG, Plavén-Sigray P, Fransson P (2018) Simulations to benchmark
        time-varying connectivity methods for fMRI. PLOS Computational Biology 14(5): e1006196.
        https://doi.org/10.1371/journal.pcbi.1006196

        Args:
            n_tr: Number of timepoints (default=1000)
            n_variables: Number of variables to simulate (default=2)
            mu_r: Mean of distribution to sample covariance (default=0.2)
            sigma_r: SD of distribution to sample covariance (default=0.1)
            alpha_time: Autocorrelation on timeseries (default = 0.8)
            alpha_cov: Autocorrelation on connectivity (default = 0.5)
            mu: Timeseries mean. Can be a single value for all variables or a list of len(n_variables) (default: 0)
            sigma: Timeseries variance Can be a single value for all variables or a list of len(n_variables) (default: 1)

        Returns:
            timeseries: time by features matrix
            covariance: feature by feature by time covariance matrix
        '''

    if isinstance(mu, (float, int)):
        mu = mu * np.ones(n_variables)
    else:
        if n_variables != len(mu):
            raise ValueError('n_variables must be same length as mu.')
        mu = np.array(mu)

    if isinstance(sigma, (float, int)):
        sigma = sigma * np.ones(n_variables)
    else:
        if n_variables != len(sigma):
            raise ValueError('n_variables must be same length as sigma.')

    ts = pd.DataFrame()
    all_cov = []
    for t in range(n_tr):
        if t < 1:
            cov = np.zeros((n_variables, n_variables))
            ts_1 = np.zeros(n_variables)
        else:
            ts_1 = ts.iloc[t-1, :]
            cov = covariance_matrix_ar1(n_variables, sigma, mu_r, sigma_r, alpha_cov, cov)
        ts = ts.append(mu + alpha_time*ts_1 + pd.Series(np.random.multivariate_normal(np.zeros(n_variables), cov, 1).squeeze()), ignore_index=True)
        all_cov.append(cov)
    return (ts, all_cov)

def gaussian_random_walk(n_tr, mn, sd, starting_value=None):
    if starting_value is None:
        return np.cumsum((np.random.randn(n_tr) * sd) + mn)
    elif starting_value == 'random':
        return np.cumsum((np.random.randn(n_tr) * sd) + mn) + np.random.randint(low=-1*n_tr/sd,high=n_tr/sd)
    elif isinstance(starting_value, int):
        return np.cumsum((np.random.randn(n_tr) * sd) + mn) + starting_value

def sim_grw(n_walks=10, n_tr=1364, rw_mn=0, rw_sd=1, starting_value=None):
    return pd.DataFrame([gaussian_random_walk(n_tr, rw_mn, rw_sd, starting_value=starting_value) for _ in range(n_walks)]).T

def gaussian_random_walk_2sd(n_tr=100, mn=0, sd1=1, sd2=5, p_switch=.7):
    rw = []
    for t in range(n_tr):
        if np.random.random() < p_switch:
            rw.append((np.random.randn() * sd1) + mn)
        else:
            rw.append((np.random.randn() * sd2) + mn)
    wt_mn = np.sum([(1-p_switch)*sd1,p_switch*sd2])/2
    return np.cumsum(rw) + np.random.randint(low=-1*n_tr/wt_mn,high=n_tr/wt_mn)

def sim_grw_2sd(n_walks=10, n_tr=1364, mn=0, sd1=1, sd2=5, p_switch=.5):
    return pd.DataFrame([gaussian_random_walk_2sd(n_tr=n_tr, mn=mn, sd1=sd1, sd2=sd2, p_switch=p_switch) for _ in range(n_walks)]).T

def DirichletEventTimeseries(alpha, T, D, N, L, beta):
    '''Simulate Discrete number of events
        Written by Jeremy Manning for Baldassano Neuron paper
    Args:
        alpha: Topic sparsity
        beta: Topic transition sharpness (higher = sharper)
        T: Number of timepoints
        D: Number of dimensions
        N: Number of hinge events
        L: Poisson parameter; higher numbers will lead to more variable segment lengths

    Returns:
        Y: Time by Features matrix
    '''

    def get_lengths(T, N, L):
#         segment_lengths = np.round(np.divide(Poisson(L, [N-1]), np.sum(Poisson(L, [N-1]))/T)).astype(int)
        segment_lengths = Poisson(L, [N-1])
        segment_lengths = np.round(np.divide(segment_lengths, np.sum(segment_lengths)/T)).astype(int)
        all_scaffold_events = np.arange(N)
        while (np.sum(segment_lengths) > T) and np.any(segment_lengths > 1):
            next_event = np.random.choice(all_scaffold_events[np.where(segment_lengths > 1)])
            segment_lengths[next_event] -= 1

        while np.sum(segment_lengths) < T:
            next_event = np.random.choice(all_scaffold_events)
            segment_lengths[next_event] += 1

        assert np.sum(segment_lengths) == T, 'failed to generate segments of the correct lengths'
        return segment_lengths

    def fade_between(a, b, n, beta): #fade from a to be in n steps, excluding a
        mu = np.power(np.random.uniform(size=[n]), beta)
        mu = np.sort(np.divide(mu, np.max(mu)))
        #mu = np.linspace(0, 1, n+1)[1:]
        Y = np.zeros([n, len(a)])
        for i in np.arange(n):
            Y[i, :] = np.multiply(mu[i], b) + np.multiply(1-mu[i], a)
        return Y

    def dirichlet_noise(x, alpha):
        a = x + alpha
        return Dir(np.divide(a, np.sum(a)))

    scaffold_events = np.zeros((N, D))
    for i in np.arange(N):
        scaffold_events[i, :] = Dir((alpha*np.ones((1, D))).tolist()[0])

    segment_lengths = get_lengths(T, N, L)
    print(f'segment_lengths: {segment_lengths}')

    Y = np.zeros((T, D))
    for i in np.arange(N-1):
        start_ind = np.sum(segment_lengths[:i]) + 1
        end_ind = start_ind + segment_lengths[i] - 1
        Y[start_ind-1, :] = scaffold_events[i, :]
        Y[start_ind:end_ind, :] = fade_between(scaffold_events[i, :], scaffold_events[i+1, :], segment_lengths[i] - 1, beta)
