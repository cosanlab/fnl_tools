from __future__ import division
'''
FNL-tools Statistics Tools
===========================

Tools to help with statistical analyses.
'''

__all__ = ['calc_fft',
           'sort_subject_clusters',
           'validity_index',
           'align_clusters_groups',
           'group_cluster_consensus',
           'delay_coord_embedding',
           'calc_spatial_temporal_correlation',
           'autocorrelation',
           'extract_max_timeseries',
           'bic',
           'PCA'
           ]
__author__ = ["Luke Chang", "Jin Hyun Cheong"]
__license__ = "MIT"


import numpy as np
import os
import glob
from copy import deepcopy
import pandas as pd
from nltools.data import Adjacency
from nltools.stats import pearson, align_states
from sklearn.metrics import pairwise_distances
from scipy.linalg import eigh
from scipy.optimize import curve_fit
from nltools.data import Adjacency

def calc_fft(signal, Fs):
    ''' Calculate FFT of signal

    Args:
        signal: time series
        Fs: sampling frequency

    Returns:
        frequency: frequency in Hz
        power spectrum: power spectrum

    '''
    fourier = np.abs(np.fft.fft(signal))**2
    n = signal.size
    timestep = 1/Fs
    freq = np.fft.fftfreq(n, d=timestep)
    return (freq[freq>0],fourier[freq>0])

def sort_subject_clusters(data, subject_col=None, cluster_col=None,
                        n_iterations=2):
    '''This function sorts individual subject clusters so that they are
        consistent across everyone

        Args:
            target: (pd.DataFrame) clusters x features target data to sort
                    must include subject
            reference: (pd.DataFrame) clusters x features to use as a
                    reference for sorting
            subject_col: (str) Variable name of target that indicates Grouping
                    variable (e.g., 'Subject')
            cluster_col: (str) Variable name of target that indicates Cluster
                    variable (e.g., 'Cluster)
            n_iterations: (int) number of iterations to re-reference to mean

        Returns:
            sorted_data: (pd.DataFrame) sorted target dataframe with Clusters
                    relabeled
    '''

    def create_index(data):
        '''This function creates a unique index from Subject and Cluster
        indicators
        '''
        sub_idx = []
        for i in data.iterrows():
            sub_idx.append('%s_%s' % (i[1][subject_col],i[1][cluster_col]))
        return sub_idx

    def sort_target_to_reference(target, reference, subject_col=subject_col,
                                cluster_col=cluster_col):
        '''This function sorts the target clusters to the reference clusters
        '''
        sub_list = target[subject_col].unique()
        cluster_list = target[cluster_col].unique()
        new_order = []
        for s in sub_list:
            sub_dat = target.loc[target[subject_col] == s,:].drop([subject_col,
                                                        cluster_col], axis=1)
            for i in reference.iterrows():
                r = pearson(i[1],sub_dat)
                next_sub = sub_dat.index[r == r.max()][0]
                new_order.append(next_sub)
                sub_dat = sub_dat.loc[sub_dat.index != next_sub,:]
        sorted_data = target.loc[new_order,:]
        sorted_data[cluster_col] = np.array([cluster_list]*len(sub_list)).flatten()
        sorted_data.index = create_index(sorted_data)
        assert target.shape[0] == len(set(sorted_data.index))
        return sorted_data

    data = data.copy()
    sub_list = data[subject_col].unique()
    data.index = create_index(data)

    # First, sort clusters based on first subject's clusters
    ref = np.random.choice(sub_list) # Randomly select a reference subject
    ref_dat = data.loc[data[subject_col] == ref,:].drop([subject_col,
                                                    cluster_col],axis=1)
    sorted_data = sort_target_to_reference(data,ref_dat,
                            subject_col=subject_col,cluster_col=cluster_col)

    # Second, iterate through all subject and use mean of previous iteration
    # as reference
    for i in range(n_iterations):
        mean_cluster = sorted_data.drop(subject_col,
                                        axis=1).groupby(cluster_col).mean()
        sorted_data = sort_target_to_reference(sorted_data,mean_cluster,
                                                subject_col=subject_col,
                                                cluster_col=cluster_col)
    return sorted_data

def validity_index(data, subject_col='Subject', cluster_col='Cluster',
                    metric='correlation'):
    '''Calculate cluster validity index.  average normalized between-within
    cluster distance across clusters
        Args:
            data: (pd.DataFrame) clusters x features target data to sort
                    must include subject
            subject_col: (str) Variable name of target that indicates Grouping
                    variable (e.g., 'Subject')
            cluster_col: (str) Variable name of target that indicates Cluster
                    variable (e.g., 'Cluster)
            metric: (str) type of distance metric to use
        Returns:
            vi: (float) validity index
    '''
    data = data.copy()
    sub_list = data[subject_col].unique()
    clust_list = data[cluster_col].unique()

    c = clust_list[0]
    vi_all = []
    for c in clust_list:
        c_dat = data.loc[data[cluster_col] == c,:].drop([subject_col,
                                                        cluster_col],axis=1)
        other_dat = data.loc[data[cluster_col] != c,:].drop([subject_col,
                                                        cluster_col],axis=1)
        within_dist = pairwise_distances(c_dat,metric=metric)
        within_dist_mn = np.mean(
                        within_dist[np.triu_indices(within_dist.shape[0],k=1)])
        between_dist = pairwise_distances(pd.concat([c_dat,other_dat],axis=0))
        between_dist = between_dist[c_dat.shape[0]:,c_dat.shape[0]:]
        between_dist_mn = np.mean(
                    between_dist[np.triu_indices(between_dist.shape[0],k=1)])
        c_vi = ((between_dist_mn-within_dist_mn)/
                    (np.maximum(between_dist_mn,within_dist_mn)))
        vi_all.append(c_vi)
    vi = np.mean(vi_all)
    return vi


def align_clusters_groups(group1, group2):
    '''Reorder group2 columns to match group1 (expects time x cluster mean)

        Args:
            group1: data x clusters dataframe for group1
            group2: data x clusters dataframe for group2 (can have less features than group1)

        Returns:
            aligned group2: Pandas DataFrame of Aligned group2
            aligned keys: Dictionary of column remapping {orig_column: new_column}
    '''

    if group1.shape[0] != group2.shape[0]:
        # Can relax this eventually for columns
        raise ValueError('Make sure groups have same number of observations.')

    group1_selected = group1.copy()
    group2_new = {}
    for i in range(group2.shape[1]):
        r_vec = pd.Series(pearson(group2.iloc[:,i],group1_selected.T))
        idx = r_vec.idxmax()
        group2_new[idx] = group2.iloc[:,i].values.flatten()
        group1_selected.iloc[:,idx] = np.nan #block column from being rematched
    group2_new = pd.DataFrame(group2_new)
    remapped_columns = {x:group2_new.columns[x] for x in range(group2_new.shape[1])}
    group2_new = group2_new.reindex(sorted(group2_new.columns), axis=1)
    group2_new.index = group2.index
    return (group2_new, remapped_columns)
 
def group_cluster_consensus(group1, group2, align=False):
    '''Calculate cluster average reliability of clusters

        Args:
            group1: data x clusters dataframe for group1
            group2: data x clusters dataframe for group1
            align: (bool) align group2 to group1 if this hasn't already been done.

        Returns:
            cluster_match: Average diagonal correlation
            cluster_unmatch: Average off diagonal correlation

    '''

    if group1.shape != group2.shape:
        raise ValueError('Make sure groups are same size.')

    if align:
        group2 = align_clusters_groups(group1, group2)

    r = group1.T.append(group2.T).T.corr()
    n_clust = group1.shape[1]
    return (np.mean(np.diag(r,k=n_clust)),np.mean([np.mean(np.diag(r,k=x)) for x in range(1,n_clust*2) if x != n_clust]))

def delay_coord_embedding(data, delay=4, dimensions=2):
    out = deepcopy(data)
    for d in range(dimensions-1):
        out = np.vstack([out,np.concatenate([data[delay:],[np.nan]*delay])])
    out = out.T
    return out[:-delay*(dimensions-1)]

def exponential_func(x, a, b, c):
    return a * np.exp(-b * x) + c

def exponential_func_2param(x, b, c):
    return b*np.exp(-b * x) + c

def exponential_func_1param(x, b):
    return b*np.exp(-b * x)

def calc_spatial_temporal_correlation(roi, base_dir, data_dir='/Volumes/Manifesto/Data/fnl/preprocessed'):
    # Load Time series for each subject
    file_list = glob.glob(os.path.join(data_dir,'roi_denoised','*','*CSF*ROI%s.csv' % (roi)))
    sub_id = [x.split('/')[-2].split('-')[1] for x in file_list]
    data = []
    for f in file_list:
        data.append(pd.read_csv(f))

    # Create adjacency matrix
    d = Adjacency()
    for s in data:
        d = d.append(Adjacency(1-pairwise_distances(s, metric='correlation'), metric='similarity'))

    # Calculate spatial autocorrelation
    autocorr = []
    for s in d:
        sub_autocorr = []
        for x in range(1,50):
            sub_autocorr.append(np.diag(s.squareform(),x).mean())
        autocorr.append(sub_autocorr)
    autocorr = pd.DataFrame(autocorr).T
    autocorr['Lag'] = autocorr.index
    autocorr_long = autocorr.melt(id_vars='Lag',var_name='Subject',value_name='Correlation')
    autocorr_long.to_csv(os.path.join(base_dir, 'Analyses','Spatiotemporal_Autocorrelation','TV_Study_Autocorrelation_ROI%s.csv' % roi))

    # Fit curve
    params={}
    for sub in autocorr_long['Subject'].unique():
        r = autocorr_long.loc[autocorr_long['Subject']==sub,:]
        popt, pcov = curve_fit(exponential_func, r['Lag'], r['Correlation'])
        params[sub] = np.concatenate([popt,np.sqrt(np.diag(pcov))])
    params = pd.DataFrame(params,index=['a','b','c','a_sd','b_sd','c_sd']).T
    params['Subject'] = params.index
    params['ROI'] = roi
    return params

def mean_weighted_by_inverse_variance(effect_size, std):
    return np.sum(effect_size*(std**2))/(np.sum(std**2))

def autocorrelation(data, delay=30):
    out = deepcopy(data)
    for d in range(1,delay+1):
        out = np.vstack([out,np.concatenate([data[d:],[np.nan]*d])])
    out = out.T
    out = out[:-delay]
    r = np.corrcoef(out.T)
    autocorr = []
    for d in range(1,delay+1):
        autocorr.append(np.mean(np.diag(r,k=d)))
    return np.array(autocorr)

def extract_max_timeseries(k, roi, analysis, base_dir):
    within_mean = pd.read_csv(os.path.join(base_dir, 'Analyses', analysis, 'HMM_WithinPatternSimilarity_k%s_ROI%s.csv' % (k,roi)),index_col=0,header=None)
    max_state = within_mean.iloc[:,0].idxmax()
    sorted_data = pd.read_csv(os.path.join(base_dir, 'Analyses', analysis, 'HMM_AlignedTimeSeries_k%s_ROI%s.csv' % (k,roi)),index_col=0)
    return sorted_data.loc[sorted_data['Cluster']==max_state,:].drop(['Cluster','Subject'],axis=1).mean()

def bic(ll,k,n):
    ''' Calculate BIC

    Args:
        ll: log-likelihood
        k: number of states
        n: number of observations
    Returns:
        bic: bayesian information criterion

    '''
    return (np.log(n)*k - 2*ll)

# def create_avg_concordance(k, roi, analysis, base_dir):
#     '''Create average concordance for a specific HMM state'''
#     sorted_data = pd.read_csv(os.path.join(base_dir, 'Analyses', analysis, 'HMM_AlignedTimeSeries_k%s_ROI%s.csv' % (k,roi)),index_col=0)
#     temporal_mn = {}
#     for i in range(k):
#         temporal_mn[i] = sorted_data.loc[sorted_data['Cluster']==i,:].drop(['Cluster','Subject'],axis=1).mean()
#     return pd.DataFrame(temporal_mn)


def create_average_concordance(data, values='Viterbi'):
    states = data.pivot(columns='Subject', values=values)
    n_states = len(np.unique(states.dropna()))
    concordance = {}
    for i in range(n_states):
        concordance[f'State_{i}'] = (states == i).mean(axis=1)
    return pd.DataFrame(concordance)

def calculate_r_square(Y, X, betas):
    '''Calculate r^2 of regression model based on Gelman & Hill pg 41'''
    sigma_hat = np.sqrt(np.sum(((Y - np.dot(X, betas))**2)/(len(Y) - X.shape[1])))
    return 1 - ((sigma_hat**2)/(np.std(Y)**2))

class PCA(object):
    '''
    Compute PCA on Correlation Matrix

    Args:
        data: n x n correlation matrix
        n_components: number of components to return

    '''
    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X):
        '''Fit PCA Model'''
        # calculate eigenvectors & eigenvalues of the covariance matrix
        # use 'eigh' rather than 'eig' since R is symmetric,
        # the performance gain is substantial
        X = np.array(X, dtype=np.float64)
        evals, evecs = eigh(X)

        # sort eigenvalue in decreasing order
        idx = np.argsort(evals)[::-1]
        self.evals = evals
        self.idx = idx
        self.explained_variance_ = evals[idx]
        self.explained_variance_ratio_ = self.explained_variance_/np.sum(evals)
        self.components_ = evecs[:,idx]
        if self.n_components is not None:
            self.explained_variance_ = self.explained_variance_[:self.n_components]
            self.explained_variance_ratio_ = self.explained_variance_ratio_[:self.n_components]
            self.components_ = self.components_[:, :self.n_components]

    def transform(self, X):
        '''Apply PCA model to X'''
        return np.dot(self.components_.T, X.T).T

    def fit_transform(self, X):
        '''Fit PCA Model and apply it to X'''
        self.fit(X)
        return self.transform(X)

def cluster_consensus(weights, align=True, metric='correlation', consensus_metric='within_between', 
                      cluster_metric='mean', verbose=True):
    '''Get the overall pattern consensus across measurements
    
    Args:
        weights: (list) list of weight matrices (state x pattern)
        align: (bool) align states if True, otherwise assume they are already aligned
        metric: (str) distance metric to use
        consensus_metric: (str) method to summarize cluster metrics. Use mean, median, 
                          or max within_between when calculating consensus over clusters.                
        cluster_metric: (str) use mean or median when computing within cluster similarity.

    Returns:
        mean: (float) average cluster consensus across states
    
    '''
    
    if consensus_metric not in ['max', 'mean', 'median', 'within_between']:
        raise ValueError("consensus_metric must be ['mean', 'median', 'max', 'within_between']")

    if cluster_metric not in ['mean', 'median']:
        raise ValueError("cluster_metric must be ['mean', 'median']")   

    if align:
        reference = weights[np.random.choice(range(len(weights)))].T

        ordered_weights = []
        for i,w in enumerate(weights):
            try:
                ordered_weights.append(align_states(reference, w.T, metric=metric, return_index=False, replace_zero_variance=True))
            except:
                if verbose:
                    print(f'Weight{i+1} not converging with hungarian algorithm')
                warnings.warn('Nans in alignment')
    else:
        ordered_weights = weights.copy()

    # Plot reordered spatial similarity
    k = weights[0].shape[0]
    n = len(ordered_weights)
    clusters = list(np.hstack([np.ones(n).astype(int)*x for x in range(k)]))

    rearranged = []
    for i in range(k):
        for w in ordered_weights:
            rearranged.append(w[:,i])
    state_similarity = Adjacency(1 - pairwise_distances(np.vstack(rearranged), metric=metric), matrix_type='similarity')

    state_mean = state_similarity.cluster_summary(clusters=clusters, metric=cluster_metric, summary='within')

    if consensus_metric == 'mean':
        consensus = np.mean(list(state_mean.values()))
    elif consensus_metric == 'median':
        consensus = np.median(list(state_mean.values()))
    elif consensus_metric == 'max':
        consensus = np.max(list(state_mean.values()))
    elif consensus_metric == 'within_between':
        consensus = np.mean(list(state_similarity.cluster_summary(clusters=clusters, metric=cluster_metric, summary='within').values())) - np.mean(list(state_similarity.cluster_summary(clusters=clusters, metric=cluster_metric, summary='between').values()))

    return consensus

def bootstrap_consensus(weights, n_bootstraps=10, align=True, consensus_metric='mean', verbose=False):
    '''Bootstrap consensus metric by resampling weights with replacement'''
    
    consensus_mean_boot = {}
    for b in range(n_bootstraps):
        bootstrap_id = np.random.choice(range(len(weights)), size=len(weights), replace=True)
        bootstrap_weights = [weights[x] for x in bootstrap_id]
        consensus_mean_boot[b] = cluster_consensus(bootstrap_weights, align=align, consensus_metric=consensus_metric, verbose=verbose)   
    return consensus_mean_boot

def hmm_bic(LL, n_states, n_observations=1364):
    k = 2*n_states + (n_states*(n_states - 1))
    return  k * np.log(n_observations) - 2*LL

def min_subarray(data):
    best_diff = 0
    current_diff = 0
    cumulative_diff = []
    for x in data:
        current_diff = current_diff - x
        best_diff = min(best_diff, current_diff)
        cumulative_diff.append(current_diff)
    return np.where(np.array(cumulative_diff) == best_diff)[0][0] + 1

def max_subarray(data):
    """Find a contiguous subarray with the largest sum."""
    best_sum = 0  # or: float('-inf')
    best_start = best_end = 0  # or: None
    current_sum = 0
    for current_end, x in enumerate(data):
        if current_sum <= 0:
            # Start a new sequence at the current element
            current_start = current_end
            current_sum = x
        else:
            # Extend the existing sequence with the current element
            current_sum += x

        if current_sum > best_sum:
            best_sum = current_sum
            best_start = current_start
            best_end = current_end + 1  # the +1 is to make 'best_end' exclusive

    return best_start, best_end

def compute_ISC_all_roi(data_dir, mask_x, episode = 'ep01'):
    '''Run ISC across each ROI for a given study'''

    roi_isc = {}
    for roi in tqdm(range(len(mask_x))):
        file_list = glob.glob(os.path.join(data_dir, f'*{episode}*ROI{roi}.csv'))
        mn = {}
        for f in file_list:
            sub = os.path.basename(f).split('_')[0]
            dat = pd.read_csv(f)
            mn[sub] = dat.T.mean()
        mn = pd.DataFrame(mn)
        roi_isc[roi] = isc(mn)

    r = pd.Series({x:roi_isc[x]['isc'] for x in roi_isc})
    p = pd.Series({x:roi_isc[x]['p'] for x in roi_isc})
    r_brain = roi_to_brain(r, mask_x=mask_x)
    p_brain = roi_to_brain(p, mask_x=mask_x)
    return (r_brain, p_brain)

def calc_r_square(Y, predicted_y):
    SS_total = np.sum((Y - np.mean(Y))**2)
    SS_residual = np.sum((Y - predicted_y)**2)
    return 1-(SS_residual/SS_total)

def exponential_func(x, a, b, c):
    return a * np.exp(-b * x) + c

def calculate_spatial_temporal_correlation(file_list, n_lags=50, target_var=0.9):
    auto_corr = {}
    for f in file_list:
        sub = os.path.basename(f).split('_')[0]
        dat = center(pd.read_csv(f))

        pca = PCA(n_components=target_var)
        reduced = pca.fit_transform(dat)
        reduced = pd.DataFrame(reduced)

        sim = Adjacency(1 - pairwise_distances(reduced, metric='correlation'), matrix_type='similarity')
        auto_corr[sub] = [np.diag(sim.squareform(), x).mean() for x in range(1, n_lags)]
    auto_corr = pd.DataFrame(auto_corr)
    auto_corr['Lag'] = auto_corr.index
    return auto_corr

def fit_exponential_function(auto_corr, maxfev=1000):
    params={}
    for x in auto_corr.drop(columns='Lag'):
        try:
            popt, pcov = curve_fit(exponential_func, auto_corr['Lag'], auto_corr[x], maxfev=maxfev)
            params[x] = np.concatenate([popt, np.sqrt(np.diag(pcov))])
        except:
            params[x] = np.repeat(np.nan, 6)
    params = pd.DataFrame(params,index=['a','b','c','a_sd','b_sd','c_sd']).T
    params['Subject'] = params.index
    return params

def time_to_correlation(params, correlation=.1, lags=50, id_column='ROI'):
    lag = np.arange(0,lags,1)
    out = {}
    for roi in params[id_column].unique():
        pred = exponential_func(lag, params.query(f'{id_column}==@roi')['a'].values[0], params.query(f'{id_column}==@roi')['b'].values[0],params.query(f'{id_column}==@roi')['c'].values[0])
        out[roi] = np.sum(pred>.1)
    return pd.Series(out)

def autocorrelation(data, delay=50):
    data = np.array(data)
    out = deepcopy(data)
    for d in range(1, delay + 1):
        out = np.vstack([out, np.concatenate([data[d:], [np.nan]*d])])
    out = out.T
    out = out[:-delay]
    r = np.corrcoef(out.T)
    autocorr = []
    for d in range(1, delay + 1):
        autocorr.append(np.mean(np.diag(r, k=d)))
    return np.array(autocorr)

def mean_weighted_by_inverse_variance(effect_size, std):
    return np.sum(effect_size*(std**2))/(np.sum(std**2)) 