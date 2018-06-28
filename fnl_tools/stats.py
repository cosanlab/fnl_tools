from __future__ import division
import numpy as np
import pandas as pd
from nltools.stats import pearson
from sklearn.metrics import pairwise_distances

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
            cluster_match: Average diagonal correlation
            cluster_unmatch: Average off diagonal correlation

    '''

    if group1.shape[0] != group2.shape[0]:
        # Can relax this eventually for columns
        raise ValueError('Make sure groups have same number of observations.')

    group1_selected = group1.copy()
    group2_new = {}
    for i in group2:
        r_vec = pd.Series(pearson(group2.iloc[:,i],group1_selected.T))
        idx = r_vec.idxmax()
        group2_new[idx] = group2.iloc[:,i].values.flatten()
        group1_selected.iloc[:,idx] = np.nan #block column from being rematched
    group2_new = pd.DataFrame(group2_new)
    group2_new = group2_new.reindex_axis(sorted(group2_new.columns), axis=1)
    group2_new.index = group2.index
    return group2_new

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
        group2 = align_clusters_groups(group1,group2)

    r = group1.T.append(group2.T).T.corr()
    n_clust = group1.shape[1]
    return (np.mean(np.diag(r,k=n_clust)),np.mean([np.mean(np.diag(r,k=x)) for x in range(1,n_clust*2) if x != n_clust]))
