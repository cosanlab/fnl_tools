from __future__ import division
'''
FNL-tools Utilities Tools
===========================

Utility functions to help analyses.
'''

__all__ = ['get_rect_coord',
           'rec_to_time',
           'create_hmm_labels',
           'clean',
           'expand_states',
           'parse_triangle',
           'load_dyad_df',
           'sort_srm',
           'align_srms',
           'grab_pairwise_dist', 
           'grab_subIDs'
           ]
__author__ = ["Luke Chang", "Jin Hyun Cheong"]
__license__ = "MIT"


import numpy as np
import pandas as pd
from sklearn.preprocessing import binarize 
from sklearn.metrics import pairwise_distances
import scipy
from scipy.stats import pearsonr
from nltools.data import Adjacency
from scipy.spatial.distance import pdist, squareform
import scipy.signal as signal
from scipy.signal import butter, filtfilt, freqz

def get_rect_coord(labels):
    '''
    This method takes a vector of labels as input and outputs
    a dictionary of the start and duration of a state.
    '''
    labels = np.array(labels)
    count_on = 0
    start = []; duration = []
    for state in set(labels):
        for i,x in enumerate(labels):
            if x == state:
                if count_on==0:
                    start.append(i)
                count_on = count_on + 1
            elif x != state:
                if count_on > 0:
                    duration.append(count_on)
                count_on = 0
            if i == len(labels)-1:
                if count_on > 0:
                    duration.append(count_on)
    return dict(zip(start,duration))

def rec_to_time(values, TR=2., fps=None):
    '''
    Returns time tick labels in mm:ss format.

    Args:
        values: Tick values.
        TR (default: 2.): TR is temporal resolution in seconds if used with fMRI data.
        fps (default: None): Frames per second
        **Either TR or fps must be specified. Specifying both will throw ValueError.
    Example:
        ax.set_xticks(range(0,2718,300))
        ax.set_xticklabels(rec_to_time(range(0,2718,300) ,fps=1))
    '''
    if TR and fps:
        raise ValueError("Must specify either TR or fps")
    if TR:
        fps = 1/TR

    times = np.array(values)/60./fps
    times = [str(int(np.floor(t))).zfill(2)+':'+str(int((t-np.floor(t))*60)).zfill(2) for t in times]
    return times

def create_hmm_labels(data, threshold=0.5):
    hmmdat = pd.DataFrame(binarize(data, threshold=0.5, copy=True))
    return pd.DataFrame([hmmdat[c[0]]*(i+1) for i,c in enumerate(hmmdat.iteritems())]).sum()

def clean(dm, replace_na=True):
    ''' This function removes nans and drops any duplicate columns
    '''
    if replace_na:
        dm.fillna(value=0, inplace=True)
    keep = []; remove = []
    for i, c in dm.iteritems():
        for j, c2 in dm.iteritems():
            if i != j:
                r = pearsonr(c,c2)[0]
                if (r > 0.99) & (j not in keep):
                    keep.append(i)
                    remove.append(j)
    dm.drop(remove, inplace=True, axis=1)
    return dm

def expand_states(data):
    ''' Given an input of vector labels from HMM, expand into separate variables'''
    out = {}
    for i in sorted(data.unique()):
        out[i] = data==i
    return pd.DataFrame(out)

def parse_triangle(df, condition='upper'):
    '''
    This function grabs the upper triangle of a correlation matrix
    by masking out the bottom triangle (tril) and returns the values.
    You can use scipy.spatial.distance.squareform to recreate matrix from upper triangle

    Args:
      df: pandas or numpy correlation matrix
      condition: 'upper': grabs the upper triangle
                 'pairs': grabs pair of subjects, skipping each diagonal
                 'nonpairs': grabs nonpairs, upper - pairs
    Returns:
      df: masked df
    '''
    try:
        assert(type(df)==np.ndarray)
    except:
        if type(df)==pd.DataFrame:
            df = df.values
        else:
            raise TypeError('Must be np.ndarray or pd.DataFrame')
    if condition =='upper':
        mask = np.triu_indices(df.shape[0], k=1)
        return df[mask]
    else:
        noDyads = int(df.shape[0]/2)
        if condition =='pairs':
            return np.diag(df,k=1)[range(0,noDyads*2,2)]
        elif condition =='nonpairs':
            mask = np.triu(np.ones(df.shape),k=1).astype(np.bool)
            ix, iy = np.arange(0,noDyads*2,2), np.arange(1,noDyads*2,2)
            for i in np.arange(0,len(ix)):
                mask[ix[i],iy[i]] = False
            return df[mask]
        else:
            raise ValueError('Condition,'+ str(condition) +' not recognized')

def load_dyad_df(dyad_file, dyads):
    '''
    This function reads the dyad data file and returns connection ratings and enjoyment ratings with the parsed dataframe.

    Args:
        dyad_file: File with dyad ratings (see example).
        dyads: list of dyads to analyze

    Returns:
        connection: dict with Episodes as keys and Average dyad connection as values in a series with dyad numbers
        connection_diff: dict with Episodes as keys and dyad connection absolute differences as values in a series with dyad numbers
        enjoy: dict with Episodes as keys and average enjoy ratings as values in a series with dyad numbers
        enjoy_diff: dict with Episodes as keys and enjoy ratings absolute differences as values in a series with dyad numbers
        dyads_dat: DataFrame with all ratings.

    Example:
        dyad_file = os.path.join('../../Data/Ratings/FilteredData/20171029_fnldyad_chartrans_dat.csv')

        all_dyads = set(np.arange(71,103,1))
        dyads_to_exclude = set([90, 91, 82, 86])
        dyads = list(all_dyads-dyads_to_exclude)

        connection,connection_diff, enjoy,enjoy_diff,dyads_dat = load_dyad_df(dyad_file, dyads = dyads)
    '''
    dyad_dat = pd.read_csv(dyad_file)
    dyad_dat['subID'] = ''
    for i in dyad_dat.index:
        dyad_dat.loc[i,'subID'] = 's' + str(dyad_dat.loc[i,'subject_ID'])+'_'+str(dyad_dat.loc[i,'id'])
    dyad_dat.index = dyad_dat['subID']
    noDyads = len(dyads)
    connection, connection_diff,connection_diff_list = {}, {},[]
    enjoy, enjoy_diff,enjoy_diff_list = {}, {},[]
    for epn in ['ep01','ep02','ep03','ep04']:
        connection[epn] = dyad_dat.groupby('subject_ID').mean()['con'+epn[-1]][dyads]
        diff_conn = np.abs(dyad_dat.groupby('subject_ID')['con'+epn[-1]].diff().dropna())
        diff_conn.index = [int(ix.split('_')[0][1:]) for ix in diff_conn.index]
        connection_diff[epn] = diff_conn[dyads]
        connection_diff_list.extend(connection_diff[epn])

        enjoy[epn] = dyad_dat.groupby('subject_ID').mean()['enjoy'+epn[-1]][dyads]
        diff_enjoy = -dyad_dat.groupby('subject_ID')['enjoy'+epn[-1]].diff().dropna()
        diff_enjoy.index = [int(ix.split('_')[0][1:]) for ix in diff_enjoy.index]
        enjoy_diff[epn] = diff_enjoy[dyads]
        enjoy_diff_list.extend(enjoy_diff[epn])
    return connection, connection_diff, enjoy, enjoy_diff,dyad_dat

def sort_srm(srm_data):
    '''
    This function sorts the Shared Responses by ISC and returns the sorted Shared Response

    Args:
        srm_data: List with (subject, k-features, time)

    Returns:
        srm_data_sorted: List with SRM sorted by max ISC (subject, k-features, time)

    Examples:
        # Compute SRM for training data
        srm_solo = brainiak.funcalign.srm.SRM(n_iter=n_iter, features=features, rand_seed=seed)
        srm_solo.fit(solo_train_data)

        # predict on left out data
        solo_test = srm_solo.transform(solo_test_data) # s subject by k feature by t time matrix

        # Sort Shared Responses based on high ISC to low ISC
        solo_test = sort_srm(solo_test)

        # compute average shared response
        solo_test_sr = np.mean(solo_test,axis=0)
    '''
    out = {'transformed': srm_data}
    n_features = np.shape(srm_data)[1]
    # For each Shared Response, calculate the Intersubject Correlation.
    # This is the temporal correlation for every pair of subjects in that group.
    a = Adjacency()
    for f in range(n_features):
        a = a.append(Adjacency(1-pairwise_distances(np.array([x[f,:] for x in out['transformed']]), metric='correlation'), metric='similarity'))
    out['isc'] = dict(zip(np.arange(n_features), a.mean(axis=1)))
    # Sort states from high to low ISC
    sorted_states =sorted(out['isc'],key=out['isc'].get, reverse=True)
    srm_data_sorted = [s[sorted_states] for s in srm_data]
    return srm_data_sorted

def align_srms(srm1, srm2):
    '''
    Align two SRM by matching the average shared responses.

    Args:
        srm1: first SRM object
        srm2: second SRM object

    Returns:
        srm2: SRM object in which the s_ and w_ are sorted.
        s2_sorted: sorted Shared Responses
        col_ind: re-organized columns to sort the srm2 columns
        max_corr: Maximum correlation from Hungarian Algorithm
    '''
    # Extract shared responses for each group and cast to DataFrame.
    s1 = pd.DataFrame(srm1.s_).T
    s1.columns = ['g1_'+str(col) for col in s1.columns]
    s2 = pd.DataFrame(srm2.s_).T
    s2.columns = ['g2_'+str(col) for col in s2.columns]

    n_features = len(s1.columns)
    assert n_features == len(s2.columns), "Number of features for both SRM must match."

    srms = pd.concat([s1, s2],axis=1)
    srms_corr = srms.corr().iloc[:n_features, n_features:] # grab the upper-right quadrant.
    # Match Shared Responses between groups using Hungarian Algorithm.
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(-1*srms_corr)
    # Grab max corr
    max_corr = np.round(np.mean(np.diag(srms_corr.iloc[row_ind,col_ind])),3)
    # Reorganize SRM2 based on sorted columns
    s2_sorted = s2.iloc[:,col_ind]
    # Reorganize data in the SRM object
    srm2.s_ = srm2.s_[col_ind]
    srm2.w_ = [s[:,col_ind] for s in srm2.w_]
    return srm2, s2_sorted, col_ind, max_corr

def grab_pairwise_dist(beh_dat, sublist, epn,dim = None,char=None,metric = 'euclidean',output='corr'):
    '''
    This function returns the pairwise distance using pdist of all subjects 
    
    output : 'corr':returns correlation values;
            'value': returns the values
    '''
    i = int(epn[-1])
    X = beh_dat[(beh_dat['episode']==i)].pivot_table(
        index='subject_id',
        columns=['dimension','character'],
        values='rating').loc[sublist]
    if dim:
        if char:
            if output=='corr':
                return pd.DataFrame(squareform(pdist(X[dim].swaplevel('dimension',
                                    'character',
                                    axis=1)[char],metric)))
            else:
                return X[dim].swaplevel('dimension','character',axis=1)[char]
        else:
            if output=='corr':
                return pd.DataFrame(squareform(pdist(X[dim], metric)))
            else:
                return X[dim]
    elif dim==None:
        if char:
            if output=='corr':
                return pd.DataFrame(squareform(pdist(X.swaplevel('dimension',
                                    'character',
                                    axis=1)[char],metric)))
            else:
                return X.swaplevel('dimension','character',axis=1)[char]
        else:
            if output=='corr':
                return pd.DataFrame(squareform(pdist(X, metric)))
            else:
                return X
            
# create a dataframe so you can make that figure with factorplot. also run lmer model. 
def grab_subIDs(df, k=0):
    """Extracts subject Ids from row indices and column names.
    
    Args:
        df: 
    """
    df_row = df.apply(lambda x: x.index)
    df_col = df_row.T
    df_row = np.hstack(df_row.mask(np.tril(np.ones(df_row.shape),k=k).astype(np.bool)).values.tolist())
    df_row = df_row[df_row!='nan']
    df_col = np.hstack(df_col.mask(np.tril(np.ones(df_col.shape),k=k).astype(np.bool)).values.tolist())
    df_col = df_col[df_col!='nan']
    return df_row, df_col

def butter_bandpass(lowcut, highcut, fs, order=5):
    """Create a Butterworth bandpass filter

    https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
    https://stackoverflow.com/questions/25191620/creating-lowpass-filter-in-scipy-understanding-methods-and-units

    Args:
        lowcut (float): Low end frequency
        highcut (float): High end frequency
        fs (float): Frequency
        order (int, optional): Order of Butterworth. Defaults to 5.

    Returns:
        b, a
    """    
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Apply function for Butterworth Bandpass filter

    Args:
        data ([type]): [description]
        lowcut ([type]): [description]
        highcut ([type]): [description]
        fs ([type]): [description]
        order (int, optional): [description]. Defaults to 5.
    """    
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = data.apply(lambda x: filtfilt(b,a, x))
#     y = filtfilt(b, a, data)
    return y