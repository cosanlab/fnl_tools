from __future__ import division
import numpy as np
import pandas as pd

def get_rect_coord(labels):
    '''
    This method takes a vector of labels as input and outputs
    a dictionary of the start and duration of a state.
    '''
    labels = np.array(labels)
    count_on = 0
    start = []; duration = [];
    for i,x in enumerate(labels):
        if x:
            if count_on==0:
                start.append(i)
            count_on = count_on + 1
        if ~x:
            if count_on > 0:
                duration.append(count_on)
            count_on = 0
        if i==len(labels)-1:
            if count_on > 0:
                duration.append(count_on)
    return dict(zip(start,duration))

def rec_to_time(values, TR=2.):
    times = np.array(values)/60.*TR
    return [str(int(np.floor(t)))+':'+str(int((t-np.floor(t))*60)) for t in times]

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
