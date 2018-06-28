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
    start = []
    duration = []
    for i,x in enumerate(labels):
        if x:
            if count_on==0:
                start.append(i)
            count_on = count_on + 1
        if ~x:
            if count_on > 0:
                duration.append(count_on)
            count_on = 0
    return dict(zip(start,duration))

def rec_to_time(values, TR=2.):
    times = np.array(values)/60.*TR
    return [str(int(np.floor(t)))+':'+str(int((t-np.floor(t))*60)) for t in times]
