from __future__ import division
import numpy as np
import pandas as pd
from nltools.stats import pearson

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


def sort_hmm_states(data):
    '''This function attempts to reorder HMM states by correlation similarity'''
    data = data.copy()
    sub_idx = np.array([x.split('_')[0] for x in data.columns])
    sub_list = list(set(sub_idx))
    new_order = []
    for s in sub_list:
        sub_dat = data.loc[:,sub_idx==s]
        other_dat = data.loc[:,sub_idx!=s]
        other_dat = other_dat.loc[:,~other_dat.columns.isin(new_order)]
        for x in sub_dat.iteritems():
            if x[0] not in new_order:
                new_order.append(x[0])
                counter = 1
                if other_dat.shape[1] > 0:
                    while counter < len(sub_list):
                        r = pearson(x[1],other_dat.T)
                        next_sub = other_dat.columns[r==r.max()][0]
                        new_order.append(next_sub)
                        counter = counter + 1
                        other_dat = other_dat.loc[:,other_dat.columns != next_sub]
    return data.loc[:,new_order]
