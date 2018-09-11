from __future__ import division
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from fnl_tools.utils import rec_to_time, get_rect_coord
from sklearn.preprocessing import MinMaxScaler

def plot_recurrence(data, labels=None, file_name=None,
                    color = None,
                    title=None, tr=2., cmap=None, vmin=-1, vmax=1):
    '''
    This function plots a recurrence plot.
    Can optionally color the states with a vector of unique labels.
    '''
    plt.figure(figsize=(10,10))
    plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax = plt.gca()

    if labels is not None:
        states = list(set(labels))
        if color is None:
            color = sns.color_palette("hls", len(states))
        else:
            if len(color) != len(states):
                raise ValueError('Make sure list of colors matches length of unique states')
        for i,s in enumerate(states):
            for start,duration in get_rect_coord(labels==s).items():
                rect = patches.Rectangle((start,start), duration, duration, linewidth=2, edgecolor=color[i], facecolor='none')
                ax.add_patch(rect)

    ax.set_xticks(range(0,data.shape[0],50))
    ax.set_xticklabels(rec_to_time(ax.get_xticks(), TR=tr),rotation=60)
    ax.set_yticks(range(0,data.shape[0],50))
    ax.set_yticklabels(rec_to_time(ax.get_yticks(), TR=tr))

    if title is not None:
        plt.title(title)
    if file_name is not None:
        plt.savefig(file_name)

def plot_raster(data, groupby=None, line_width=3, color='skyblue', file_name=None, tr=2.,xticklabels=None):
    '''
    Plot a heatmap raster plot with histogram above.

    Args:
        data: data by time pandas object.
        groupby: column to organize rows and plot separately
        tr: repetition time for plotting minutes

    Returns:
        fig: figure handle
        axes: axes handle

    '''
    data = data.copy()
    grid = dict(height_ratios=[data.shape[0]/5,data.shape[0]], width_ratios=[data.shape[1],data.shape[1]/10 ])
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(20,10),gridspec_kw = grid)
    if groupby is not None:
        data.sort_values(groupby,inplace=True)
        group_idx = data[groupby]
        groups = data[groupby].unique()
        data.drop(groupby, axis=1, inplace=True)
        im = axes[1,1].imshow(np.array(group_idx, ndmin=2).T, aspect="auto")
        color = im.cmap(im.norm(groups))
        for i,x in enumerate(groups):
            data.loc[group_idx==x,:].mean().plot(kind='line',ax=axes[0,0], color=tuple(color[i]), linewidth=line_width)
    else:
        data.mean().plot(kind='line',ax=axes[0,0], color=color, linewidth=line_width)
    sns.heatmap(data,ax=axes[1,0], cbar=False, xticklabels=False)
    axes[0,0].margins(x=0)
    axes[0,0].axis("off")
    axes[1,1].axis("off")
    axes[0,1].axis("off")
    if xticklabels is not None:
        sns.heatmap(data,ax=axes[1,0], cbar=False, xticklabels=xticklabels)
    else:
        axes[0,0].set_xticks(range(data.columns.min(),data.columns.max(),200))
        axes[1,0].set_xticks(range(data.columns.min(),data.columns.max(),50))
        axes[1,0].set_xticklabels(rec_to_time(range(data.columns.min(),data.columns.max(),50),TR=tr),rotation=0,fontsize=14)

    plt.tight_layout()
    if file_name is not None:
        plt.savefig(file_name)
    return (fig, axes)

def plot_wavelet(data, n_bins=50, n_decimal=4, title=None, file_name=None):
    '''
    Plot a heatmap of a wavelet decomposition

    Args:
        data: data by time pandas object.
        n_bins: number of frequency bins
        n_decimal: number of decimal places to round
        title: Title for plot
        file_name: name of filename to save plot
        legend: (bool) include legend

    Returns:
        fig: figure handle
        axes: axes handle

    '''
    data = data.copy()
    plt.figure(figsize=(15,10))
    cA, cD = pywt.cwt(data,np.arange(1,n_bins),'morl')
    sns.heatmap(cA**2,yticklabels=False,xticklabels=False)
    ax = plt.gca()
    ax.set_yticks(range(0,len(cD),5))
    ax.set_yticklabels(np.round(cD[range(0,len(cD),5)],decimals=n_decimals),fontsize=14)
    plt.ylabel('Frequency (Hz)',fontsize=16)
    ax.set_xticks(range(0,len(data),50))
    ax.set_xticklabels(rec_to_time(range(0,len(data),50),TR=tr),rotation=60,fontsize=14)
    plt.xlabel('Time',fontsize=16)
    if title is not None:
        plt.title(title, fontsize=18)
    plt.tight_layout()
    if file_name is not None:
        plt.savefig(file_name)

def plot_avg_state_timeseries(data, groupby=None, line_width=3, overlay=True, color='skyblue', file_name=None, tr=2., ylim=[0,.4],ax=None, legend=True):
    '''
    Plot average state timeseries

    Args:
        data: data by time pandas object.
        groupby: column to organize rows and plot separately
        tr: repetition time for plotting minutes
        line_width: width of lines
        color: vector of colors that matches group

    Returns:
        fig: figure handle
        axes: axes handle

    '''
    if groupby is not None:
        data = data.copy()
        data.sort_values(groupby,inplace=True)
        group_idx = data[groupby]
        groups = data[groupby].unique()
        data.drop(groupby, axis=1, inplace=True)
        if ax is not None:
            if not overlay:
                raise NotImplemented('Setting non overlay plots to a specific axis is not implemented yet.')
            for i,x in enumerate(groups):
                data.loc[group_idx==x,:].mean().plot(kind='line',ax=ax, color=color[i], linewidth=line_width)
                ax.set_xticks(range(data.columns.min(),data.columns.max(),50))
                ax.set_xticklabels(rec_to_time(range(data.columns.min(),data.columns.max(),50),TR=tr),rotation=60,fontsize=14)
                if legend:
                    ax.legend(groups+1,title='State',fontsize=14,loc='upper left')
                ax.set_ylabel('State Probability',fontsize=16)
        else:
            if overlay:
                fig,axes = plt.subplots(figsize=(20,5))
            else:
                fig,axes = plt.subplots(nrows=len(groups),figsize=(20,len(groups)*2),sharex=True)

            for i,x in enumerate(groups):
                if overlay:
                    data.loc[group_idx==x,:].mean().plot(kind='line',ax=axes, color=color[i], linewidth=line_width)
                    axes.set_xticks(range(data.columns.min(),data.columns.max(),50))
                    axes.set_xticklabels(rec_to_time(range(data.columns.min(),data.columns.max(),50),TR=tr),rotation=60,fontsize=14)
                    if legend:
                        plt.legend(groups+1,title='State',fontsize=14,loc='upper left')
                    plt.ylabel('State Probability',fontsize=16)
                else:
                    data.loc[group_idx==x,:].mean().plot(kind='line',ax=axes[i], color=color[i], linewidth=line_width)
                    axes[i].set_xticks(range(data.columns.min(),data.columns.max(),50))
                    axes[i].set_xticklabels(rec_to_time(range(data.columns.min(),data.columns.max(),50),TR=tr),rotation=60,fontsize=14)
                    axes[i].set_ylim(ylim)
                    axes[i].set_ylabel('State %s' % (i + 1),fontsize=14)

            plt.xlabel('Time',fontsize=16)
            plt.tight_layout()

        if ax is None:
            if file_name is not None:
                plt.savefig(file_name)
            return (fig, axes)

    else:
        raise NotImplementedError('This is not implemented yet.')

def _create_opacity(color, intensity, opacity=True):
    if opacity:
        out = []
        for i in intensity:
            out.append(color + (i,))
    else:
        out = [color + (1,)] * len(intensity)
    return out

def _create_simulation_pdf(data):
    p = {}
    for i in np.arange(0,1.01,.01):
        p[np.round(i, decimals=2)] = np.mean(data['Prediction']>i)
    return p

def plot_concordance(data, sim_data=None, fontsize=18, p_threshold=0.05, tr=2.0,
                     opacity=True, legend=True):

    samples = [i for i in data.columns if isinstance(i, (float,int))]
    colors = sns.color_palette("hls", len(data['Cluster'].unique()))
    fig, axs = plt.subplots(1, sharex=True, sharey=True, figsize=(15,3))

    if sim_data is not None:
        p = _create_simulation_pdf(sim_data)
        pp = pd.Series(p)
        ci = pp.keys()[pp.values < p_threshold][0]

        for cluster in sorted(data['Cluster'].unique()):
            y = data.loc[data.loc[:,"Cluster"]==cluster,:].drop('Cluster', axis=1).mean()
            x = y.index
            y = y.values

            z = np.array([(p[np.round(i,decimals=2)]) for i in list(y)])
            z = 1-(z-pp.min())/(pp.max()-pp.min())
            color_list = _create_opacity(colors[cluster], z, opacity=opacity)

            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            lc = LineCollection(segments, colors=color_list, cmap=None)
            lc.set_linewidth(2)
            axs.add_collection(lc)

        plt.axhline(ci, linestyle='--', alpha=.5, color='grey')
    else:
        for cluster in sorted(data['Cluster'].unique()):
            y = data.loc[data.loc[:,"Cluster"]==cluster,:].drop('Cluster', axis=1).mean()
            x = y.index
            y = y.values
            color_list = _create_opacity(colors[cluster], y, opacity=opacity)

            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            lc = LineCollection(segments, colors=color_list, cmap=None)
            lc.set_linewidth(2)
            axs.add_collection(lc)
    if legend:
        axs.legend(sorted(data['Cluster'].unique()),fontsize=fontsize, loc='center left', bbox_to_anchor=(1, 0.5))

    axs.set_xlim(x.min(), x.max())
    axs.set_ylim([0, 1])
    axs.set_xticks(range(min(samples),max(samples),50))
    axs.set_xticklabels(rec_to_time(range(min(samples),max(samples),50),TR=tr),rotation=60,fontsize=fontsize)
    axs.set_ylabel('Concordance', fontsize=fontsize)
    plt.tight_layout()
    return (fig, axs)

def plot_weighted_concordance(data, weighting_dict=None, fontsize=18, tr=2.0,
                                normalize=True, legend=True):

    if weighting_dict is None:
        weighting_dict = {x:1 for x in data['Cluster'].unique()}

    if normalize:
        weights = np.array(list(weighting_dict.values())).reshape(-1,1)
        if len(set(weights.flatten())) > 1:
            scaler = MinMaxScaler()
            weights = scaler.fit_transform(weights)
        for i,x in enumerate(weighting_dict):
            weighting_dict.update({x: weights.flatten()[i]})
    samples = [i for i in data.columns if isinstance(i, (float,int))]
    colors = sns.color_palette("hls", len(data['Cluster'].unique()))
    fig, axs = plt.subplots(1, sharex=True, sharey=True, figsize=(15,3))

    for cluster in sorted(data['Cluster'].unique()):
        y = data.loc[data.loc[:,"Cluster"]==cluster,:].drop('Cluster', axis=1).mean()
        x = y.index
        y = y.values
        c = colors[cluster] + (weighting_dict[cluster],)
        axs.plot(x,y,color=c)
    if legend:
        axs.legend(sorted(data['Cluster'].unique()),fontsize=fontsize, loc='center left', bbox_to_anchor=(1, 0.5))
    axs.set_xlim(x.min(), x.max())
    axs.set_ylim([0, 1])
    axs.set_xticks(range(min(samples),max(samples),50))
    axs.set_xticklabels(rec_to_time(range(min(samples),max(samples),50),TR=tr),rotation=60,fontsize=fontsize)
    axs.set_ylabel('Concordance', fontsize=fontsize)
    plt.tight_layout()
    return (fig, axs)
