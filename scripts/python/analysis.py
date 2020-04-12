import sys
from glob import glob

import numpy as np
import networkx as nx
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from matplotlib.collections import LineCollection
from matplotlib import lines, colors

from scipy.stats.kde import gaussian_kde
from scipy.signal import savgol_filter

from sklearn import preprocessing
from sklearn.decomposition import PCA

sys.path.append('../multiviewtracks/MultiViewTracks/')
# download at https://github.com/pnuehrenberg/multiviewtracks

from utils import load, save, plot_tracks_2d
from tracks import tracks_to_pooled, tracks_from_pooled

def get_speed_distribution(tracks, fps):
    '''Calculates the speed distribution of given tracks.

    Parameters
    ----------
    tracks : dict
        A tracks dictionary
    fps : float or int
        Frames per second, sampling rate of tracks

    Returns
    -------
    np.ndarray
        The full speed distribution (pixel per second) that was sampled from the tracks
    '''

    speed_distribution = []
    for i in tracks['IDENTITIES']:
        d_pos = np.sqrt(np.square(np.diff(tracks[str(i)]['SPINE'][:, 3, :], axis=0)).sum(axis=1))
        frame_idx = tracks[str(i)]['FRAME_IDX']
        speed = savgol_filter(d_pos / np.diff(frame_idx), int(fps) if fps % 2 != 0 else int(fps + 1), 1) * fps # smooth individual speed to compensate tracking noise (1 second)
        speed_distribution.append(speed)
    speed_distribution = np.concatenate(speed_distribution)
    return speed_distribution

def plot_speed_distribution(speed_distribution, q=None):
    '''Plots the speed distribution, annotates a quantile when specified.'''

    fig, ax = plt.subplots()
    ax.hist(speed_distribution, color='gray', bins=40, density=True)
    ax.set_ylabel('normed frequency')
    ax.set_xlabel(r'speed in $px * s^{-1}$')
    if q is not None:
        ax.axvline(q, c='b', label='90 %')
        ax.legend()
    plt.show()

def get_intervals(a):
    '''Returns list of contigues intervals without gaps by cutting the given array where the difference of subsequent values is greater than one.'''

    breaks = np.argwhere(np.diff(a) > 1).ravel() + 1
    breaks = np.insert(breaks, 0, 0)
    breaks = np.append(breaks, -1)
    intervals = []
    for start, end in zip(breaks[:-1], breaks[1:]):
        if a[start:end].size == 0:
            continue
        intervals.append((a[start:end][0], a[start:end][-1]))
    return intervals

def interval_overlap(interval_1, interval_2):
    '''Calculates the overlap of two intervals (tuples of int or float).'''

    overlap = min(interval_1[1], interval_2[1]) - max(interval_1[0], interval_2[0])
    overlap = 0 if overlap < 0 else overlap
    return overlap

def threshold_speed(tracks, q, fps, plot=False, dom_id=None, figsize=(30, 5), xlim=None, xticks=None, xticklabels=None, xlabel=None, ylabel=None, q_label=None, scale=None, legend_loc='upper left', legend_bbox_to_anchor=None):
    '''Thresholds the speed of individual trajectories, and optionally visualizes accordingly.

    Parameters
    ----------
    tracks : dict
        A tracks dictionary
    q : float or int
        The threshold for the speeds
    fps : float or int
        Frames per second, sampling rate of tracks
    plot : bool, optional
        Visualize the thresholding. Defaults to False
    dom_id : int, optional
        Index and id of dominant individual. Defaults to None
    figsize : (int, int), optional
        Size of the figure. Defaults to (30, 5)
    xticks : np.ndarray or list
        X tick positions for the figure. Defaults to None
    xticklabels : np.ndarray or list
        X tick labels for the figure. Defaults to None
    xlabel : string, optional
        Label of the x axis. Defaults to None
    ylabel : string, optional
        Label of the y axis. Defaults to None
    q_label : string, optional
        Label of the speed threshold. Defaults to None
    scale : float
        Pixel per meter ratio for conversion. Defaults to None
    legend_loc : string, optional
        Matplotlib legend location. Defaults to 'upper left'
    legend_bbox_to_anchor : (float, float), optional
        Matplotlib bbox_to_anchor for legend. Defaults to None

    Returns
    -------
    list
        list of lists containing the per-individual intervals above the specified speed threshold

    '''

    if xlim is not None: # subset tracks for visualization
        pooled = tracks_to_pooled(tracks)
        mask = (pooled['FRAME_IDX'] >= xlim[0]) & (pooled['FRAME_IDX'] <= xlim[1])
        pooled = {key: pooled[key][mask] for key in pooled}
        tracks = tracks_from_pooled(pooled)
    if plot:
        dom_color = tuple(v / 255 for v in (255, 109, 69))
        sub_color = tuple(v / 255 for v in (39, 170, 214))
        N = 9
        vals = np.ones((N, 4))
        vals[:, 0] = np.linspace(126 / 255, 24 / 255, N)
        vals[:, 1] = np.linspace(209 / 255, 101 / 255, N)
        vals[:, 2] = np.linspace(237 / 255, 128 / 255, N)
        cmap = ListedColormap(vals)
        y1_bar_dom = [0.6] * 2
        y2_bar_dom = [0.9] * 2
        y1_bar = [0.1] * 2
        y2_bar = [0.4] * 2
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=figsize, gridspec_kw={'height_ratios': [0.2, 0.7]})
        ax_bars, ax = axes
        ax_bars.tick_params(axis='y', width=0, labelsize=14)
        ax_bars.get_xaxis().set_visible(False)
        ax_bars.set_yticks([0.25, 0.75])
        ax_bars.set_ylim((0, 1))
        ax_bars.set_yticklabels([r'$Sub$', r'$Dom$'])
    intervals_above = []
    sub_idx = 0
    for i in tracks['IDENTITIES']:
        d_pos = np.sqrt(np.square(np.diff(tracks[str(i)]['SPINE'][:, 3, :], axis=0)).sum(axis=1))
        frame_idx = tracks[str(i)]['FRAME_IDX']
        speed = savgol_filter(d_pos / np.diff(frame_idx), window_length=int(fps) if fps % 2 != 0 else int(fps + 1), polyorder=1) * fps
        above_thresh = np.argwhere(speed > q).ravel()
        intervals = get_intervals(above_thresh)
        intervals_above.append(intervals)
        if scale is not None:
            speed = speed / scale
        if plot:
            for start, end in intervals:
                if end - start < 2:
                    continue
                ax_bars.fill_between([frame_idx[start], frame_idx[end]],
                                     y1_bar_dom if i == dom_id else y1_bar,
                                     y2_bar_dom if i == dom_id else y2_bar,
                                     facecolor=dom_color if i == dom_id else sub_color,
                                     linewidth=0, zorder=0)
                ax_bars.fill_between([frame_idx[start], frame_idx[end]],
                                     y1_bar_dom if i == dom_id else y1_bar,
                                     y2_bar_dom if i == dom_id else y2_bar,
                                     facecolor=tuple([0, 0, 0, 0]),
                                     linewidth=1, edgecolor='k', zorder=1)
            ax.plot(frame_idx[1:], speed, c=dom_color if i == dom_id else cmap.colors[sub_idx], alpha=1,
                    label=r'$Dom$' if i == dom_id else r'$Sub$' if sub_idx == 4 else '', zorder=1 if i == dom_id else 0)
            if i != dom_id:
                sub_idx += 1
    if plot:
        ax.set_xticks(np.arange(0, 7800, fps * 60))
        ax.set_xticklabels(np.arange(np.arange(0, 7800, fps * 60).size).astype(np.int))
        ax.set_xlabel(r'time in $min$')
        ax.set_ylabel(r'speed in $px * s^{-1}$' if scale is None else r'speed in $m * s^{-1}$')
        ax.set_xlim((min([tracks[str(i)]['FRAME_IDX'].min() for i in tracks['IDENTITIES']]) - fps,
                          max([tracks[str(i)]['FRAME_IDX'].max() for i in tracks['IDENTITIES']])))
        ax.axhline(q / scale if scale is not None else q, c='gray', dashes=(5, 2), label=r'$Q_{90}$' if q_label is None else q_label)
        ax.legend(edgecolor='k', fancybox=True, borderaxespad=0, fontsize=14, loc=legend_loc, bbox_to_anchor=legend_bbox_to_anchor)
        if xticks is not None:
            ax.set_xticks(xticks)
        if xticklabels is not None:
            ax.set_xticklabels(xticklabels)
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=14)
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=14)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        fig.tight_layout()
        plt.show()
    return intervals_above

def get_pairwise_distances(tracks, scale):
    '''Calculates the mean pairwise distance (m) matrix from given tracks and pixel per meter scale.

    Parameters
    ----------
    tracks : dict
        A tracks dictionary
    scale : float
        Pixel per meter ratio for conversion

    Returns
    -------
    np.ndarray
        The mean pairwise distance matrix
    '''

    pairwise_distances = np.zeros((tracks['IDENTITIES'].size, tracks['IDENTITIES'].size))
    for u, i in enumerate(tracks['IDENTITIES']):
        for v, j in enumerate(tracks['IDENTITIES']):
            if i == j:
                pairwise_distances[u, v] = 0
                continue
            mask_ij = np.isin(tracks[str(i)]['FRAME_IDX'], tracks[str(j)]['FRAME_IDX'])
            mask_ji = np.isin(tracks[str(j)]['FRAME_IDX'], tracks[str(i)]['FRAME_IDX'])
            d_pos = tracks[str(i)]['SPINE'][mask_ij, 1, :] - tracks[str(j)]['SPINE'][mask_ji, 1, :] # head to head distance in frames with both individuals
            distance = np.sqrt(np.square(d_pos).sum(axis=1)).mean() / scale
            pairwise_distances[u, v] = distance
    return pairwise_distances

def plot_pairwise_matrix(pairwise_matrix, dom_id=None):
    '''Plots a pairwise metric matrix, annotates dominant when dom_id is specified.'''

    plt.matshow(pairwise_matrix)
    if dom_id is not None:
        ax = plt.gca()
        ax.set_xticks([dom_id])
        ax.set_yticks([dom_id])
        ax.set_xticklabels(['dom'])
        ax.set_yticklabels(['dom'])
    plt.show()

def get_above_thresh_matrix(tracks, intervals_above):
    '''Calculates the pairwise durations that individuals share speeding above the speed threshold.

    Parameters
    ----------
    tracks : dict
        A tracks dictionary
    intervals_above : list
        Returned from threshold_speed()

    Returns
    -------
    np.ndarray
        The pairwise shared speeding duration matrix
    '''

    above_thresh_matrix = np.zeros((tracks['IDENTITIES'].size, tracks['IDENTITIES'].size))
    for u, i in enumerate(tracks['IDENTITIES']):
        for v, j in enumerate(tracks['IDENTITIES']):
            if i == j:
                above_thresh_matrix[u, v] = 0
                continue
            for interval_i in intervals_above[u]:
                for interval_j in intervals_above[v]:
                    overlap = interval_overlap(interval_i, interval_j)
                    above_thresh_matrix[u, v] += overlap
        above_thresh_matrix[u, :] /= np.sum([end - start for start, end in intervals_above[(np.argwhere(tracks['IDENTITIES'] == i).ravel()[0])]])
    return above_thresh_matrix

def get_rel_time_above(tracks, intervals_above):
    '''Calculates the proportion of time that individuals spent above the speed threshold.

    Parameters
    ----------
    tracks : dict
        A tracks dictionary
    intervals_above : list
        Returned from threshold_speed()

    Returns
    -------
    np.ndarray
        Proportion of time that individuals spent speeding
    '''

    time_above = np.array([np.sum([end - start for start, end in intervals]) for i, intervals in enumerate(intervals_above)])
    time_trial = max([tracks[str(i)]['FRAME_IDX'].max() for i in tracks['IDENTITIES']])
    return time_above / time_trial

def get_data(tracks, intervals_above, scale):
    '''Calculate proportion of time that individuals spent speeding, their pairwise distance matrix and social speeding events.
    From the speeding events, calculate delay times and corresponding ids and individual network centrality.

    Parameters
    ----------
    tracks : dict
        A tracks dictionary
    intervals_above : list
        Returned from threshold_speed()
    scale : float
        Pixel per meter ratio for conversion

    Returns
    -------
    np.ndarray
        The proportion of time spent speeding
    np.ndarray
        The pairwise distance matrix
    NoneType
        Should be replaced with visual connectivity matrix for further analysis
    np.ndarray
        Individual network centrality
    np.ndarray
        Event delay times
    np.ndarray
        Event ids
    '''

    rel_time_above = get_rel_time_above(tracks, intervals_above)
    pairwise_distances = get_pairwise_distances(tracks, scale)
    event_time, event_ids = get_event_response(tracks, intervals_above)
    centrality = get_social_influence(tracks, event_ids, plot=False, event_time=event_time)
    return rel_time_above, pairwise_distances, None, centrality, event_time, event_ids

def group_data(tracks, data, dom_id, min_delay_time=0):
    '''Groups data (dom and sub) that was obtained with get_data(). Optionally filter events without clear initiator or responder (min_delay_time).
    Returns data as tuples (dom and sub) according to input.'''

    rel_time_above_dom = data[0][tracks['IDENTITIES'] == dom_id]
    rel_time_above_sub = data[0][tracks['IDENTITIES'] != dom_id]
    pairwise_dist_dom = data[1].mean(axis=0)[tracks['IDENTITIES'] == dom_id]
    pairwise_dist_sub = data[1].mean(axis=0)[tracks['IDENTITIES'] != dom_id]
    aa_out_dom = data[2].mean(axis=0)[tracks['IDENTITIES'] == dom_id]
    aa_out_sub = data[2].mean(axis=0)[tracks['IDENTITIES'] != dom_id]
    centrality_dom = data[3][tracks['IDENTITIES'] == dom_id]
    centrality_sub = data[3][tracks['IDENTITIES'] != dom_id]
    data[5] = data[5][np.diff(data[4], axis=1).ravel() >= min_delay_time]
    data[4] = data[4][np.diff(data[4], axis=1).ravel() >= min_delay_time]
    event_times_dom = data[4].ravel()[data[5].ravel() == dom_id]
    event_times_sub = [data[4].ravel()[data[5].ravel() == i] for i in tracks['IDENTITIES'][tracks['IDENTITIES'] != dom_id]]
    event_ids_dom = np.repeat(dom_id, event_times_dom.size)
    event_ids_sub = [np.repeat(i, data[4].ravel()[data[5].ravel() == i].size) for i in tracks['IDENTITIES'][tracks['IDENTITIES'] != dom_id]]
    return (centrality_dom, centrality_sub), \
           (pairwise_dist_dom, pairwise_dist_sub), \
           (aa_out_dom, aa_out_sub), \
           (rel_time_above_dom, rel_time_above_sub), \
           (event_times_dom, event_times_sub), \
           (event_ids_dom, event_ids_sub)

def get_event_response(tracks, intervals_above):
    '''Determines event initiators and responders, and delay times.

    Parameters
    ----------
    tracks : dict
        A tracks dictionary
    intervals_above : list
        Returned from threshold_speed()

    Returns
    -------
    np.ndarray
        Event delay times
    np.ndarray
        Event ids
    '''

    intervals_trial = np.concatenate(intervals_above, axis=0)
    intervals_identities = np.concatenate([np.repeat(i, len(intervals)) for i, intervals in zip(tracks['IDENTITIES'], intervals_above)])
    sort_idx = np.argsort(intervals_trial[:, 0])
    intervals_identities = intervals_identities[sort_idx]
    intervals_trial = intervals_trial[sort_idx, :]
    intervals_identities = intervals_identities[(intervals_trial > 5).any(axis=1)] # remove first few frames because it would not be clear who is responder
    intervals_trial = intervals_trial[(intervals_trial > 5).any(axis=1), :] # remove first few frames because it would not be clear who is responder
    intervals_identities = intervals_identities.tolist()
    intervals_trial = [tuple(interval) for interval in intervals_trial]
    events = []
    while len(intervals_trial) > 1:
        event = []
        event_ids = []
        initiator_start, end = intervals_trial.pop(0)
        initiator_id = intervals_identities.pop(0)
        event.append(initiator_start)
        event_ids.append(initiator_id)
        start = intervals_trial[0][0]
        while start < end:
            responder_start, responder_end = intervals_trial.pop(0)
            responder_id = intervals_identities.pop(0)
            if responder_end > end:
                end = responder_end
            event.append(responder_start)
            event_ids.append(responder_id)
            start = intervals_trial[0][0]
        events.append((event, event_ids))
        events = [event for event in events if len(event[0]) > 1] # needs at least one responder to be counted as event
    event_ids = np.array([event[1][:2] for event in events]) # we are only interested in dyadic initiator - first responder interactions
    event_time = np.array([np.array(event[0])[:2] - event[0][0] for event in events]) # actual time to delay time
    return event_time, event_ids

def get_social_influence(tracks, event_ids, dom_id=None, plot=False, plot_network=False, figsize=(5, 2.5), event_time=None, min_delay_time=0):
    '''Calculates social influence as network (out-edges Katz) centrality. Optionally visualizes network and centrality.

    Parameters
    ----------
    tracks : dict
        A tracks dictionary
    event_ids : np.ndarray
        The event_ids
    dom_id : int, optional
        The id of the dominant individual for visualization. Defaults to None
    plot : bool, optional
        Visualize social influence matrix? Defaults to False
    plot_network : bool, optional
        Visualize network? Defaults to False
    figsize : (int, int), optional
        Figure size for network visualization. Defaults to (5. 2.5)
    event_time : np.ndarray, optional
        Corresponding event times (for event ids). Necessary when only counting events with a minimum delay time. Defaults to None
    min_delay_time : float
        Minimum delay time between initiator and responder to define them as clear initiator and responder. Defaults to 0

    Returns
    -------
    np.ndarray
        Network centrality for all individuals
    '''

    social_influence = np.zeros((tracks['IDENTITIES'].size, tracks['IDENTITIES'].size))
    for idx, (initiator, responder) in enumerate(event_ids):
        social_influence[initiator, responder] += 1
        if event_time is not None and np.diff(event_time[idx]) < min_delay_time:
            social_influence[responder, initiator] += 1 # if there is no clear responder, add undirected weight (we did not use this)
    if plot:
        plot_pairwise_matrix(social_influence, dom_id=dom_id)
    G = nx.from_numpy_array(social_influence, create_using=nx.DiGraph) # directed graph from social influence matrix
    centrality = np.array(list(nx.centrality.katz_centrality(G.reverse(), weight='weight').values())) # out-edges centrality from the reversed graph
    if plot or plot_network:
        N = 50
        vals = np.ones((N * 4, 4))
        vals[:N, 0] = np.linspace(22 / 255, 47 / 255)
        vals[:N, 1] = np.linspace(71 / 255, 156 / 255)
        vals[:N, 2] = np.linspace(0 / 255, 0 / 255)
        vals[N:, 0] = np.linspace(47 / 255, 255 / 255, N * 3)
        vals[N:, 1] = np.linspace(156 / 255, 230 / 255, N * 3)
        vals[N:, 2] = np.linspace(0 / 255, 66 / 255, N * 3)
        cmap = ListedColormap(vals)
        c = centrality
        vmin = c.min() - c.min() % 0.1
        vmax = 0.1 + c.max() - c.max() % 0.1
        G = nx.from_numpy_array(social_influence, create_using=nx.Graph)
        fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [0.8, 0.025]})
        ax, cax = axes
        ax.axis('OFF')
        pos = nx.spring_layout(G, k=0.7, scale=100)
        nodes = nx.draw_networkx_nodes(G, pos=pos,
                                       node_color=c,
                                       vmin=vmin,
                                       vmax=vmax,
                                       cmap=cmap,
                                       node_size=500,
                                       ax=ax)
        nodes.set_edgecolor('k')
        for edge in zip(np.repeat(np.arange(len(G)), len(G)),
                        np.tile(np.arange(len(G)), len(G))):
            edge_weight = social_influence[edge[0], edge[1]]
            if edge_weight == 0:
                continue
            ax.annotate('',
                        xy=pos[edge[1]],
                        xycoords='data',
                        xytext=pos[edge[0]],
                        textcoords='data',
                        arrowprops=dict(arrowstyle='->',
                                        linewidth=(1 + edge_weight / np.max(social_influence)) / 2,
                                        color=[0.5 * (1 - edge_weight / np.max(social_influence))] * 3,
                                        shrinkA=14,
                                        shrinkB=14,
                                        patchA=None,
                                        patchB=None,
                                        connectionstyle='arc3,rad=-0.15'))
        nx.draw_networkx_labels(G, pos=pos,
                                labels={idx: r'$Dom$' if i == dom_id else '' \
                                        for idx, i in enumerate(tracks['IDENTITIES'])},
                                font_size=8.5, ax=ax,
                                verticalalignment='center')
        ax.set_ylim((ax.get_ylim()[0] - 10, ax.get_ylim()[1] + 10))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=colors.Normalize(vmin=vmin, vmax=vmax))
        sm._A = []
        cbar = plt.colorbar(sm, cax=cax)
        cax.set_ylabel('network centrality', labelpad=20, rotation=270, fontsize=14)
        cbar.set_ticks(np.round(np.arange(vmin, vmax + 0.1, 0.1), 1))
        cbar.set_ticklabels(np.round(np.arange(vmin, vmax + 0.1, 0.1), 1))
        fig.tight_layout()
        plt.show()
    return centrality

def enumerate_identities(tracks, dom_id):
    '''Sort (ascending) and relabel trajectory identities. Returns relabeled tracks and new id of dominant individual.'''

    dom_id = np.argwhere(tracks['IDENTITIES'] == dom_id).ravel()[0]
    for idx, i in enumerate(np.sort(tracks['IDENTITIES'])):
        tracks[str(idx)] = tracks.pop(str(i))
    tracks['IDENTITIES'] = np.arange(tracks['IDENTITIES'].size, dtype=np.int)
    return tracks, dom_id

def plot_network_randomization(avg_metric_sub, avg_metric_dom, metric_sub, metric_dom, ylabel, xlim_hist, alpha_level=0.05, dom_color=None, sub_color=None):
    '''Visualize network randomization test and calculate two-tailed p-value. Refer to the example notebook for network randomization.

    Parameters
    ----------
    avg_metric_sub : np.ndarray
        Contains the metric of sub obtained from network randomizations
    avg_metric_dom : np.ndarray
        Contains the metric of dom obtained from network randomizations
    metric_sub : np.ndarray
        Contains the observed metric of sub
    metric_dom : np.ndarray
        Contains the observed metric of dom
    ylabel : string
        Y axis label for the randomization plot
    xlim_hist : (float, float)
        X axis limits for the histogram.
    alpha_level : float, optional
        Significance level for test visualization in the range of [0, 1]. Defaults to 0.05
    dom_color : (float, float, float), optional
        RGB color for dom in the range of [0, 1]
    sub_color : (float, float, float), optional
        RGB color for sub in the range of [0, 1]

    Returns
    -------
    string
        A formatted p-value
    '''

    if dom_color is None:
        dom_color = tuple(v / 255 for v in (255, 109, 69))
    if sub_color is None:
        sub_color = tuple(v / 255 for v in (39, 170, 214))
    mean_values_sub = []
    for dist in np.concatenate([trial for trial in np.array(avg_metric_sub).reshape(6, 1000, -1)], axis=1):
        dist = dist[np.isfinite(dist)]
        mean_values_sub.append(dist.mean())
    mean_values_dom = []
    for dist in np.concatenate([trial for trial in np.array(avg_metric_dom).reshape(6, 1000, -1)], axis=1):
        dist = dist[np.isfinite(dist)]
        mean_values_dom.append(dist.mean())
    mean_values_dom = np.array(mean_values_dom)
    mean_values_sub = np.array(mean_values_sub)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={'width_ratios': [0.5, 0.4]})
    lc = LineCollection(np.transpose([np.repeat(1, mean_values_dom.size),
                                      mean_values_dom,
                                      np.repeat(4, mean_values_sub.size),
                                      mean_values_sub]).reshape(-1, 2, 2),
                        lw=0.5,
                        alpha=0.1,
                        color=(0.2, 0.2, 0.2), zorder=0,
                        capstyle='butt')
    axes[0].add_collection(lc)
    axes[0].scatter(np.random.uniform(0.1, 0.9, mean_values_dom.size),
                    mean_values_dom,
                    s=5,
                    facecolor=(0.5, 0.5, 0.5),
                    edgecolor='k',
                    lw=0.4)
    axes[0].scatter(np.random.uniform(4.1, 4.9, mean_values_sub.size),
                    mean_values_sub,
                    s=5,
                    facecolor=(0.5, 0.5, 0.5),
                    edgecolor='k',
                    lw=0.4)
    axes[0].plot([1, 4], [metric_dom.mean(), metric_sub.mean()], '--', color='k', solid_capstyle='butt')
    axes[0].scatter([0.5, 4.5], [metric_dom.mean(), metric_sub.mean()], s=20,
                    marker='o', facecolor=np.array([dom_color, sub_color]), edgecolor='k')
    axes[0].set_ylabel(ylabel, fontsize=14)
    axes[0].set_xticks([0.5, 4.5])
    axes[0].set_xticklabels([r'$Dom$', r'$Sub$'], fontsize=14)
    differences = mean_values_dom - mean_values_sub
    pdf = gaussian_kde(differences)
    padding = (differences.max() - differences.min())
    x = np.linspace(differences.min() - padding , differences.max() + padding, 1000)
    cdf = np.cumsum(pdf(x)) * np.diff(x)[0]
    left = np.argwhere(cdf <= alpha_level / 2).ravel().max()
    right = np.argwhere(cdf >= 1 - alpha_level / 2).ravel().min()
    observed = metric_dom.mean() - metric_sub.mean()
    idx = np.argmin(np.abs(x - observed))
    # calculate p value ( * 2) because two-sided
    if np.abs(observed - x[left]) <= np.abs(observed - x[right]):
        p_value = 2 * cdf[idx + 1]
    else:
        p_value = 2 * (1 - cdf[idx])
    axes[1].hist(mean_values_dom - mean_values_sub, bins=30, density=True,
                 facecolor=(0, 0, 0, 0.1), edgecolor=(0, 0, 0, 0.4))
    axes[1].fill_between(x[:left + 1], pdf(x[:left + 1]), facecolor='#7CB939', alpha=0.75)
    axes[1].fill_between(x[left:right + 1], pdf(x[left:right + 1]), facecolor='k', alpha=0.25)
    axes[1].fill_between(x[right:], pdf(x[right:]), facecolor='#7CB939', alpha=0.75)
    axes[1].plot([x[left]] * 2, [0, pdf(x[left])], c='k', alpha=0.75, lw=0.5, solid_capstyle='butt')
    axes[1].plot([x[right]] * 2, [0, pdf(x[right])], c='k', alpha=0.75, lw=0.5, solid_capstyle='butt')
    axes[1].plot(x, pdf(x), alpha=1, lw=0.5, c='k', solid_capstyle='butt')
    axes[1].axvline(observed, linestyle='--', color='k', solid_capstyle='butt', ymax=0.9)
    axes[1].set_xlim(xlim_hist)
    axes[1].set_xlabel('mean difference', fontsize=14)
    axes[1].set_ylabel('density', fontsize=14)
    for ax in axes.ravel():
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
    fig.tight_layout()
    plt.show()
    return 'p-value: {}'.format(p_value)
