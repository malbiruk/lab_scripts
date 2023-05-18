'''
module with objects related to mhp clusterization
'''

from collections import namedtuple
from pathlib import Path

import numpy as np
from modules.general import flatten
from scipy.ndimage import gaussian_filter, label, uniform_filter, zoom


def mhp_single_clust(mapp: np.array, box_side: float,
                     option: str = 'hydrophobic') -> namedtuple:
    '''
    mapp - 2-dimensional array
    box_side - box side size
    oprion: 'hydrophobic' or 'hydrophilic'

    obtain clusters of MHP for single frame using
    connected-component labeling with periodic boundary conditions
    '''
    Result = namedtuple('Result', ['counts', 'labels', 'clusters'])

    matrix = gaussian_filter(mapp, sigma=1)
    shrinkby = mapp.shape[0] / box_side
    matrix_filtered = uniform_filter(input=matrix, size=shrinkby)
    matrix = zoom(input=matrix_filtered, zoom=1. / shrinkby, order=0)

    # MHP>=0.5 => hydrophobic, MHP<=-0.5 => hydrophilic
    if option == 'hydrophilic':
        threshold_indices = matrix > -0.5
    if option == 'hydrophobic':
        threshold_indices = matrix < 0.5
    matrix[threshold_indices] = 0
    s = [[1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]]
    clusters, _ = label(matrix, structure=s)

    # periodic boundary conditions
    for y in range(clusters.shape[0]):
        if clusters[y, 0] > 0 and clusters[y, -1] > 0:
            clusters[clusters == clusters[y, -1]] = clusters[y, 0]
    for x in range(clusters.shape[1]):
        if clusters[0, x] > 0 and clusters[-1, x] > 0:
            clusters[clusters == clusters[-1, x]] = clusters[0, x]
    labels, counts = np.unique(
        clusters[clusters > 0], return_counts=True)
    return Result(counts, labels, clusters)


def mhp_clust_sizes(mapp: np.array, box_side: float,
                    option: str = 'hydrophobic') -> namedtuple:
    '''
    perform mhp_single_clust for each of frames in mapp (3d-d array)
    and collect results into arrays/lists

    returns:
    Result.counts - np.array
    Result.labels - list
    Result.clusters - list
    '''

    Result = namedtuple('Result', ['counts', 'labels', 'clusters'])

    counts_arr = np.array([])
    labels_list = []
    clusters_list = []

    for ts in mapp:
        counts, labels, clusters = mhp_single_clust(ts, box_side, option)
        counts_arr = np.hstack((counts_arr, counts))
        labels_list.append(labels.tolist())
        clusters_list.append(clusters)

    return Result(counts_arr, labels_list, clusters_list)


class NotSameClusterError(Exception):
    '''
    raised when trying to update clusters
    with mismatching positions and/or sizes
    '''


class Cluster:
    '''
    MHP Cluster instance. is used for lifetimes calculation
    '''
    # pylint: disable=too-many-instance-attributes
    ID = 1

    def __init__(self, clusters_list_slice, l, area_threshold):
        '''
        clusters_list_slice: clusters_list on current timestep
        l: initial label of cluster
        area_threshold: threshold for intersecting area of clusters.
        if > area_threshold intersection on next timestep => same clusters
        '''
        self.id = Cluster.ID
        self.init_frame = clusters_list_slice
        self.label = l
        self.position = np.transpose(
            np.nonzero(self.init_frame == self.label))  # list of indices
        self.flat_position = np.flatnonzero(
            self.init_frame == self.label)  # flat list of indices
        self.size = len(self.position)
        self.lifetime = 0
        self.area_threshold = area_threshold
        Cluster.ID += 1

    def __repr__(self):
        a = np.zeros_like(self.init_frame)
        np.put(a, self.flat_position, 1)
        return (f'Cluster {self.id}:\n('
                + np.array2string(a) + f', lt={self.lifetime})')

    def same_as(self, other) -> bool:
        '''
        returns True if intersection (of flat indices) of clusters
        (multiplied by 1/area_threshold)
        is bigger than each of clusters in comparison
        '''
        return (
            len(set(self.flat_position)
                & set(other.flat_position)) * (1 / self.area_threshold)
            > len(self.flat_position)
            and len(set(self.flat_position)
                    & set(other.flat_position)) * (1 / self.area_threshold)
            > len(other.flat_position))

    def update_cluster(self, other, dt: int) -> None:
        '''
        update cluster from self to other and increment lifetime by dt

        dt: timestep between frames from which clusters are from
        '''
        if not self.same_as(other):
            raise NotSameClusterError('trying to update cluster '
                                      'with mismatching positions and/or sizes')
        self.init_frame = other.init_frame
        self.label = other.label
        self.lifetime += dt
        self.flat_position = np.flatnonzero(
            self.init_frame == self.label)
        self.position = np.transpose(
            np.nonzero(self.init_frame == self.label))
        self.size = len(self.position)


def calculate_cluster_lifetimes(clusters: list, labels: list, dt: int,
                                area_threshold: int = 0.5) -> np.array:
    '''
    arguments:
    clusters - list of clusters (arrays),
    labels - list of labels (arrays),
    dt - timestep between frames in clusters and labels
    area_threshold - minimum area intersection of same cluster
    on adjacent frames

    returns array of lifetimes for all clusters
    '''
    # {timestep: [clusters which died on current timestep]}
    all_clusters = {ts: [] for ts in range(len(clusters))}

    # initiate clusters on frame 0
    all_clusters[0] = [Cluster(clusters[0], labl, area_threshold)
                       for labl in labels[0]]

    for ts in range(1, len(clusters)):
        new_clusters = [Cluster(clusters[ts], labl, area_threshold)
                        for labl in labels[ts]]
        old_clusters = all_clusters[ts - 1]

        # updating all_clusters dict
        for c_old in old_clusters:
            for c_new in new_clusters:
                if c_old.same_as(c_new):
                    c_old.update_cluster(c_new, dt)
                    all_clusters[ts].append(c_old)
                    all_clusters[ts - 1].remove(c_old)
                    c_new.updated = True

        all_clusters[ts].extend(
            [c for c in new_clusters if not hasattr(c, 'updated')])

    return np.array([i.lifetime for i in flatten(all_clusters.values())])


def get_mhp_clusters_sizes_lifetimes(
        nmp_file: Path, box_side: float, dt: int,
        option='hydrophobic', area_threshold: int = 0.5):
    '''
    wrapper for mhp_clust_sizes() and calculate_cluster_lifetimes()
    functions, which returns only cluster sizes and lifetimes
    '''

    Result = namedtuple('Result', ['sizes', 'lifetimes'])

    mapp = np.load(nmp_file)['data']
    counts, labels, clusters = mhp_clust_sizes(
        mapp, box_side, option)
    lifetimes = calculate_cluster_lifetimes(
        clusters, labels, dt, area_threshold)

    return Result(counts, lifetimes)
