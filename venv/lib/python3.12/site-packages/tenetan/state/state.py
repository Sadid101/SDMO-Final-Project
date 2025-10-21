from tenetan.networks import SnapshotGraph
import numpy as np
from itertools import combinations
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, cut_tree
from typing import Callable

from tenetan.static.distance import *

__all__ = ["MasudaHolme", "DunnIndex"]


def DunnIndex(D: np.ndarray, labels: np.ndarray) -> float:
    """
    Dunn Index of a clustering of objects.

    Computed as the minimum distance between two clusters divided by the maximum
    diameter of a single cluster.
    Parameters
    ----------
    D: np.ndarray,
        Distance (N,N) matrix giving pairwise distances between N objects.
    labels: np.ndarray,
        Vector of cluster labels, i.e., labels[i] given the integer label of cluster of i.

    Returns
    -------
    dunn: float,
        Dunn Index
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters < 2:
        return 0

    intra_dists = []
    inter_dists = []

    for i in unique_labels:
        members_i = np.where(labels == i)[0]
        if len(members_i) <= 1:
            intra_dists.append(0)
        else:
            intra = D[np.ix_(members_i, members_i)]
            intra_dists.append(np.max(intra))

        for j in unique_labels:
            if i < j:
                members_j = np.where(labels == j)[0]
                inter = D[np.ix_(members_i, members_j)]
                inter_dists.append(np.min(inter))

    max_intra = max(intra_dists)
    min_inter = min(inter_dists)

    return min_inter / max_intra if max_intra > 0 else 0


def MasudaHolme(G: SnapshotGraph, dist: Callable = GraphEditDistance):
    """
    State detection in Snapshot temporal network by method of Masuda & Holme.
    Masuda, N., & Holme, P. (2019). Detecting sequences of system states in temporal networks.
    Scientific reports, 9(1), 795.

    State detection is performed by clustering snapshots of the temporal network,
    i.e. all possible timestamps, by calculating distances between snapshot
    adjacency (weight) matrices and applying hierarchical clustering to the resulting
    distance matrix. Best clustering is selected using Dunn Index.

    In case clustering (scipy linkage) computation fails, return only the distance
    matrix (all other returns are None).

    Parameters
    ----------
    G: SnapshotGraph,
    dist: Callable,
        Distance function to use for distance computation between snapshot (default GraphEditDistance).
        Must accept two adjacency matrices A1 and A2 as the only parameters
        (NumPy arrays of shape (N,N)).
        If a function accepts more parameters, they should be initialized with
        functools.partial.

    Returns
    -------
        best_C: int
            Index of the best cluster assignment and best Dunn Index
            (i.e., argmax(dunn_scores)).
            Since indexes are 0-based, the amount of clusters is best_C+1.
        labels: np.ndarray
            Array of shape (T,T) giving the assignment of T timestamps to different
            number of clusters;
            labels[t,c] gives the cluster of t when c+1 clusters
            are created;
            labels[:, best_C] gives the best cluster assignment according
            to Dunn index.
        dunn_scores: np.ndarray
            Vector of size T; dunn_scores[c] gives the Dunn index of c+1 clusters;
            dunn_scores[best_C] gives the best Dunn index.
        distance_matrix: np.ndarray
            Matrix of shape (T,T) encoding distances between all pairs of timestamps.
        linkage_matrix: np.ndarray
            Linkage matrix for clustering, given by scipy.cluster.hierarchy.linkage
    """

    N, T = G.N, G.T
    A = G.tensor
    distance_matrix = np.zeros((T,T))
    for t1, t2 in combinations(range(T), 2):
        A1 = A[:, :, t1]
        A2 = A[:, :, t2]
        distance_matrix[t1,t2] = distance_matrix[t2, t1] = dist(A1, A2)
    distance_vector = squareform(distance_matrix)
    try:
        linkage_matrix = linkage(distance_vector)
    except:
        return None, None, None, distance_matrix, None

    dunn_scores = np.zeros(T)
    labels = cut_tree(linkage_matrix)
    labels = np.flip(labels, axis=1)
    for C in range(1, T+1):
        dunn = DunnIndex(distance_matrix, labels[:, C-1])
        dunn_scores[C-1] = dunn

    best_C = int(np.argmax(dunn_scores))

    return best_C, labels, dunn_scores, distance_matrix, linkage_matrix


if __name__ == "__main__":

    G = SnapshotGraph()
    G.load_csv("../datasets/eg_taylor.csv",
               source="i", target="j", timestamp="t", weight="w",
               sort_vertices=True, sort_timestamps=True)
    best_C, labels, dunn_scores, distance_matrix, linkage_matrix = MasudaHolme(G, dist=SpectralDistance)
    print(best_C)
    print(dunn_scores)
    print(labels)
    print(distance_matrix)
    print(linkage_matrix)


