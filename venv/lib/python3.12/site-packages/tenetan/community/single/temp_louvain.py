import numpy as np
import networkx as nx
from networkx.algorithms.community import louvain_communities
from tenetan.networks import SnapshotGraph
from collections import defaultdict
from typing import List, Set, Dict, Any
from itertools import pairwise, product

__all__ = ["StaticLouvain", "TemporalLouvain", "StepwiseLouvain", "DynamicCommunities"]

def _renumber_labels(labels: np.ndarray) -> np.ndarray:
    """Make labels contiguous (0..K-1) in a stable encounter order."""
    uniq = {}
    out = np.empty_like(labels, dtype=int)
    next_id = 0
    for i, c in enumerate(labels):
        if c not in uniq:
            uniq[c] = next_id
            next_id += 1
        out[i] = uniq[c]
    return out


def DynamicCommunities(
    G,
    labels_matrix: np.ndarray,
    sim_threshold: float = 0.5,
) -> nx.DiGraph:
    """
    Construct dynamic communities by matching step communities across snapshots via Jaccard.

    Parameters
    ----------
    G : SnapshotGraph
    labels_matrix : np.ndarray
        Shape (N, T). labels_matrix[n, t] is the step-community label of node n at time t.
        Labels at different t are *snapshot-local* (no assumed correspondence across t).
    sim_threshold : float
        Jaccard threshold. If J(C_{t-1,i}, C_{t,j}) >= theta, we add an edge (t-1,i) → (t,j).

    Returns
    -------
    evolution : nx.DiGraph
        A directed graph over step communities. Each node is a tuple (t, label_value_at_t)
        with attributes:
            - 'nodes': set of member indices at snapshot t
            - 'size':  int size of the community
        For each t>0, an edge exists (t-1, li) -> (t, lj) with Jaccard similarity
        as weight if the Jaccard similarity >= sim_threshold.
    """
    N, T = labels_matrix.shape
    timestamps = G.timestamps
    vertices = G.vertices

    evolution = nx.DiGraph()

    # --- 1) Build step communities per snapshot: { (t, label) -> set(node_indices) } ---
    # Also keep an indexable list per t for easier iteration.
    step_sets_per_t: List[Dict[int, Set[int]]] = []
    for t in range(T):
        labels_t = labels_matrix[:, t]
        comms_t: Dict[int, Set[int]] = {}
        for cid in np.unique(labels_t):
            members = set(np.where(labels_t == cid)[0])
            members = {vertices[i] for i in members}
            comms_t[int(cid)] = members
            evolution.add_node(
                (timestamps[t], int(cid)),
                members=members,
                size=len(members),
            )
        step_sets_per_t.append(comms_t)

    # --- 2) Make evolution graph over step communities ---
    # Jaccard helper
    def _jac(a: Set[int], b: Set[int]) -> float:
        if not a and not b:
            return 1.0
        inter = len(a & b)
        if inter == 0:
            return 0.0
        return inter / (len(a) + len(b) - inter)

    # Add edges between consecutive snapshots when Jaccard >= theta
    for t, (prev_comms, curr_comms) in enumerate(pairwise(step_sets_per_t), start=1):
        for (li, Si), (lj, Sj) in product(prev_comms.items(), curr_comms.items()):
                jacc = _jac(Si, Sj)
                if jacc >= sim_threshold:
                    evolution.add_edge((timestamps[t - 1], li), (timestamps[t], lj), weight=float(jacc))

    return evolution

def StaticLouvain(W: np.ndarray, threshold: float = 1e-07, resolution: float = 1.0, seed=None) -> np.ndarray:
    """
    Run Louvain community detection (Blondel et al.) on a weighted directed graph given by adjacency array W.

    Parameters
    ----------
    W: np.ndarray
        Adjacency (weight) matrix
    threshold: float
        Threshold parameter for networkx.algorithms.community.louvain_communities.
        Default 1e-07.
    resolution: float
        Resolution parameter for networkx.algorithms.community.louvain_communities.
        Default 1.0.
    seed: Any | None
        Random seed for networkx.algorithms.community.louvain_communities.
    """

    n = W.shape[0]
    G = nx.from_numpy_array(W, create_using=nx.DiGraph)
    comms = louvain_communities(G, weight="weight", threshold=threshold, resolution=resolution, seed=seed)
    labels = np.empty(n, dtype=int)
    for cid, nodes in enumerate(comms):
        for v in nodes:
            labels[int(v)] = cid
    # return _renumber_labels(labels)
    return labels

def TemporalLouvain(
    G,
    *,
    louvain_threshold: float = 1e-07,
    louvain_resolution: float = 1.0,
    louvain_seed=None,
    change_tol: float = 0.0
):
    """
    Temporal Louvain algorithm for community detection by analysis of changed edges.
    He, J., & Chen, D. (2015). A fast algorithm for community detection in temporal network.
    Physica A: Statistical Mechanics and its Applications, 429, 87-94.

    Parameters
    ----------
    G : SnapshotGraph
    louvain_threshold: float
        Threshold parameter. Passed to NetworkX Louvain.
        Default 1e-07.
    louvain_resolution : float
        Louvain resolution (default 1.0). Passed to NetworkX Louvain.
        Default 1.0.
    louvain_seed : int or None
        Random seed for Louvain algorithm. Passed to NetworkX Louvain.
    change_tol : float
        Absolute tolerance for detecting a node's connection change between
        t-1 and t (any incident edge |Δ| > change_tol -> node considered "changed").
        Default 0.0, i.e. any change in connection or weight is considered a change edge.
        Increase to permit small variations in edge weights.

    Returns
    -------
    communities : List[Dict[int, List[Any]]]
        For each t, a dict: {label -> [nodes]}.
        Community labels are integers.
    node_labels : List[Dict[Any, int]]
        For each t, a dict: {node -> [label]}.
        Community labels are integers.
    labels: np.ndarray
        Shape (N, T), contains the label of node with index n at time t

    Notes
    -----
    Implements the two-step loop from He & Chen: (1) run Louvain on t=0;
    (2) for t>0, compress unchanged nodes (within their prev community) into
    supernodes to build a small graph, run Louvain there, then expand.
    """
    # --- Inputs & checks
    A = G.tensor
    N, T = G.N, G.T

    def W(t):
        return A[:, :, t]

    labels_t = []
    # First iteration t=0: static Louvain
    labels = StaticLouvain(A[:, :, 0], threshold=louvain_threshold,
                           resolution=louvain_resolution, seed=louvain_seed)
    labels_t.append(labels)

    # Temporal Louvain
    for t in range(1,T):
        Wt = W(t)
        prev_labels = labels_t[t-1]

        # Detect node changes between t-1 and t, up to a certain tolerance
        diff = np.abs(Wt - W(t - 1))
        changed = (diff > change_tol).any(axis=1)

        # Supernode construction per He & Chen:
        #   within each previous community, one supernode for all UNCHANGED nodes,
        #   and one supernode per CHANGED node.
        groups = []
        next_gid = 0
        for c in np.unique(prev_labels):
            idx = np.where(prev_labels == c)[0]
            # unchanged subset within this community
            u = idx[~changed[idx]]
            if u.size > 0:
                groups.append(u)
                next_gid += 1
            # each changed node becomes its own group
            for i in idx[changed[idx]]:
                groups.append(np.array([i], dtype=int))
                next_gid += 1

        K = len(groups)

        # Dense membership matrix (OK for moderate N; use sparse if needed)
        M = np.zeros((N, K), dtype=float)
        for k, nodes in enumerate(groups):
            M[nodes, k] = 1.0

        # Compressed adjacency
        B = M.T @ Wt @ M

        # Louvain on compressed graph
        group_labels = StaticLouvain(B, threshold=louvain_threshold,
                                     resolution=louvain_resolution, seed=louvain_seed)

        # Assign labels to original nodes
        labels = np.empty(N, dtype=int)
        for k, nodes in enumerate(groups):
            labels[nodes] = group_labels[k]
        # labels = _renumber_labels(labels)
        labels_t.append(labels)

    # Construct mappings from node to label and from label to list of nodes
    node_labels = []
    communities = []
    for labels in labels_t:
        node_labels_t = {}
        communities_t = defaultdict(list)
        for i, node in enumerate(G.vertices):
            label = int(labels[i])
            node_labels_t[node] = label
            communities_t[label].append(node)
        node_labels.append(node_labels_t)
        communities.append(communities_t)

    # (N,T) matrix of node labels
    labels = np.stack(labels_t, axis=1)

    return communities, node_labels, labels


def StepwiseLouvain(
    G,
    *,
    direction="both",
    louvain_threshold: float = 1e-07,
    louvain_resolution: float = 1.0,
    louvain_seed=None,
):
    """
    Stepwise Louvain algorithm for community detection by analysis of changed edges within submodules.

    He, J., Chen, D., Sun, C., Fu, Y., & Li, W. (2017). Efficient stepwise detection of communities in temporal networks.
    Physica A: Statistical Mechanics and its Applications, 469, 438-446.

    Parameters
    ----------
    G : SnapshotGraph
    direction : str
        Which direction of connections to consider when determining the neighbours
        of the node, "in", "out" or "both". Defaults to "both".
        For undirected networks use "both".
    louvain_threshold: float
        Threshold parameter. Passed to NetworkX Louvain.
        Default 1e-07.
    louvain_resolution : float
        Louvain resolution (default 1.0). Passed to NetworkX Louvain.
        Default 1.0.
    louvain_seed : int or None
        Random seed for Louvain algorithm. Passed to NetworkX Louvain.

    Returns
    -------
    communities : List[Dict[int, List[Any]]]
        For each t, a dict: {label -> [nodes]}.
        Community labels are integers.
    node_labels : List[Dict[Any, int]]
        For each t, a dict: {node -> [label]}.
        Community labels are integers.
    labels: np.ndarray
        Shape (N, T), contains the label of node with index n at time t

    Notes
    -----
    Implements He et al. (2017) two-stage loop:
      • t=0: Louvain on full snapshot.
      • t>0: Division via ΔQ rule, then partition remaining nodes into modules (Louvain),
              Agglomeration on module graph (Louvain), Expand to node communities.
    """
    A = G.tensor
    N, T = G.N, G.T

    if direction not in ["in", "out", "both"]:
        raise ValueError("direction must be 'in', 'out', or 'both'")

    def A_t(t): return A[:, :, t]

    labels_t: List[np.ndarray] = []

    # ---- t=0: Louvain on full snapshot ----
    labels0 = StaticLouvain(A_t(0), threshold=louvain_threshold,
                            resolution=louvain_resolution, seed=louvain_seed)
    labels_t.append(labels0)

    # (N, N, T)
    w_pq = np.zeros_like(A)
    w_p = np.zeros((N, T))
    if direction in ["out", "both"]:
        w_pq += A  # w_pq = A[p,q]
        w_p += A.sum(axis=1)  # out-strength (rows) -> shape (N, T)
    if direction in ["in", "both"]:
        w_pq += A.transpose(1, 0, 2)  # w_pq = A[q,p]
        w_p += A.sum(axis=0)  # in-strength (cols) -> shape (N, T)

    # total weight per snapshot (sum of effective pair weights)
    w_G = w_pq.sum(axis=(0, 1))  # (T,)
    nbr_mask_all = (w_pq > 0)  # neighbor if either dir
    # never allow self as neighbor
    diag_mask = np.eye(N, dtype=bool)[:, :, None]  # (N, N, 1) broadcast over T
    nbr_mask_all &= ~diag_mask

    # guard division-by-zero using where; fill with -inf when w_G[t]==0
    w_G_safe = np.where(w_G > 0, w_G, np.nan)  # shape (T,)
    dQ_all = (w_pq / w_G_safe[None, None, :]) - (
            (w_p[:, None, :] * w_p[None, :, :]) / (2.0 * (w_G_safe[None, None, :] ** 2))
    )  # shape (N, N, T)
    dQ_all = np.where(np.isnan(dQ_all), -np.inf, dQ_all)

    # neighbor if edge exists either direction
    dQ = np.where(nbr_mask_all, dQ_all, -np.inf)  # (N, N, T)

    # best neighbor index for each p,t (argmax over q)
    best_neighbor = dQ.argmax(axis=1)  # (N, T)
    # corresponding ΔQ value (useful to test validity)
    best_dQ = dQ.max(axis=1)  # (N, T)


    # ---- t>0: stepwise detection (division → agglomeration) ----
    for t in range(1, T):
        At = A_t(t)
        prev_labels = labels_t[t - 1]

        # ---- Division stage ----
        modules: List[Set[int]] = []

        for i_prev, cid in enumerate(np.unique(prev_labels)):
            # membership mask for fast "in community" checks
            in_comm = prev_labels == cid
            C_prev = np.where(in_comm)[0]
            # best neighbor per node (restricted to this community subset)
            best_q_comm = best_neighbor[C_prev, t]  # shape (|C_prev|,)
            dq_pick = best_dQ[C_prev, t]  # ΔQ value actually used for each p

            # valid if p had any neighbor (masked row not all -inf)
            valid = np.isfinite(dq_pick)

            # among valid p's, mark those whose best neighbor is OUTSIDE the community
            outside = ~in_comm[best_q_comm]

            # nodes to remove: p in C_prev_idx where valid & outside
            to_remove_idx = C_prev[valid & outside]  # array of node ids

            # 1) add singleton modules for removed nodes
            if to_remove_idx.size:
                modules.extend([{int(v)} for v in to_remove_idx])

            # 2) partition the remaining nodes into modules
            if to_remove_idx.size < C_prev.size:
                remaining = np.setdiff1d(C_prev, to_remove_idx, assume_unique=True)
                subW = At[np.ix_(remaining, remaining)]
                sub_labels = StaticLouvain(
                    subW,
                    threshold=louvain_threshold,
                    resolution=louvain_resolution,
                    seed=louvain_seed,
                )
                for sub_cid in np.unique(sub_labels):
                    nodes = remaining[np.where(sub_labels == sub_cid)[0]]
                    modules.append(set(map(int, nodes.tolist())))

        # No modules → fall back to plain Louvain on Wt
        if not modules:
            labels = StaticLouvain(At, threshold=louvain_threshold,
                                   resolution=louvain_resolution, seed=louvain_seed)
            labels_t.append(labels)
            continue

        # ---- Agglomeration stage ----
        K = len(modules)
        # one-hot membership M (N×K)
        M = np.zeros((N, K), dtype=float)
        for k, mod in enumerate(modules):
            idx = np.fromiter(mod, dtype=int)
            M[idx, k] = 1.0

        # compressed (directed) adjacency; diagonal now = sum of intra-module weights
        H = M.T @ At @ M

        mod_labels = StaticLouvain(
            H, threshold=louvain_threshold, resolution=louvain_resolution, seed=louvain_seed
        )

        labels = np.empty(N, dtype=int)
        for k, mod in enumerate(modules):
            labels[list(mod)] = int(mod_labels[k])
        labels_t.append(labels)

    # -------------------------------------------------------------------------
    # Build per-snapshot outputs identical to TemporalLouvain
    # -------------------------------------------------------------------------
    communities: List[Dict[int, List[Any]]] = []
    node_labels: List[Dict[Any, int]] = []
    for labels in labels_t:
        node_to_label: Dict[Any, int] = {}
        label_to_nodes: Dict[int, List[Any]] = defaultdict(list)
        for i, node in enumerate(G.vertices):
            lab = int(labels[i])
            node_to_label[node] = lab
            label_to_nodes[lab].append(node)
        node_labels.append(node_to_label)
        communities.append(label_to_nodes)

    labels_matrix = np.stack(labels_t, axis=1)  # (N, T)

    return communities, node_labels, labels_matrix

# ---------- Example usage ----------
if __name__ == "__main__":
    N, T = 10, 3
    A = np.zeros((N, N, T), dtype=float)


    def add_undirected(A, u, v, w, t):
        A[u, v, t] += w
        A[v, u, t] += w


    # t=0: Two clear clusters, weak bridge 4-5
    for t in [0]:
        for u in [0, 1, 2, 3, 4]:
            for v in [0, 1, 2, 3, 4]:
                if u < v:
                    add_undirected(A, u, v, 2.0, t)
        for u in [5, 6, 7, 8, 9]:
            for v in [5, 6, 7, 8, 9]:
                if u < v:
                    add_undirected(A, u, v, 2.0, t)
        add_undirected(A, 4, 5, 0.2, t)

    # t=1: Strengthen 4's ties to C2
    A[:, :, 1] = A[:, :, 0]
    for v in [6, 7]:
        add_undirected(A, 4, v, 1.2, 1)
    for v in [0, 1]:
        A[4, v, 1] *= 0.5;
        A[v, 4, 1] *= 0.5

    # t=2: Make 5 overlap into C1, strengthen bridge
    A[:, :, 2] = A[:, :, 1]
    for v in [1, 2]:
        add_undirected(A, 5, v, 1.5, 2)
    add_undirected(A, 4, 5, 1.5, 2)

    # ------------------- Run DOMLPA -------------------
    G = SnapshotGraph(A)
    G.vertices = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    G.timestamps = ['v1', 'v2', 'v3']
    #
    # comms, nl, l = TemporalLouvain(G)
    # print(comms)
    # print(nl)
    # print(l)

    comms, nl, l = StepwiseLouvain(G)
    print(comms)
    print(nl)
    print(l)
    g = DynamicCommunities(G, l)
    print(len(g))