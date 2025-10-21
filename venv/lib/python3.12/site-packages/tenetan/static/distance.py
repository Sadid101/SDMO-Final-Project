import numpy as np

from .linalg import NormalizedLaplacian, Laplacian

__all__ = ["DeltaCon", "SpectralDistance", "NormalizedSpectralDistance", "GraphEditDistance"]

def DeltaCon(A1: np.ndarray, A2: np.ndarray, direction: str = "in") -> np.floating:
    """
    DeltaCon graph distance, as defined in:
    Koutra, D., Shah, N., Vogelstein, J. T., Gallagher, B., & Faloutsos, C. (2016)
    Deltacon: Principled massive-graph similarity function with attribution.
    ACM Transactions on Knowledge Discovery from Data (TKDD), 10(3), 1-43.

    Parameters
    ----------
    A1,A2: np.ndarray
        Adjacency (weight) matrices. Must be of same shape.
    direction: str
        Whether to compute in- or out-degree. If "in" (default), uses in-degree,
        otherwise out-degree.

    Returns
    -------
    np.float: DeltaCon distance
    """
    D1 = np.diag(A1.sum(axis=0 if direction=="in" else 1))
    D2 = np.diag(A2.sum(axis=0 if direction=="in" else 1))
    N = A1.shape[0]

    eps_1 = 1 / (1 + np.max(D1))
    eps_2 = 1 / (1 + np.max(D2))

    S1 = np.linalg.inv(np.eye(N) + (eps_1 ** 2) * D1 - eps_1 * A1)
    S2 = np.linalg.inv(np.eye(N) + (eps_2 ** 2) * D2 - eps_2 * A2)

    # Matusita Distance
    return np.sqrt(np.sum(np.square(np.sqrt(S1) - np.sqrt(S2))))


def SpectralDistance(A1: np.ndarray, A2: np.ndarray, direction: str = "in",
                     normalized_laplacian: bool = False,
                     n_eig: int = None) -> np.floating:
    """
    Spectral distance of the graph, given by the L2-norm of the vector
    of eigenvalue differences of the Laplacian matrices of the networks.

    Parameters
    ----------
    A1,A2: np.ndarray
        Adjacency (weight) matrices. Must be of same shape.
    direction: str
        Whether to compute in- or out-degree. If "in" (default), uses in-degree,
        otherwise out-degree.
    normalized_laplacian: bool
        If True, uses Normalized Laplacian (tenetan.static.linalg.NormalizedLaplacian)
        for eigenvalue computation, otherwise regular Laplacian
        (tenetan.static.linalg.Laplacian). Default False.
    n_eig: int
        Number of eigenvalues to use, starting from the leading eigenvalue.
        If None (default), uses all eigenvalues, i.e., the size of A1.

    Returns
    -------
    dist: np.float, spectral distance
    """
    if n_eig is None:
        n_eig = A1.shape[0]
    L = NormalizedLaplacian if normalized_laplacian else Laplacian
    eig1 = np.linalg.eigvals(L(A1, direction=direction))
    eig1.sort()
    eig2 = np.linalg.eigvals(L(A2, direction=direction))
    eig2.sort()
    eig1 = np.flip(eig1)
    eig2 = np.flip(eig2)
    eig1 = eig1[:n_eig]
    eig2 = eig2[:n_eig]
    return np.linalg.norm(eig1-eig2)


def NormalizedSpectralDistance(A1: np.ndarray, A2: np.ndarray, direction: str = "in",
                               normalized_laplacian: bool = False,
                               n_eig: int = None) -> np.floating:
    # TODO Add switch min/max
    """
    Normalized spectral distance of the graph.
    Same as Spectral Distance, but the value is normalized by sqrt() of minimum (maximum)
    sum of the square of the eigenvalues of either Laplacian.

    Parameters
    ----------
    A1,A2: np.ndarray
        Adjacency (weight) matrices. Must be of same shape.
    direction: str
        Whether to compute in- or out-degree. If "in" (default), uses in-degree,
        otherwise out-degree.
    normalized_laplacian: bool
        If True, uses Normalized Laplacian (tenetan.static.linalg.NormalizedLaplacian)
        for eigenvalue computation, otherwise regular Laplacian
        (tenetan.static.linalg.Laplacian). Default False.
    n_eig: int
        Number of eigenvalues to use, starting from the leading eigenvalue.
        If None (default), uses all eigenvalues, i.e., the size of A1.

    Returns
    -------
    dist: np.float, spectral distance
    """
    if n_eig is None:
        n_eig = A1.shape[0]
    L = NormalizedLaplacian if normalized_laplacian else Laplacian
    eig1 = np.linalg.eigvals(L(A1, direction=direction))
    eig1.sort()
    eig2 = np.linalg.eigvals(L(A2, direction=direction))
    eig2.sort()
    eig1 = np.flip(eig1)
    eig2 = np.flip(eig2)
    eig1 = eig1[:n_eig]
    eig2 = eig2[:n_eig]
    m1 = np.sum(np.square(eig1))
    m2 = np.sum(np.square(eig2))
    m = max(m1, m2)
    return np.linalg.norm(eig1-eig2)/np.sqrt(m)


def GraphEditDistance(A1: np.ndarray, A2: np.ndarray) -> int:
    """
    Graph edit distance between graph adjacency matrices A1 and A2, given bu
    N(A1) +N(A2) − 2N(A1∩A2) + E(A1) + E(A2) − 2E(A1∩A2),
    where N(A) is the number of nodes and E(A) is the number of edges in the graph

    Parameters
    ----------
    A1,A2: np.ndarray
        Adjacency (weight) matrices. Must be of same shape.

    Returns
    -------
    dist: graph edit distance
    """

    # Nodes with any in or out edge (non-zero rows or cols)
    nodes_A = np.any(A1 != 0, axis=0) | np.any(A1 != 0, axis=1)
    nodes_B = np.any(A2 != 0, axis=0) | np.any(A2 != 0, axis=1)

    N_A = np.count_nonzero(nodes_A)
    N_B = np.count_nonzero(nodes_B)
    N_common = np.count_nonzero(np.logical_and(nodes_A, nodes_B))

    # Edge counts = number of non-zero entries
    E_A = np.count_nonzero(A1)
    E_B = np.count_nonzero(A2)
    E_common = np.count_nonzero(np.logical_and(A1 != 0, A2 != 0))

    GED = (N_A + N_B - 2 * N_common) + (E_A + E_B - 2 * E_common)
    return int(GED)
