import numpy as np

__all__ = ["LiuCentralityFunction", "HubCentralityFunction", "AuthorityCentralityFunction", "PageRankCentralityFunction"]

def HubCentralityFunction(A):
    return A@A.T


def AuthorityCentralityFunction(A):
    return A.T@A


def PageRankCentralityFunction(A, p):
    """
    Compute the PageRank matrix (Google matrix) given an adjacency matrix A.

    Parameters:
    - A: numpy.ndarray
        The adjacency matrix (NxN), where A[i, j] means a link from node j to node i.
    - p: float
        The teleportation parameter.

    Returns:
    - G: numpy.ndarray
        The PageRank (Google) matrix of size (N, N).
    """

    # Ensure column stochastic matrix (normalize columns)
    column_sums = A.sum(axis=0)
    N = column_sums.size
    column_sums[column_sums == 0] = 1  # Avoid division by zero (dangling nodes)
    M = A / column_sums  # Normalize columns (M is the transition matrix)

    # Google matrix formula
    G = p * M + (1 - p) / N * np.ones((N, N))

    return G

def LiuCentralityFunction(A):
    # Compute degrees (count non-zero elements in each row)
    d = np.count_nonzero(A, axis=1)  # Shape (N,)

    # Compute W using broadcasting
    W = d[:, None] + d[None, :]  # Outer sum of degrees
    # Set values to zero where A is zero
    W[A == 0] = 0
    return W
