import numpy as np

__all__ = ["Laplacian", "NormalizedLaplacian"]

def Laplacian(A: np.ndarray, direction:str = "in") -> np.ndarray:
    """
    Laplacian matrix of the graph defined by adjacency (weight) matrix A.

    L = D - A, where D is the diagonal matrix of degree and A is the adjacency matrix.

    Parameters
    ----------
    A: np.ndarray
        Adjacency (weight) matrix.
    direction: str
        Whether to compute in- or out-degree. If "in" (default), uses in-degree,
        otherwise out-degree.

    Returns
    -------
    L: np.ndarray
        Laplacian matrix
    """
    D = A.sum(axis=0 if direction=="in" else 1)
    return  np.diag(D) - A


def NormalizedLaplacian(A: np.ndarray, direction: str = "in") -> np.ndarray:
    """
    Normalized Laplacian matrix of the graph defined by adjacency (weight) matrix A.

    Ln = D^{-1/2}LD^{-1/2}, where D is the diagonal matrix of degree and L is the regular Laplacian.

    Parameters
    ----------
    A: np.ndarray
        Adjacency (weight) matrix.
    direction: str
        Whether to compute in- or out-degree. If "in" (default), uses in-degree,
        otherwise out-degree.

    Returns
    -------
    Ln: np.ndarray
        Normalized Laplacian matrix
    """
    # Degree vector
    D = A.sum(axis=0 if direction=="in" else 1)
    N = A.shape[0]

    # Inverse sqrt of degrees, avoid division by zero
    with np.errstate(divide='ignore'):
        inv_sqrt_deg = 1.0 / np.sqrt(D)
    inv_sqrt_deg[np.isinf(inv_sqrt_deg)] = 0.0  # replace inf with 0 for isolated nodes

    # Diagonal matrix of inverse sqrt degrees
    D_inv_sqrt = np.diag(inv_sqrt_deg)

    # Normalized Laplacian
    return np.eye(N) - D_inv_sqrt @ A @ D_inv_sqrt
