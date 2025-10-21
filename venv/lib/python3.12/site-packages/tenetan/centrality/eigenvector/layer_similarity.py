import numpy as np
import statsmodels.api as sm

__all__ = ["LiuLayerSimilarity", "YinLayerSimilarity", "HuangLayerSimilarity"]


def YinLayerSimilarity(snapshot):
    N = snapshot.shape[0]
    T = snapshot.shape[2]
    NT = N*T

    # Compute C[j, t] for all j and t
    C_j_t = np.sum(snapshot[:, :, :-1] * snapshot[:, :, 1:], axis=0)  # Shape (N, T-1)
    # Initialize the big NT x NT matrix
    inter_layer_similarity = np.zeros((NT, NT))

    # Place C_j_t at off-diagonal (t, t+1) and (t+1, t) positions
    for t in range(T - 1):
        start_t = t * N
        start_t1 = (t + 1) * N

        inter_layer_similarity[start_t:start_t + N, start_t1:start_t1 + N] = np.diag(C_j_t[:, t])  # t to t+1
        inter_layer_similarity[start_t1:start_t1 + N, start_t:start_t + N] = np.diag(C_j_t[:, t])  # t+1 to t

    return inter_layer_similarity


def LiuLayerSimilarity(snapshot):
    N = snapshot.shape[0]
    T = snapshot.shape[2]
    NT = N*T
    # Compute numerator: sum over j of A[i,j,t] * A[i,j,t+1]
    numerator = np.sum(snapshot[:, :, :-1] * snapshot[:, :, 1:], axis=1)  # Shape (N, T-1)

    # Compute denominators separately
    denominator_t1 = np.sum(snapshot[:, :, 1:], axis=1)  # Sum over j for A[i,j,t+1] (Shape: N, T-1)
    denominator_t = np.sum(snapshot[:, :, :-1], axis=1)  # Sum over j for A[i,j,t] (Shape: N, T-1)

    # Avoid division by zero
    denominator_t1 = np.where(denominator_t1 == 0, 1e-10, denominator_t1)
    denominator_t = np.where(denominator_t == 0, 1e-10, denominator_t)

    # Compute S matrices
    S_t_t1 = numerator / denominator_t1  # S_{i,t,t+1} uses denominator of t+1
    S_t1_t = numerator / denominator_t  # S_{i,t+1,t} uses denominator of t

    # Initialize the big NT x NT matrix
    inter_layer_similarity = np.zeros((NT, NT))

    # Place S_t_t1 at (t, t+1) and S_t1_t at (t+1, t)
    for t in range(T - 1):
        start_t = t * N
        start_t1 = (t + 1) * N

        inter_layer_similarity[start_t:start_t + N, start_t1:start_t1 + N] = np.diag(S_t_t1[:, t])  # t → t+1
        inter_layer_similarity[start_t1:start_t1 + N, start_t:start_t + N] = np.diag(S_t1_t[:, t])  # t+1 → t
    return inter_layer_similarity


def HuangLayerSimilarity(snapshot):
    """
    Fits an ARMA (ARIMA with d=0) model to the degree sequence of each node from a series of adjacency matrices
    and constructs a block matrix where each block W[t, t-i] is a diagonal matrix containing
    the i-th coefficient of the ARMA model for each node at length t.

    Parameters:
    - adj_matrices: numpy.ndarray (NxNxT)
        A sequence of T adjacency matrices of size NxN.

    Returns:
    - inter_layer_similarity: numpy.ndarray
        The (N*T, N*T) block matrix with ARMA coefficients.
    """
    N, _, T = snapshot.shape  # Extract dimensions

    # Compute degree sequences for each node over time
    degree_sequences = np.sum(snapshot, axis=1)  # Shape: (N, T), summing over columns (node degrees)

    # Dictionary to store ARMA models for each node
    arma_models = {i: [] for i in range(N)}

    # Fit ARMA (via ARIMA with d=0) models for each node
    for i in range(N):  # Iterate over nodes
        for end_t in range(1, T + 1):  # From first step up to full sequence
            time_series = degree_sequences[i, :end_t]  # Extract sequence for node i up to end_t

            if len(time_series) > 1:  # ARMA needs at least two points
                model_order = (end_t - 1, 0, end_t - 1)  # (p, d, q) with d=0 (ARMA equivalent)
                try:
                    model = sm.tsa.ARIMA(time_series, order=model_order).fit()
                    arma_models[i].append(model)  # Store the model
                except Exception as e:
                    print(f"ARIMA (ARMA equivalent) failed for node {i} with sequence length {end_t}: {e}")
                    arma_models[i].append(None)  # Append None if model fitting fails

    # Initialize the (NT x NT) block matrix
    inter_layer_similarity = np.zeros((N * T, N * T))

    # Construct the block matrix using AR coefficients
    for t in range(1, T):  # Start from t=1 because t=0 has no previous coefficients
        for i in range(1, t + 1):  # Iterate over possible lags (t-i)
            start_t = t * N
            start_ti = (t - i) * N  # Previous time step block

            # Extract AR coefficients for time length t
            diag_entries = np.zeros(N)  # Default to zero if no coefficient available

            for node in range(N):
                if len(arma_models[node]) >= t and arma_models[node][t-1] is not None:
                    ar_coeffs = arma_models[node][t-1].arparams  # Extract AR coefficients
                    if len(ar_coeffs) >= i:
                        diag_entries[node] = ar_coeffs[i-1]  # Get the i-th coefficient

            # Fill diagonal block W[t, t-i] with the extracted coefficients
            inter_layer_similarity[start_t:start_t+N, start_ti:start_ti+N] = np.diag(diag_entries)

    return inter_layer_similarity
