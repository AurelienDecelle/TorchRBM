import torch


def get_freq_single_point(data : torch.Tensor, weights : torch.Tensor, pseudo_count : float) -> torch.Tensor:
    """Computes the single point frequences of the input MSA.

    Args:
        data (Array): Input one-hot encoded matrix (M, L, q).
        weights (Array): Array of weights to assign to the sequences.
        pseudo_count (float): Pseudo count for the single and two points statistics. Acts as a regularization.

    Returns:
        Array: Array of single-point frequences (L, q).
    """
    M, _, q = data.shape
    if weights is not None:
        norm_weights = weights.reshape(M, 1, 1) / weights.sum()
    else:
        norm_weights = torch.ones(size=(M, 1, 1), device=data.device) / M
    frequences = torch.clamp((data * norm_weights).sum(0), min=1e-6, max=1. - 1e-6)
    return (1. - pseudo_count) * frequences + (pseudo_count / q)


def get_freq_two_points(data : torch.Tensor, weights : torch.Tensor=None, pseudo_count : float=0.) -> torch.Tensor:
    """Computes the 2-points statistics of the input MSA.

    Args:
        data (Array): Input one-hot encoded matrix (M, L, q).
        weights (Array): Array of weights to assign to the sequences.
        pseudo_count (float): Pseudo count for the single and two points statistics. Acts as a regularization.

    Returns:
        Array: Two-point frequences matrix of the MSA (L, q, L, q).
    """
    M, L, q = data.shape
    data_oh = data.reshape(M, q * L)
    if weights is not None:
        norm_weights = weights.reshape(M, 1) / weights.sum()
    else:
        norm_weights = torch.ones(size=(M, 1), device=data.device) / M
        
    fij = ((data_oh * norm_weights).T @ data_oh)
    fij = torch.clamp(fij.reshape(L, q, L, q), min=1e-6, max=1. - 1e-6)
    # Set the auto-correlations to zero
    fij[torch.arange(L), :, torch.arange(L), :] = torch.zeros(size=(q, q), device=data.device)
    # Apply the pseudo-count
    fij = fij.reshape(L * q, L * q)
    fij = (1. - pseudo_count) * fij + (pseudo_count / q**2)
    # Diagonal term
    frequences = get_freq_single_point(data=data, weights=weights, pseudo_count=pseudo_count).flatten()
    fij_diag = (1. - pseudo_count) * frequences + (pseudo_count / q)
    fij[torch.arange(L * q), torch.arange(L * q)] = fij_diag
    fij = fij.reshape(L, q, L, q)
    return fij


def get_correlation_two_points(fij : torch.Tensor, pij: torch.Tensor, fi : torch.Tensor, pi : torch.Tensor) -> torch.Tensor:
    """Computes the Pearson coefficient between the two-point frequencies of data and chains.

    Args:
        fij (Array): Two-point frequencies of the data, shape=(L, q, L, q).
        pij (Array): Two-point frequencies of the chains, shape=(L, q, L, q).
        fi (Array): Single-point frequencies of the data, shape=(L, q).
        pi (Array): Single-point frequencies of the chains, shape=(L, q).

    Returns:
        Array: Pearson correlation coefficient.
    """
    L, q = fij.shape[0], fij.shape[1]
    # Compute the covariance matrices
    cov_data = fij - torch.tensordot(fi.reshape(L, q, 1), fi.reshape(1, L, q), dims=[[-1], [0]])
    cov_chains = pij - torch.tensordot(pi.reshape(L, q, 1), pi.reshape(1, L, q), dims=[[-1], [0]])
    # Extract only the entries of half the matrix and out of the diagonal blocks
    row_extract, col_extract = torch.tril_indices(L, L, offset=-1)
    fij_extract = cov_data[row_extract, :, col_extract, :].flatten()
    pij_extract = cov_chains[row_extract, :, col_extract, :].flatten()
    return torch.corrcoef(torch.vstack([fij_extract, pij_extract]))[0, 1]