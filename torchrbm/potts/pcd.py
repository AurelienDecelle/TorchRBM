from typing import Dict, Tuple
import torch

from torchrbm.potts.grad import compute_gradient
from torchrbm.potts.sampling import sample_hiddens, sample_state


def update_parameters(
    data : dict[str, torch.Tensor],
    chains : dict[str, torch.Tensor],
    params : dict[str, torch.Tensor],
    lr : float,
    centered : bool=True
    ) -> Dict[str, torch.Tensor]:
    """Computes the gradient of the log-likelihood and updates the parameters of the model.

    Args:
        data (Dict[str, torch.Tensor]): Observed data.
        chains (Dict[str, torch.Tensor]): Monte Carlo chains.
        params (Dict[str, torch.Tensor]): Parameters of the model.
        lr (float): Learning rate.
        centered (bool, optional): Whether to center the gradient or not. Defaults to True.

    Returns:
        Dict[str, torch.Tensor]: Updated parameters.
    """
    num_colors = params["weight_matrix"].shape[1]
    # Compute the gradient of the log-likelihood
    grad = compute_gradient(data=data, chains=chains, num_colors=num_colors, centered=centered)
    
    # Update the parameters
    params["vbias"] += lr * grad["vbias"]
    params["hbias"] += lr * grad["hbias"]
    params["weight_matrix"] += lr * grad["weight_matrix"]
    
    # Zero-sum gauge
    params["weight_matrix"] -= params["weight_matrix"].mean(1, keepdim=True)
    return params

@torch.jit.script
def fit_batch(
    data : Dict[str, torch.Tensor],
    chains : Dict[str, torch.Tensor],
    params : Dict[str, torch.Tensor],
    gibbs_steps : int,
    lr : float,
    centered : bool=True
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Takes a batch of data and updates the parameters of the model and the Monte Carlo chains.

    Args:
        data (Dict[str, torch.Tensor]): Observed data.
        chains (Dict[str, torch.Tensor]): Monte Carlo chains.
        params (Dict[str, torch.Tensor]): Parameters of the model.
        gibbs_steps (int): Number of Monte Carlo updates to be performed.
        lr (float): Learning rate.
        centered (bool, optional): Whether to center the gradient or not. Defaults to True.

    Returns:
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]: Updated parallel_chains and parameters.
    """

    # Get the hidden activation given by the data
    data = sample_hiddens(chains=data, params=params)
    
    # Update the state of the model
    chains = sample_state(chains=chains, params=params, gibbs_steps=gibbs_steps)

    # Update the parameters
    params = update_parameters(data=data, chains=chains, params=params, lr=lr, centered=centered)
    
    return chains, params