from typing import Dict, Tuple
import torch

from torchrbm.potts.grad import compute_gradient
from torchrbm.potts.sampling import sample_hiddens, sample_state


def update_parameters(
    data : dict[str, torch.Tensor],
    chains : dict[str, torch.Tensor],
    params : dict[str, torch.Tensor],
    lr : float,
    centered : bool
    ) -> dict[str, torch.Tensor]:
    """Computes the gradient of the log-likelihood and updates the parameters of the model. Implements the centered version of the gradient,
    which normally improveds the quality of the learning.

    Args:
        data (Tuple[Tensor, Tensor, Tensor]): (v, h, w_data) Observed data.
        parallel_chains (Tuple[Tensor, Tensor]): (v, h) Parallel chains.
        params (Tuple[Tensor, Tensor, Tensor]): (vbias, hbias, weight_matrix) Parameters of the model.
        learning_rate (float): Learning rate.
        device (torch.device): Device.

    Returns:
        Tensor: Updated parameters.
    """
    # Compute the gradient of the log-likelihood
    grad = compute_gradient(data=data, chains=chains, params=params, centered=centered)
    
    # Update the parameters
    params["vbias"] += lr * grad["vbias"]
    params["hbias"] += lr * grad["hbias"]
    params["weight_matrix"] += lr * grad["weight_matrix"]
    
    # Zero-sum gauge
    params["weight_matrix"] -= params["weight_matrix"].mean(1, keepdim=True)
    return params

#@torch.jit.script
def fit_batch(
    data : Dict[str, torch.Tensor],
    chains : Dict[str, torch.Tensor],
    params : Dict[str, torch.Tensor],
    gibbs_steps : int,
    lr : float,
    centered : bool
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Takes a batch of data and updates the parameters of the model.

    Args:
        batch (Tuple[Tensor, Tensor]): (v, w_data) Data batch.
        parallel_chains (Tuple[Tensor, Tensor]): (v, h) Current state of the model.
        params (Tuple[Tensor, Tensor, Tensor]): (vbias, hbias, weight_matrix) Parameters of the model.
        gibbs_steps (int): Number of Monte Carlo updates.
        learning_rate (float): Learning rate.
        device (torch.device): Device.

    Returns:
        Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]: Updated parallel_chains and parameters.
    """

    # Get the hidden activation given by the data
    data = sample_hiddens(chains=data, params=params)

    # Update the parameters
    params = update_parameters(data=data, chains=chains, params=params, lr=lr, centered=centered)
    
    # Update the state of the model
    chains = sample_state(chains=chains, params=params, gibbs_steps=gibbs_steps)
    
    return chains, params