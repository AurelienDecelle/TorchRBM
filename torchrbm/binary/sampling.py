
from typing import Dict
import torch

from torchrbm.custom_fn import one_hot


def sample_hiddens(
    chains : Dict[str, torch.Tensor],
    params : Dict[str, torch.Tensor],
    beta : float=1.
    ) -> Dict[str, torch.Tensor]:
    """Samples the hidden layer conditioned on the state of the visible layer.

    Args:
        chains (Dict[str, torch.Tensor]): Monte Carlo chains.
        params (Dict[str, torch.Tensor]): Parameters of the model.
        beta (float, optional): Inverse temperature. Defaults to 1.

    Returns:
        Dict[str, torch.Tensor]: Updated chains.
    """
    chains["mh"] = torch.sigmoid(beta * (params["hbias"] + chains["v"] @ params["weight_matrix"]))
    chains["h"] = torch.bernoulli(chains["mh"])
    return chains


def sample_visibles(
    chains : Dict[str, torch.Tensor],
    params : Dict[str, torch.Tensor],
    beta : float=1.
    ) -> Dict[str, torch.Tensor]:
    """Samples the visible layer conditioned on the hidden layer.

    Args:
        chains (Dict[str, torch.Tensor]): Monte Carlo chains.
        params (Dict[str, torch.Tensor]): Parameters of the model.
        beta (float, optional): Inverse temperature. Defaults to 1.

    Returns:
        Dict[str, torch.Tensor]: Updated chains.
    """
    chains["mv"] = torch.sigmoid(beta * (params["vbias"] + chains["h"] @ params["weight_matrix"].T))
    chains["v"] = torch.bernoulli(chains["mv"])
    return chains


@torch.jit.script
def sample_state(
    chains : Dict[str, torch.Tensor],
    params : Dict[str, torch.Tensor],
    gibbs_steps : int,
    beta : float=1.
    ) -> Dict[str, torch.Tensor]:
    """Generates data sampled from the model by performing "gibbs_steps" Monte Carlo updates.

    Args:
        Args:
        chains (Dict[str, torch.Tensor]): Monte Carlo chains.
        params (Dict[str, torch.Tensor]): Parameters of the model.
        gibbs_steps (int): Number of Monte Carlo updates.
        beta (float, optional): Inverse temperature. Defaults to 1.

    Returns:
        Dict[str, torch.Tensor]: Updated chains.
    """
    for _ in range(gibbs_steps):
        chains = sample_hiddens(chains=chains, params=params, beta=beta)
        chains = sample_visibles(chains=chains, params=params, beta=beta)
    return chains