
from typing import Dict
import torch

from rbm.custom_fn import one_hot

@torch.jit.script
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
    num_visibles, num_states, num_hiddens = params["weight_matrix"].shape
    weight_matrix_oh = params["weight_matrix"].reshape(num_visibles * num_states, num_hiddens)
    v_oh = one_hot(chains["v"], num_classes=num_states).reshape(-1, num_visibles * num_states)
    chains["mh"] = torch.sigmoid(beta * (params["hbias"] + v_oh @ weight_matrix_oh))
    chains["h"] = torch.bernoulli(chains["mh"])
    return chains

@torch.jit.script
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
    num_visibles, num_states, _ = params["weight_matrix"].shape
    chains["mv"] = torch.softmax(beta * (params["vbias"] + torch.tensordot(chains["h"], params["weight_matrix"], dims=[[1], [2]])), dim=-1)
    chains["v"] = torch.multinomial(chains["mv"].reshape(-1, num_states), 1).reshape(-1, num_visibles)
    return chains


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