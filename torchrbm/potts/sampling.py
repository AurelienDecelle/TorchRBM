
from typing import Dict
import torch

from torchrbm.custom_fn import one_hot


def sample_hiddens(
    chains : Dict[str, torch.Tensor],
    params : Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
    """Samples the hidden layer conditioned on the state of the visible layer.

    Args:
        v (Tensor): Visible layer.
        hbias (Tensor): Hidden bias.
        weight_matrix (Tensor): Weight matrix.

    Returns:
        Tuple[Tensor, Tensor]: Hidden units and magnetizations.
    """
    num_visibles, num_states, num_hiddens = params["weight_matrix"].shape
    weight_matrix_oh = params["weight_matrix"].reshape(num_visibles * num_states, num_hiddens)
    v_oh = one_hot(chains["v"], num_classes=num_states).reshape(-1, num_visibles * num_states)
    chains["mh"] = torch.sigmoid(params["hbias"] + v_oh @ weight_matrix_oh)
    chains["h"] = torch.bernoulli(chains["mh"])
    return chains


def sample_visibles(
    chains : Dict[str, torch.Tensor],
    params : Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
    """Samples the visible layer conditioned on the hidden layer.

    Args:
        h (Tensor): Hidden layer.
        vbias (Tensor): Visible bias.
        weight_matrix (Tensor): Weight matrix.

    Returns:
        Tensor: Visible units.
    """
    num_visibles, num_states, _ = params["weight_matrix"].shape
    chains["mv"] = torch.softmax(params["vbias"] + torch.tensordot(chains["h"], params["weight_matrix"], dims=[[1], [2]]), dim=-1)
    chains["v"] = torch.multinomial(chains["mv"].reshape(-1, num_states), 1).reshape(-1, num_visibles)
    return chains


@torch.jit.script
def sample_state(
    chains : Dict[str, torch.Tensor],
    params : Dict[str, torch.Tensor],
    gibbs_steps : int
    ) -> Dict[str, torch.Tensor]:
    """Generates data sampled from the model by performing "gibbs_steps" Monte Carlo updates.

    Args:
        parallel_chains (Tuple[Tensor, Tensor]): (v, h) Initial visible state.
        params (Tuple[Tensor, Tensor, Tensor]): (vbias, hbias, weight_matrix) Parameters of the model.
        gibbs_steps (int): Number of Monte Carlo updates.

    Returns:
        Tuple[Tensor, Tensor]: Updated parallel chains.
    """
    for _ in range(gibbs_steps):
        chains = sample_hiddens(chains=chains, params=params)
        chains = sample_visibles(chains=chains, params=params)
    return chains