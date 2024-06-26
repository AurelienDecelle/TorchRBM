import torch
from typing import Optional, Union, List, Any, Tuple, Dict

from torchrbm.custom_fn import one_hot
from torchrbm.potts.sampling import sample_hiddens, sample_visibles

def init_chains(
    num_chains : int,
    num_visibles : int,
    num_hiddens : int,
    num_colors : int,
    device : torch.device
) -> Dict[str, torch.Tensor]:
    
    chains = {}
    chains["v"] = torch.randint(0, num_colors, size=(num_chains, num_visibles), device=device, dtype=torch.int32)
    chains["h"] = torch.randint(0, 2, size=(num_chains, num_hiddens), device=device, dtype=torch.float32)
    chains["mv"] = torch.zeros(size=(num_chains, num_visibles, num_colors), device=device, dtype=torch.float32)
    chains["mh"] = torch.zeros(size=(num_chains, num_hiddens), device=device, dtype=torch.float32)
    return chains

def init_parameters(
    data : torch.Tensor,
    num_visibles : int,
    num_hiddens : int,
    device : torch.device
) -> Dict[str, torch.Tensor]:
    
    eps = 1e-4
    init_std = 1e-4
    num_visibles = data.shape[1]
    num_colors = torch.max(data) + 1
    dataset_oh = one_hot(data, num_classes=num_colors).to(device)
    frequencies = dataset_oh.mean(0)
    frequencies = torch.clamp(frequencies, min=eps, max=(1. - eps))

    params = {}
    params["vbias"] = torch.log(frequencies) - 1. / num_colors * torch.sum(torch.log(frequencies), 0)
    params["hbias"] = torch.zeros(num_hiddens, device=device, dtype=torch.float32)
    params["weight_matrix"] = torch.randn(size=(num_visibles, num_colors, num_hiddens), device=device) * init_std
    return params


@torch.jit.script
def compute_energy(
    chains : Dict[str, torch.Tensor],
    params : Dict[str, torch.Tensor]
    ) -> torch.Tensor:
    """Returns the energy of the model computed on the input data.

    Args:
        v (Tensor): Visible data.
        h (Tensor): Hidden data.
        vbias (Tensor): Visible bias.
        hbias (Tensor): Hidden bias.
        weight_matrix (Tensor): Weight matrix.

    Returns:
        Tensor: Energies of the data points.
    """
    num_visibles, num_states, num_hiddens = params["weight_matrix"].shape
    v_oh = one_hot(chains["v"], num_classes=num_states).reshape(-1, num_visibles * num_states)
    vbias_oh = params["vbias"].flatten()
    weight_matrix_oh = params["weight_matrix"].reshape(num_visibles * num_states, num_hiddens)
    fields = (v_oh @ vbias_oh) + (chains["h"] @ chains["hbias"])
    interaction = ((v_oh @ weight_matrix_oh) * chains["h"]).sum(1)
    return - fields - interaction

@torch.jit.script
def compute_energy_visibles(
    chains : Dict[str, torch.Tensor],
    params : Dict[str, torch.Tensor]
    ) -> torch.Tensor:
    """Returns the energy of the model computed on the input data.

    Args:
        v (Tensor): Visible data.
        params (Tuple[Tensor, Tensor, Tensor]): (vbias, hbias, weight_matrix) Parameters of the model.

    Returns:
        Tensor: Energies of the data points.
    """
    num_visibles, num_states, num_hiddens = params["weight_matrix"].shape
    v_oh = one_hot(chains["v"], num_classes=num_states).reshape(-1, num_visibles * num_states)
    vbias_oh = params["vbias"].flatten()
    weight_matrix_oh = params["weight_matrix"].reshape(num_visibles * num_states, num_hiddens)
    field = v_oh @ vbias_oh
    exponent = params["hbias"] + (v_oh @ weight_matrix_oh)
    log_term = torch.where(exponent < 10, torch.log(1. + torch.exp(exponent)), exponent)
    return - field - log_term.sum(1)


def compute_partition_function_ais(
    num_chains: int,
    num_beta: int,
    params: Dict[str, torch.Tensor],
    device: torch.device,
) -> float:
    """Estimates the partition function of the model using Annealed Importance Sampling.

    Args:
        num_chains (int): Number of parallel chains.
        num_beta (int): Number of inverse temperatures that define the trajectories.
        params (Tuple[Tensor, Tensor, Tensor])): (vbias, hbias, weight_matrix) Parameters of the model.
        device (device): device.

    Returns:
        float: Estimate of the log-partition function.
    """

    num_visibles, num_states, num_hiddens = params["weight_matrix"].shape
    E = torch.zeros(num_chains, device=device, dtype=torch.float64)
    beta_list = torch.linspace(0.0, 1.0, num_beta)
    dB = 1.0 / num_beta

    # initialize the chains
    hbias0 = torch.zeros(size=(num_hiddens,), device=device)
    energy0 = torch.zeros(num_chains, device=device, dtype=torch.float64)
    v = torch.randint(0, num_states, size=(1, num_visibles), device=device).repeat(num_chains, 1)
    chains = init_chains(num_samples=num_chains, params=params, start_v=v)
    chains.mean_hidden = torch.sigmoid(hbias0).repeat(num_chains, 1)
    chains.hidden = torch.bernoulli(chains.mean_hidden)

    energy1 = compute_energy(chains.visible, chains.hidden, params)
    E += energy1 - energy0
    for beta in beta_list:
        chains = sample_hiddens(chains=chains, params=params, beta=beta)
        chains = sample_visibles(chains=chains, params=params, beta=beta)
        E += compute_energy(chains.visible, chains.hidden, params)

    # Subtract the average for avoiding overflow
    W = -dB * E
    W_ave = W.mean()
    logZ0 = (num_visibles + num_hiddens) * torch.log(2)
    logZ = logZ0 + torch.log(torch.mean(torch.exp(W - W_ave))) + W_ave
    return logZ


def compute_log_likelihood_ais(
    data : Dict[str, torch.Tensor],
    params: Dict[str, torch.Tensor],
    num_chains: int,
    num_beta: int,
    device: torch.device,
) -> float:
    """Estimates the log-likelihood of the model using Annealed Importance Sampling (AIS).

    Args:
        v_data (Tensor): Visible data.
        params (Tuple[Tensor, Tensor, Tensor])): (vbias, hbias, weight_matrix) Parameters of the model.
        num_chains (int): Number of parallel chains.
        num_beta (int): Number of inverse temperatures that define the trajectories.
        device (device): device.

    Returns:
        float: Estimate of the log-likelihood.
    """
    return (
        - compute_partition_function_ais(num_chains, num_beta, params, device)
        - compute_energy_visibles(chains=data, params=params).mean()
    )