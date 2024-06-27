import torch
from typing import Dict

from rbm.binary.sampling import sample_hiddens, sample_visibles

def init_chains(
    num_chains : int,
    num_visibles : int,
    num_hiddens : int,
    device : torch.device
) -> Dict[str, torch.Tensor]:
    
    chains = {}
    chains["v"] = torch.randint(0, 2, size=(num_chains, num_visibles), device=device, dtype=torch.float32)
    chains["h"] = torch.randint(0, 2, size=(num_chains, num_hiddens), device=device, dtype=torch.float32)
    chains["mv"] = torch.zeros(size=(num_chains, num_visibles), device=device, dtype=torch.float32)
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
    frequencies = data.to(device).mean(0)
    frequencies = torch.clamp(frequencies, min=eps, max=(1. - eps))

    params = {}
    params["vbias"] = torch.log(frequencies) - torch.log(1. - frequencies)
    params["hbias"] = torch.zeros(num_hiddens, device=device, dtype=torch.float32)
    params["weight_matrix"] = torch.randn(size=(num_visibles, num_hiddens), device=device) * init_std
    return params


@torch.jit.script
def compute_energy(
    chains : Dict[str, torch.Tensor],
    params : Dict[str, torch.Tensor]
    ) -> torch.Tensor:
    """Returns the energy of the model computed on the input data.

    Args:
        chains (Dict[str, torch.Tensor]): Input data.
        params (Dict[str, torch.Tensor]): Parameters of the model.

    Returns:
        torch.Tensor: Energies of the data points.
    """
    fields = (chains["v"] @ params["vbias"]) + (chains["h"] @ params["hbias"])
    interaction = ((chains["v"] @ params["weight_matrix"]) * chains["h"]).sum(1)
    return - fields - interaction

@torch.jit.script
def compute_energy_visibles(
    chains : Dict[str, torch.Tensor],
    params : Dict[str, torch.Tensor]
    ) -> torch.Tensor:
    """Returns the energy of the model computed on the visible input data.

    Args:
        chains (Dict[str, torch.Tensor]): Input data.
        params (Dict[str, torch.Tensor]): Parameters of the model.

    Returns:
        torch.Tensor: Energies of the visible data points.
    """
    
    field = chains["v"] @ params["vbias"]
    exponent = params["hbias"] + (chains["v"] @ params["weight_matrix"])
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
        params (Dict[str, torch.Tensor]): Parameters of the model.
        device (device): device.

    Returns:
        float: Estimate of the log-partition function.
    """

    num_visibles, num_hiddens = params["weight_matrix"].shape
    E = torch.zeros(num_chains, device=device, dtype=torch.float64)
    beta_list = torch.linspace(0.0, 1.0, num_beta)
    dB = 1.0 / num_beta

    # initialize the chains
    vbias0 = torch.zeros(size=(num_visibles,), device=device)
    hbias0 = torch.zeros(size=(num_hiddens,), device=device)
    energy0 = torch.zeros(num_chains, device=device, dtype=torch.float64)
    v = torch.bernoulli(torch.sigmoid(vbias0)).repeat(num_chains, 1)
    chains = {"v" : v}
    chains["mh"] = torch.sigmoid(hbias0).repeat(num_chains, 1)
    chains["h"] = torch.bernoulli(chains["mv"])

    energy1 = compute_energy(chains, params)
    E += energy1 - energy0
    for beta in beta_list:
        chains = sample_hiddens(chains=chains, params=params, beta=beta)
        chains = sample_visibles(chains=chains, params=params, beta=beta)
        E += compute_energy(chains, params).type(torch.float64)

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
        data (Dict[str, torch.Tensor]): Input data.
        params (Dict[str, torch.Tensor]): Parameters of the model.
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