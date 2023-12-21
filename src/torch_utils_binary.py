import torch
import numpy as np
from typing import Optional, Union, List, Any, Tuple

Tensor = torch.Tensor

@torch.jit.script
def sample_hiddens(v : Tensor, hbias : Tensor, weight_matrix : Tensor, beta : float=1.) -> Tuple[Tensor, Tensor]:
    """Samples the hidden layer conditioned on the state of the visible layer.

    Args:
        v (Tensor): Visible layer.
        hbias (Tensor): Hidden bias.
        weight_matrix (Tensor): Weight matrix.
        beta (float, optional): Inverse temperature. Defaults to 1..

    Returns:
        Tuple[Tensor, Tensor]: Hidden units, hidden magnetizations.
    """
    mh = torch.sigmoid(beta * (hbias + (v @ weight_matrix)))
    h = torch.bernoulli(mh)
    return (h, mh)

@torch.jit.script
def sample_visibles(h : Tensor, vbias : Tensor, weight_matrix : Tensor, beta : float=1.) -> Tuple[Tensor, Tensor]:
    """Samples the visible layer conditioned on the hidden layer.

    Args:
        h (Tensor): Hidden layer.
        vbias (Tensor): Visible bias.
        weight_matrix (Tensor): Weight matrix.
        beta (float, optional): Inverse temperature. Defaults to 1..

    Returns:
        Tuple[Tensor, Tensor]: Visible units, visible magnetizations.
    """
    mv = torch.sigmoid(beta * (vbias + (h @ weight_matrix.T)))
    v = torch.bernoulli(mv)
    return (v, mv)

def sample_state(parallel_chains : Tuple[Tensor, Tensor], params : Tuple[Tensor, Tensor, Tensor], gibbs_steps : int) -> Tuple[Tensor, Tensor]:
    """Generates data sampled from the model by performing gibbs_steps Monte Carlo updates.

    Args:
        parallel_chains (Tuple[Tensor, Tensor]): (v, h) Initial visible state.
        params (Tuple[Tensor, Tensor, Tensor]): (vbias, hbias, weight_matrix) Parameters of the model.
        gibbs_steps (int): Number of Monte Carlo updates.

    Returns:
        Tuple[Tensor, Tensor]: Generated visibles, generated hiddens
    """
    # Unpacking the arguments
    v, h = parallel_chains
    vbias, hbias, weight_matrix = params
    
    for _ in torch.arange(gibbs_steps):
        h, _ = sample_hiddens(v=v, hbias=hbias, weight_matrix=weight_matrix)
        v, _ = sample_visibles(h=h, vbias=vbias, weight_matrix=weight_matrix)
    return (v, h)

def compute_energy(v : Tensor, h : Tensor, params : Tuple[Tensor, Tensor, Tensor]) -> Tensor:
        """Computes the Hamiltonian on the visible (v) and hidden (h) variables.

        Args:
            v (Tensor): Visible units.
            h (Tensor): Hidden units.
            params (Tuple[Tensor, Tensor, Tensor])): (vbias, hbias, weight_matrix) Parameters of the model.

        Returns:
            Tensor: Energy of the data points.
        """
        vbias, hbias, weight_matrix = params
        fields = torch.tensordot(vbias, v, dims=[[0], [1]]) + torch.tensordot(hbias, h, dims=[[0], [1]])
        interaction = torch.multiply(v, torch.tensordot(h, weight_matrix, dims=[[1], [1]])).sum(1)

        return - fields - interaction
    
def compute_energy_visibles(v : Tensor, params : Tuple[Tensor, Tensor, Tensor]) -> Tensor:
    """Returns the energy of the model computed on the input data.

    Args:
        v (Tensor): Visible data.
        params (Tuple[Tensor, Tensor, Tensor]): (vbias, hbias, weight_matrix) Parameters of the model.

    Returns:
        Tensor: Energies of the data points.
    """
    vbias, hbias, weight_matrix = params
    field = v @ vbias
    exponent = hbias + (v @ weight_matrix)
    log_term = torch.where(exponent < 10, torch.log(1. + torch.exp(exponent)), exponent)
    return - field - log_term.sum(1)

def compute_partition_function_AIS(num_chains : int, num_beta : int, params : Tuple[Tensor, Tensor, Tensor], device : torch.device) -> float:
        """Estimates the partition function of the model using Annealed Importance Sampling.

        Args:
            num_chains (int): Number of parallel chains.
            num_beta (int): Number of inverse temperatures that define the trajectories.
            params (Tuple[Tensor, Tensor, Tensor])): (vbias, hbias, weight_matrix) Parameters of the model.
            device (device): device.

        Returns:
            float: Estimate of the log-partition function.
        """
        vbias, hbias, weight_matrix = params
        num_visibles = vbias.shape[0]
        num_hiddens = hbias.shape[0]
        E = torch.zeros(num_chains, device=device, dtype=torch.float64)
        beta_list = np.linspace(0., 1., num_beta)
        dB = 1. / num_beta
        
        # initialize the chains
        vbias0 = torch.zeros(size=(num_visibles,), device=device)
        hbias0 = torch.zeros(size=(num_hiddens,), device=device)
        energy0 = torch.zeros(num_chains, device=device, dtype=torch.float64)
        v = torch.bernoulli(torch.sigmoid(vbias0)).repeat(num_chains, 1)
        h = torch.bernoulli(torch.sigmoid(hbias0)).repeat(num_chains, 1)
        energy1 = compute_energy(v, h, params).type(torch.float64)
        E += energy1 - energy0
        for beta in beta_list:
            h, _ = sample_hiddens(v, hbias, weight_matrix, beta=beta)
            v, _ = sample_visibles(h, vbias, weight_matrix, beta=beta)
            E += compute_energy(v, h, params).type(torch.float64)
   
        # Subtract the average for avoiding overflow
        W = (-dB * E)
        W_ave = W.mean()
        logZ0 = (num_visibles + num_hiddens) * np.log(2)
        logZ = logZ0 + torch.log(torch.mean(torch.exp(W - W_ave))) + W_ave
        return logZ
    
def compute_log_likelihood(v_data : Tensor, params : Tuple[Tensor, Tensor, Tensor], free_energy : Tensor) -> float:
    """Estimates the log-likelihood of the model using pop-MC.

    Args:
        v_data (Tensor): Visible data.
        params (Tuple[Tensor, Tensor, Tensor])): (vbias, hbias, weight_matrix) Parameters of the model.
        free_energy (Tensor): Free energy of the model.

    Returns:
        float: Estimate of the log-likelihood.
    """
    return free_energy - compute_energy_visibles(v_data, params).mean()

def compute_log_likelihood_AIS(v_data : Tensor, params : Tuple[Tensor, Tensor, Tensor],
                               num_chains : int, num_beta : int, device : torch.device) -> float:
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
    return - compute_partition_function_AIS(num_chains, num_beta, params, device) - compute_energy_visibles(v_data, params).mean()