import torch
from typing import Dict


def compute_gradient(
    data: Dict[str, torch.Tensor],
    chains: Dict[str, torch.Tensor],
    centered: bool = True,
    pseudocount: bool= False,
) -> Dict[str, torch.Tensor]:
    """Computes the gradient of the log-likelihood. By default implements the centered version of the gradient,
    which normally improveds the quality of the learning.

    Args:
        data (Dict[str, torch.Tensor]): Observed data.
        chains (Dict[str, torch.Tensor]): Monte Carlo chains.
        params (Dict[str, torch.Tensor]): Parameters of the model.
        centered (bool, optional): Whether to center the gradient or not. Defaults to True. 

    Returns:
        Dict[str, torch.Tensor]: Gradient of the log-likelihood.
    """
    num_chains = len(chains["v"])

    if pseudocount:
        device=data["v"].device
        old_values=data["v"].clone() # to perturbe for the pseudocount
        Neff=data["weights"].sum()
        random_probs = torch.rand(data["v"].shape,device=device)
        change_mask = random_probs < 1./Neff/num_colors
        new_values = torch.randint(0, num_colors, data["v"].shape,device=device)

        data_new = torch.where(change_mask,new_values,old_values)
    else:
        data_new=data["v"].clone()
    
    # Averages over data and generated samples
    v_data_mean = (data_new * data["weights"]).sum(0) / data["weights"].sum()
    torch.clamp_(v_data_mean, min=1e-4, max=(1. - 1e-4))
    h_data_mean = (data["mh"] * data["weights"]).sum(0) / data["weights"].sum()
    v_gen_mean = chains["v"].mean(0)
    torch.clamp_(v_gen_mean, min=1e-4, max=(1. - 1e-4))
    h_gen_mean = chains["h"].mean(0)
    grad = {}
    
    if centered:
        # Centered variables
        v_data_centered =data_new- v_data_mean
        h_data_centered = data["mh"] - h_data_mean
        v_gen_centered = chains["v"] - v_data_mean
        h_gen_centered = chains["h"] - h_data_mean

        # Gradient
        grad["weight_matrix"] = ((v_data_centered * data["weights"]).T @ h_data_centered) / data["weights"].sum() \
            - (v_gen_centered.T @ h_gen_centered) / num_chains
        grad["vbias"] = v_data_mean - v_gen_mean - (grad["weight_matrix"] @ h_data_mean)
        grad["hbias"] = h_data_mean - h_gen_mean - (v_data_mean @ grad["weight_matrix"])
        
    else:
        # Gradient
        grad["weight_matrix"] = ((data["v"] @ data["weights"]).T @ data["mh"]) / data["weights"].sum() \
            - (chains["v"].T @ chains["h"]) / num_chains
        grad["vbias"] = v_data_mean - v_gen_mean
        grad["hbias"] = h_data_mean - h_gen_mean
    
    return grad