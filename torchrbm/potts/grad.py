import torch
from typing import Dict
from torchrbm.custom_fn import one_hot


def compute_gradient(
    data: Dict[str, torch.Tensor],
    chains: Dict[str, torch.Tensor],
    num_colors : int,
    centered: bool = True,
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
    # One-hot representation of the data
    v_data_one_hot = one_hot(data["v"], num_classes=num_colors)
    v_gen_one_hot = one_hot(chains["v"], num_classes=num_colors)
    
    # Averages over data and generated samples
    v_data_mean = (v_data_one_hot * data["weights"].unsqueeze(-1)).sum(0) / data["weights"].sum()
    torch.clamp_(v_data_mean, min=1e-4, max=(1. - 1e-4))
    h_data_mean = (data["mh"] * data["weights"]).sum(0) / data["weights"].sum()
    v_gen_mean = v_gen_one_hot.mean(0)
    torch.clamp_(v_gen_mean, min=1e-4, max=(1. - 1e-4))
    h_gen_mean = chains["h"].mean(0)
    grad = {}
    
    if centered:
        # Centered variables
        v_data_centered = v_data_one_hot - v_data_mean
        h_data_centered = data["mh"] - h_data_mean
        v_gen_centered = v_gen_one_hot - v_data_mean
        h_gen_centered = chains["h"] - h_data_mean

        # Gradient
        grad["weight_matrix"] = (
            torch.tensordot(
                (v_data_centered * data["weights"].unsqueeze(-1)),
                h_data_centered,
                dims=[[0], [0]],
            )
            / data["weights"].sum()
            - torch.tensordot(v_gen_centered, h_gen_centered, dims=[[0], [0]])
            / num_chains
        )
        grad["vbias"] = v_data_mean - v_gen_mean - (grad["weight_matrix"] @ h_data_mean)
        grad["hbias"] = (
            h_data_mean
            - h_gen_mean
            - torch.tensordot(v_data_mean, grad["weight_matrix"], dims=[[0, 1], [0, 1]])
        )
        
    else:
        # Gradient
        grad["weight_matrix"] = (
            torch.tensordot(
                (v_data_one_hot * data["weights"].unsqueeze(-1)),
                data["mh"],
                dims=[[0], [0]],
            )
            / data["weights"].sum()
            - torch.tensordot(v_gen_one_hot, chains["h"], dims=[[0], [0]])
            / num_chains
        )
        grad["vbias"] = v_data_mean - v_gen_mean
        grad["hbias"] = h_data_mean - h_gen_mean
    
    return grad