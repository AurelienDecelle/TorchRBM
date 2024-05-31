from typing import Optional, Tuple
import time
from pathlib import Path
from h5py import File
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

Tensor = torch.Tensor

def init_parameters(num_visibles : int,
                    num_hiddens : int,
                    dataset : Tensor,
                    device : torch.device) -> Tuple[Tensor, Tensor, Tensor]:
    """Initialize the parameters of the RBM.

    Args:
        num_visibles (int): Number of visible units.
        num_hiddens (int): Number of hidden units.
        dataset (Tensor): Matrix of data.
        device (torch.device): Device.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: visible bias, hidden bias, weight matrix
    """
    eps = 1e-4
    init_std = 1e-4
    frequencies = torch.tensor(dataset.mean(0))
    frequencies = torch.clamp(frequencies, min=eps, max=(1. - eps))
    vbias = (torch.log(frequencies) - torch.log(1. - frequencies)).to(device)
    hbias = torch.zeros(num_hiddens, device=device)
    weight_matrix = torch.randn(size=(num_visibles, num_hiddens), device=device) * init_std
    return (vbias, hbias, weight_matrix)

def init_parallel_chains(num_chains : int,
                         num_visibles : int,
                         num_hiddens : int,
                         device : torch.device) -> Tuple[Tensor, Tensor]:
    """Initialize the parallel chains of the RBM.

    Args:
        num_chains (int): Number of parallel chains.
        num_visibles (int): Number of visible units.
        num_hiddens (int): Number of hidden units.
        device (torch.device): Device.

    Returns:
        Tuple[Tensor, Tensor]: Initial visible and hidden units.
    """
    v = torch.randint(0, 2, size=(num_chains, num_visibles), device=device).type(torch.float32)
    h = torch.randint(0, 2, size=(num_chains, num_hiddens), device=device).type(torch.float32)
    return (v, h)

def sample_hiddens(v : Tensor, hbias : Tensor, weight_matrix : Tensor) -> Tuple[Tensor, Tensor]:
    """Samples the hidden layer conditioned on the state of the visible layer.

    Args:
        v (Tensor): Visible layer.
        hbias (Tensor): Hidden bias.
        weight_matrix (Tensor): Weight matrix.

    Returns:
        Tuple[Tensor, Tensor]: Hidden units and magnetizations.
    """
    mh = torch.sigmoid(hbias + v @ weight_matrix)
    h = torch.bernoulli(mh)
    return (h, mh)

def sample_visibles(h : Tensor, vbias : Tensor, weight_matrix : Tensor) -> Tensor:
    """Samples the visible layer conditioned on the hidden layer.

    Args:
        h (Tensor): Hidden layer.
        vbias (Tensor): Visible bias.
        weight_matrix (Tensor): Weight matrix.

    Returns:
        Tensor: Visible units.
    """
    mv = torch.sigmoid(vbias + h @ weight_matrix.T)
    v = torch.bernoulli(mv)
    return v

def compute_gradient(data : Tuple[Tensor, Tensor, Tensor], parallel_chains : Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
    """Computes the gradient of the log-likelihood. Implements the centered version of the gradient,
    which normally improveds the quality of the learning.

    Args:
        data (Tuple[Tensor, Tensor, Tensor]): (v, h, data_weights) Observed data.
        parallel_chains (Tuple[Tensor, Tensor]): (v, h) Parallel chains.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: (grad_vbias, grad_hbias, grad_weight_matrix).
    """
    # Unpacking the arguments
    v_data, h_data, w_data = data
    v_gen, h_gen = parallel_chains
    num_chains = v_gen.shape[0]
    
    # Averages over data and generated samples
    v_data_mean = (v_data * w_data).sum(0) / w_data.sum()
    torch.clamp_(v_data_mean, min=1e-4, max=(1. - 1e-4))
    h_data_mean = (h_data * w_data).sum(0) / w_data.sum()
    v_gen_mean = v_gen.mean(0)
    torch.clamp_(v_gen_mean, min=1e-4, max=(1. - 1e-4))
    h_gen_mean = h_gen.mean(0)
    
    # Centered variables
    v_data_centered = v_data - v_data_mean
    h_data_centered = h_data - h_data_mean
    v_gen_centered = v_gen - v_data_mean
    h_gen_centered = h_gen - h_data_mean
    
    # Gradient
    grad_weight_matrix = ((v_data_centered * w_data).T @ h_data_centered) / w_data.sum() - (v_gen_centered.T @ h_gen_centered) / num_chains
    grad_vbias = v_data_mean - v_gen_mean - (grad_weight_matrix @ h_data_mean)
    grad_hbias = h_data_mean - h_gen_mean - (v_data_mean @ grad_weight_matrix)
    return (grad_vbias, grad_hbias, grad_weight_matrix)

def update_parameters(data : Tuple[Tensor, Tensor, Tensor],
                      parallel_chains : Tuple[Tensor, Tensor],
                      params : Tuple[Tensor, Tensor, Tensor],
                      learning_rate : float) -> Tuple[Tensor, Tensor, Tensor]:
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
    grad = compute_gradient(data=data, parallel_chains=parallel_chains)
    # Update the parameters
    vbias_new = params[0] + learning_rate * grad[0]
    hbias_new = params[1] + learning_rate * grad[1]
    weight_matrix_new = params[2] + learning_rate * grad[2]
    params_new = (vbias_new, hbias_new, weight_matrix_new)
    return params_new

def sample_state(parallel_chains : Tuple[Tensor, Tensor], params : Tuple[Tensor, Tensor, Tensor], gibbs_steps : int) -> Tuple[Tensor, Tensor]:
    """Generates data sampled from the model by performing "gibbs_steps" Monte Carlo updates.

    Args:
        parallel_chains (Tuple[Tensor, Tensor]): (v, h) Initial visible state.
        params (Tuple[Tensor, Tensor, Tensor]): (vbias, hbias, weight_matrix) Parameters of the model.
        gibbs_steps (int): Number of Monte Carlo updates.

    Returns:
        Tuple[Tensor, Tensor]: Updated parallel chains.
    """
    v, h = parallel_chains
    vbias, hbias, weight_matrix = params
    for _ in range(gibbs_steps):
        h, _ = sample_hiddens(v=v, hbias=hbias, weight_matrix=weight_matrix)
        v = sample_visibles(h=h, vbias=vbias, weight_matrix=weight_matrix)
    return (v, h)

@torch.jit.script
def fit_batch(batch : Tuple[Tensor, Tensor],
              parallel_chains : Tuple[Tensor, Tensor],
              params : Tuple[Tensor, Tensor, Tensor],
              gibbs_steps : int,
              learning_rate : float) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
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
    _, hbias, weight_matrix = params
    v_data, w_data = batch
    _, h_data = sample_hiddens(v=v_data, hbias=hbias, weight_matrix=weight_matrix)
    data = (v_data, h_data, w_data)

    # Update the parameters
    params = update_parameters(data=data, parallel_chains=parallel_chains, params=params, learning_rate=learning_rate)
    # Update the state of the model
    parallel_chains = sample_state(parallel_chains=parallel_chains, params=params, gibbs_steps=gibbs_steps)
    return (parallel_chains, params)
    
def fit(dataset : Dataset,
        epochs : int,
        num_hiddens : Optional[int] = 100,
        training_mode : Optional[str] = "PCD",
        num_chains : Optional[int] = 500,
        learning_rate : Optional[float] = 0.01,
        batch_size : Optional[int] = 500,
        gibbs_steps : Optional[int] = 50,
        filename : Optional[str] = "RBM.h5",
        checkpoints : Optional[list] = None) -> None:
    """Fits an RBM model on the training data and saves the results in a file.

    Args:
        dataset (Dataset): Training data.
        epochs (int): Number of epochs to be performed.
        num_hiddens (int, optional): Number of hidden units. Defaults to 100.
        training_mode (str, optional): Training scheme to be used. Available options are "PCD", "CD" and "Rdm". Defaults to "PCD".
        num_chains (int, optional): Number of parallel chains. Defaults to 500.
        learning_rate (float, optional): Learning rate. Defaults to 0.01.
        batch_size (int, optional): Batch size. Defaults to 500.
        gibbs_steps (int, optional): Maximum number of Monte Carlo steps to update the state of the model. Defaults to 50.
        filename (str, optional): Path of the file where to store the trained model. Defaults to "RBM.h5".
        checkpoints (list, optional): List of epochs at which storing the model. Defaults to None.
    """
    # Setup the device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Load the data
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2, drop_last=True)
    num_visibles = dataset.get_num_visibles()
    
    # Validate checkpoints
    if checkpoints is None:
        checkpoints = [epochs]
    checkpoints = list(checkpoints)
    
    # Create file for saving the model
    filename = Path(filename)
    filename = filename.parent / Path(filename.name)
    file_model = File(filename, "w")
    hyperparameters = file_model.create_group("hyperparameters")
    hyperparameters["epochs"] = epochs
    hyperparameters["num_hiddens"] = num_hiddens
    hyperparameters["num_visibles"] = num_visibles
    hyperparameters["training_mode"] = training_mode
    hyperparameters["num_chains"] = num_chains
    hyperparameters["batch_size"] = batch_size
    hyperparameters["gibbs_steps"] = gibbs_steps
    hyperparameters["learning_rate"] = learning_rate
    hyperparameters["filename"] = str(filename)
    params = init_parameters(num_visibles=num_visibles, num_hiddens=num_hiddens, dataset=dataset.data, device=device)
    parallel_chains = init_parallel_chains(num_chains=num_chains, num_visibles=num_visibles, num_hiddens=num_hiddens, device=device)
    file_model["parallel_chains"] = parallel_chains[0].cpu().numpy()
    file_model.close()
    
    # Training the model
    pbar = tqdm(initial=0, total=epochs, colour="red", dynamic_ncols=True, ascii="-#")
    pbar.set_description("Training RBM")
    num_updates = 0
    while num_updates < epochs:
        for batch in dataloader:
            num_updates += 1
            batch = (batch[0].to(device), batch[1].to(device))
            if training_mode == "Rdm":
                parallel_chains = init_parallel_chains(num_chains=num_chains, num_visibles=num_visibles, num_hiddens=num_hiddens, device=device)
            elif training_mode == "CD":
                parallel_chains_v = batch[0].to(device)
                _, h = sample_hiddens(parallel_chains_v, hbias, weight_matrix)
                parallel_chains = (parallel_chains_v, h)
            parallel_chains, params = fit_batch(batch=batch, parallel_chains=parallel_chains, params=params, gibbs_steps=gibbs_steps, learning_rate=learning_rate)
            pbar.update(1)
            
        # Save the model if a checkpoint is reached
        if num_updates in checkpoints:
            vbias, hbias, weight_matrix = params
            file_model = File(filename, "r+")
            checkpoint = file_model.create_group(f"epoch_{num_updates}")
            checkpoint["vbias"] = vbias.cpu().numpy()
            checkpoint["hbias"] = hbias.cpu().numpy()
            checkpoint["weight_matrix"] = weight_matrix.cpu().numpy()
            checkpoint['torch_rng_state'] = torch.get_rng_state()
            checkpoint['numpy_rng_arg0'] = np.random.get_state()[0]
            checkpoint['numpy_rng_arg1'] = np.random.get_state()[1]
            checkpoint['numpy_rng_arg2'] = np.random.get_state()[2]
            checkpoint['numpy_rng_arg3'] = np.random.get_state()[3]
            checkpoint['numpy_rng_arg4'] = np.random.get_state()[4]
            del file_model["parallel_chains"]
            file_model["parallel_chains"] = parallel_chains[0].cpu().numpy()
            file_model.close()
       
 
def restore_training(filename : str,
                     dataset : Dataset,
                     epochs : int,
                     checkpoints : Optional[list] = None):
    """Ripristinate an interrupted training.

    Args:
        filename (str): Path of the file containing the model.
        dataset (Dataset): Training dataset.
        epochs (int): New total number of epochs to be reached.
        checkpoints (list, optional): List of epochs at which saving the model. Defaults to None.

    Raises:
        RuntimeError: If the specified number of epochs is smaller than the one already reached in a previous training.
    """
    # Setup the device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    # Retrieve the the number of training epochs already performed on the model
    file_model = File(filename, 'r+')
    num_epochs = 0
    for file_key in file_model.keys():
        if "epoch" in file_key:
            epoch = int(file_key.split('_')[1])
            if num_epochs < epoch:
                num_epochs = epoch
    if epochs <= num_epochs:
        raise RuntimeError(f"The parameter /'epochs/' ({epochs}) must be greater than the previous number of epochs ({num_epochs}).")
    last_file_key = f"epoch_{num_epochs}"
    
    # Load the last checkpoint
    torch.set_rng_state(torch.tensor(np.array(file_model[last_file_key]['torch_rng_state'])))
    np_rng_state = tuple([file_model[last_file_key]['numpy_rng_arg0'][()].decode('utf-8'),
                            file_model[last_file_key]['numpy_rng_arg1'][()],
                            file_model[last_file_key]['numpy_rng_arg2'][()],
                            file_model[last_file_key]['numpy_rng_arg3'][()],
                            file_model[last_file_key]['numpy_rng_arg4'][()]])
    np.random.set_state(np_rng_state)
    weight_matrix = torch.tensor(file_model[last_file_key]["weight_matrix"][()], device=device)
    vbias = torch.tensor(file_model[last_file_key]["vbias"][()], device=device)
    hbias = torch.tensor(file_model[last_file_key]["hbias"][()], device=device)
    parallel_chains_v = torch.tensor(file_model["parallel_chains"][()], device=device)
    batch_size = int(file_model["hyperparameters"]["batch_size"][()])
    gibbs_steps = int(file_model["hyperparameters"]["gibbs_steps"][()])
    learning_rate = float(file_model["hyperparameters"]["learning_rate"][()])
    training_mode = file_model["hyperparameters"]["training_mode"][()].decode('utf-8')
    num_chains = int(file_model["hyperparameters"]["num_chains"][()])
    num_hiddens = int(file_model["hyperparameters"]["num_hiddens"][()])
    num_updates = num_epochs
    file_model.close()
    
    # Initialize the chains, import the data
    params = (vbias, hbias, weight_matrix)
    _, parallel_chains_h = sample_hiddens(parallel_chains_v, hbias, weight_matrix)
    parallel_chains = (parallel_chains_v, parallel_chains_h)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2, drop_last=True)
    num_visibles = dataset.get_num_visibles()
    
    # Validate checkpoints
    if checkpoints is None:
        checkpoints = [epochs]
    checkpoints = list(checkpoints)
    
    # Continue the training
    pbar = tqdm(initial=num_epochs, total=epochs, colour="red", dynamic_ncols=True, ascii="-#")
    pbar.set_description("Training RBM")
    while num_updates < epochs:
        for batch in dataloader:
            num_updates += 1
            batch = (batch[0].to(device), batch[1].to(device))
            if training_mode == "Rdm":
                parallel_chains = init_parallel_chains(num_chains=num_chains, num_visibles=num_visibles, num_hiddens=num_hiddens, device=device)
            elif training_mode == "CD":
                parallel_chains_v = batch[0].to(device)
                _, h = sample_hiddens(parallel_chains_v, hbias, weight_matrix)
                parallel_chains = (parallel_chains_v, h)
            parallel_chains, params = fit_batch(batch=batch, parallel_chains=parallel_chains, params=params, gibbs_steps=gibbs_steps, learning_rate=learning_rate)
            pbar.update(1)
            
        # Save the model if a checkpoint is reached
        if num_updates in checkpoints:
            vbias, hbias, weight_matrix = params
            file_model = File(filename, 'r+')
            checkpoint = file_model.create_group(f"epoch_{num_updates}")
            checkpoint["vbias"] = vbias.cpu().numpy()
            checkpoint["hbias"] = hbias.cpu().numpy()
            checkpoint["weight_matrix"] = weight_matrix.cpu().numpy()
            checkpoint['torch_rng_state'] = torch.get_rng_state()
            checkpoint['numpy_rng_arg0'] = np.random.get_state()[0]
            checkpoint['numpy_rng_arg1'] = np.random.get_state()[1]
            checkpoint['numpy_rng_arg2'] = np.random.get_state()[2]
            checkpoint['numpy_rng_arg3'] = np.random.get_state()[3]
            checkpoint['numpy_rng_arg4'] = np.random.get_state()[4]
            del file_model["parallel_chains"]
            file_model["parallel_chains"] = parallel_chains[0].cpu().numpy()
            file_model.close()    
    
    file_model = File(filename, 'r+')
    del file_model["hyperparameters"]["epochs"]
    file_model["hyperparameters"]["epochs"] = epoch
    file_model.close()