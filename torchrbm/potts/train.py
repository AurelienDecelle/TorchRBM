import time
from pathlib import Path
from h5py import File
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from itertools import cycle

from torchrbm.potts.functions import init_chains, init_parameters
from torchrbm.potts.pcd import fit_batch
from torchrbm.potts.functions import init_chains, init_parameters
from torchrbm.io import save_checkpoint, load_model
from torchrbm.utils import get_checkpoints

  
def train(
    filename : str,
    dataset : Dataset,
    num_updates : int,
    num_hiddens : int,
    training_mode : str,
    num_chains : int,
    lr : float,
    batch_size : int,
    gibbs_steps : int,
    checkpoints : list,
    device : torch.device
    ) -> None:
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
    
    # Load the data
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    dataloader = cycle(dataloader)
    num_visibles = dataset.get_num_visibles()
    num_colors = dataset.get_num_states()
    
    # Validate checkpoints
    if checkpoints is None:
        checkpoints = [num_updates]
    checkpoints = list(checkpoints)
    
    # Initialize parameters and chains
    params = init_parameters(data=dataset.data, num_visibles=num_visibles, num_hiddens=num_hiddens, device=device)
    chains = init_chains(num_chains=num_chains, num_visibles=num_visibles, num_hiddens=num_hiddens, num_colors=num_colors, device=device)
    
    # Create file for saving the model
    filename = Path(filename)
    filename = filename.parent / Path(filename.name)
    file_model = File(filename, "w")
    hyperparameters = file_model.create_group("hyperparameters")
    hyperparameters["num_hiddens"] = num_hiddens
    hyperparameters["num_visibles"] = num_visibles
    hyperparameters["training_mode"] = training_mode
    hyperparameters["num_chains"] = num_chains
    hyperparameters["batch_size"] = batch_size
    hyperparameters["gibbs_steps"] = gibbs_steps
    hyperparameters["learning_rate"] = lr
    file_model["parallel_chains"] = chains["v"].cpu().numpy()
    file_model.close()
    
    # Training the model
    pbar = tqdm(initial=0, total=num_updates, colour="red", dynamic_ncols=True, ascii="-#")
    pbar.set_description("Training RBM")

    with torch.no_grad():
        for upd in range(1, num_updates + 1):
            pbar.update(1)
            batch = next(dataloader)
            if training_mode == "Rdm":
                chains = init_chains(num_chains=num_chains, num_visibles=num_visibles, num_hiddens=num_hiddens, num_colors=num_colors, device=device)
            elif training_mode == "CD":
                chains = batch.clone()
            chains, params = fit_batch(
                data=batch,
                chains=chains,
                params=params,
                gibbs_steps=gibbs_steps,
                lr=lr,
                centered=True
            )
            
            #from torch.linalg import svdvals
            #weight_matrix = params["weight_matrix"].reshape(-1, params["weight_matrix"].shape[-1])
            #eig = svdvals(weight_matrix)
            #print(eig)
            
            if upd in checkpoints:
                save_checkpoint(
                    fname=filename,
                    group_name=f"update_{upd}",
                    params=params,
                    chains=chains
                )
       
 
def restore_training(
    filename : str,
    dataset : Dataset,
    num_updates : int,
    checkpoints : list,
    device : torch.device):
    """Ripristinate an interrupted training.

    Args:
        filename (str): Path of the file containing the model.
        dataset (Dataset): Training dataset.
        epochs (int): New total number of epochs to be reached.
        checkpoints (list, optional): List of epochs at which saving the model. Defaults to None.

    Raises:
        RuntimeError: If the specified number of epochs is smaller than the one already reached in a previous training.
    """
        
    # Retrieve the the number of training updates already performed on the model
    last_checkpoint = get_checkpoints(filename)[-1]
    if num_updates <= last_checkpoint:
        raise RuntimeError(f"The parameter /'num_updates/' ({num_updates}) must be greater than the previous number of updates ({updates_done}).")
    
    
    # Load the last checkpoint
    chains, params, hyperparams = load_model(fname=filename, checkpoint=last_checkpoint, device=device, set_rng_state=True)
    
    dataloader = DataLoader(dataset, batch_size=hyperparams["batch_size"], shuffle=True, drop_last=True)
    num_visibles = dataset.get_num_visibles()
    num_colors = dataset.get_num_states()
    
    # Validate checkpoints
    if checkpoints is None:
        checkpoints = [num_updates]
    checkpoints = list(checkpoints)
    
    # Continue the training
    pbar = tqdm(initial=last_checkpoint, total=num_updates, colour="red", dynamic_ncols=True, ascii="-#")
    pbar.set_description("Training RBM")
    
    with torch.no_grad():
        for upd in range(last_checkpoint + 1, num_updates + 1):
            pbar.update(1)
            batch = next(dataloader)
            if hyperparams["training_mode"] == "Rdm":
                chains = init_chains(num_chains=hyperparams["num_chains"], num_visibles=num_visibles, num_hiddens=hyperparams["num_hiddens"], num_colors=num_colors)
            elif hyperparams["training_mode"] == "CD":
                chains = batch.clone()
            chains, params = fit_batch(
                data=batch,
                chains=chains,
                params=params,
                gibbs_steps=hyperparams["gibbs_steps"],
                lr=hyperparams["lr"],
                centered=True
            )
            if upd in checkpoints:
                save_checkpoint(
                    fname=filename,
                    group_name=f"update_{upd}",
                    params=params,
                    chains=chains
                )