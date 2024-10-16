import time
from pathlib import Path
from h5py import File
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from itertools import cycle

from rbm.potts.functions import init_chains, init_parameters
from rbm.potts.pcd import fit_batch
from rbm.potts.functions import init_chains, init_parameters
from rbm.potts.sampling import sample_hiddens
from rbm.io import save_checkpoint, load_model
from rbm.utils import get_checkpoints

  
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
    centered : bool,
    pseudocount: bool,
    checkpoints : list,
    device : torch.device
    ) -> None:
    """Fits an RBM model on the training data and saves the results in a file.

    Args:
        filename (str): Path of the file where to store the trained model.
        dataset (Dataset): Training data.
        num_updates (int): Number of gradient updates to be performed.
        num_hiddens (int): Number of hidden units.
        training_mode (str): Training scheme to be used. Available options are "PCD", "CD" and "Rdm".
        num_chains (int): Number of Monte Carlo chains.
        lr (float): Learning rate.
        batch_size (int): Batch size.
        gibbs_steps (int): Number of Monte Carlo steps for updating the chains.
        checkpoints (list): List of checkpoints at which storing the model.
        device (torch.device): Device.
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
    t_start = time.time()

    with torch.no_grad():
        for upd in range(1, num_updates + 1):
            
            if upd % 100 == 0:
                pbar.update(100)
                
            batch = next(dataloader)
            
            if training_mode == "Rdm":
                chains = init_chains(
                    num_chains=num_chains,
                    num_visibles=num_visibles,
                    num_hiddens=num_hiddens,
                    num_colors=num_colors,
                    device=device
                )
                
            elif training_mode == "CD":
                chains = {k : c.clone() for k, c in batch.items()}
                chains = sample_hiddens(chains=chains, params=params)
                
            chains, params = fit_batch(
                data=batch,
                chains=chains,
                params=params,
                gibbs_steps=gibbs_steps,
                lr=lr,
                centered=centered,
                pseudocount=pseudocount
            )
            
            if upd in checkpoints:
                save_checkpoint(
                    fname=filename,
                    group_name=f"update_{upd}",
                    params=params,
                    chains=chains,
                    time=time.time() - t_start
                )
       
 
def restore_training(
    filename : str,
    dataset : Dataset,
    num_updates : int,
    checkpoints : list,
    centered : bool,
    pseudocount: bool,
    device : torch.device):
    """Restores a training from the last saved checkpoint.

    Args:
        filename (str): Path of the file where to store the trained model.
        dataset (Dataset): Training data.
        num_updates (int): Number of gradient updates to be performed.
        checkpoints (list): List of checkpoints at which storing the model.
        device (torch.device): Device.
    """
        
    # Retrieve the the number of training updates already performed on the model
    last_checkpoint = get_checkpoints(filename)[-1]
    if num_updates <= last_checkpoint:
        raise RuntimeError(f"The parameter /'num_updates/' ({num_updates}) must be greater than the previous number of updates ({last_checkpoint}).")
    
    
    # Load the last checkpoint
    chains, params, hyperparams, elapsed_time = load_model(fname=filename, checkpoint=last_checkpoint, device=device, set_rng_state=True)
    chains = sample_hiddens(chains=chains, params=params)
    
    dataloader = DataLoader(dataset, batch_size=hyperparams["batch_size"], shuffle=True, drop_last=True)
    dataloader = cycle(dataloader)
    num_visibles = dataset.get_num_visibles()
    num_colors = dataset.get_num_states()
    
    # Validate checkpoints
    if checkpoints is None:
        checkpoints = [num_updates]
    checkpoints = list(checkpoints)
    
    # Continue the training
    pbar = tqdm(initial=last_checkpoint, total=num_updates, colour="red", dynamic_ncols=True, ascii="-#")
    pbar.set_description("Training RBM")
    t_start = time.time()
    
    with torch.no_grad():
        for upd in range(last_checkpoint + 1, num_updates + 1):
            
            if upd % 100 == 0:
                pbar.update(100)
                
            batch = next(dataloader)
            
            if hyperparams["training_mode"] == "Rdm":
                chains = init_chains(
                    num_chains=hyperparams["num_chains"],
                    num_visibles=num_visibles,
                    num_hiddens=hyperparams["num_hiddens"],
                    num_colors=num_colors,
                    device=device
                )
            
            elif hyperparams["training_mode"] == "CD":
                chains = {k : c.clone() for k, c in batch.items()}
                chains = sample_hiddens(chains=chains, params=params)
                
            chains, params = fit_batch(
                data=batch,
                chains=chains,
                params=params,
                gibbs_steps=hyperparams["gibbs_steps"],
                lr=hyperparams["lr"],
                centered=centered,
                pseudocount=pseudocount,
            )
            
            if upd in checkpoints:
                save_checkpoint(
                    fname=filename,
                    group_name=f"update_{upd}",
                    params=params,
                    chains=chains,
                    time=time.time() - t_start + elapsed_time
                )
