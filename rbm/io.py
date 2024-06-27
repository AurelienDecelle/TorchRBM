import numpy as np
from typing import Tuple, Dict
import torch
import h5py

def load_params(
    fname : str,
    device : torch.device="cpu",
    checkpoint : int=None) -> Dict[str, torch.Tensor]:
    """Load the parameters of a previously saved RBM model.

    Args:
        fname (str): Path to the model.
        device (torch.device, optional): Device. Defaults to "cpu".
        checkpoint (int, optional): Checkpoint to be loaded. If None, the last one is taken. Defaults to None.

    Returns:
        Dict[str, torch.Tensor]: Parameters of the model.
    """
    file_model = h5py.File(fname, 'r+')
    # Automatically selects the last checkpoint if not given
    if not checkpoint:
        chp = 0
        for key in file_model.keys():
            if "update" in key:
                n = int(key.split('_')[1])
                if chp < n:
                    chp = n
    else:
        chp = checkpoint
        
    key_chp = f"update_{chp}"
    
    # Load parameters
    params = {}
    params["weight_matrix"] = torch.tensor(file_model[key_chp]["weight_matrix"][()], device=device)
    params["vbias"] = torch.tensor(file_model[key_chp]["vbias"][()], device=device)
    params["hbias"] = torch.tensor(file_model[key_chp]["hbias"][()], device=device)
    return params


def load_model(
    fname : str,
    checkpoint : int,
    device : torch.device,
    set_rng_state : bool
):
    last_file_key = f"update_{checkpoint}"
    
    chains = {}
    params = {}
    hyperparams = {}
    
    file_model = h5py.File(fname, 'r+')
    
    params["weight_matrix"] = torch.tensor(file_model[last_file_key]["weight_matrix"][()], device=device)
    params["vbias"] = torch.tensor(file_model[last_file_key]["vbias"][()], device=device)
    params["hbias"] = torch.tensor(file_model[last_file_key]["hbias"][()], device=device)
    chains["v"] = torch.tensor(file_model["parallel_chains"][()], device=device)
    # Elapsed time
    start = np.array(file_model[last_file_key]["time"]).item()
    
    hyperparams["batch_size"] = int(file_model["hyperparameters"]["batch_size"][()])
    hyperparams["gibbs_steps"] = int(file_model["hyperparameters"]["gibbs_steps"][()])
    hyperparams["lr"] = float(file_model["hyperparameters"]["learning_rate"][()])
    hyperparams["training_mode"] = file_model["hyperparameters"]["training_mode"][()].decode('utf-8')
    hyperparams["num_chains"] = int(file_model["hyperparameters"]["num_chains"][()])
    hyperparams["num_hiddens"] = int(file_model["hyperparameters"]["num_hiddens"][()])
    
    if set_rng_state:
        torch.set_rng_state(torch.tensor(np.array(file_model[last_file_key]['torch_rng_state'])))
        np_rng_state = tuple([file_model[last_file_key]['numpy_rng_arg0'][()].decode('utf-8'),
                                file_model[last_file_key]['numpy_rng_arg1'][()],
                                file_model[last_file_key]['numpy_rng_arg2'][()],
                                file_model[last_file_key]['numpy_rng_arg3'][()],
                                file_model[last_file_key]['numpy_rng_arg4'][()]])
        np.random.set_state(np_rng_state)
        
    file_model.close()
    
    return chains, params, hyperparams, start  


def save_checkpoint(
    fname : str,
    group_name : str,
    params : Dict[str, torch.Tensor],
    chains : Dict[str, torch.Tensor],
    time : int
):
    file_model = h5py.File(fname, 'r+')
    checkpoint = file_model.create_group(group_name)
    checkpoint["vbias"] = params["vbias"].cpu().numpy()
    checkpoint["hbias"] = params["hbias"].cpu().numpy()
    checkpoint["weight_matrix"] = params["weight_matrix"].cpu().numpy()
    checkpoint["time"] = time
    checkpoint['torch_rng_state'] = torch.get_rng_state()
    checkpoint['numpy_rng_arg0'] = np.random.get_state()[0]
    checkpoint['numpy_rng_arg1'] = np.random.get_state()[1]
    checkpoint['numpy_rng_arg2'] = np.random.get_state()[2]
    checkpoint['numpy_rng_arg3'] = np.random.get_state()[3]
    checkpoint['numpy_rng_arg4'] = np.random.get_state()[4]
    del file_model["parallel_chains"]
    file_model["parallel_chains"] = chains["v"].cpu().numpy()
    file_model.close()