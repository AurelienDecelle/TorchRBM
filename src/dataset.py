from typing import Union, List, Any
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import torch
import utils

class DatasetRBM(Dataset):
    def __init__(self,
                 path_data : Union[str, Path],
                 alphabet : str="protein",
                 compute_weights : bool=False,
                 th : float=0.8):
        """Initialize the dataset.

        Args:
            path_data (Union[str, Path]): Path to the file (plain text or fasta).
            alphabet (str, optional): Selects the type of encoding of the sequences. Default choices are ("protein", "rna", "dna"). Defaults to "protein".
            compute_weights (bool, optional): Whether to assign weights to the imput data. The weight of each data is 1 / n_clust, where 'n_clust' is the number of sequences
                                              that have a sequence identity with 's' >= th. Defaults to False.
            th (float, optional) : Sequence identity threshold for computing the weights of the sequences. Defaults to 0.8.
        """
        self.names = []
        self.data = []
        self.tokens = None # Only needed for protein sequence data
        
        # Automatically detects if the file is in fasta format and imports the data
        with open(path_data, "r") as f:
            first_line = f.readline()
        if first_line.startswith(">"):
            # Select the proper encoding
            self.tokens = utils.get_tokens(alphabet)
            names, sequences = utils.import_from_fasta(path_data)
            utils.validate_alphabet(sequences=sequences, tokens=self.tokens)
            self.names = np.array(names)
            self.data = np.vectorize(utils.encode_sequence, excluded=["tokens"], signature="(), () -> (n)")(sequences, self.tokens)
        else:
            with open(path_data, "r") as f:
                for line in f:
                    self.data.append(line.strip().split())
            self.data = np.array(self.data, dtype=np.float32)
            self.names = np.arange(len(self.data))
        num_data = len(self.data)
        
        # Computes the weights to be assigned to the data
        if compute_weights:
            print("Automatically computing the sequence weights...")
            self.weights = utils.compute_weights(data=self.data, th=th)
        else:
            self.weights = np.ones((num_data, 1), dtype=np.float32)
        
        # Shuffle the data
        perm = np.random.permutation(num_data)
        self.data = self.data[perm]
        self.names = self.names[perm]
        self.weights = self.weights[perm]
        
        # Binary or categorical dataset?
        if self.get_num_states() == 2:
            self.is_binary = True
        else:
            self.is_binary = False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx : int) -> Any:
        visible = torch.from_numpy(self.data[idx])
        weight = torch.from_numpy(self.weights[idx])
        return (visible, weight)
    
    def get_num_visibles(self) -> int:
        return self.data.shape[1]
    
    def get_num_states(self) -> int:
        return np.max(self.data) + 1
    
    def get_effective_size(self) -> int:
        return int(self.weights.sum())
    
    def get_covariance_matrix(self, device : torch.device=torch.device("cpu"), num_data : int=None) -> torch.Tensor:
        """Returns the covariance matrix of the data. If path_clu was specified, the weighted covariance matrix is computed.

        Args:
            device (torch.device, optional): Device. Defaults to torch.device("cpu").
            num_data (int, optional): Number of data to extract for computing the covariance matrix.

        Returns:
            Tensor: Covariance matrix of the dataset.
        """
        num_states = self.get_num_states()
        num_visibles = self.get_num_visibles()
        if num_data is not None:
            idxs = torch.multinomial(input=(torch.ones(self.__len__(), device=device) / self.__len__()), num_samples=num_data, replacement=False)
            data = torch.tensor(self.data[idxs], device=device)
            weights = torch.tensor(self.weights[idxs], device=device, dtype=torch.float32)
        else:
            data = torch.tensor(self.data, device=device)
            weights = torch.tensor(self.weights, device=device, dtype=torch.float32)
        if num_states != 2:
            data_oh = torch.eye(num_states, device=device)[data].reshape(-1, num_states * num_visibles).float()
        else:
            data_oh = data
        norm_weights = weights.reshape(-1, 1) / weights.sum()
        data_mean = (data_oh * norm_weights).sum(0, keepdim=True)
        cov_matrix = ((data_oh * norm_weights).mT @ data_oh) - (data_mean.mT @ data_mean)
        return cov_matrix