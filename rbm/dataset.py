from typing import Union, Dict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import torch
import rbm.fasta_utils as fasta_utils

class DatasetRBM(Dataset):
    def __init__(self,
                 path_data : Union[str, Path],
                 alphabet : str="protein",
                 compute_weights : bool=False,
                 th : float=0.8,
                 device : torch.device="cpu"):
        """Initialize the dataset.

        Args:
            path_data (Union[str, Path]): Path to the file (plain text or fasta).
            alphabet (str, optional): Selects the type of encoding of the sequences. Default choices are ("protein", "rna", "dna"). Defaults to "protein".
            compute_weights (bool, optional): Whether to assign weights to the imput data. The weight of each data is 1 / n_clust, where 'n_clust' is the number of sequences
                                              that have a sequence identity with 's' >= th. Defaults to False.
            th (float, optional) : Sequence identity threshold for computing the weights of the sequences. Defaults to 0.8.
            device (torch.device, optional): Device.
        """
        self.names = []
        self.data = []
        self.tokens = None # Only needed for protein sequence data
        self.device = device
        
        # Automatically detects if the file is in fasta format and imports the data
        with open(path_data, "r") as f:
            first_line = f.readline()
        if first_line.startswith(">"):
            # Select the proper encoding
            self.tokens = fasta_utils.get_tokens(alphabet)
            names, sequences = fasta_utils.import_from_fasta(path_data)
            fasta_utils.validate_alphabet(sequences=sequences, tokens=self.tokens)
            self.names = np.array(names)
            self.data = np.vectorize(fasta_utils.encode_sequence, excluded=["tokens"], signature="(), () -> (n)")(sequences, self.tokens)
        else:
            self.data = np.loadtxt(path_data, dtype=np.float32)
            self.names = np.arange(len(self.data))
        num_data = len(self.data)
        
        # Computes the weights to be assigned to the data
        if compute_weights:
            print("Automatically computing the sequence weights...")
            self.weights = fasta_utils.compute_weights(data=self.data, th=th, device=self.device)
        else:
            self.weights = np.ones((num_data, 1), dtype=np.float32)
            
        self.num_states = self.data.max() + 1
        
        # Shuffle the data
        perm = np.random.permutation(num_data)
        self.data = self.data[perm]
        self.names = self.names[perm]
        self.weights = self.weights[perm]
        
        # Binary or categorical dataset?
        if self.num_states == 2:
            self.is_binary = True
            self.dtype = torch.float32
        else:
            self.is_binary = False
            self.dtype = torch.int32
            
        # Load the data on the device
        self.data = torch.from_numpy(self.data).to(self.device).to(self.dtype)
        self.weights = torch.from_numpy(self.weights).to(self.device).to(self.dtype)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        return {
            "v": self.data[index],
            "weights": self.weights[index],
        }
    
    def get_num_visibles(self) -> int:
        return self.data.shape[1]
    
    def get_num_states(self) -> int:
        return self.num_states
    
    def get_effective_size(self) -> int:
        return int(self.weights.sum())